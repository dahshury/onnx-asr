"""FunAudioLLM SenseVoice multilingual CTC ASR (sherpa-onnx / FunASR export).

SenseVoice is a CTC-based multilingual model from FunAudioLLM / FunASR with
first-class support for Mandarin (zh), English (en), Japanese (ja), Korean
(ko) and Cantonese (yue). It does not fit the shared kaldi/nemo preprocessor
abstractions — it needs an 80-mel FBANK followed by Low-Frame-Rate (LFR)
stacking and a model-supplied CMVN — so this class runs its own NumPy
front-end (``_preprocessor_name == "identity"``: the resolver hands raw
waveforms straight through) and feeds the stacked features to the ONNX graph.

Two graph shapes are handled, auto-detected from metadata / input arity:

* **Full SenseVoice** (sherpa-onnx export) — four inputs
  ``(feat, x_length, language, text_norm)``; the model prepends four control
  tokens (language / emotion / event / itn) which we strip on decode. CMVN
  ``neg_mean`` / ``inv_stddev`` ride in the ONNX ``custom_metadata_map``.
* **FunASR Nano** — single ``feat`` input, base64-encoded ``tokens.txt``,
  no control tokens, no CMVN.

Files: ``model.onnx`` / ``model.int8.onnx`` + ``tokens.txt`` at the repo root
(no ``config.json``). Greedy CTC decode is done inline (the control-token
strip + base64 vocab don't fit the shared ``_AsrWithCtcDecoding`` path).

Ported verbatim (for numerical parity) from WinSTT's standalone SenseVoice
adapter, which itself follows transcribe-rs' ``sense_voice_mod.rs``.
"""

from __future__ import annotations

import base64
import binascii
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import onnxruntime as rt

from onnx_asr.asr import BaseAsr, ModelCapabilities, Preprocessor, TimestampedResult
from onnx_asr.onnx import OnnxSessionOptions

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)

# ── FBANK / LFR constants (Kaldi compute-fbank-feats, wespeaker profile) ──
_SAMPLE_RATE = 16_000
_NUM_MELS = 80
_N_FFT = 400
_HOP_LENGTH = 160
_WIN_LENGTH = 400
_PRE_EMPHASIS = 0.97
_F_MIN = 20.0

#: 4 control tokens (language / emotion / event / itn) prepended by the full
#: model before the transcript proper.
_NUM_CONTROL_TOKENS = 4

#: lang code → model language id, when metadata carries no ``lang_*`` table.
_DEFAULT_LANG_IDS: dict[str, int] = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12}
_LANGUAGE_KEY_MAP: dict[str, str] = {
    "auto": "auto",
    "": "auto",
    "zh": "zh",
    "zh-Hans": "zh",
    "zh-Hant": "zh",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "yue": "yue",
}
_DEFAULT_WITH_ITN_ID = 14
_DEFAULT_WITHOUT_ITN_ID = 15


def _build_mel_filterbank() -> npt.NDArray[np.float32]:
    """HTK-style triangular mel filterbank, shape ``(n_fft//2 + 1, n_mels)``."""
    n_freqs = _N_FFT // 2 + 1
    fmax = _SAMPLE_RATE / 2.0
    all_freqs = np.linspace(0.0, _SAMPLE_RATE / 2.0, n_freqs)
    m_min = 2595.0 * np.log10(1.0 + _F_MIN / 700.0)
    m_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    m_pts = np.linspace(m_min, m_max, _NUM_MELS + 2)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    f_diff = np.diff(f_pts)
    slopes = f_pts[None, :] - all_freqs[:, None]
    down_slopes = -slopes[:, :-2] / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    fb = np.maximum(np.zeros_like(down_slopes), np.minimum(down_slopes, up_slopes))
    return fb.astype(np.float32)


def _compute_fbank(samples: npt.NDArray[np.float32], fbanks: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """80-mel log-magnitude FBANK with Hamming window + pre-emphasis 0.97.

    snip_edges=True (no padding): ``T = 1 + (N - win) // hop``. Returns a
    ``(T, n_mels)`` float32 array.
    """
    if samples.size < _WIN_LENGTH:
        return np.zeros((0, _NUM_MELS), dtype=np.float32)
    num_frames = 1 + (samples.size - _WIN_LENGTH) // _HOP_LENGTH
    strided = np.lib.stride_tricks.sliding_window_view(samples, _WIN_LENGTH)[::_HOP_LENGTH][:num_frames]
    strided = strided.astype(np.float32, copy=True)
    if _PRE_EMPHASIS != 0.0:
        offset = np.pad(strided, ((0, 0), (1, 0)), mode="edge")
        strided = strided - _PRE_EMPHASIS * offset[..., :-1]
    window = np.hamming(_WIN_LENGTH).astype(np.float32)
    strided = strided * window
    spectrum = np.abs(np.fft.rfft(strided, _N_FFT)).astype(np.float32) ** 2
    mel_energies = np.matmul(spectrum, fbanks)
    eps = float(np.finfo(np.float32).eps)
    result: npt.NDArray[np.float32] = np.log(np.maximum(mel_energies, eps)).astype(np.float32, copy=False)
    return result


def _apply_lfr(features: npt.NDArray[np.float32], window_size: int, window_shift: int) -> npt.NDArray[np.float32]:
    """Low-Frame-Rate stacking — ``window_size`` frames per row, step ``window_shift``.

    The final partial window is right-padded with its own last frame (FunASR
    ``apply_lfr`` behaviour), so a non-empty input always emits ≥1 row.
    """
    if features.shape[0] == 0:
        return np.zeros((0, features.shape[1] * window_size), dtype=np.float32)
    in_frames, mel_dim = features.shape
    out_frames = max(1, 1 + (in_frames - 1) // window_shift)
    out = np.zeros((out_frames, mel_dim * window_size), dtype=np.float32)
    for i in range(out_frames):
        start = i * window_shift
        end = start + window_size
        if end <= in_frames:
            chunk = features[start:end]
        else:
            last_idx = in_frames - 1
            chunk = np.concatenate(
                [features[start:in_frames], np.tile(features[last_idx : last_idx + 1], (end - in_frames, 1))]
            )
        out[i] = chunk.reshape(-1)
    return out


def _apply_cmvn(
    features: npt.NDArray[np.float32],
    neg_mean: npt.NDArray[np.float32],
    inv_stddev: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """``(features + neg_mean) * inv_stddev`` (broadcast along time)."""
    return ((features + neg_mean) * inv_stddev).astype(np.float32, copy=False)


def _ctc_greedy_decode(logits: npt.NDArray[np.float32], num_frames: int, blank_id: int) -> list[int]:
    """Single-utterance CTC greedy decode: argmax, drop blanks, collapse repeats."""
    if logits.shape[0] == 0:
        return []
    scan = logits[:num_frames]
    ids = scan.argmax(axis=-1).astype(np.int64)
    out: list[int] = []
    prev = -1
    for token in ids.tolist():
        if token != blank_id and token != prev:
            out.append(int(token))
        prev = int(token)
    return out


def _load_tokens(tokens_path: Path, *, base64_encoded: bool) -> dict[int, str]:
    """Load a SenseVoice ``tokens.txt`` (``<symbol> <id>`` per line) → ``{id: symbol}``.

    rsplit so symbols containing whitespace stay intact; Nano symbols are
    base64-decoded.
    """
    out: dict[int, str] = {}
    for line in tokens_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.rsplit(None, 1)
        if len(parts) != 2:
            continue
        symbol, id_str = parts
        try:
            token_id = int(id_str)
        except ValueError:
            continue
        if base64_encoded:
            try:
                symbol = base64.b64decode(symbol.encode("ascii")).decode("utf-8")
            except (binascii.Error, UnicodeDecodeError, UnicodeEncodeError):
                logger.debug("base64 decode failed for token %r — keeping raw", symbol)
        out[token_id] = symbol
    return out


def _meta_int(meta: dict[str, str], key: str, default: int | None = None) -> int | None:
    raw = meta.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _meta_float_vec(meta: dict[str, str], key: str) -> npt.NDArray[np.float32]:
    raw = meta.get(key)
    if raw is None:
        return np.zeros(0, dtype=np.float32)
    # FunASR exports separate by whitespace (incl. newline) or comma; normalise
    # to spaces and let NumPy parse the vector in one shot (matches dolphin.py).
    return np.fromstring(raw.replace(",", " "), sep=" ", dtype=np.float32)


def _parse_metadata(meta: dict[str, str]) -> dict[str, Any]:
    """Resolve SenseVoice-relevant ONNX metadata into a plain dict."""
    is_nano = "Nano" in (meta.get("comment", "") or "")
    vocab_size = _meta_int(meta, "vocab_size")
    if vocab_size is None:
        msg = "SenseVoice model metadata is missing required 'vocab_size'"
        raise ValueError(msg)
    blank_id = _meta_int(meta, "blank_id", 0) or 0
    lfr_window_size = _meta_int(meta, "lfr_window_size", 7) or 7
    lfr_window_shift = _meta_int(meta, "lfr_window_shift", 6) or 6
    normalize_samples = (_meta_int(meta, "normalize_samples", 0) or 0) != 0
    if is_nano:
        with_itn_id, without_itn_id = 14, 15
        lang2id: dict[str, int] = {}
        neg_mean = np.zeros(0, dtype=np.float32)
        inv_stddev = np.zeros(0, dtype=np.float32)
    else:
        with_itn_id = _meta_int(meta, "with_itn", _DEFAULT_WITH_ITN_ID) or _DEFAULT_WITH_ITN_ID
        without_itn_id = _meta_int(meta, "without_itn", _DEFAULT_WITHOUT_ITN_ID) or _DEFAULT_WITHOUT_ITN_ID
        lang2id = {}
        for lang_code, meta_key in (
            ("auto", "lang_auto"),
            ("zh", "lang_zh"),
            ("en", "lang_en"),
            ("ja", "lang_ja"),
            ("ko", "lang_ko"),
            ("yue", "lang_yue"),
        ):
            lang_id = _meta_int(meta, meta_key)
            if lang_id is not None:
                lang2id[lang_code] = lang_id
        if not lang2id:
            lang2id = dict(_DEFAULT_LANG_IDS)
        neg_mean = _meta_float_vec(meta, "neg_mean")
        inv_stddev = _meta_float_vec(meta, "inv_stddev")
    return {
        "vocab_size": vocab_size,
        "blank_id": blank_id,
        "lfr_window_size": lfr_window_size,
        "lfr_window_shift": lfr_window_shift,
        "normalize_samples": normalize_samples,
        "with_itn_id": with_itn_id,
        "without_itn_id": without_itn_id,
        "lang2id": lang2id,
        "neg_mean": neg_mean,
        "inv_stddev": inv_stddev,
        "is_funasr_nano": is_nano,
    }


def _format_result_text(tokens: list[int], symbols: dict[int, str], *, is_nano: bool) -> str:
    """Render decoded ids to text: strip the 4 control tokens (non-Nano), ``▁``→space."""
    start = 0 if is_nano else _NUM_CONTROL_TOKENS
    pieces = [symbols.get(tid, "").replace("▁", " ") for tid in tokens[start:] if symbols.get(tid, "")]
    text = "".join(pieces).strip()
    return text.replace(" '", "'").replace(" ▁'", "'")


class SenseVoiceCtc(BaseAsr):
    """FunAudioLLM SenseVoice CTC model (FBANK + LFR + CMVN front-end in NumPy)."""

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(is_multilingual=True)

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        self._model = rt.InferenceSession(model_files["model"], **onnx_options)
        self._input_names = [inp.name for inp in self._model.get_inputs()]
        meta = dict(self._model.get_modelmeta().custom_metadata_map or {})
        self._metadata = _parse_metadata(meta)
        self._fbanks = _build_mel_filterbank()
        self._symbols = _load_tokens(model_files["vocab"], base64_encoded=self._metadata["is_funasr_nano"])

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        return []

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        # Flat repo root: ``model.onnx`` / ``model.int8.onnx`` + ``tokens.txt``.
        # ``?`` matches the ``.`` (or ``_``) before the quant suffix.
        suffix = "?" + quantization if quantization else ""
        return {"model": f"model{suffix}.onnx", "vocab": "tokens.txt"}

    @property
    def _preprocessor_name(self) -> str:
        # SenseVoice runs its own FBANK+LFR+CMVN in recognize_batch; the
        # resolver's identity preprocessor passes raw waveforms straight through.
        return "identity"

    def _resolve_language_id(self, language: str) -> int:
        canonical = _LANGUAGE_KEY_MAP.get(language, "auto")
        lang2id: dict[str, int] = self._metadata["lang2id"]
        return lang2id.get(canonical, lang2id.get("auto", 0))

    def _features_for(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if self._metadata["normalize_samples"]:
            audio = (audio * 32768.0).astype(np.float32, copy=False)
        features = _compute_fbank(audio, self._fbanks)
        features = _apply_lfr(features, self._metadata["lfr_window_size"], self._metadata["lfr_window_shift"])
        if not self._metadata["is_funasr_nano"] and self._metadata["neg_mean"].size > 0:
            features = _apply_cmvn(features, self._metadata["neg_mean"], self._metadata["inv_stddev"])
        return features

    def _run(self, features: npt.NDArray[np.float32], language: str) -> npt.NDArray[np.float32]:
        feat = features[np.newaxis, ...]
        if self._metadata["is_funasr_nano"]:
            outputs = self._model.run(None, {self._input_names[0]: feat})
        else:
            feeds = {
                self._input_names[0]: feat,
                self._input_names[1]: np.array([features.shape[0]], dtype=np.int32),
                self._input_names[2]: np.array([self._resolve_language_id(language)], dtype=np.int32),
                self._input_names[3]: np.array([self._metadata["with_itn_id"]], dtype=np.int32),
            }
            outputs = self._model.run(None, feeds)
        return np.asarray(outputs[0], dtype=np.float32)

    def _transcribe_one(self, audio: npt.NDArray[np.float32], language: str) -> str:
        if audio.size == 0:
            return ""
        features = self._features_for(audio)
        if features.shape[0] == 0:
            return ""
        is_nano: bool = self._metadata["is_funasr_nano"]
        logits = self._run(features, language)[0]
        num_frames = int(logits.shape[0]) if is_nano else features.shape[0] + _NUM_CONTROL_TOKENS
        token_ids = _ctc_greedy_decode(logits, num_frames, int(self._metadata["blank_id"]))
        return _format_result_text(token_ids, self._symbols, is_nano=is_nano)

    def recognize_batch(
        self,
        waveforms: npt.NDArray[np.float32],
        waveforms_len: npt.NDArray[np.int64],
        /,
        **kwargs: object | None,
    ) -> Iterator[TimestampedResult]:
        """Recognize a batch of waveforms (the identity preprocessor passes them raw)."""
        language = str(kwargs.get("language") or "")
        results: list[TimestampedResult] = []
        for i in range(waveforms.shape[0]):
            length = int(waveforms_len[i])
            audio = np.ascontiguousarray(waveforms[i, :length], dtype=np.float32)
            results.append(TimestampedResult(text=self._transcribe_one(audio, language)))
        return iter(results)
