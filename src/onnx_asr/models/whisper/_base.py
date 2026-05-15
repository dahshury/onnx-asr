"""Whisper base class — shared init / tokenizer / preprocessing / orchestration.

Plan item #7: the ``_Whisper`` abstract base (formerly ``_WhisperBase`` per the
plan; both names are exported for source compatibility) owns:

* vocab + added-token loading
* GPT-2 byte_decoder construction
* the canonical static decoder prompts (with / without timestamps)
* the segment-timestamp parser (``_extract_segments``)
* the word-alignment orchestrator (``_align_word_timestamps``)
* ``recognize_batch`` dispatch (timestamps / word-timestamps / plain)
* ``_transcribe_single`` helper used by streaming wrappers
* ``create_stream`` factory

Concrete subclasses (:class:`WhisperOrt`, :class:`WhisperHf`) live in
sibling modules; each only adds the encoder/decoder ORT-session setup and
the model-specific ``_decoding`` greedy autoregressive loop.
"""

from __future__ import annotations

import json
import typing
from abc import abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import numpy.typing as npt

from onnx_asr.asr import (
    BaseAsr,
    ModelCapabilities,
    Preprocessor,
    TimestampedResult,
    WordResult,
)
from onnx_asr.onnx import OnnxSessionOptions, TensorRtOptions

if TYPE_CHECKING:
    from onnxruntime import OrtValue

    from onnx_asr.models.whisper._stream import WhisperStream


@typing.no_type_check
def bytes_to_unicode() -> dict[int, str]:
    """Magic func copied from transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))  # noqa: B905


class _Whisper(BaseAsr):
    """Abstract base for Whisper-family ONNX exports.

    Subclasses must override ``_decoding`` (and optionally ``_encode``,
    ``_decoding_with_cross_attention``, ``supports_word_timestamps``).
    """

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        streaming_native=True,
        is_multilingual=True,
    )

    def __init__(
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)

        with model_files["vocab"].open("rt", encoding="utf-8") as f:
            self._tokens: dict[str, int] = json.load(f)

        with model_files["added_tokens"].open("rt", encoding="utf-8") as f:
            self._tokens |= json.load(f)

        self._vocab = {id: token for token, id in self._tokens.items()}
        self._bos_token_id = self._tokens["<|startoftranscript|>"]
        self._eos_token_id = self._tokens["<|endoftext|>"]
        self._byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
        self._transcribe_input = np.array(
            [
                [
                    self._bos_token_id,
                    self._eos_token_id,
                    self._tokens["<|transcribe|>"],
                    self._tokens["<|notimestamps|>"],
                ]
            ],
            dtype=np.int64,
        )
        self._detect_lang_input = np.array([[self._bos_token_id]], dtype=np.int64)
        # Timestamp tokens occupy a contiguous range starting at ``<|0.00|>`` with a 0.02 s step.
        # When ``return_timestamps=True`` we drop ``<|notimestamps|>`` from the prompt so the
        # decoder can emit segment timestamp tokens.
        self._timestamp_begin_id: int | None = self._tokens.get("<|0.00|>")
        self._timestamp_step_s = 0.02
        self._transcribe_input_with_timestamps = np.array(
            [
                [
                    self._bos_token_id,
                    self._eos_token_id,
                    self._tokens["<|transcribe|>"],
                ]
            ],
            dtype=np.int64,
        )

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        return TensorRtOptions.get_provider_names()

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        return {"vocab": "vocab.json", "added_tokens": "added_tokens.json"}

    def _encode(self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]) -> OrtValue:
        from onnxruntime import OrtValue  # noqa: PLC0415

        input_features, _ = self._preprocessor(waveforms, waveforms_len)
        return OrtValue.ortvalue_from_numpy(input_features)

    @abstractmethod
    def _decoding(
        self, input_features: OrtValue, tokens: npt.NDArray[np.int64], max_length: int = 448
    ) -> npt.NDArray[np.int64]: ...

    def _decode_text(self, tokens: npt.NDArray[np.int64] | list[int]) -> str:
        text = "".join(token for id in tokens if (token := self._vocab[int(id)]) and not token.startswith("<|"))
        return bytearray([self._byte_decoder[c] for c in text]).decode("utf-8", errors="replace").removeprefix(" ")

    def _decode_tokens(self, tokens: npt.NDArray[np.int64]) -> TimestampedResult:
        return TimestampedResult(self._decode_text(tokens))

    def _extract_segments(self, tokens: npt.NDArray[np.int64]) -> list[tuple[float, float, str]]:
        """Parse the Whisper timestamp token stream into ``(start_s, end_s, text)`` segments.

        Whisper emits paired ``<|t|>`` timestamp tokens around each segment::

            <|0.00|> hello world <|2.34|> <|2.34|> how are you <|4.56|>

        Tokens with id ``>= _timestamp_begin_id`` are treated as timestamp markers; the time in
        seconds is ``(id - _timestamp_begin_id) * 0.02``. Unpaired or empty segments are skipped.
        """
        if self._timestamp_begin_id is None:
            return []

        begin_id = self._timestamp_begin_id
        step = self._timestamp_step_s
        segments: list[tuple[float, float, str]] = []
        i = 0
        while i < len(tokens):
            tok = int(tokens[i])
            if tok < begin_id:
                i += 1
                continue
            start = (tok - begin_id) * step
            j = i + 1
            while j < len(tokens) and int(tokens[j]) < begin_id:
                if int(tokens[j]) == self._eos_token_id:
                    break
                j += 1
            if j >= len(tokens) or int(tokens[j]) < begin_id:
                break
            end = (int(tokens[j]) - begin_id) * step
            text = self._decode_text(tokens[i + 1 : j])
            if text:
                segments.append((start, end, text.strip()))
            i = j + 1
        return segments

    @property
    def supports_word_timestamps(self) -> bool:
        """Whether this model exports cross-attention (required for word-DTW)."""
        return False

    def _decoding_with_cross_attention(
        self,
        input_features: OrtValue,
        tokens: npt.NDArray[np.int64],
        max_length: int = 448,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """Decode while collecting per-step cross-attention.

        Subclasses that support word timestamps override this method; the
        default raises ``NotImplementedError`` so callers know the model
        lacks the required outputs.

        Returns ``(token_ids, cross_attentions)`` where ``cross_attentions``
        has shape ``(num_layers, num_heads, num_decoder_tokens, num_encoder_frames)``.
        """
        msg = f"{type(self).__name__} does not export cross-attention; word timestamps unavailable."
        raise NotImplementedError(msg)

    def _align_word_timestamps(
        self,
        cross_attentions: npt.NDArray[np.float32],
        generated_tokens: list[int],
        *,
        prompt_length: int,
        num_audio_frames: int,
        language: str | None = None,
    ) -> list[WordResult]:
        """Run cross-attention DTW to recover word timings. See :mod:`word_timestamps`."""
        from onnx_asr.word_timestamps import align_words, lookup_alignment_heads  # noqa: PLC0415

        num_layers = int(cross_attentions.shape[0])
        num_heads = int(cross_attentions.shape[1])
        vocab_size = max(self._tokens.values()) + 1
        heads_mask = lookup_alignment_heads(num_layers, num_heads, vocab_size)

        def decode_one(ids: list[int]) -> str:
            return self._decode_text(np.asarray(ids, dtype=np.int64))

        timings = align_words(
            cross_attentions,
            heads_mask,
            text_tokens=generated_tokens,
            decode_one=decode_one,
            eot_id=self._eos_token_id,
            prompt_length=prompt_length,
            num_audio_frames=num_audio_frames,
            language=language,
        )
        return [WordResult(text=t.word, start=t.start, end=t.end) for t in timings]

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        input_encoding = self._encode(waveforms, waveforms_len)
        return_timestamps = bool(kwargs.get("return_timestamps"))
        return_word_timestamps = bool(kwargs.get("return_word_timestamps")) and self.supports_word_timestamps
        prompt = self._transcribe_input_with_timestamps if return_timestamps else self._transcribe_input
        input_tokens = np.repeat(prompt, len(waveforms), axis=0)

        language_raw = kwargs.get("language")
        language = str(language_raw) if isinstance(language_raw, str) else None
        if language:
            input_tokens[:, 1] = self._tokens[f"<|{language}|>"]
        else:
            input_tokens_detect_lang = np.repeat(self._detect_lang_input, len(waveforms), axis=0)
            input_tokens[:, 1] = self._decoding(input_encoding, input_tokens_detect_lang, 3)[:, 1]

        prompt_length = int(input_tokens.shape[1])
        num_audio_frames = int(waveforms_len[0]) // 160  # HOP_LENGTH = 160

        if return_word_timestamps:
            decoded, cross_attentions = self._decoding_with_cross_attention(input_encoding, input_tokens)
            for batch_idx, row in enumerate(decoded):
                text = self._decode_text(row)
                segments = self._extract_segments(row) if return_timestamps else None
                generated = [int(t) for t in row[prompt_length:] if int(t) != self._eos_token_id]
                generated.append(self._eos_token_id)
                words = self._align_word_timestamps(
                    cross_attentions[batch_idx] if cross_attentions.ndim == 5 else cross_attentions,
                    generated,
                    prompt_length=prompt_length,
                    num_audio_frames=num_audio_frames,
                    language=language,
                )
                yield TimestampedResult(text=text, segments=segments, words=words)
            return

        decoded = self._decoding(input_encoding, input_tokens)
        for row in decoded:
            text = self._decode_text(row)
            segments = self._extract_segments(row) if return_timestamps else None
            yield TimestampedResult(text=text, segments=segments)

    def _transcribe_single(
        self,
        waveform: npt.NDArray[np.float32],
        *,
        language: str | None = None,
        with_timestamps: bool = False,
    ) -> tuple[npt.NDArray[np.int64], str]:
        """Run one full decode on a single audio array. Returns (raw_token_ids, decoded_text).

        Internal helper for streaming wrappers that need both the integer token
        sequence (for LCP / commit policies) and the rendered text. When
        ``with_timestamps=True`` the prompt drops ``<|notimestamps|>`` so the
        decoder emits ``<|t|>`` segment markers; callers can then extract
        segment times via :meth:`_extract_segments` to drive audio-buffer
        trimming (bounded streaming compute).
        """
        wf = waveform.reshape(1, -1).astype(np.float32, copy=False)
        wf_len = np.array([wf.shape[1]], dtype=np.int64)

        input_encoding = self._encode(wf, wf_len)
        prompt = self._transcribe_input_with_timestamps if with_timestamps else self._transcribe_input
        input_tokens = np.repeat(prompt, 1, axis=0)
        if language:
            input_tokens[:, 1] = self._tokens[f"<|{language}|>"]
        else:
            input_tokens_detect_lang = np.repeat(self._detect_lang_input, 1, axis=0)
            input_tokens[:, 1] = self._decoding(input_encoding, input_tokens_detect_lang, 3)[:, 1]

        output = self._decoding(input_encoding, input_tokens)
        token_ids = output[0]
        text = self._decode_tokens(token_ids).text
        return token_ids, text

    def create_stream(
        self,
        *,
        sample_rate: int = 16_000,
        min_chunk_size_s: float = 1.0,
        language: str | None = None,
        trim_after_s: float = 8.0,
    ) -> WhisperStream:
        """Open a LocalAgreement-2 streaming session over this Whisper instance.

        See :class:`~onnx_asr.models.whisper.WhisperStream` for the algorithm.
        The returned object satisfies the :class:`~onnx_asr.asr.AsrStream` protocol.

        Args:
            sample_rate: Input PCM sample rate (must match what push_audio sends).
            min_chunk_size_s: Minimum buffered audio before each decode step.
            language: Force a Whisper language code (skips lang autodetect).
            trim_after_s: Trim the audio buffer up to the latest committed
                ``<|t|>`` boundary once the buffer exceeds this many seconds.
                Keeps the LocalAgreement-2 decode bounded; defaults to 8 s.
                Set to a large number (e.g. 600.0) to disable trimming.

        """
        from onnx_asr.models.whisper._stream import WhisperStream  # noqa: PLC0415

        return WhisperStream(
            self,
            sample_rate=sample_rate,
            min_chunk_size_s=min_chunk_size_s,
            language=language,
            trim_after_s=trim_after_s,
        )


#: Plan #7 alias — the plan calls the base ``_WhisperBase``. We keep the original
#: ``_Whisper`` name (since existing imports use it) and alias for both.
_WhisperBase = _Whisper
