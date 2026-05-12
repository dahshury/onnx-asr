"""Whisper model implementations."""

import json
import typing
import zlib
from abc import abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt
from onnxruntime import OrtValue

from onnx_asr.asr import BaseAsr, Preprocessor, TimestampedResult
from onnx_asr.onnx import OnnxSessionOptions, TensorRtOptions, get_onnx_device
from onnx_asr.utils import is_float32_array, is_int32_array

DEFAULT_TEMPERATURES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_NO_SPEECH_THRESHOLD: float = 0.6
DEFAULT_COMPRESSION_RATIO_THRESHOLD: float = 2.4
DEFAULT_LOGPROB_THRESHOLD: float = -1.0


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


@dataclass(frozen=True)
class _DecodingResult:
    """Per-batch decoding output with quality-guard metadata."""

    tokens: npt.NDArray[np.int64]  # (batch, seq_len)
    avg_logprob: npt.NDArray[np.float32]  # (batch,) — mean log-prob over non-prompt tokens
    no_speech_prob: npt.NDArray[np.float32]  # (batch,) — P(<|nospeech|>) at first decoder step


class _Whisper(BaseAsr):
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
        self._no_speech_token_id: int | None = self._tokens.get("<|nospeech|>") or self._tokens.get("<|nocaptions|>")
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
        self._rng = np.random.default_rng()

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        return TensorRtOptions.get_provider_names()

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        return {"vocab": "vocab.json", "added_tokens": "added_tokens.json"}

    def _encode(self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]) -> OrtValue:
        input_features, _ = self._preprocessor(waveforms, waveforms_len)
        return OrtValue.ortvalue_from_numpy(input_features)

    @abstractmethod
    def _decoding(
        self,
        input_features: OrtValue,
        tokens: npt.NDArray[np.int64],
        max_length: int = 448,
        *,
        temperature: float = 0.0,
    ) -> _DecodingResult: ...

    def _decode_tokens(self, tokens: npt.NDArray[np.int64]) -> TimestampedResult:
        text = "".join(token for id in tokens if (token := self._vocab[id]) and not token.startswith("<|"))
        return TimestampedResult(
            bytearray([self._byte_decoder[c] for c in text]).decode("utf-8", errors="replace").removeprefix(" ")
        )

    @staticmethod
    def _compression_ratio(text: str) -> float:
        """Return gzip compression ratio of ``text``. Higher = more repetitive."""
        if not text:
            return 0.0
        raw = text.encode("utf-8")
        return len(raw) / len(zlib.compress(raw))

    def _decode_with_fallback(
        self,
        input_features: OrtValue,
        tokens: npt.NDArray[np.int64],
        *,
        temperatures: tuple[float, ...],
        no_speech_threshold: float | None,
        compression_ratio_threshold: float | None,
        logprob_threshold: float | None,
        max_length: int = 448,
    ) -> _DecodingResult:
        """Try temperatures in order until a result clears the quality guards.

        Mirrors OpenAI Whisper's transcribe-time fallback recipe (Radford et al.
        2022): on each attempt, decode at the next temperature; if the output
        looks repetitive (compression-ratio guard) or low-confidence
        (logprob guard) AND it isn't simply silence (no-speech override),
        retry. Returns the last attempt if the ladder is exhausted.
        """
        last: _DecodingResult | None = None
        for temp in temperatures:
            result = self._decoding(input_features, tokens, max_length, temperature=temp)
            last = result
            text = self._decode_tokens(result.tokens[0]).text
            ratio = self._compression_ratio(text)
            avg_lp = float(result.avg_logprob[0])
            no_speech = float(result.no_speech_prob[0])

            needs_fallback = False
            if compression_ratio_threshold is not None and ratio > compression_ratio_threshold:
                needs_fallback = True
            if logprob_threshold is not None and avg_lp < logprob_threshold:
                needs_fallback = True
            # Silence override: a low-confidence silent segment isn't worth retrying.
            if (
                no_speech_threshold is not None
                and no_speech > no_speech_threshold
                and logprob_threshold is not None
                and avg_lp < logprob_threshold
            ):
                needs_fallback = False

            if not needs_fallback:
                return result

        assert last is not None
        return last

    @staticmethod
    def _parse_temperatures(raw: object) -> tuple[float, ...]:
        if isinstance(raw, (int, float)):
            return (float(raw),)
        if isinstance(raw, (tuple, list)) and raw:
            return tuple(float(t) for t in raw)
        return DEFAULT_TEMPERATURES

    @staticmethod
    def _parse_optional_float(raw: object, default: float) -> float | None:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return float(raw)
        return default

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        input_encoding = self._encode(waveforms, waveforms_len)
        input_tokens = np.repeat(self._transcribe_input, len(waveforms), axis=0)

        temperatures = self._parse_temperatures(kwargs.get("temperature"))
        no_speech_threshold = (
            self._parse_optional_float(kwargs.get("no_speech_threshold"), DEFAULT_NO_SPEECH_THRESHOLD)
            if "no_speech_threshold" in kwargs
            else DEFAULT_NO_SPEECH_THRESHOLD
        )
        compression_ratio_threshold = (
            self._parse_optional_float(kwargs.get("compression_ratio_threshold"), DEFAULT_COMPRESSION_RATIO_THRESHOLD)
            if "compression_ratio_threshold" in kwargs
            else DEFAULT_COMPRESSION_RATIO_THRESHOLD
        )
        logprob_threshold = (
            self._parse_optional_float(kwargs.get("logprob_threshold"), DEFAULT_LOGPROB_THRESHOLD)
            if "logprob_threshold" in kwargs
            else DEFAULT_LOGPROB_THRESHOLD
        )

        language = kwargs.get("language")
        if language:
            input_tokens[:, 1] = self._tokens[f"<|{language}|>"]
        else:
            input_tokens_detect_lang = np.repeat(self._detect_lang_input, len(waveforms), axis=0)
            lang_result = self._decoding(input_encoding, input_tokens_detect_lang, 3, temperature=0.0)
            input_tokens[:, 1] = lang_result.tokens[:, 1]

        if len(temperatures) == 1:
            result = self._decoding(input_encoding, input_tokens, temperature=temperatures[0])
            return map(self._decode_tokens, result.tokens)

        return self._batch_with_fallback(
            input_encoding,
            input_tokens,
            temperatures=temperatures,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
        )

    def _batch_with_fallback(
        self,
        input_encoding: OrtValue,
        input_tokens: npt.NDArray[np.int64],
        *,
        temperatures: tuple[float, ...],
        no_speech_threshold: float | None,
        compression_ratio_threshold: float | None,
        logprob_threshold: float | None,
    ) -> Iterator[TimestampedResult]:
        enc_np = input_encoding.numpy()
        for i in range(input_tokens.shape[0]):
            single_enc = OrtValue.ortvalue_from_numpy(enc_np[i : i + 1])
            single_tokens = input_tokens[i : i + 1]
            result = self._decode_with_fallback(
                single_enc,
                single_tokens,
                temperatures=temperatures,
                no_speech_threshold=no_speech_threshold,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
            )
            yield self._decode_tokens(result.tokens[0])


class WhisperOrt(_Whisper):
    """Whisper (exported via onnxruntime) model implementation."""

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        self._model = rt.InferenceSession(model_files["model"], **onnx_options)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"whisper-*_beamsearch{suffix}.onnx"} | _Whisper._get_model_files(quantization)

    @property
    def _preprocessor_name(self) -> str:
        return f"whisper{self.config.get('features_size', 80)}"

    def _decoding(
        self,
        input_features: OrtValue,
        tokens: npt.NDArray[np.int64],
        max_length: int = 448,
        *,
        temperature: float = 0.0,
    ) -> _DecodingResult:
        # The packaged beam-search ONNX graph doesn't expose per-token logprobs
        # or the no-speech head, so the quality guards are no-ops for WhisperOrt.
        # Temperature is ignored (the graph is deterministic greedy/beam).
        _ = temperature
        (sequences,) = self._model.run(
            ["sequences"],
            {
                "input_features": input_features,
                "max_length": [max_length],
                "min_length": [0],
                "num_beams": [1],
                "num_return_sequences": [1],
                "length_penalty": [1.0],
                "repetition_penalty": [1.0],
                "decoder_input_ids": tokens.astype(np.int32),
            },
        )
        assert is_int32_array(sequences)
        out_tokens = sequences[:, 0, :].astype(np.int64)
        batch = out_tokens.shape[0]
        return _DecodingResult(
            tokens=out_tokens,
            avg_logprob=np.zeros(batch, dtype=np.float32),
            no_speech_prob=np.zeros(batch, dtype=np.float32),
        )


class WhisperHf(_Whisper):
    """Whisper (exported via optimum) model implementation."""

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        self._encoder = rt.InferenceSession(model_files["encoder"], **onnx_options)
        self._decoder = rt.InferenceSession(model_files["decoder"], **onnx_options)
        self._device_type, self._device_id = get_onnx_device(self._encoder)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"**/encoder_model{suffix}.onnx",
            "decoder": f"**/decoder_model_merged{suffix}.onnx",
        } | _Whisper._get_model_files(suffix)

    @property
    def _preprocessor_name(self) -> str:
        return f"whisper{self.config.get('num_mel_bins', 80)}"

    def _encode(self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]) -> OrtValue:
        input_features = super()._encode(waveforms, waveforms_len)
        binding = self._encoder.io_binding()
        binding.bind_ortvalue_input("input_features", input_features)
        binding.bind_output("last_hidden_state", self._device_type, self._device_id)
        self._encoder.run_with_iobinding(binding)
        last_hidden_state: OrtValue = binding.get_outputs()[0]
        return last_hidden_state

    def _create_state(self) -> dict[str, OrtValue]:
        return {
            x.name: OrtValue.ortvalue_from_numpy(np.zeros((0, x.shape[1], 0, x.shape[3]), dtype=np.float32))
            for x in self._decoder.get_inputs()
            if x.name.startswith("past_key_values.")
        }

    def _decode(
        self,
        tokens: npt.NDArray[np.int64],
        prev_state: dict[str, OrtValue],
        encoder_out: OrtValue,
    ) -> tuple[npt.NDArray[np.float32], dict[str, OrtValue]]:
        use_cache = any(x.shape()[0] for x in prev_state.values())

        binding = self._decoder.io_binding()
        binding.bind_cpu_input("input_ids", tokens[:, -1:] if use_cache else tokens)
        binding.bind_ortvalue_input("encoder_hidden_states", encoder_out)
        binding.bind_output("logits")
        if prev_state:
            binding.bind_cpu_input("use_cache_branch", np.array([use_cache]))
            for key, value in prev_state.items():
                binding.bind_ortvalue_input(key, value)
                binding.bind_output(key.replace("past_key_values.", "present."), self._device_type, self._device_id)

        self._decoder.run_with_iobinding(binding)
        outputs = binding.get_outputs()
        logits = outputs[0].numpy()
        assert is_float32_array(logits)
        return logits, {
            key: next_value if next_value.shape()[0] else prev_value
            for (key, prev_value), next_value in zip(prev_state.items(), outputs[1:], strict=True)
        }

    @staticmethod
    def _stable_log_softmax(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Numerically stable log-softmax over the last axis."""
        shifted = logits - logits.max(axis=-1, keepdims=True)
        result: npt.NDArray[np.float32] = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        return result

    def _sample_next_token(self, logits: npt.NDArray[np.float32], temperature: float) -> npt.NDArray[np.int64]:
        """Pick next token per batch row — argmax at T=0, multinomial sample at T>0."""
        if temperature <= 0.0:
            argmax_int64: npt.NDArray[np.int64] = logits.argmax(axis=-1).astype(np.int64)
            return argmax_int64
        scaled = logits / temperature
        scaled = scaled - scaled.max(axis=-1, keepdims=True)
        probs = np.exp(scaled)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        return np.asarray(
            [self._rng.choice(probs.shape[-1], p=probs[i]) for i in range(probs.shape[0])],
            dtype=np.int64,
        )

    def _decoding(
        self,
        input_features: OrtValue,
        tokens: npt.NDArray[np.int64],
        max_length: int = 448,
        *,
        temperature: float = 0.0,
    ) -> _DecodingResult:
        batch = tokens.shape[0]
        state = self._create_state()
        sum_logprobs = np.zeros(batch, dtype=np.float32)
        emitted_count = np.zeros(batch, dtype=np.int64)
        no_speech_prob = np.zeros(batch, dtype=np.float32)
        first_step = True

        for _ in range(tokens.shape[-1], max_length):
            logits, state = self._decode(tokens, state, input_features)
            last = logits[:, -1, :]

            if first_step and self._no_speech_token_id is not None:
                # Probability that this segment is silence — read from the dedicated head.
                shifted = last - last.max(axis=-1, keepdims=True)
                probs = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
                no_speech_prob = probs[:, self._no_speech_token_id].astype(np.float32)
                first_step = False

            log_probs = self._stable_log_softmax(last)
            next_tokens = self._sample_next_token(last, temperature)
            finished_mask = tokens[:, -1] == self._eos_token_id
            next_tokens = np.where(finished_mask, self._eos_token_id, next_tokens)

            # Track logprob for non-finished beams.
            chosen_lp = np.take_along_axis(log_probs, next_tokens[:, None], axis=-1).squeeze(-1)
            chosen_lp = np.where(finished_mask, 0.0, chosen_lp).astype(np.float32)
            sum_logprobs = sum_logprobs + chosen_lp
            emitted_count = emitted_count + np.where(finished_mask, 0, 1).astype(np.int64)

            tokens = np.hstack((tokens, next_tokens[:, None]))
            if (tokens[:, -1] == self._eos_token_id).all():
                break

        # avg over emitted (non-prompt, non-finished-padding) tokens.
        denom = np.maximum(emitted_count, 1).astype(np.float32)
        avg_logprob = (sum_logprobs / denom).astype(np.float32)
        return _DecodingResult(tokens=tokens, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob)
