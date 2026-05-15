"""Whisper model implementations."""

from __future__ import annotations

import json
import typing
from abc import abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import onnxruntime as rt
from onnxruntime import OrtValue

from onnx_asr.asr import BaseAsr, ModelCapabilities, Preprocessor, StreamingResult, TimestampedResult
from onnx_asr.onnx import OnnxSessionOptions, TensorRtOptions, get_onnx_device
from onnx_asr.utils import is_float32_array, is_int32_array


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
        # Skip prompt prefix (BOS/lang/transcribe) — handled by caller passing only generated output.
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

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        input_encoding = self._encode(waveforms, waveforms_len)
        return_timestamps = bool(kwargs.get("return_timestamps"))
        prompt = self._transcribe_input_with_timestamps if return_timestamps else self._transcribe_input
        input_tokens = np.repeat(prompt, len(waveforms), axis=0)

        language = kwargs.get("language")
        if language:
            input_tokens[:, 1] = self._tokens[f"<|{language}|>"]
        else:
            input_tokens_detect_lang = np.repeat(self._detect_lang_input, len(waveforms), axis=0)
            input_tokens[:, 1] = self._decoding(input_encoding, input_tokens_detect_lang, 3)[:, 1]

        decoded = self._decoding(input_encoding, input_tokens)
        for row in decoded:
            text = self._decode_text(row)
            segments = self._extract_segments(row) if return_timestamps else None
            yield TimestampedResult(text=text, segments=segments)

    def _transcribe_single(
        self, waveform: npt.NDArray[np.float32], *, language: str | None = None
    ) -> tuple[npt.NDArray[np.int64], str]:
        """Run one full decode on a single audio array. Returns (raw_token_ids, decoded_text).

        Internal helper for streaming wrappers that need both the integer
        token sequence (for LCP / commit policies) and the rendered text.
        """
        wf = waveform.reshape(1, -1).astype(np.float32, copy=False)
        wf_len = np.array([wf.shape[1]], dtype=np.int64)

        input_encoding = self._encode(wf, wf_len)
        input_tokens = np.repeat(self._transcribe_input, 1, axis=0)
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
    ) -> WhisperStream:
        """Open a LocalAgreement-2 streaming session over this Whisper instance.

        See :class:`WhisperStream` for the algorithm. The returned object
        satisfies the :class:`~onnx_asr.asr.AsrStream` protocol.
        """
        return WhisperStream(
            self,
            sample_rate=sample_rate,
            min_chunk_size_s=min_chunk_size_s,
            language=language,
        )


class WhisperStream:
    """LocalAgreement-2 streaming wrapper around a Whisper model.

    The classic UFAL whisper_streaming policy, ONNX-friendly. Each ``step()``:

    1. Re-decode the whole audio buffer with the underlying greedy decoder.
    2. Compute the longest common prefix (token-level, integer IDs) of the new
       token sequence against the previous step's sequence.
    3. Any token at a prefix position that matches across two consecutive
       decodes is considered **committed** — it won't change in future
       snapshots — and goes into ``StreamingResult.committed_text``.
    4. The full decoded text goes into ``StreamingResult.text`` so callers
       can render the live preview that may still be revised.

    On ``finish()``, the next ``step()`` commits everything (every token
    becomes committed) and the snapshot has ``is_partial=False`` and
    ``is_endpoint=True``.

    Bounded compute (audio-buffer trim at committed segment boundary) is
    deferred — depends on Whisper segment-timestamp extraction (`<|t|>`
    tokens, plan item #5). Until that lands, the buffer grows over the
    course of an utterance; the caller should call ``reset()`` on VAD
    endpoint to drop it.
    """

    def __init__(
        self,
        asr: _Whisper,
        *,
        sample_rate: int = 16_000,
        min_chunk_size_s: float = 1.0,
        language: str | None = None,
    ) -> None:
        """Bind the stream to ``asr``. See :meth:`_Whisper.create_stream` for kwargs."""
        self._asr = asr
        self._sample_rate = sample_rate
        self._min_chunk_samples = max(1, int(min_chunk_size_s * sample_rate))
        self._language = language
        self._buffer: npt.NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._prev_token_ids: list[int] = []
        self._committed_count = 0
        self._finished = False
        self._endpoint = False
        self._segment_id = 0

    def push_audio(self, samples: npt.NDArray[np.float32], sample_rate: int = 16_000) -> None:
        """Append PCM samples to the input buffer."""
        if sample_rate != self._sample_rate:
            msg = f"WhisperStream sample_rate mismatch: expected {self._sample_rate}, got {sample_rate}"
            raise ValueError(msg)
        flat = np.asarray(samples, dtype=np.float32).ravel()
        self._buffer = np.concatenate([self._buffer, flat])

    def finish(self) -> None:
        """Mark the input as closed. Next ``step()`` commits everything."""
        self._finished = True

    def is_ready(self) -> bool:
        """Return True once buffer has enough audio for a meaningful decode."""
        if self._buffer.size == 0:
            return False
        if self._finished:
            return True
        return self._buffer.size >= self._min_chunk_samples

    def step(self) -> StreamingResult | None:
        """Re-decode, update commit prefix via LocalAgreement-2, return snapshot."""
        if not self.is_ready():
            return None

        token_ids, text = self._asr._transcribe_single(self._buffer, language=self._language)
        token_ids_list = [int(t) for t in token_ids]

        if self._finished:
            self._committed_count = len(token_ids_list)
            self._endpoint = True
        else:
            lcp = self._longest_common_prefix(token_ids_list, self._prev_token_ids)
            # Committed count grows monotonically — never shrink even if a later
            # decode briefly disagrees with itself at the boundary.
            self._committed_count = max(self._committed_count, min(lcp, len(token_ids_list)))

        committed_ids = token_ids_list[: self._committed_count]
        committed_text = (
            self._asr._decode_tokens(np.asarray(committed_ids, dtype=np.int64)).text if committed_ids else ""
        )
        token_strs = [
            self._asr._vocab[tid] for tid in token_ids_list if not self._asr._vocab.get(tid, "").startswith("<|")
        ]

        self._prev_token_ids = token_ids_list

        return StreamingResult(
            text=text,
            tokens=token_strs,
            timestamps=None,
            is_partial=not self._finished,
            segment_id=self._segment_id,
            committed_text=committed_text,
        )

    def reset(self, *, keep_audio: bool = False) -> None:
        """Zero decode state; bump segment id. Drops audio buffer unless ``keep_audio=True``."""
        if not keep_audio:
            self._buffer = np.empty(0, dtype=np.float32)
        self._prev_token_ids = []
        self._committed_count = 0
        self._finished = False
        self._endpoint = False
        self._segment_id += 1

    @property
    def is_endpoint(self) -> bool:
        """Whether the last ``step()`` produced the final/committed snapshot."""
        return self._endpoint

    @property
    def buffered_samples(self) -> int:
        """Current buffer size in samples."""
        return int(self._buffer.size)

    @staticmethod
    def _longest_common_prefix(a: list[int], b: list[int]) -> int:
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n


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
        self, input_features: OrtValue, tokens: npt.NDArray[np.int64], max_length: int = 448
    ) -> npt.NDArray[np.int64]:
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
        return sequences[:, 0, :].astype(np.int64)


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

    def _decoding(
        self, input_features: OrtValue, tokens: npt.NDArray[np.int64], max_length: int = 448
    ) -> npt.NDArray[np.int64]:
        state = self._create_state()
        for _ in range(tokens.shape[-1], max_length):
            logits, state = self._decode(tokens, state, input_features)
            next_tokens = logits[:, -1].argmax(axis=-1)
            next_tokens[tokens[:, -1] == self._eos_token_id] = self._eos_token_id
            tokens = np.hstack((tokens, next_tokens[:, None]))
            if (tokens[:, -1] == self._eos_token_id).all():
                break

        return tokens
