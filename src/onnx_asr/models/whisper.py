"""Whisper model implementations."""

import json
import typing
from abc import abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt
from onnxruntime import OrtValue

from onnx_asr.asr import BaseAsr, Preprocessor, TimestampedResult, WordResult
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
        # ``vocab_size`` distinguishes English-only (51 864) from multilingual.
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
                # Trailing EOT is needed by ``align_words`` to anchor the last word.
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

    @property
    def supports_word_timestamps(self) -> bool:
        """True iff the decoder export exposes ``cross_attentions.*`` outputs."""
        return any(o.name.startswith("cross_attentions.") for o in self._decoder.get_outputs())

    def _cross_attention_output_names(self) -> list[str]:
        """Sorted list of cross-attention output names from the decoder session.

        Sorted by the trailing layer index so the stacked tensor is in
        canonical ``(layer 0, layer 1, ...)`` order.
        """
        names = [o.name for o in self._decoder.get_outputs() if o.name.startswith("cross_attentions.")]
        return sorted(names, key=lambda n: int(n.removeprefix("cross_attentions.")))

    def _decode_collect_attention(
        self,
        tokens: npt.NDArray[np.int64],
        prev_state: dict[str, OrtValue],
        encoder_out: OrtValue,
        cross_attn_names: list[str],
    ) -> tuple[npt.NDArray[np.float32], dict[str, OrtValue], list[npt.NDArray[np.float32]]]:
        """Like :meth:`_decode` but also returns per-layer cross-attention arrays.

        Cross-attention is bound as a CPU output (rather than via io_binding's
        device-typed bind_output) because we'll concatenate it across decode
        steps on the CPU side. The performance hit is tolerable since word
        timestamps are an opt-in feature run after the audio is committed.
        """
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
        for name in cross_attn_names:
            binding.bind_output(name)

        self._decoder.run_with_iobinding(binding)
        outputs = binding.get_outputs()
        logits = outputs[0].numpy()
        assert is_float32_array(logits)
        # Outputs layout: [logits, present.*..., cross_attentions.*...]
        num_state = len(prev_state)
        next_state = {
            key: next_value if next_value.shape()[0] else prev_value
            for (key, prev_value), next_value in zip(prev_state.items(), outputs[1 : 1 + num_state], strict=True)
        }
        cross_attns_step: list[npt.NDArray[np.float32]] = []
        for i, _name in enumerate(cross_attn_names):
            arr = outputs[1 + num_state + i].numpy()
            assert is_float32_array(arr)
            cross_attns_step.append(arr)
        return logits, next_state, cross_attns_step

    def _decoding_with_cross_attention(
        self,
        input_features: OrtValue,
        tokens: npt.NDArray[np.int64],
        max_length: int = 448,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """Autoregressive decode that collects cross-attention across all steps.

        Returns ``(token_ids, cross_attentions)`` where cross_attentions has
        shape ``(batch, num_layers, num_heads, num_decoder_tokens, num_encoder_frames)``.
        """
        cross_attn_names = self._cross_attention_output_names()
        if not cross_attn_names:
            msg = "Decoder export does not include cross_attentions.* outputs."
            raise RuntimeError(msg)

        state = self._create_state()
        # Per-layer running buffers — list of (batch, heads, dec_step_len, enc_frames).
        # Concatenated along the decoder-sequence axis as we generate.
        per_layer_attn: list[list[npt.NDArray[np.float32]]] = [[] for _ in cross_attn_names]
        for _ in range(tokens.shape[-1], max_length):
            logits, state, attn_step = self._decode_collect_attention(tokens, state, input_features, cross_attn_names)
            for li, arr in enumerate(attn_step):
                per_layer_attn[li].append(arr)
            next_tokens = logits[:, -1].argmax(axis=-1)
            next_tokens[tokens[:, -1] == self._eos_token_id] = self._eos_token_id
            tokens = np.hstack((tokens, next_tokens[:, None]))
            if (tokens[:, -1] == self._eos_token_id).all():
                break

        # Stack each layer's per-step attention along the decoder-sequence axis.
        stacked_per_layer = [np.concatenate(layer_steps, axis=2) for layer_steps in per_layer_attn]
        # Stack layers → (batch, num_layers, num_heads, num_dec_tokens, num_enc_frames).
        full = np.stack(stacked_per_layer, axis=1).astype(np.float32, copy=False)
        return tokens, full

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
