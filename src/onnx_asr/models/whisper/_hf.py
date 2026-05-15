"""Whisper exported via HuggingFace Optimum — split encoder/decoder.

The Optimum export is split into:

* ``encoder_model.onnx`` — int16 PCM mel → contextual hidden states
* ``decoder_model_merged.onnx`` — autoregressive cross-attended decoder
  with optional past_key_values / present cache and, for the
  ``*_timestamped`` variants, ``cross_attentions.{0..N-1}`` outputs

We run the encoder once per clip, then drive the decoder step-by-step in
Python so we can:

* observe per-step state for streaming (:class:`WhisperStream`)
* collect cross-attention for word timestamps
  (:meth:`_decoding_with_cross_attention`)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt
from onnxruntime import OrtValue

from onnx_asr.asr import Preprocessor
from onnx_asr.models.whisper._base import _Whisper
from onnx_asr.onnx import OnnxSessionOptions, get_onnx_device
from onnx_asr.utils import is_float32_array


class WhisperHf(_Whisper):
    """Whisper (exported via optimum) model implementation."""

    def __init__(
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
