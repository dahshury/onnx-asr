"""Cohere Transcribe ASR — Conformer encoder + lightweight Transformer decoder.

The HuggingFace ``onnx-community/cohere-transcribe-03-2026-ONNX`` export
(2B parameters, 14 languages) ships in the modern merged-decoder layout:

* ``onnx/encoder_model.onnx`` (+ ``_fp16`` / ``_q4`` / ``_q4f16`` /
  ``_quantized``): ``input_features (B, T, 128)`` → ``last_hidden_state (B, T', 1024)``.
  The mel features are *time-first* (unlike Whisper's ``(B, 128, T)``); see
  :class:`~onnx_asr.preprocessors.numpy_preprocessor.CohereAsrPreprocessorNumpy`.
* ``onnx/decoder_model_merged.onnx`` (+ siblings): autoregressive cross-
  attended decoder with ``input_ids`` / ``attention_mask`` / ``position_ids`` /
  ``num_logits_to_keep`` / ``encoder_hidden_states`` and KV-cache I/O. The
  *merged* qualifier here is the new ORT shape — KV-cache branch selection
  is implicit in the past-tensor shapes rather than gated by an explicit
  ``use_cache_branch`` flag (no such input exists on this export).

Tokenizer is a SentencePiece-style BPE with ``byte_fallback`` (NOT GPT-2
byte-level), decoded by replacing ``▁`` → space, then resolving any
``<0xXX>`` bytes through a fuse step. We hand-roll the decoder because the
upstream tokenizer relies on ``tokenizers`` / ``transformers`` — neither of
which is an onnx-asr dependency.

Prompt format (from
``transformers/models/cohere_asr/processing_cohere_asr.py`` at HF revision
``89c0c2e``):

    ["▁", "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
     "<|lang|>", "<|lang|>", "<|pnc|>"|"<|nopnc|>",
     "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>"]

The lang token is emitted twice (source + target = same language; the model
exposes no MT capability in this release). Default punctuation is enabled.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import BaseAsr, ModelCapabilities, Preprocessor, TimestampedResult
from onnx_asr.onnx import OnnxSessionOptions, TensorRtOptions, get_onnx_device
from onnx_asr.utils import is_float32_array

if TYPE_CHECKING:
    from onnxruntime import OrtValue


#: Languages the model is trained on; passing any other code falls back to ``<|en|>``.
_COHERE_LANGUAGES: frozenset[str] = frozenset(
    {"ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh"}
)


def _strip_special(token: str) -> bool:
    """Return True iff ``token`` is a special / control token to skip on decode."""
    # Control tokens come in two flavours in this tokenizer:
    #   * ``<|...|>`` — task / language / speaker / spltoken markers
    #   * ``<unk>`` / ``<pad>`` — sentinel words without the ``|`` syntax
    return (token.startswith("<|") and token.endswith("|>")) or token in {"<unk>", "<pad>"}


class CohereAsr(BaseAsr):
    """Cohere Transcribe ASR model — Conformer encoder + Transformer decoder.

    Tokenizer is SentencePiece-BPE with ``byte_fallback`` decoded in pure
    Python (no ``tokenizers`` / ``sentencepiece`` dependency). Decoding is
    greedy / argmax; the official ``generation_config`` calls for beam=1 too.
    """

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        streaming_native=False,
        is_multilingual=True,
    )
    #: Hard cap on decoder steps. ``transf_decoder.max_sequence_length`` in
    #: config.json is 1024; we leave headroom for the 10-token prompt.
    _max_decode_length: ClassVar[int] = 1024
    #: ``decoder_start_token_id`` from generation_config.json — id of ``▁``.
    _decoder_start_token_id: ClassVar[int] = 13764

    def __init__(
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        """Initialize the Cohere ASR model with encoder, decoder, and tokenizer."""
        super().__init__(model_files, preprocessor_factory, onnx_options)
        self._encoder = rt.InferenceSession(model_files["encoder"], **onnx_options)
        self._decoder = rt.InferenceSession(model_files["decoder"], **onnx_options)
        self._device_type, self._device_id = get_onnx_device(self._encoder)

        with model_files["tokenizer"].open("rt", encoding="utf-8") as f:
            tok_json = json.load(f)

        vocab_obj = tok_json.get("model", {}).get("vocab", {})
        if isinstance(vocab_obj, dict):
            self._token_to_id: dict[str, int] = {str(k): int(v) for k, v in vocab_obj.items()}
        elif isinstance(vocab_obj, list):  # pragma: no cover — onnx-community export uses dict shape
            self._token_to_id = {str(t[0]): int(i) for i, t in enumerate(vocab_obj)}
        else:  # pragma: no cover — defensive
            msg = f"Unexpected tokenizer.json vocab shape: {type(vocab_obj).__name__}"
            raise TypeError(msg)
        self._id_to_token: dict[int, str] = {i: t for t, i in self._token_to_id.items()}

        with model_files["tokenizer_config"].open("rt", encoding="utf-8") as f:
            tok_cfg = json.load(f)
        self._bos_token_id: int = int(tok_cfg.get("bos_token_id", self._token_to_id.get("<|startoftranscript|>", 4)))
        self._eos_token_id: int = int(tok_cfg.get("eos_token_id", self._token_to_id.get("<|endoftext|>", 3)))
        self._pad_token_id: int = int(tok_cfg.get("pad_token_id", self._token_to_id.get("<pad>", 2)))

        # Pre-resolve byte_fallback tokens (``<0x00>`` … ``<0xFF>``) → byte values.
        self._byte_fallback_id_to_byte: dict[int, int] = {}
        for b in range(256):
            tok = f"<0x{b:02X}>"
            tid = self._token_to_id.get(tok)
            if tid is not None:
                self._byte_fallback_id_to_byte[tid] = b

        # Decoder declares ``past_key_values.<L>.{decoder|encoder}.{key|value}``;
        # collect them once for fast iteration during decode steps.
        self._past_input_names: list[str] = sorted(
            i.name for i in self._decoder.get_inputs() if i.name.startswith("past_key_values.")
        )
        self._present_output_names: list[str] = sorted(
            o.name for o in self._decoder.get_outputs() if o.name.startswith("present.")
        )
        # The Cohere export has 8 decoder layers x 4 (dec.k, dec.v, enc.k, enc.v) = 32 past tensors.
        # Head count (8) and head_dim (128) are read from the input shape; first two dims are dynamic.
        first_past = next(i for i in self._decoder.get_inputs() if i.name.startswith("past_key_values."))
        # Shape layout: (batch, num_heads, seq, head_dim).
        self._num_heads: int = int(first_past.shape[1])
        self._head_dim: int = int(first_past.shape[3])

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        # TensorRT is intolerant of dynamic-shape KV-cache layouts on the
        # merged decoder; mirror :class:`WhisperHf`'s policy.
        return TensorRtOptions.get_provider_names()

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"**/encoder_model{suffix}.onnx",
            "decoder": f"**/decoder_model_merged{suffix}.onnx",
            "tokenizer": "tokenizer.json",
            "tokenizer_config": "tokenizer_config.json",
        }

    @property
    def _preprocessor_name(self) -> str:
        return "cohere_asr_128mel"

    def _resolve_lang_token(self, language: str | None) -> str:
        """Return the language-marker token string (with ``<|...|>``) for the prompt."""
        if language and language.lower() in _COHERE_LANGUAGES:
            return f"<|{language.lower()}|>"
        # ``<|unklang|>`` is the model's "unknown / out-of-range" sentinel.
        return "<|unklang|>" if language is None else "<|en|>"

    def _build_prompt(self, language: str | None, *, punctuation: bool = True) -> list[int]:
        """Build the decoder prompt token-id sequence per HF ``CohereAsrProcessor``."""
        pnc_token = "<|pnc|>" if punctuation else "<|nopnc|>"
        lang_token = self._resolve_lang_token(language)
        tokens = [
            "▁",
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            lang_token,
            lang_token,
            pnc_token,
            "<|noitn|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
        ]
        prompt: list[int] = []
        for tok in tokens:
            tid = self._token_to_id.get(tok)
            if tid is None:
                msg = f"Cohere tokenizer is missing required prompt token {tok!r}."
                raise RuntimeError(msg)
            prompt.append(tid)
        return prompt

    def _encode(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]
    ) -> OrtValue:
        input_features, _features_lens = self._preprocessor(waveforms, waveforms_len)
        # The preprocessor already produces (B, T, 128) to match the encoder's
        # declared shape — no transpose needed.
        binding = self._encoder.io_binding()
        binding.bind_cpu_input("input_features", input_features)
        binding.bind_output("last_hidden_state", self._device_type, self._device_id)
        self._encoder.run_with_iobinding(binding)
        last_hidden_state: OrtValue = binding.get_outputs()[0]
        return last_hidden_state

    def _create_empty_state(self, batch_size: int) -> dict[str, OrtValue]:
        """Allocate empty KV-cache tensors for the first decoder call.

        Both decoder.* and encoder.* past tensors start with sequence length 0;
        the merged decoder switches branches based on the past length rather
        than an explicit ``use_cache_branch`` flag.
        """
        from onnxruntime import OrtValue  # noqa: PLC0415

        empty = np.zeros((batch_size, self._num_heads, 0, self._head_dim), dtype=np.float32)
        return {name: OrtValue.ortvalue_from_numpy(empty) for name in self._past_input_names}

    def _decode_step(
        self,
        input_ids: npt.NDArray[np.int64],
        attention_mask: npt.NDArray[np.int64],
        position_ids: npt.NDArray[np.int64],
        encoder_out: OrtValue,
        prev_state: dict[str, OrtValue],
    ) -> tuple[npt.NDArray[np.float32], dict[str, OrtValue]]:
        """Run one decoder pass; return ``(last_step_logits, next_state)``."""
        binding = self._decoder.io_binding()
        binding.bind_cpu_input("input_ids", input_ids)
        binding.bind_cpu_input("attention_mask", attention_mask)
        binding.bind_cpu_input("position_ids", position_ids)
        binding.bind_cpu_input("num_logits_to_keep", np.array(1, dtype=np.int64))
        binding.bind_ortvalue_input("encoder_hidden_states", encoder_out)
        for name in self._past_input_names:
            binding.bind_ortvalue_input(name, prev_state[name])
        binding.bind_output("logits")
        for name in self._present_output_names:
            binding.bind_output(name, self._device_type, self._device_id)

        self._decoder.run_with_iobinding(binding)
        outputs = binding.get_outputs()
        logits = outputs[0].numpy()
        assert is_float32_array(logits)

        # outputs layout: [logits, present.0.dec.k, present.0.dec.v, present.0.enc.k, present.0.enc.v, ...]
        next_state: dict[str, OrtValue] = {}
        for past_name, present_name, ort_val in zip(
            self._past_input_names, self._present_output_names, outputs[1:], strict=True
        ):
            # ``past_key_values.N.{decoder|encoder}.{key|value}`` ↔ ``present.N.{decoder|encoder}.{key|value}``
            _ = present_name  # captured only so strict=True validates ordering
            next_state[past_name] = ort_val
        return logits, next_state

    def _decode_text(self, tokens: list[int] | npt.NDArray[np.int64]) -> str:
        """Decode token ids to text via SentencePiece-style ``▁``-replace + byte fuse.

        Mirrors tokenizer.json ``decoder`` sequence: Replace ``▁`` → space →
        ByteFallback (collect ``<0xXX>`` ids into a single byte buffer that's
        decoded as UTF-8 with ``errors='replace'``) → Fuse (concatenate).
        """
        out_parts: list[str] = []
        byte_buf = bytearray()
        for raw_id in tokens:
            tid = int(raw_id)
            if tid in self._byte_fallback_id_to_byte:
                byte_buf.append(self._byte_fallback_id_to_byte[tid])
                continue
            if byte_buf:
                out_parts.append(byte_buf.decode("utf-8", errors="replace"))
                byte_buf.clear()
            token = self._id_to_token.get(tid)
            if token is None or _strip_special(token):
                continue
            out_parts.append(token.replace("▁", " "))
        if byte_buf:
            out_parts.append(byte_buf.decode("utf-8", errors="replace"))
        return "".join(out_parts).removeprefix(" ")

    def _decoding(
        self,
        input_encoding: OrtValue,
        prompts: npt.NDArray[np.int64],
        max_length: int | None = None,
    ) -> list[list[int]]:
        """Greedy autoregressive decode for a batch of prompts.

        Each row of ``prompts`` is a complete prompt (e.g. the 10-token Cohere
        prompt). Generation runs per-row independently because the decoder
        graph does not support per-batch early termination; once any row hits
        EOS, that row's tokens are frozen to ``eos_token_id`` for the
        remaining steps and the loop exits when all rows are EOS.
        """
        batch_size, prompt_len = prompts.shape
        max_len = max_length or self._max_decode_length

        # First call: full prompt as input, zero past, position_ids = 0..prompt_len-1.
        state = self._create_empty_state(batch_size)
        attention_mask = np.ones((batch_size, prompt_len), dtype=np.int64)
        position_ids = np.tile(np.arange(prompt_len, dtype=np.int64), (batch_size, 1))
        logits, state = self._decode_step(prompts, attention_mask, position_ids, input_encoding, state)
        next_tokens = logits[:, -1].argmax(axis=-1).astype(np.int64)

        generated: list[list[int]] = [[int(next_tokens[i])] for i in range(batch_size)]
        finished = next_tokens == self._eos_token_id
        current_pos = prompt_len  # position of the token JUST emitted (= prompt_len)

        for _step in range(prompt_len + 1, max_len):
            if finished.all():
                break
            # Feed only the last token; cache supplies the rest.
            input_ids = next_tokens.reshape(batch_size, 1)
            attention_mask = np.ones((batch_size, current_pos + 1), dtype=np.int64)
            position_ids = np.full((batch_size, 1), current_pos, dtype=np.int64)
            logits, state = self._decode_step(input_ids, attention_mask, position_ids, input_encoding, state)
            next_tokens = logits[:, -1].argmax(axis=-1).astype(np.int64)
            # Freeze finished rows.
            next_tokens = np.where(finished, self._eos_token_id, next_tokens)
            for i in range(batch_size):
                if not finished[i]:
                    generated[i].append(int(next_tokens[i]))
            finished = finished | (next_tokens == self._eos_token_id)
            current_pos += 1

        return generated

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        """Run ASR on a batch of mono 16 kHz waveforms; yield per-clip results."""
        batch_size = int(waveforms.shape[0])
        input_encoding = self._encode(waveforms, waveforms_len)

        lang_raw = kwargs.get("language")
        language = str(lang_raw) if isinstance(lang_raw, str) else None
        pnc_raw = kwargs.get("punctuation")
        punctuation = True if pnc_raw is None else bool(pnc_raw)
        prompt = self._build_prompt(language, punctuation=punctuation)
        prompts = np.tile(np.asarray(prompt, dtype=np.int64), (batch_size, 1))

        max_len_raw = kwargs.get("max_new_tokens")
        max_new_tokens = int(max_len_raw) if isinstance(max_len_raw, int) else None
        max_length = (
            min(self._max_decode_length, len(prompt) + max_new_tokens)
            if max_new_tokens is not None
            else self._max_decode_length
        )

        generated = self._decoding(input_encoding, prompts, max_length=max_length)
        for row in generated:
            # Drop trailing EOS sentinels (kept earlier only to flag finished rows).
            trimmed: list[int] = []
            for tid in row:
                if tid == self._eos_token_id:
                    break
                trimmed.append(tid)
            yield TimestampedResult(self._decode_text(trimmed))
