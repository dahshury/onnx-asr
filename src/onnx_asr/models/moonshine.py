"""Moonshine ASR (Useful Sensors / onnx-community export).

Moonshine is a tiny encoder-decoder ASR (~27M params for ``tiny``,
~61M for ``base``) trained to ingest **raw 16 kHz waveform** directly —
its first encoder block does the time-frequency analysis internally, so
there is no external mel-spectrogram preprocessor. That makes it a
notably cheap drop-in for short-form English (and per-language
``moonshine-{tiny,base}-{zh,ja,ko,ar,vi}`` variants).

The ``onnx-community`` exports follow the same three-graph layout as
HuggingFace Optimum's split decoder family:

* ``onnx/encoder_model.onnx`` — raw audio ``(batch, num_samples)`` →
  ``last_hidden_state (batch, enc_T, hidden)``
* ``onnx/decoder_model.onnx`` — first decode step. Inputs
  ``input_ids`` + ``encoder_hidden_states``; emits ``logits`` plus
  ``present.{0..L-1}.{decoder,encoder}.{key,value}``.
* ``onnx/decoder_with_past_model.onnx`` — subsequent steps. Inputs
  ``input_ids`` (last token only) plus the previously cached
  ``past_key_values.{0..L-1}.{decoder,encoder}.{key,value}``; emits
  ``logits`` plus ``present.{0..L-1}.decoder.{key,value}`` ONLY (encoder
  K/V are static across the decode and feed straight back from the
  first-step output).

Unlike Whisper, Moonshine has no ``decoder_model_merged.onnx`` /
``use_cache_branch`` switch in this layout, which keeps each graph small
but means we orchestrate the first-step / past-step swap in Python.

The tokenizer is a SentencePiece BPE with byte-fallback (LlamaTokenizer-
style) shipped as ``tokenizer.json``. We parse it directly with the std
``json`` module — no ``tokenizers`` / ``sentencepiece`` / ``transformers``
runtime dependency. Decode pipeline mirrors the JSON ``decoder`` field:

1. id → token string lookup
2. byte-fallback (``<0xNN>`` → raw byte) then UTF-8 decode
3. SentencePiece ``▁`` (U+2581) → ASCII space
4. drop the single SentencePiece-prepended leading space

Each Moonshine checkpoint is already language-specialized at training
time, so the decoder prompt is just ``[bos_token_id]`` — no language /
task tokens are injected.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import (
    BaseAsr,
    ModelCapabilities,
    Preprocessor,
    TimestampedResult,
)
from onnx_asr.onnx import OnnxSessionOptions
from onnx_asr.utils import is_float32_array

if TYPE_CHECKING:
    pass


# SentencePiece "underscore" — what tokenizers/SentencePiece uses as the
# visible substitute for an ASCII space inside a token piece. The Moonshine
# decoder pipeline (parsed from ``tokenizer.json``) maps it back to a real
# space before fusing tokens, so we do the same when rendering text.
_SP_SPACE: str = "▁"
# Default max decode length when the caller doesn't supply one. Moonshine's
# ``max_position_embeddings`` is 512 in the published configs; 448 matches
# Whisper's classic cap and is plenty for a short-form ASR — going to 512
# wastes time on a runaway decode if the model gets stuck in a loop.
_DEFAULT_MAX_LENGTH: int = 448


class Moonshine(BaseAsr):
    """Useful Sensors / onnx-community Moonshine ASR.

    Works against the 3-graph ``onnx-community/moonshine-*-ONNX`` exports
    (tiny / base, English-default plus per-language ``zh/ja/ko/ar/vi``
    variants). Each variant ships the same file layout and the same
    SentencePiece tokenizer; only the trained weights differ.
    """

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        streaming_native=False,
        is_multilingual=False,
    )

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        self._encoder = rt.InferenceSession(model_files["encoder"], **onnx_options)
        self._decoder = rt.InferenceSession(model_files["decoder"], **onnx_options)
        self._decoder_with_past = rt.InferenceSession(model_files["decoder_with_past"], **onnx_options)

        # Cached layout of the past-step decoder so we can build / round-trip
        # the KV state without re-querying the session on every decode step.
        self._past_input_names: list[str] = [
            x.name for x in self._decoder_with_past.get_inputs() if x.name.startswith("past_key_values.")
        ]
        self._present_output_names: list[str] = [
            o.name for o in self._decoder.get_outputs() if o.name.startswith("present.")
        ]

        # Tokenizer state — parsed straight from the rust-tokenizers JSON.
        self._id_to_token: dict[int, str] = {}
        self._special_token_ids: set[int] = set()
        self._bos_id: int
        self._eos_id: int
        self._load_tokenizer(model_files["tokenizer"], model_files.get("tokenizer_config"))

    # ------------------------------------------------------------------ files

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"**/encoder_model{suffix}.onnx",
            "decoder": f"**/decoder_model{suffix}.onnx",
            "decoder_with_past": f"**/decoder_with_past_model{suffix}.onnx",
            "tokenizer": "tokenizer.json",
            "tokenizer_config": "tokenizer_config.json",
        }

    @property
    def _preprocessor_name(self) -> str:
        # Moonshine takes raw float32 PCM at 16 kHz — no mel / feature extractor.
        # The :class:`IdentityPreprocessor` (loader.py) maps this name to a pass-through.
        return "identity"

    # -------------------------------------------------------------- tokenizer

    def _load_tokenizer(self, tokenizer_path: Path, tokenizer_config_path: Path | None) -> None:  # noqa: C901
        """Parse ``tokenizer.json`` (+ optional ``tokenizer_config.json``) into the maps used by :meth:`_decode_text`.

        We avoid pulling in ``tokenizers``/``sentencepiece``/``transformers`` —
        the published Moonshine tokenizer is a static BPE-with-byte-fallback
        and we only need the id → text direction (the encoder is fed raw audio,
        never text). For decode, we:

        * build ``id_to_token`` from ``model.vocab`` (dict ``{piece: id}``)
        * overlay the ``added_tokens`` array (covers ``<s>``, ``</s>``,
          ``<<ST_*>>``…) so we know every reserved id
        * pull bos/eos ids and the special-token id set
        """
        with tokenizer_path.open("rt", encoding="utf-8") as f:
            tok = json.load(f)

        model = tok.get("model") or {}
        vocab = model.get("vocab") or {}
        # ``vocab`` for a BPE model is ``{piece_str: id}`` (the rust-tokenizers
        # canonical layout). Invert into id → piece. If a future export ever
        # ships it as a list (SentencePiece-unigram style) accept that too.
        if isinstance(vocab, dict):
            self._id_to_token = {int(idx): str(piece) for piece, idx in vocab.items()}
        elif isinstance(vocab, list):
            for i, entry in enumerate(vocab):
                piece = entry[0] if isinstance(entry, (list, tuple)) else entry
                self._id_to_token[i] = str(piece)

        # Added tokens (specials + the <<ST_*>> timestamp markers Moonshine
        # uses for segment boundaries) live OUTSIDE ``model.vocab``. Merge
        # them in so id lookups can't miss, and track which ids are special
        # so plain-text decode skips them.
        bos_id: int | None = None
        eos_id: int | None = None
        for entry in tok.get("added_tokens", []) or []:
            tid = int(entry["id"])
            content = str(entry.get("content", ""))
            self._id_to_token[tid] = content
            if entry.get("special"):
                self._special_token_ids.add(tid)
            if content == "<s>":
                bos_id = tid
            elif content == "</s>":
                eos_id = tid

        # tokenizer_config.json's ``added_tokens_decoder`` is the same data
        # in a slightly different shape — read it too as a belt-and-braces
        # measure in case a variant ever ships only one of the two files.
        if tokenizer_config_path is not None and tokenizer_config_path.exists():
            with tokenizer_config_path.open("rt", encoding="utf-8") as f:
                cfg = json.load(f)
            for tid_str, entry in (cfg.get("added_tokens_decoder") or {}).items():
                tid = int(tid_str)
                content = str(entry.get("content", ""))
                self._id_to_token.setdefault(tid, content)
                if entry.get("special"):
                    self._special_token_ids.add(tid)
                if content == "<s>" and bos_id is None:
                    bos_id = tid
                elif content == "</s>" and eos_id is None:
                    eos_id = tid

        # Fall back to the canonical Moonshine ids if the JSON didn't name
        # the tokens — every published variant uses ``<s>=1`` / ``</s>=2``.
        self._bos_id = bos_id if bos_id is not None else 1
        self._eos_id = eos_id if eos_id is not None else 2

    def _decode_text(self, ids: npt.NDArray[np.int64] | list[int]) -> str:
        """Render decoder token ids to plain text.

        Mirrors the JSON ``decoder`` chain shipped in ``tokenizer.json``:

        1. **id → piece** lookup (skipping any id flagged ``special``).
        2. **Byte fallback** — pieces of the form ``<0xNN>`` carry a raw
           byte; we buffer them and decode the run as UTF-8 once a non-byte
           piece (or end-of-stream) breaks the run. This is how Moonshine
           handles characters outside the BPE vocab (e.g. emoji, CJK
           glyphs that the BPE didn't merge as a single piece).
        3. **SentencePiece space** — replace ``U+2581`` with an ASCII space.
        4. **Leading-space strip** — SentencePiece always prepends one
           ``▁`` to the first non-empty piece, which surfaces as a leading
           space after step 3.
        """
        byte_buf: list[int] = []
        out_chars: list[str] = []

        def _flush_byte_buf() -> None:
            if byte_buf:
                out_chars.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                byte_buf.clear()

        for raw_id in ids:
            tid = int(raw_id)
            if tid in self._special_token_ids:
                # Bos/eos/timestamp tokens contribute no characters to plain text.
                _flush_byte_buf()
                continue
            piece = self._id_to_token.get(tid)
            if piece is None:
                _flush_byte_buf()
                continue
            # Byte-fallback pieces look like ``<0xNN>`` — three hex chars
            # between angle-bracket-zero-x and a closing bracket.
            if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
                try:
                    byte_buf.append(int(piece[3:5], 16))
                    continue
                except ValueError:
                    pass
            _flush_byte_buf()
            out_chars.append(piece)

        _flush_byte_buf()
        text = "".join(out_chars).replace(_SP_SPACE, " ")
        return text.removeprefix(" ")

    # -------------------------------------------------------------- inference

    def _encode(self, waveforms: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Run the encoder once for the whole utterance.

        Moonshine eats the raw waveform — no spectrogram, no padding to
        a fixed window. We pass ``(batch, num_samples)`` straight through.
        """
        (last_hidden_state,) = self._encoder.run(["last_hidden_state"], {"input_values": waveforms})
        assert is_float32_array(last_hidden_state)
        return last_hidden_state

    def _first_decode_step(
        self, input_ids: npt.NDArray[np.int64], encoder_hidden_states: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], dict[str, npt.NDArray[np.float32]]]:
        """Run ``decoder_model.onnx`` (no past) to seed the KV cache."""
        outputs = self._decoder.run(
            ["logits", *self._present_output_names],
            {"input_ids": input_ids, "encoder_hidden_states": encoder_hidden_states},
        )
        logits = outputs[0]
        assert is_float32_array(logits)
        # Match the ``present.*`` outputs against the ``past_key_values.*``
        # input names — they share the same layer/decoder-or-encoder/key-or-value
        # suffix; only the ``present`` / ``past_key_values`` prefix differs.
        present_arrays = outputs[1:]
        state: dict[str, npt.NDArray[np.float32]] = {}
        for present_name, arr in zip(self._present_output_names, present_arrays, strict=True):
            assert is_float32_array(arr)
            past_name = present_name.replace("present.", "past_key_values.", 1)
            state[past_name] = arr
        return logits, state

    def _past_decode_step(
        self, next_token: npt.NDArray[np.int64], state: dict[str, npt.NDArray[np.float32]]
    ) -> tuple[npt.NDArray[np.float32], dict[str, npt.NDArray[np.float32]]]:
        """Run ``decoder_with_past_model.onnx`` for one autoregressive step."""
        feeds: dict[str, npt.NDArray[np.float32] | npt.NDArray[np.int64]] = {"input_ids": next_token}
        for name in self._past_input_names:
            feeds[name] = state[name]
        outputs = self._decoder_with_past.run(["logits", *self._past_decoder_present_names()], feeds)
        logits = outputs[0]
        assert is_float32_array(logits)
        # Only decoder-self-attn K/V are emitted on past-steps; encoder K/V
        # are static and carried over from the first-step output unchanged.
        new_state = dict(state)
        for present_name, arr in zip(self._past_decoder_present_names(), outputs[1:], strict=True):
            assert is_float32_array(arr)
            past_name = present_name.replace("present.", "past_key_values.", 1)
            new_state[past_name] = arr
        return logits, new_state

    def _past_decoder_present_names(self) -> list[str]:
        """Return the ``present.*`` outputs the past-step decoder actually emits.

        ``decoder_with_past_model.onnx`` only recomputes the decoder-self-attn
        K/V each step (the cross-attn K/V are static once the encoder is run).
        """
        return [o.name for o in self._decoder_with_past.get_outputs() if o.name.startswith("present.")]

    def _decode_greedy(
        self,
        encoder_hidden_states: npt.NDArray[np.float32],
        max_length: int,
    ) -> npt.NDArray[np.int64]:
        """Greedy autoregressive decode for a single waveform (batch_size = 1 supported).

        Moonshine's ``recognize`` path is dominated by the encoder and the
        first decoder step; the per-step past loop is cheap. We don't
        bother with beam search — it's not part of the published baseline
        and would compound the cost.
        """
        batch_size = int(encoder_hidden_states.shape[0])
        # Seed with bos for every row in the batch.
        prompt = np.full((batch_size, 1), self._bos_id, dtype=np.int64)
        logits, state = self._first_decode_step(prompt, encoder_hidden_states)

        # argmax over the last (and only) decoded step → next token per row.
        next_tokens = logits[:, -1].argmax(axis=-1).astype(np.int64)
        tokens = np.concatenate([prompt, next_tokens[:, None]], axis=1)
        finished = next_tokens == self._eos_id

        while int(tokens.shape[1]) < max_length and not bool(finished.all()):
            step_in = next_tokens[:, None]
            logits, state = self._past_decode_step(step_in, state)
            next_tokens = logits[:, -1].argmax(axis=-1).astype(np.int64)
            # Once a row has emitted eos, freeze its emission so downstream
            # batch-stack logic stays well-defined for the (rare) batched calls.
            next_tokens = np.where(finished, self._eos_id, next_tokens)
            tokens = np.concatenate([tokens, next_tokens[:, None]], axis=1)
            finished = finished | (next_tokens == self._eos_id)

        return tokens

    # ----------------------------------------------------------- public API

    def recognize_batch(
        self,
        waveforms: npt.NDArray[np.float32],
        waveforms_len: npt.NDArray[np.int64],
        /,
        **kwargs: object | None,
    ) -> Iterator[TimestampedResult]:
        """Encode + greedy-decode every row in ``waveforms``.

        ``waveforms`` is the padded ``(batch, max_samples)`` array; rows are
        already 16-kHz float32 PCM with the per-row valid length given by
        ``waveforms_len``. We trim each row to its own length so the encoder
        doesn't waste compute (and emit timestamp-space tokens) on padding.
        """
        max_length_raw = kwargs.get("max_length")
        max_length = int(max_length_raw) if isinstance(max_length_raw, int) else _DEFAULT_MAX_LENGTH

        for i in range(int(waveforms.shape[0])):
            valid_len = int(waveforms_len[i])
            if valid_len <= 0:
                yield TimestampedResult(text="")
                continue
            wf = waveforms[i : i + 1, :valid_len].astype(np.float32, copy=False)
            encoder_out = self._encode(wf)
            tokens = self._decode_greedy(encoder_out, max_length=max_length)
            # Strip the prompt (bos) before rendering.
            text = self._decode_text(tokens[0, 1:])
            yield TimestampedResult(text=text)
