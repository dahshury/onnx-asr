r"""IBM Granite Speech ASR — Conformer encoder + Q-Former projector + Granite LLM decoder.

The HuggingFace ``onnx-community/granite-4.0-1b-speech-ONNX`` export (and any
future ``granite_speech`` ``model_type`` checkpoint, e.g. ``granite-speech-4.1``)
ships a three-graph layout where the multimodal embed-injection trick is
externalised to Python:

* ``onnx/audio_encoder.onnx`` (+ ``_fp16`` / ``_q4`` / ``_q4f16`` /
  ``_quantized``) — ``input_features (B, T, 160)`` → ``audio_features (B, T_a,
  hidden_size)``. ``T`` is the **post-stack** mel-frame count (the HF feature
  extractor concatenates two adjacent 80-mel frames into one 160-d frame, so
  ``T = mel_length // 2``). The Q-Former / BLIP-2 projector is fused into this
  graph, hence the output is already in the LLM's hidden-state space
  (``hidden_size = 2048`` for ``granite-4.0-1b``). ``T_a`` is the projector's
  output length: ``3 * ((T + 14) // 15)`` per the published config
  (``projector_downsample_rate=5``, ``projector_window_size=15``).
* ``onnx/embed_tokens.onnx`` (+ siblings) — ``input_ids (B, L)`` →
  ``inputs_embeds (B, L, hidden_size)``. The text embedding table is split
  out from the decoder graph so the large 100k-token table can be
  externalised / quantised independently.
* ``onnx/decoder_model_merged.onnx`` (+ siblings) — the autoregressive
  Granite-LLM body. Inputs: ``inputs_embeds (B, L, hidden_size)``,
  ``attention_mask (B, total_seq_len)`` and ``past_key_values.<N>.{key,value}``
  KV cache tensors with grouped-query layout (4 KV heads, ``head_dim=128``).
  Outputs: ``logits (B, L, vocab=100353)`` plus the updated ``present.*`` KV.
  Notably **no ``position_ids`` input** — RoPE position is implicit in the KV-
  cache length. The merged switch (no-cache vs. with-cache branch) is
  implicit in the past-tensor sequence length.

Tokenizer is a GPT-2-style **byte-level BPE** (``ByteLevel`` pre-tokenizer,
``BPE`` model, no ``byte_fallback`` — every byte is reachable via the
``bytes_to_unicode`` mapping so the vocab covers all 256 bytes natively).
The hand-rolled encoder/decoder lives in this file, mirroring the
``transformers``/``tokenizers`` library behaviour byte-for-byte for the
canonical chat prompt (verified against ``tokenizers.Tokenizer.from_file``).

Prompt format from ``chat_template.jinja``::

    USER: {{ content }}\n ASSISTANT:

For pure transcription the user content is the single ``<|audio|>`` placeholder
token (id ``100352``). Decoded tokens for the prompt
``"USER: <|audio|>\n ASSISTANT:"`` are::

    [6584, 25, 220, 100352, 198, 36660, 3931, 2891, 25]
    ['USER', ':', 'Ġ', '<|audio|>', 'Ċ', 'ĠASS', 'IST', 'ANT', ':']

The audio injection step then splices ``T_a`` audio embeddings into
``inputs_embeds`` at the position of the ``<|audio|>`` id, producing the final
decoder input sequence.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import BaseAsr, ModelCapabilities, Preprocessor, TimestampedResult
from onnx_asr.onnx import OnnxSessionOptions, TensorRtOptions
from onnx_asr.utils import is_float32_array

# GPT-2's ``bytes_to_unicode`` (vendored here so we don't import from the
# Whisper subpackage and keep concerns separate). The map sends every byte
# 0..255 to a printable Unicode codepoint that the BPE pieces are stored as.
#
# Inverse direction (``unicode_char → byte``) is used by :meth:`_decode_text`
# to reconstruct the UTF-8 byte stream before decoding to a Python ``str``.


def _bytes_to_unicode() -> dict[int, str]:
    """Build the GPT-2 byte→printable-unicode mapping.

    Pulled out as a private helper so this module is self-contained (no
    cross-module import from :mod:`onnx_asr.models.whisper._base`).
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs], strict=True))


# GPT-2 pre-tokenisation regex, restricted to **ASCII** letter/number classes.
# The original ``tokenizers`` ByteLevel pre-tokenizer uses Unicode classes
# (``\p{L}``, ``\p{N}``) but Python's stdlib ``re`` lacks Unicode-property
# support. For the static chat prompt — pure ASCII — the substitution
# produces an identical split. Generated *text* is decoded id→string, never
# re-tokenised, so this asymmetry is safe in practice.
_GPT2_PRE_TOKEN_PAT: re.Pattern[str] = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)


class GraniteSpeech(BaseAsr):
    """IBM Granite Speech ASR — Conformer encoder + Q-Former projector + Granite LLM decoder.

    Supports any ``onnx-community/granite-*-speech-ONNX`` export whose
    ``config.json`` reports ``model_type == "granite_speech"``. The same
    class transparently handles future parameter-count bumps
    (e.g. ``granite-speech-4.1-2b``) — only the bundled weights change.

    The class composes three ONNX graphs (encoder, embed-tokens, merged
    decoder) plus a hand-rolled byte-level BPE encoder/decoder. There is no
    ``tokenizers``/``transformers``/``sentencepiece`` runtime dependency.
    """

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        streaming_native=False,
        is_multilingual=True,
    )

    #: Hard cap on decoder steps. Granite-4.0-1b ``text_config.max_position_embeddings``
    #: is 4096; we leave headroom for the (prompt + audio_embed) prefix that's
    #: variable per utterance and cap at a more modest value so a runaway
    #: decode doesn't burn 30 s of compute.
    _max_decode_length: ClassVar[int] = 1024

    def __init__(
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ) -> None:
        """Initialize the Granite Speech model — three ORT sessions + tokenizer state."""
        super().__init__(model_files, preprocessor_factory, onnx_options)
        self._audio_encoder = rt.InferenceSession(model_files["audio_encoder"], **onnx_options)
        self._embed_tokens = rt.InferenceSession(model_files["embed_tokens"], **onnx_options)
        self._decoder = rt.InferenceSession(model_files["decoder"], **onnx_options)

        # ---------------- tokenizer parsing ----------------
        with model_files["tokenizer"].open("rt", encoding="utf-8") as f:
            tok_json = json.load(f)

        # Vocab is ``{piece_str: id}``; merges are ``[[a, b], ...]`` in rank order.
        vocab_obj = tok_json["model"]["vocab"]
        if not isinstance(vocab_obj, dict):  # pragma: no cover — defensive
            msg = f"Expected dict-shape vocab, got {type(vocab_obj).__name__}"
            raise TypeError(msg)
        self._token_to_id: dict[str, int] = {str(k): int(v) for k, v in vocab_obj.items()}
        self._id_to_token: dict[int, str] = {i: t for t, i in self._token_to_id.items()}
        # Folded-in specials (``<|pad|>``, ``<|audio|>`` …) live in ``added_tokens``
        # rather than ``model.vocab``; merge them into the id↔token maps so
        # decode can find them. Track which ids are "special" so plain-text
        # decode skips them.
        self._special_token_ids: set[int] = set()
        for entry in tok_json.get("added_tokens", []) or []:
            tid = int(entry["id"])
            content = str(entry.get("content", ""))
            self._id_to_token[tid] = content
            self._token_to_id.setdefault(content, tid)
            if entry.get("special"):
                self._special_token_ids.add(tid)
        # Merge rank table (rank = position in the merges list; lower rank wins).
        self._bpe_ranks: dict[tuple[str, str], int] = {
            (str(m[0]), str(m[1])): rank for rank, m in enumerate(tok_json["model"]["merges"])
        }

        # ---------------- tokenizer config (eos/pad/audio ids) ----------------
        if "tokenizer_config" in model_files:
            with model_files["tokenizer_config"].open("rt", encoding="utf-8") as f:
                tok_cfg = json.load(f)
        else:  # pragma: no cover — every published export ships this file
            tok_cfg = {}

        def _id_from_cfg(tok_field: str, default_str: str) -> int:
            """Resolve eos/pad/bos id from tokenizer_config — accepts str OR dict."""
            raw = tok_cfg.get(tok_field, default_str)
            if isinstance(raw, dict):
                raw = raw.get("content", default_str)
            return int(self._token_to_id.get(str(raw), self._token_to_id[default_str]))

        self._bos_token_id: int = _id_from_cfg("bos_token", "<|end_of_text|>")
        self._eos_token_id: int = _id_from_cfg("eos_token", "<|end_of_text|>")
        self._pad_token_id: int = _id_from_cfg("pad_token", "<|pad|>")
        # ``<|audio|>`` placeholder id — verified against config.json's
        # ``audio_token_index`` field too (both should agree).
        audio_tok_id_raw = 0
        if "config" in model_files:
            with model_files["config"].open("rt", encoding="utf-8") as f:
                model_cfg = json.load(f)
            audio_tok_id_raw = int(model_cfg.get("audio_token_index", 0))
        # Fallback: look up the literal "<|audio|>" string in the vocab.
        self._audio_token_id: int = audio_tok_id_raw or int(self._token_to_id["<|audio|>"])

        # ---------------- byte-level codec ----------------
        self._byte_encoder: dict[int, str] = _bytes_to_unicode()
        self._byte_decoder: dict[str, int] = {ch: b for b, ch in self._byte_encoder.items()}

        # ---------------- decoder KV-cache shape ----------------
        # The merged Granite decoder emits ``past_key_values.<N>.{key,value}``
        # with grouped-query layout ``(batch, num_kv_heads, seq, head_dim)``.
        self._past_input_names: list[str] = sorted(
            i.name for i in self._decoder.get_inputs() if i.name.startswith("past_key_values.")
        )
        self._present_output_names: list[str] = sorted(
            o.name for o in self._decoder.get_outputs() if o.name.startswith("present.")
        )
        first_past = next(i for i in self._decoder.get_inputs() if i.name.startswith("past_key_values."))
        self._num_kv_heads: int = int(first_past.shape[1])
        self._head_dim: int = int(first_past.shape[3])

        # Pre-encode the static chat prompt once. The audio placeholder gets
        # spliced in at recognise-time via :meth:`_inject_audio_embeds` —
        # all that the BPE encoder needs to know is *where* the placeholder
        # ends up in the id stream.
        self._prompt_ids: list[int] = self._encode_chat_prompt()

    # ------------------------------------------------------------ static infra

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        # TensorRT EP doesn't tolerate the dynamic-shape KV cache + audio
        # injection layout; same policy as ``WhisperHf`` / ``CohereAsr``.
        return TensorRtOptions.get_provider_names()

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "audio_encoder": f"**/audio_encoder{suffix}.onnx",
            "embed_tokens": f"**/embed_tokens{suffix}.onnx",
            "decoder": f"**/decoder_model_merged{suffix}.onnx",
            "tokenizer": "tokenizer.json",
            "tokenizer_config": "tokenizer_config.json",
        }

    @property
    def _preprocessor_name(self) -> str:
        return "granite_speech_80mel"

    # ------------------------------------------------------- BPE encode helpers

    @staticmethod
    def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
        """Return the set of adjacent piece pairs in ``word``."""
        pairs: set[tuple[str, str]] = set()
        prev = word[0]
        for ch in word[1:]:
            pairs.add((prev, ch))
            prev = ch
        return pairs

    def _bpe(self, token: str) -> list[str]:
        """Greedy BPE merge: repeatedly fuse the lowest-rank adjacent pair.

        Pure-Python clone of ``tokenizers``'s rust BPE — verified to produce
        identical id sequences against the reference encoder for the static
        chat prompt.
        """
        if len(token) <= 1:
            return [token]
        word: tuple[str, ...] = tuple(token)
        pairs = self._get_pairs(word)
        if not pairs:
            return list(word)
        while True:
            # Pick the pair with the lowest rank (= highest merge priority).
            best = min(pairs, key=lambda p: self._bpe_ranks.get(p, len(self._bpe_ranks)))
            if best not in self._bpe_ranks:
                break
            first, second = best
            new_word: list[str] = []
            i = 0
            while i < len(word):
                # Skip ahead to the next occurrence of ``first``.
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
        return list(word)

    def _encode_text_segment(self, segment: str) -> list[int]:
        """Byte-level-BPE-encode an ASCII-or-UTF-8 text segment to ids.

        Steps mirror the rust ``ByteLevel`` + ``BPE`` pipeline:

        1. Pre-tokenize: split on the GPT-2 regex into "words" / runs.
        2. UTF-8-encode each pre-token; remap every byte through the
           ``bytes_to_unicode`` table into a printable-unicode string.
        3. Greedy BPE merge against ``merges``.
        4. Look every piece up in ``vocab``.

        Raises:
            RuntimeError: a BPE piece is missing from the vocab — would
                indicate non-ASCII input the ASCII pre-tokenizer regex can't
                split. Should never fire for the static chat prompt.

        """
        out: list[int] = []
        for pre_token in _GPT2_PRE_TOKEN_PAT.findall(segment):
            encoded = "".join(self._byte_encoder[b] for b in pre_token.encode("utf-8"))
            for piece in self._bpe(encoded):
                tid = self._token_to_id.get(piece)
                if tid is None:  # pragma: no cover — unreachable for ASCII prompts
                    msg = f"Granite tokenizer: BPE piece {piece!r} (from {pre_token!r}) is not in vocab."
                    raise RuntimeError(msg)
                out.append(tid)
        return out

    def _encode_chat_prompt(self) -> list[int]:
        r"""Tokenize the static chat prompt ``USER: <|audio|>\n ASSISTANT:``.

        Splits on the ``<|audio|>`` special marker (it's an added-tokens
        entry, not part of ``model.vocab``, so the BPE encoder can't reach
        it); byte-level-BPE-encodes the two text halves around it; stitches
        them back together with the audio token id in the middle.

        The chat template from ``tokenizer_config.json`` is::

            USER: {{ content }}\n ASSISTANT:

        Verified token sequence (against tokenizers @ 0.22)::

            [6584, 25, 220, 100352, 198, 36660, 3931, 2891, 25]
            ['USER', ':', 'Ġ', '<|audio|>', 'Ċ', 'ĠASS', 'IST', 'ANT', ':']
        """
        before, after = "USER: ", "\n ASSISTANT:"
        return [*self._encode_text_segment(before), self._audio_token_id, *self._encode_text_segment(after)]

    # --------------------------------------------------------- BPE decode (id→text)

    def _decode_text(self, ids: list[int] | npt.NDArray[np.int64]) -> str:
        """Render generated token ids to plain text (byte-level inverse).

        Mirrors the rust ``ByteLevel`` decoder: concatenate every non-special
        piece into a unicode string, map each char back through
        ``bytes_to_unicode`` to recover the original UTF-8 byte stream, then
        decode as UTF-8 (replace errors — robust to mid-multibyte cuts on
        early-stopped decodes).
        """
        chars: list[str] = []
        for raw in ids:
            tid = int(raw)
            if tid in self._special_token_ids:
                continue
            piece = self._id_to_token.get(tid)
            if piece is None:
                continue
            chars.append(piece)
        unicode_str = "".join(chars)
        try:
            raw_bytes = bytes(self._byte_decoder[c] for c in unicode_str)
        except KeyError:
            # Defensive: if a non-byte-level char slips through, fall back to
            # UTF-8-encoding the unicode string directly. Should never fire
            # for the published Granite tokenizer.
            return unicode_str
        return raw_bytes.decode("utf-8", errors="replace")

    # ------------------------------------------------------------ audio embed

    def _encode_audio(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float32]:
        """Run mel preprocessing + audio_encoder graph → projected audio embeddings.

        Returns:
            ``(B, T_a, hidden_size)`` float32 — the Q-Former projector output
            in the LLM's hidden-state space.

        """
        input_features, _feature_lens = self._preprocessor(waveforms, waveforms_len)
        # Audio encoder expects ``(B, T, 160)`` — the preprocessor produces
        # this layout natively (stack-by-2 of the 80-mel sequence).
        (audio_features,) = self._audio_encoder.run(["audio_features"], {"input_features": input_features})
        assert is_float32_array(audio_features)
        return audio_features

    # ---------------------------------------------------------- embed lookup

    def _embed(self, input_ids: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
        """Run the embed_tokens lookup table — ``(B, L)`` → ``(B, L, hidden_size)``."""
        (inputs_embeds,) = self._embed_tokens.run(["inputs_embeds"], {"input_ids": input_ids})
        assert is_float32_array(inputs_embeds)
        return inputs_embeds

    # ----------------------------------------------------- audio injection

    def _inject_audio_embeds(
        self,
        text_embeds: npt.NDArray[np.float32],
        audio_embeds: npt.NDArray[np.float32],
        prompt_ids: list[int],
    ) -> npt.NDArray[np.float32]:
        """Splice ``audio_embeds`` into ``text_embeds`` at the ``<|audio|>`` slot.

        ``text_embeds`` has shape ``(1, L, H)`` with one position equal to the
        ``<|audio|>`` placeholder. Replace that single position with the
        ``(1, T_a, H)`` audio embedding sequence and return ``(1, L - 1 + T_a, H)``.

        Only the first ``<|audio|>`` occurrence is honoured (Granite Speech is
        single-audio per prompt).
        """
        audio_pos = prompt_ids.index(self._audio_token_id)
        return np.concatenate(
            [
                text_embeds[:, :audio_pos, :],
                audio_embeds,
                text_embeds[:, audio_pos + 1 :, :],
            ],
            axis=1,
        )

    # ----------------------------------------------------------- decoding

    def _create_empty_kv_state(self, batch_size: int) -> dict[str, npt.NDArray[np.float32]]:
        """Allocate ``past_sequence_length=0`` KV cache tensors for the first decoder call."""
        empty = np.zeros((batch_size, self._num_kv_heads, 0, self._head_dim), dtype=np.float32)
        return dict.fromkeys(self._past_input_names, empty)

    def _decode_step(
        self,
        inputs_embeds: npt.NDArray[np.float32],
        attention_mask: npt.NDArray[np.int64],
        prev_state: dict[str, npt.NDArray[np.float32]],
    ) -> tuple[npt.NDArray[np.float32], dict[str, npt.NDArray[np.float32]]]:
        """One decoder forward pass.

        Inputs:
            inputs_embeds: ``(B, L_step, H)`` — the LLM hidden-state slice for
                this step. ``L_step == prompt_len + T_a`` on step 0, ``1``
                thereafter.
            attention_mask: ``(B, total_seq_len)`` — 1s for valid positions.
                Includes both the cached prefix and the current step.
            prev_state: ``past_key_values.*`` numpy arrays.

        Returns:
            ``(logits, next_state)`` — ``logits`` shape ``(B, L_step, vocab)``.

        """
        feeds: dict[str, npt.NDArray[np.float32] | npt.NDArray[np.int64]] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }
        for name in self._past_input_names:
            feeds[name] = prev_state[name]

        outputs = self._decoder.run(["logits", *self._present_output_names], feeds)
        logits = outputs[0]
        assert is_float32_array(logits)
        # Map present.<N>.{key,value} → past_key_values.<N>.{key,value}.
        next_state: dict[str, npt.NDArray[np.float32]] = {}
        for present_name, arr in zip(self._present_output_names, outputs[1:], strict=True):
            assert is_float32_array(arr)
            past_name = present_name.replace("present.", "past_key_values.", 1)
            next_state[past_name] = arr
        return logits, next_state

    def _decode_greedy(
        self,
        prefix_embeds: npt.NDArray[np.float32],
        max_new_tokens: int,
    ) -> list[int]:
        """Greedy autoregressive decode for a single utterance.

        ``prefix_embeds`` is ``(1, L_prefix, H)`` — the prompt-plus-audio
        embedding stream. Returns the list of generated token ids (excluding
        the prefix, including any trailing EOS).
        """
        state = self._create_empty_kv_state(batch_size=1)
        prefix_len = int(prefix_embeds.shape[1])
        attention_mask = np.ones((1, prefix_len), dtype=np.int64)
        logits, state = self._decode_step(prefix_embeds, attention_mask, state)
        next_token = int(logits[0, -1].argmax())
        generated: list[int] = [next_token]
        current_pos = prefix_len  # length INCLUDING the just-emitted token's "slot"

        for _step in range(max_new_tokens - 1):
            if next_token == self._eos_token_id:
                break
            # Embed the single next token, then run one more decoder step.
            step_embeds = self._embed(np.array([[next_token]], dtype=np.int64))
            attention_mask = np.ones((1, current_pos + 1), dtype=np.int64)
            logits, state = self._decode_step(step_embeds, attention_mask, state)
            next_token = int(logits[0, -1].argmax())
            generated.append(next_token)
            current_pos += 1

        return generated

    # ----------------------------------------------------------- public API

    def recognize_batch(
        self,
        waveforms: npt.NDArray[np.float32],
        waveforms_len: npt.NDArray[np.int64],
        /,
        **kwargs: object | None,
    ) -> Iterator[TimestampedResult]:
        """Encode → inject → autoregressively decode every waveform in the batch.

        Granite Speech is batch-of-one in practice — the audio embed length is
        per-utterance and the decoder graph isn't padding-aware on
        ``inputs_embeds``. We iterate rows to keep semantics clean; the cost
        is identical to a true batched decode (the LLM dominates either way).
        """
        max_new_tokens_raw = kwargs.get("max_new_tokens")
        max_new_tokens = (
            int(max_new_tokens_raw) if isinstance(max_new_tokens_raw, int) else self._max_decode_length
        )

        prompt_ids = np.asarray(self._prompt_ids, dtype=np.int64)[None, :]  # (1, L_prompt)

        for i in range(int(waveforms.shape[0])):
            valid_len = int(waveforms_len[i])
            if valid_len <= 0:
                yield TimestampedResult(text="")
                continue
            row_wf = waveforms[i : i + 1, :valid_len].astype(np.float32, copy=False)
            row_len = np.array([valid_len], dtype=np.int64)

            # Audio path: mel → encoder → (1, T_a, H)
            audio_embeds = self._encode_audio(row_wf, row_len)
            # Text path: prompt_ids → embed_tokens → (1, L_prompt, H)
            text_embeds = self._embed(prompt_ids)
            # Splice: replace <|audio|> position with the audio_embeds run.
            prefix_embeds = self._inject_audio_embeds(text_embeds, audio_embeds, self._prompt_ids)

            generated = self._decode_greedy(prefix_embeds, max_new_tokens=max_new_tokens)

            # Strip trailing EOS sentinel before rendering.
            trimmed: list[int] = []
            for tid in generated:
                if tid == self._eos_token_id:
                    break
                trimmed.append(tid)
            yield TimestampedResult(text=self._decode_text(trimmed))
