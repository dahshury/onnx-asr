"""Uppercase-vocab detection drives lowercasing of ALL-CAPS LibriSpeech models.

icefall / Kaldi zipformer-en (and similar) ship an all-uppercase BPE vocab, so
the model can only emit ALL-CAPS text. ``_AsrWithDecoding`` lowercases such
output; mixed/lowercase/non-Latin vocabs (Whisper, NeMo, Vosk, CJK) are left
untouched.
"""

from __future__ import annotations

from onnx_asr.asr import _vocab_is_uppercase


def test_librispeech_style_uppercase_vocab_is_flagged() -> None:
    vocab = {0: "<blk>", 1: " THE", 2: " AND", 3: "S", 4: "ING", 5: "1"}
    assert _vocab_is_uppercase(vocab) is True


def test_lowercase_vocab_is_not_flagged() -> None:
    vocab = {0: "<blk>", 1: " the", 2: " and", 3: "s", 4: "ing"}
    assert _vocab_is_uppercase(vocab) is False


def test_mixed_case_vocab_is_not_flagged() -> None:
    # A model that distinguishes proper-noun casing must never be lowercased.
    vocab = {0: " The", 1: " quick", 2: " Brown", 3: " fox", 4: " London"}
    assert _vocab_is_uppercase(vocab) is False


def test_caseless_vocab_is_not_flagged() -> None:
    # CJK / digits / control tokens have no cased letters → never flagged.
    vocab = {0: "<blk>", 1: "的", 2: "是", 3: "1", 4: "▁"}
    assert _vocab_is_uppercase(vocab) is False


def test_empty_vocab_is_not_flagged() -> None:
    assert _vocab_is_uppercase({}) is False
