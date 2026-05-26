"""Unit tests for :class:`onnx_asr.models.moonshine.Moonshine`'s hand-rolled tokenizer.

We never touch the ONNX sessions here — these tests focus on the SentencePiece /
byte-fallback decode pipeline parsed from ``tokenizer.json``, which is the only
non-numeric bit of glue that's plausibly easy to get subtly wrong.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from onnx_asr.models.moonshine import Moonshine


def _make_tokenizer_files(tmp_path: Path) -> tuple[Path, Path]:
    """Write a minimal but realistic moonshine-style tokenizer.json + config to disk."""
    tokenizer = {
        "model": {
            "type": "BPE",
            "vocab": {
                "<unk>": 0,
                "<s>": 1,
                "</s>": 2,
                # byte-fallback pieces
                "<0x41>": 3,  # 'A'
                "<0xE2>": 4,  # first byte of U+2603 (snowman)
                "<0x98>": 5,
                "<0x83>": 6,
                # normal pieces
                "▁Hello": 7,
                "▁world": 8,
                "!": 9,
                # a piece that happens to be 6 chars but is NOT byte-fallback
                "<wow!>": 10,
            },
        },
        "added_tokens": [
            {"id": 0, "content": "<unk>", "special": True},
            {"id": 1, "content": "<s>", "special": True},
            {"id": 2, "content": "</s>", "special": True},
            {"id": 32000, "content": "<<ST_0>>", "special": True},
        ],
    }
    tok_path = tmp_path / "tokenizer.json"
    tok_path.write_text(json.dumps(tokenizer), encoding="utf-8")

    cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "added_tokens_decoder": {
            "32001": {"content": "<<ST_1>>", "special": True},
        },
    }
    cfg_path = tmp_path / "tokenizer_config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return tok_path, cfg_path


@pytest.fixture
def moonshine_no_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Moonshine:
    """Build a :class:`Moonshine` instance whose ONNX session constructor is stubbed out.

    We never call any inference method in these tests — we just need a live object
    with the tokenizer maps populated so we can exercise :meth:`_decode_text`.
    """
    tok_path, cfg_path = _make_tokenizer_files(tmp_path)
    encoder_path = tmp_path / "encoder.onnx"
    decoder_path = tmp_path / "decoder.onnx"
    decoder_with_past_path = tmp_path / "decoder_with_past.onnx"
    for p in (encoder_path, decoder_path, decoder_with_past_path):
        p.write_bytes(b"")

    fake_session = types.SimpleNamespace(
        get_inputs=list,
        get_outputs=list,
    )
    import onnxruntime as rt  # noqa: PLC0415

    monkeypatch.setattr(rt, "InferenceSession", lambda *_args, **_kwargs: fake_session)

    def fake_preproc_factory(_name: str) -> object:
        return object()

    model_files = {
        "encoder": encoder_path,
        "decoder": decoder_path,
        "decoder_with_past": decoder_with_past_path,
        "tokenizer": tok_path,
        "tokenizer_config": cfg_path,
    }
    return Moonshine(model_files, fake_preproc_factory, {})


def test_decode_plain_text(moonshine_no_sessions: Moonshine) -> None:
    """SentencePiece ``▁`` becomes a space and the leading space is stripped."""
    text = moonshine_no_sessions._decode_text([7, 8, 9])  # "▁Hello" "▁world" "!"
    assert text == "Hello world!"


def test_decode_strips_specials(moonshine_no_sessions: Moonshine) -> None:
    """``<s>``, ``</s>`` and ``<<ST_*>>`` are never rendered into plain-text output."""
    text = moonshine_no_sessions._decode_text([1, 7, 8, 32000, 9, 2])
    assert text == "Hello world!"


def test_decode_byte_fallback_ascii(moonshine_no_sessions: Moonshine) -> None:
    """A single byte-fallback ``<0xNN>`` piece round-trips through UTF-8."""
    text = moonshine_no_sessions._decode_text([3])  # <0x41> = 'A'
    assert text == "A"


def test_decode_byte_fallback_multibyte(moonshine_no_sessions: Moonshine) -> None:
    """A run of byte-fallback pieces decodes as a single UTF-8 codepoint."""
    text = moonshine_no_sessions._decode_text([4, 5, 6])  # ☃
    assert text == "☃"


def test_decode_byte_fallback_does_not_swallow_lookalike_token(
    moonshine_no_sessions: Moonshine,
) -> None:
    """A 6-char piece that isn't ``<0x..>`` shape is emitted verbatim, not parsed as a byte."""
    text = moonshine_no_sessions._decode_text([10])
    assert text == "<wow!>"


def test_bos_eos_ids_resolved(moonshine_no_sessions: Moonshine) -> None:
    """``<s>`` / ``</s>`` ids are picked up from the JSON metadata, not hard-coded."""
    assert moonshine_no_sessions._bos_id == 1
    assert moonshine_no_sessions._eos_id == 2


def test_added_tokens_decoder_overlay(moonshine_no_sessions: Moonshine) -> None:
    """Specials referenced only in ``tokenizer_config.json`` are still recognized as specials."""
    # 32001 only appears in tokenizer_config.json's added_tokens_decoder, not in
    # tokenizer.json's added_tokens. We must still skip it on plain-text decode.
    text = moonshine_no_sessions._decode_text([7, 32001, 9])
    assert text == "Hello!"


def test_capabilities_flags() -> None:
    """Moonshine ships as English-only / no native streaming wrapper (today)."""
    caps = Moonshine.capabilities
    assert caps.streaming_native is False
    assert caps.is_multilingual is False


@pytest.mark.skipif(sys.version_info < (3, 10), reason="match statement is 3.10+")
def test_get_model_files_quantization() -> None:
    """The file-glob map honours the ``?<quant>`` suffix convention."""
    no_quant = Moonshine._get_model_files()
    int8 = Moonshine._get_model_files("int8")
    assert no_quant["encoder"] == "**/encoder_model.onnx"
    assert int8["encoder"] == "**/encoder_model?int8.onnx"
    assert no_quant["tokenizer"] == "tokenizer.json"
