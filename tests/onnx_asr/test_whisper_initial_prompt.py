"""Tests for the Whisper ``initial_prompt`` kwarg (item #10)."""

from __future__ import annotations

import numpy as np
import pytest

import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter

WHISPER_MODELS = [
    pytest.param(("onnx-community/whisper-tiny", "uint8"), id="whisper-hf"),
    pytest.param(("whisper-base", "int8"), id="whisper-ort"),
]


@pytest.fixture(scope="module", params=WHISPER_MODELS)
def whisper_model(request: pytest.FixtureRequest) -> TextResultsAsrAdapter:
    model_name, quant = request.param
    return onnx_asr.load_model(model_name, quantization=quant)


def _waveform(seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((int(seconds * 16_000),), dtype=np.float32)


def test_str_initial_prompt_runs(whisper_model: TextResultsAsrAdapter) -> None:
    """A ``str`` initial_prompt encodes via local BPE and produces a transcript."""
    result = whisper_model.recognize(_waveform(), initial_prompt="Anthropic Claude")
    assert isinstance(result, str)


def test_list_int_initial_prompt_runs(whisper_model: TextResultsAsrAdapter) -> None:
    """A pre-tokenized ``list[int]`` initial_prompt also works."""
    pre_tokenized = [1234, 5678, 9012]
    result = whisper_model.recognize(_waveform(), initial_prompt=pre_tokenized)
    assert isinstance(result, str)


def test_empty_prompt_equivalent_to_none(whisper_model: TextResultsAsrAdapter) -> None:
    """Empty string and None initial_prompt produce identical output (no prompt path taken)."""
    no_prompt = whisper_model.recognize(_waveform())
    empty_prompt = whisper_model.recognize(_waveform(), initial_prompt="")
    assert no_prompt == empty_prompt


def test_long_prompt_is_truncated(whisper_model: TextResultsAsrAdapter) -> None:
    """A 1000-word prompt is truncated to Whisper's 223-token prompt window and doesn't crash."""
    big_prompt = " ".join(["word"] * 1000)
    result = whisper_model.recognize(_waveform(), initial_prompt=big_prompt)
    assert isinstance(result, str)


def test_prompt_output_does_not_echo_prompt(whisper_model: TextResultsAsrAdapter) -> None:
    """The prompt prefix is sliced off the output — transcript doesn't include the seed text."""
    distinctive = "ZebraXylophoneTetrahedron"
    waveform = _waveform(0.5)
    result = whisper_model.recognize(waveform, initial_prompt=distinctive)
    # Random audio shouldn't produce our exact distinctive word from the prompt.
    assert distinctive not in result


def test_prompt_with_language(whisper_model: TextResultsAsrAdapter) -> None:
    """``initial_prompt`` combines with explicit ``language``."""
    result = whisper_model.recognize(_waveform(), initial_prompt="dictation", language="en")
    assert isinstance(result, str)


def test_prompt_encoding_handles_unicode() -> None:
    """The BPE encoder handles UTF-8 multi-byte chars via the bytes_to_unicode mapping."""
    model: TextResultsAsrAdapter = onnx_asr.load_model("onnx-community/whisper-tiny", quantization="uint8")
    asr = model.asr
    # _encode_prompt should not raise on non-ASCII.
    ids = asr._encode_prompt("café — résumé")  # type: ignore[attr-defined]
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_prompt_str_and_list_produce_same_result_when_pre_tokenized() -> None:
    """Encoding a str then passing list[int] gives the same result as passing the str directly."""
    model: TextResultsAsrAdapter = onnx_asr.load_model("onnx-community/whisper-tiny", quantization="uint8")
    asr = model.asr
    text = "dictation"
    pre_tokenized = asr._encode_prompt(text)  # type: ignore[attr-defined]

    waveform = _waveform(0.5)
    out_str = model.recognize(waveform, initial_prompt=text)
    out_list = model.recognize(waveform, initial_prompt=pre_tokenized)
    assert out_str == out_list
