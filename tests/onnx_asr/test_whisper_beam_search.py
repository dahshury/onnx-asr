"""Tests for Whisper beam-search decoding."""

import numpy as np
import pytest

import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.asr import TimestampedResult

WHISPER_MODELS = [
    pytest.param(("onnx-community/whisper-tiny", "uint8"), id="whisper-hf"),
    pytest.param(("whisper-base", "int8"), id="whisper-ort"),
]


@pytest.fixture(scope="module", params=WHISPER_MODELS)
def whisper_model(request: pytest.FixtureRequest) -> TextResultsAsrAdapter:
    model_name, quant = request.param
    return onnx_asr.load_model(model_name, quantization=quant)


def _waveform(seconds: float = 1.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((int(seconds * 16_000),), dtype=np.float32)


def test_beam_size_one_matches_default(whisper_model: TextResultsAsrAdapter) -> None:
    """Explicit beam_size=1 must produce the same output as the default (greedy)."""
    waveform = _waveform()
    default = whisper_model.recognize(waveform)
    beam_1 = whisper_model.recognize(waveform, beam_size=1)
    assert default == beam_1


def test_beam_size_five_returns_string(whisper_model: TextResultsAsrAdapter) -> None:
    """Beam search with num_beams=5 returns a string (no crash, no malformed tokens)."""
    waveform = _waveform()
    result = whisper_model.recognize(waveform, beam_size=5)
    assert isinstance(result, str)


def test_beam_search_batch(whisper_model: TextResultsAsrAdapter) -> None:
    """Beam search over a list of waveforms returns one result per input."""
    waveforms = [_waveform(1.0, seed=1), _waveform(2.0, seed=2)]
    results = whisper_model.recognize(waveforms, beam_size=3)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)


def test_beam_search_with_timestamps(whisper_model: TextResultsAsrAdapter) -> None:
    """Beam search composes with the timestamps adapter."""
    waveform = _waveform()
    result = whisper_model.with_timestamps().recognize(waveform, beam_size=5)
    assert isinstance(result, TimestampedResult)


def test_beam_search_length_penalty_does_not_crash(whisper_model: TextResultsAsrAdapter) -> None:
    """Non-default length_penalty values run without error."""
    waveform = _waveform()
    short_pref = whisper_model.recognize(waveform, beam_size=3, length_penalty=0.1)
    long_pref = whisper_model.recognize(waveform, beam_size=3, length_penalty=2.0)
    assert isinstance(short_pref, str)
    assert isinstance(long_pref, str)


def test_beam_search_with_language(whisper_model: TextResultsAsrAdapter) -> None:
    """beam_size combines with an explicit language hint."""
    waveform = _waveform()
    result = whisper_model.recognize(waveform, beam_size=3, language="en")
    assert isinstance(result, str)
