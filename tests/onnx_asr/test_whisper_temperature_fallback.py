"""Tests for Whisper temperature fallback and quality guards."""

import numpy as np
import pytest

import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.asr import TimestampedResult
from onnx_asr.models.whisper import (
    DEFAULT_COMPRESSION_RATIO_THRESHOLD,
    DEFAULT_LOGPROB_THRESHOLD,
    DEFAULT_NO_SPEECH_THRESHOLD,
    DEFAULT_TEMPERATURES,
    _Whisper,
)

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


def test_defaults_match_openai_whisper() -> None:
    """The default ladder + thresholds line up with OpenAI Whisper's published recipe."""
    assert DEFAULT_TEMPERATURES == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert DEFAULT_NO_SPEECH_THRESHOLD == 0.6
    assert DEFAULT_COMPRESSION_RATIO_THRESHOLD == 2.4
    assert DEFAULT_LOGPROB_THRESHOLD == -1.0


def test_single_temperature_zero_matches_greedy(whisper_model: TextResultsAsrAdapter) -> None:
    """temperature=0.0 (single value) takes the no-fallback fast path and matches the default greedy output."""
    waveform = _waveform()
    default = whisper_model.recognize(waveform)
    explicit_greedy = whisper_model.recognize(waveform, temperature=0.0)
    assert default == explicit_greedy


def test_compression_ratio_helper() -> None:
    """Compression-ratio guard: repetition produces a high ratio; varied text a low one."""
    repeating = _Whisper._compression_ratio("the the the the the the the the the the")
    varied = _Whisper._compression_ratio("the quick brown fox jumps over the lazy dog")
    empty = _Whisper._compression_ratio("")
    assert repeating > varied
    assert empty == 0.0


def test_temperature_ladder_runs(whisper_model: TextResultsAsrAdapter) -> None:
    """The default temperature ladder produces a string on random audio without crashing."""
    waveform = _waveform()
    result = whisper_model.recognize(waveform, temperature=DEFAULT_TEMPERATURES)
    assert isinstance(result, str)


def test_single_temperature_above_zero(whisper_model: TextResultsAsrAdapter) -> None:
    """Single non-zero temperature uses the sampling path (no fallback ladder)."""
    waveform = _waveform()
    result = whisper_model.recognize(waveform, temperature=0.4)
    assert isinstance(result, str)


def test_temperature_ladder_batch(whisper_model: TextResultsAsrAdapter) -> None:
    """Per-item fallback works across a batch."""
    waveforms = [_waveform(1.0, seed=1), _waveform(2.0, seed=2)]
    results = whisper_model.recognize(waveforms, temperature=DEFAULT_TEMPERATURES)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)


def test_disabling_all_guards(whisper_model: TextResultsAsrAdapter) -> None:
    """Setting all thresholds to None disables fallback retries (still loops but always accepts)."""
    waveform = _waveform()
    result = whisper_model.recognize(
        waveform,
        temperature=DEFAULT_TEMPERATURES,
        no_speech_threshold=None,
        compression_ratio_threshold=None,
        logprob_threshold=None,
    )
    assert isinstance(result, str)


def test_with_language_and_temperature_ladder(whisper_model: TextResultsAsrAdapter) -> None:
    """Explicit language hint combines with the temperature fallback path."""
    waveform = _waveform()
    result = whisper_model.recognize(waveform, language="en", temperature=DEFAULT_TEMPERATURES)
    assert isinstance(result, str)


def test_temperature_ladder_with_timestamps(whisper_model: TextResultsAsrAdapter) -> None:
    """Temperature ladder composes with the timestamps adapter."""
    waveform = _waveform()
    result = whisper_model.with_timestamps().recognize(waveform, temperature=DEFAULT_TEMPERATURES)
    assert isinstance(result, TimestampedResult)


def test_strict_thresholds_dont_crash(whisper_model: TextResultsAsrAdapter) -> None:
    """Very strict guard values force the ladder to walk through every temperature."""
    waveform = _waveform()
    result = whisper_model.recognize(
        waveform,
        temperature=DEFAULT_TEMPERATURES,
        compression_ratio_threshold=0.5,  # almost any text exceeds this
        logprob_threshold=10.0,  # impossible to satisfy
    )
    assert isinstance(result, str)
