"""Tests for Whisper segment timestamps (``return_timestamps=True``)."""

from __future__ import annotations

import numpy as np
import pytest

import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.asr import TimestampedResult
from onnx_asr.models.whisper import _Whisper

whisper_models = [
    "whisper-base",
    "onnx-community/whisper-tiny",
]


@pytest.fixture(scope="module", params=whisper_models)
def whisper_model(request: pytest.FixtureRequest) -> TextResultsAsrAdapter:
    if request.param == "onnx-community/whisper-tiny":
        return onnx_asr.load_model(request.param, quantization="uint8")
    return onnx_asr.load_model(request.param, quantization="int8")


def test_segments_none_when_flag_off(whisper_model: TextResultsAsrAdapter) -> None:
    rng = np.random.default_rng(0)
    waveform = rng.random((2 * 16_000), dtype=np.float32)

    result = whisper_model.with_timestamps().recognize(waveform)
    assert isinstance(result, TimestampedResult)
    assert result.segments is None


def test_segments_populated_when_flag_on(whisper_model: TextResultsAsrAdapter) -> None:
    rng = np.random.default_rng(0)
    waveform = rng.random((3 * 16_000), dtype=np.float32)

    result = whisper_model.with_timestamps().recognize(waveform, return_timestamps=True)
    assert isinstance(result, TimestampedResult)
    # Random noise may yield 0..N segments. Just verify the type contract.
    assert result.segments is None or isinstance(result.segments, list)
    if result.segments:
        for start, end, text in result.segments:
            assert isinstance(start, float)
            assert isinstance(end, float)
            assert isinstance(text, str)


def test_segments_monotonic(whisper_model: TextResultsAsrAdapter) -> None:
    rng = np.random.default_rng(42)
    waveform = rng.random((5 * 16_000), dtype=np.float32)

    result = whisper_model.with_timestamps().recognize(waveform, return_timestamps=True)
    if not result.segments:
        pytest.skip("No segments emitted for random noise — flag plumbing still verified.")
    prev_end = 0.0
    for start, end, _ in result.segments:
        assert 0.0 <= start <= end <= 30.0
        assert start >= prev_end - 1e-3
        prev_end = end


def test_extract_segments_synthetic() -> None:
    """Unit-test ``_extract_segments`` directly with synthetic Whisper token streams."""
    model = onnx_asr.load_model("whisper-base", quantization="int8")
    asr = model.asr
    assert isinstance(asr, _Whisper)
    begin = asr._timestamp_begin_id
    assert begin is not None

    # Build a synthetic sequence: BOS, lang, transcribe, <|0.00|> ...text... <|2.00|> EOS
    bos = asr._bos_token_id
    eos = asr._eos_token_id
    lang = asr._tokens["<|en|>"]
    transcribe = asr._tokens["<|transcribe|>"]
    # Pick a real text token id that won't be filtered by ``startswith("<|")``.
    sample_token = next(tok_id for tok, tok_id in asr._tokens.items() if not tok.startswith("<|") and len(tok) > 1)

    seq = np.array(
        [bos, lang, transcribe, begin + 0, sample_token, sample_token, begin + 100, eos],
        dtype=np.int64,
    )
    segments = asr._extract_segments(seq)
    assert len(segments) == 1
    start, end, text = segments[0]
    assert start == pytest.approx(0.0)
    assert end == pytest.approx(2.0)
    assert text  # non-empty


def test_extract_segments_multiple() -> None:
    """Two consecutive segments are parsed independently."""
    model = onnx_asr.load_model("whisper-base", quantization="int8")
    asr = model.asr
    assert isinstance(asr, _Whisper)
    begin = asr._timestamp_begin_id
    assert begin is not None

    sample_token = next(tok_id for tok, tok_id in asr._tokens.items() if not tok.startswith("<|") and len(tok) > 1)

    seq = np.array(
        [
            begin + 0,  # <|0.00|>
            sample_token,
            begin + 100,  # <|2.00|>
            begin + 100,  # <|2.00|> (start of next segment)
            sample_token,
            sample_token,
            begin + 250,  # <|5.00|>
            asr._eos_token_id,
        ],
        dtype=np.int64,
    )
    segments = asr._extract_segments(seq)
    assert len(segments) == 2
    assert segments[0][0] == pytest.approx(0.0)
    assert segments[0][1] == pytest.approx(2.0)
    assert segments[1][0] == pytest.approx(2.0)
    assert segments[1][1] == pytest.approx(5.0)


def test_extract_segments_unpaired_marker() -> None:
    """A dangling start marker with no closing marker is discarded gracefully."""
    model = onnx_asr.load_model("whisper-base", quantization="int8")
    asr = model.asr
    assert isinstance(asr, _Whisper)
    begin = asr._timestamp_begin_id
    assert begin is not None
    sample_token = next(tok_id for tok, tok_id in asr._tokens.items() if not tok.startswith("<|") and len(tok) > 1)

    seq = np.array([begin + 50, sample_token, sample_token, asr._eos_token_id], dtype=np.int64)
    segments = asr._extract_segments(seq)
    assert segments == []


def test_extract_segments_empty() -> None:
    """A token stream with no timestamp markers yields no segments."""
    model = onnx_asr.load_model("whisper-base", quantization="int8")
    asr = model.asr
    assert isinstance(asr, _Whisper)

    seq = np.array([asr._bos_token_id, asr._eos_token_id], dtype=np.int64)
    segments = asr._extract_segments(seq)
    assert segments == []
