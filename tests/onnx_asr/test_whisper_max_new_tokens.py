"""Tests for Whisper ``max_new_tokens`` kwarg."""

from __future__ import annotations

import numpy as np
import pytest

import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.models.whisper import WHISPER_DEFAULT_MAX_LENGTH, _Whisper
from onnx_asr.utils import InvalidMaxNewTokensError

whisper_models = [
    "whisper-base",
    "onnx-community/whisper-tiny",
]


@pytest.fixture(scope="module", params=whisper_models)
def whisper_model(request: pytest.FixtureRequest) -> TextResultsAsrAdapter:
    if request.param == "onnx-community/whisper-tiny":
        return onnx_asr.load_model(request.param, quantization="uint8")
    return onnx_asr.load_model(request.param, quantization="int8")


def test_default_max_length_unchanged() -> None:
    """``_resolve_max_length`` with no kwarg returns the canonical Whisper default (448)."""
    model = onnx_asr.load_model("whisper-base", quantization="int8")
    asr = model.asr
    assert isinstance(asr, _Whisper)
    assert asr._resolve_max_length(4, {}) == WHISPER_DEFAULT_MAX_LENGTH


def test_max_new_tokens_adds_to_prompt_len() -> None:
    """``max_new_tokens=N`` translates to total max_length = prompt_len + N (clamped at 448)."""
    model = onnx_asr.load_model("whisper-base", quantization="int8")
    asr = model.asr
    assert isinstance(asr, _Whisper)
    assert asr._resolve_max_length(4, {"max_new_tokens": 10}) == 14
    assert asr._resolve_max_length(4, {"max_new_tokens": 100}) == 104


def test_max_new_tokens_clamped_to_448() -> None:
    """Even with very large ``max_new_tokens``, total length is capped at the model maximum."""
    model = onnx_asr.load_model("whisper-base", quantization="int8")
    asr = model.asr
    assert isinstance(asr, _Whisper)
    assert asr._resolve_max_length(4, {"max_new_tokens": 1000}) == WHISPER_DEFAULT_MAX_LENGTH


def test_invalid_max_new_tokens_rejected() -> None:
    model = onnx_asr.load_model("whisper-base", quantization="int8")
    asr = model.asr
    assert isinstance(asr, _Whisper)
    with pytest.raises(InvalidMaxNewTokensError):
        asr._resolve_max_length(4, {"max_new_tokens": 0})
    with pytest.raises(InvalidMaxNewTokensError):
        asr._resolve_max_length(4, {"max_new_tokens": -5})
    with pytest.raises(InvalidMaxNewTokensError):
        asr._resolve_max_length(4, {"max_new_tokens": "ten"})


def test_recognize_with_max_new_tokens(whisper_model: TextResultsAsrAdapter) -> None:
    """End-to-end: a tiny ``max_new_tokens`` produces a short result (no hang, no error)."""
    rng = np.random.default_rng(0)
    waveform = rng.random((2 * 16_000), dtype=np.float32)

    result = whisper_model.recognize(waveform, max_new_tokens=5)
    assert isinstance(result, str)


def test_recognize_with_large_max_new_tokens(whisper_model: TextResultsAsrAdapter) -> None:
    """``max_new_tokens`` larger than the budget still works (gets clamped)."""
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000), dtype=np.float32)
    result = whisper_model.recognize(waveform, max_new_tokens=10_000)
    assert isinstance(result, str)
