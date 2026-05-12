"""Tests for Whisper ``task`` kwarg (transcribe vs translate)."""

from __future__ import annotations

import numpy as np
import pytest

import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter
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


def test_default_task_is_transcribe(whisper_model: TextResultsAsrAdapter) -> None:
    """Without explicit ``task``, the prompt uses ``<|transcribe|>``."""
    asr = whisper_model.asr
    assert isinstance(asr, _Whisper)
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000), dtype=np.float32)
    # Smoke: no kwargs ⇒ transcribe behavior (no error).
    result = whisper_model.recognize(waveform)
    assert isinstance(result, str)


def test_explicit_transcribe(whisper_model: TextResultsAsrAdapter) -> None:
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000), dtype=np.float32)
    result = whisper_model.recognize(waveform, task="transcribe")
    assert isinstance(result, str)


def test_translate_task(whisper_model: TextResultsAsrAdapter) -> None:
    """``task="translate"`` selects the ``<|translate|>`` decoder mode."""
    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000), dtype=np.float32)
    result = whisper_model.recognize(waveform, task="translate", language="fr")
    assert isinstance(result, str)


def test_invalid_task_raises(whisper_model: TextResultsAsrAdapter) -> None:
    from onnx_asr.utils import InvalidTaskError  # noqa: PLC0415

    rng = np.random.default_rng(0)
    waveform = rng.random((1 * 16_000), dtype=np.float32)
    with pytest.raises(InvalidTaskError):
        whisper_model.recognize(waveform, task="invalid")  # type: ignore[arg-type]


def test_task_token_swapped_in_prompt() -> None:
    """Verify the prompt array has the correct task token at position 2."""
    model = onnx_asr.load_model("whisper-base", quantization="int8")
    asr = model.asr
    assert isinstance(asr, _Whisper)

    transcribe_id = asr._tokens["<|transcribe|>"]
    translate_id = asr._tokens["<|translate|>"]
    assert transcribe_id != translate_id
    # The static default prompt uses transcribe.
    assert asr._transcribe_input[0, 2] == transcribe_id
