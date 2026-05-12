"""Tests for Whisper-specific LocalAgreement-2 streaming.

Verifies that ``_Whisper.create_stream()`` returns a working ``AsrStream``
implementation that applies the LocalAgreement-2 commit policy: committed
text grows monotonically, finishing commits everything, reset bumps the
segment id, etc.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

import onnx_asr
from onnx_asr import AsrStream, StreamingResult
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.models.whisper import WhisperStream

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


def test_create_stream_returns_whisper_stream(whisper_model: TextResultsAsrAdapter) -> None:
    """Whisper's overridden ``create_stream`` returns a ``WhisperStream``, not the BaseAsr stub."""
    stream = whisper_model.asr.create_stream()
    assert isinstance(stream, WhisperStream)


def test_whisper_stream_satisfies_asr_stream_protocol(whisper_model: TextResultsAsrAdapter) -> None:
    """Structural check via the runtime-checkable Protocol."""
    stream = whisper_model.asr.create_stream()
    assert isinstance(stream, AsrStream)


def test_whisper_capabilities_advertise_streaming(whisper_model: TextResultsAsrAdapter) -> None:
    """Whisper class sets ``capabilities.streaming_native=True`` and ``is_multilingual=True``."""
    caps = whisper_model.asr.capabilities
    assert caps.streaming_native is True
    assert caps.is_multilingual is True


def test_not_ready_until_min_chunk(whisper_model: TextResultsAsrAdapter) -> None:
    """``is_ready()`` only returns True once enough audio is buffered."""
    stream = whisper_model.asr.create_stream(min_chunk_size_s=1.0)
    assert stream.is_ready() is False
    stream.push_audio(_waveform(0.4))
    assert stream.is_ready() is False
    stream.push_audio(_waveform(0.7))
    assert stream.is_ready() is True


def test_step_returns_streaming_result_with_committed_text(whisper_model: TextResultsAsrAdapter) -> None:
    """A step before ``finish()`` returns a partial snapshot with the new ``committed_text`` field."""
    stream = whisper_model.asr.create_stream(min_chunk_size_s=0.5)
    stream.push_audio(_waveform(0.5))
    snap = stream.step()
    assert snap is not None
    assert isinstance(snap, StreamingResult)
    assert snap.is_partial is True
    assert isinstance(snap.text, str)
    assert isinstance(snap.committed_text, str)
    # Without prior decode, LCP is 0 → committed_text is empty on first step.
    assert snap.committed_text == ""


def test_finish_commits_everything(whisper_model: TextResultsAsrAdapter) -> None:
    """After ``finish()``, ``step()`` returns ``committed_text == text`` and ``is_partial=False``."""
    stream = whisper_model.asr.create_stream(min_chunk_size_s=0.5)
    stream.push_audio(_waveform(1.0))
    stream.step()  # partial — establishes prev_token_ids
    stream.finish()
    final = stream.step()
    assert final is not None
    assert final.is_partial is False
    assert final.committed_text == final.text
    assert stream.is_endpoint is True


def test_committed_text_is_prefix_of_text(whisper_model: TextResultsAsrAdapter) -> None:
    """``committed_text`` is always a (possibly empty) prefix of ``text``."""
    stream = whisper_model.asr.create_stream(min_chunk_size_s=0.5)
    stream.push_audio(_waveform(0.5, seed=1))
    snap_a = stream.step()
    stream.push_audio(_waveform(0.5, seed=2))
    snap_b = stream.step()
    stream.finish()
    snap_c = stream.step()
    for snap in (snap_a, snap_b, snap_c):
        assert snap is not None
        assert snap.text.startswith(snap.committed_text)


def test_committed_text_grows_monotonically(whisper_model: TextResultsAsrAdapter) -> None:
    """LocalAgreement-2 guarantee: committed_text length never shrinks."""
    stream = whisper_model.asr.create_stream(min_chunk_size_s=0.5)
    lengths: list[int] = []
    for _ in range(3):
        stream.push_audio(_waveform(0.5))
        snap = stream.step()
        assert snap is not None
        lengths.append(len(snap.committed_text))
    stream.finish()
    snap = stream.step()
    assert snap is not None
    lengths.append(len(snap.committed_text))

    for prev, curr in pairwise(lengths):
        assert curr >= prev, f"committed_text length shrank: {lengths}"


def test_reset_bumps_segment_id_and_clears_state(whisper_model: TextResultsAsrAdapter) -> None:
    """``reset()`` zeros buffer and decode state, bumps segment id."""
    stream = whisper_model.asr.create_stream(min_chunk_size_s=0.5)
    stream.push_audio(_waveform(0.5))
    stream.finish()
    first = stream.step()
    assert first is not None
    assert first.segment_id == 0

    stream.reset()
    assert stream.is_endpoint is False
    assert stream.is_ready() is False
    assert stream.buffered_samples == 0

    stream.push_audio(_waveform(0.5))
    stream.finish()
    second = stream.step()
    assert second is not None
    assert second.segment_id == 1


def test_reset_keep_audio(whisper_model: TextResultsAsrAdapter) -> None:
    """``reset(keep_audio=True)`` preserves the buffered audio."""
    stream = whisper_model.asr.create_stream(min_chunk_size_s=0.5)
    stream.push_audio(_waveform(0.5))
    stream.reset(keep_audio=True)
    assert stream.buffered_samples == 8000
    assert stream.is_ready() is True


def test_rejects_sample_rate_mismatch(whisper_model: TextResultsAsrAdapter) -> None:
    """Sample rate must match the constructor argument."""
    stream = whisper_model.asr.create_stream(sample_rate=16_000)
    with pytest.raises(ValueError, match="sample_rate mismatch"):
        stream.push_audio(_waveform(0.1), sample_rate=8000)


def test_language_kwarg_skips_lang_detect(whisper_model: TextResultsAsrAdapter) -> None:
    """Passing ``language='en'`` runs the stream without the lang-detect step."""
    stream = whisper_model.asr.create_stream(min_chunk_size_s=0.5, language="en")
    stream.push_audio(_waveform(0.5))
    snap = stream.step()
    assert snap is not None
    assert isinstance(snap.text, str)


def test_step_returns_none_when_not_ready(whisper_model: TextResultsAsrAdapter) -> None:
    """``step()`` is a no-op poll when the buffer is below the threshold."""
    stream = whisper_model.asr.create_stream(min_chunk_size_s=2.0)
    stream.push_audio(_waveform(0.5))
    assert stream.step() is None
