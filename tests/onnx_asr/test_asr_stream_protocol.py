"""Tests for the AsrStream protocol scaffolding (item #4).

These tests don't exercise a real streaming model — they verify the protocol
shape, defaults, and the negative path (a non-streaming model raises clearly).
A small in-memory ``FakeAsrStream`` doubles as a reference implementation for
future adapters (sherpa-onnx-style native streaming for Parakeet, or a
LocalAgreement wrapper for Whisper).
"""

from __future__ import annotations

import numpy as np
import pytest

import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.asr import AsrStream, ModelCapabilities, StreamingResult
from onnx_asr.utils import StreamingNotSupportedError


class FakeAsrStream:
    """In-memory reference implementation of ``AsrStream`` used by the test suite."""

    def __init__(self, chunk_size: int = 1600) -> None:
        self._chunk_size = chunk_size
        self._buffer: list[float] = []
        self._finished = False
        self._endpoint = False
        self._segment_id = 0
        self._step_count = 0

    def push_audio(self, samples: np.ndarray, sample_rate: int = 16_000) -> None:
        del sample_rate
        self._buffer.extend(samples.tolist())

    def finish(self) -> None:
        self._finished = True

    def is_ready(self) -> bool:
        return len(self._buffer) >= self._chunk_size or (self._finished and bool(self._buffer))

    def step(self) -> StreamingResult | None:
        if not self.is_ready():
            return None
        take = min(self._chunk_size, len(self._buffer))
        self._buffer = self._buffer[take:]
        self._step_count += 1
        is_final = self._finished and not self._buffer
        if is_final:
            self._endpoint = True
        return StreamingResult(
            text=f"chunk-{self._step_count}",
            tokens=[f"chunk-{self._step_count}"],
            timestamps=None,
            is_partial=not is_final,
            segment_id=self._segment_id,
        )

    def reset(self, *, keep_audio: bool = False) -> None:
        if not keep_audio:
            self._buffer = []
        self._finished = False
        self._endpoint = False
        self._segment_id += 1
        self._step_count = 0

    @property
    def is_endpoint(self) -> bool:
        return self._endpoint


def test_model_capabilities_defaults_to_offline_batch() -> None:
    """A plain ``ModelCapabilities()`` describes a non-streaming, English-only, no-knobs model."""
    cap = ModelCapabilities()
    assert cap.streaming_native is False
    assert cap.supports_timestamps is False
    assert cap.supports_word_timestamps is False
    assert cap.supports_beam_search is False
    assert cap.supports_temperature_fallback is False
    assert cap.is_multilingual is False


def test_model_capabilities_is_frozen() -> None:
    """Capabilities are immutable so callers can rely on cached reads."""
    cap = ModelCapabilities(streaming_native=True)
    with pytest.raises(Exception):  # noqa: B017, PT011  # FrozenInstanceError
        cap.streaming_native = False  # type: ignore[misc]


def test_streaming_result_fields() -> None:
    """``StreamingResult`` carries the minimum metadata callers need to render a UI."""
    r = StreamingResult(text="hi", tokens=["hi"], timestamps=[0.0], is_partial=True, segment_id=0)
    assert r.text == "hi"
    assert r.tokens == ["hi"]
    assert r.timestamps == [0.0]
    assert r.is_partial is True
    assert r.segment_id == 0


def test_fake_stream_satisfies_asr_stream_protocol() -> None:
    """``FakeAsrStream`` is structurally an ``AsrStream`` (Protocol check)."""
    stream: AsrStream = FakeAsrStream()
    assert isinstance(stream, AsrStream)


def test_fake_stream_lifecycle() -> None:
    """End-to-end: push → step (partial) → finish → step (final) → is_endpoint."""
    stream = FakeAsrStream(chunk_size=4)
    audio = np.zeros(10, dtype=np.float32)

    assert stream.is_ready() is False
    stream.push_audio(audio)
    assert stream.is_ready() is True

    first = stream.step()
    assert first is not None
    assert first.is_partial is True
    assert first.segment_id == 0

    stream.finish()
    while stream.is_ready():
        last = stream.step()
        assert last is not None

    assert last is not None
    assert last.is_partial is False
    assert stream.is_endpoint is True


def test_fake_stream_reset_increments_segment_id() -> None:
    """``reset()`` zeros decoder state, bumps segment id, drops audio by default."""
    stream = FakeAsrStream(chunk_size=4)
    stream.push_audio(np.zeros(10, dtype=np.float32))
    stream.reset()
    assert stream.is_endpoint is False
    assert stream.is_ready() is False

    stream.push_audio(np.zeros(10, dtype=np.float32))
    assert stream.step() is not None
    # Segment id was bumped by reset, so the new snapshot reflects segment 1.
    next_snap = stream.step()
    assert next_snap is None or next_snap.segment_id == 1


def test_fake_stream_reset_keep_audio() -> None:
    """``reset(keep_audio=True)`` preserves buffered samples (sherpa-onnx pattern)."""
    stream = FakeAsrStream(chunk_size=4)
    stream.push_audio(np.zeros(10, dtype=np.float32))
    stream.reset(keep_audio=True)
    assert stream.is_ready() is True


def test_non_streaming_model_raises_clearly() -> None:
    """A model that doesn't override ``create_stream()`` raises ``StreamingNotSupportedError`` with the model name."""
    model = onnx_asr.load_model("alphacep/vosk-model-small-ru", quantization="int8")
    inner = model.asr if hasattr(model, "asr") else model
    assert isinstance(inner, object)

    with pytest.raises(StreamingNotSupportedError) as excinfo:
        inner.create_stream()  # type: ignore[attr-defined]
    assert "capabilities" in str(excinfo.value)


def test_default_model_capabilities_on_loaded_model() -> None:
    """A model that doesn't explicitly set capabilities inherits the all-false defaults."""
    model: TextResultsAsrAdapter = onnx_asr.load_model("alphacep/vosk-model-small-ru", quantization="int8")
    inner = model.asr
    caps = inner.capabilities  # type: ignore[attr-defined]
    assert isinstance(caps, ModelCapabilities)
    assert caps.streaming_native is False
