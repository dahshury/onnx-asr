"""Tests for the buffered-streaming wrapper (item #2)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

import onnx_asr
from onnx_asr import AsrStream, BufferedAsrStream, StreamingResult, create_buffered_stream
from onnx_asr.asr import Asr, ModelCapabilities, TimestampedResult


class _FakeAsr:
    """Minimal in-memory ASR for testing the streaming wrapper without downloads."""

    capabilities = ModelCapabilities()

    def __init__(self, scripted_texts: list[str] | None = None) -> None:
        self._scripted = scripted_texts or []
        self.call_count = 0
        self.last_waveform_size = 0

    @staticmethod
    def _get_sample_rate() -> int:
        return 16_000

    def recognize_batch(
        self, waveforms: np.ndarray, waveforms_len: np.ndarray, /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        del kwargs
        self.call_count += 1
        self.last_waveform_size = int(waveforms.shape[-1])
        text = self._scripted[min(self.call_count - 1, len(self._scripted) - 1)] if self._scripted else "fake"
        yield TimestampedResult(text=text, tokens=text.split(), timestamps=None)

    def create_stream(self) -> AsrStream:
        # Not implementing real streaming — the test exercises the wrapper, not this hook.
        raise NotImplementedError


@pytest.fixture
def fake_asr() -> _FakeAsr:
    return _FakeAsr(["hello", "hello world", "hello world today"])


def test_buffered_stream_implements_asr_stream_protocol(fake_asr: _FakeAsr) -> None:
    """Structural check: ``BufferedAsrStream`` satisfies ``AsrStream``."""
    stream: AsrStream = BufferedAsrStream(fake_asr)
    assert isinstance(stream, AsrStream)


def test_not_ready_until_min_chunk(fake_asr: _FakeAsr) -> None:
    """``is_ready()`` is False until we've buffered ``min_chunk_size_s`` worth of samples."""
    stream = BufferedAsrStream(fake_asr, min_chunk_size_s=0.5)
    assert stream.is_ready() is False
    stream.push_audio(np.zeros(8000, dtype=np.float32))  # 0.5 s
    assert stream.is_ready() is True


def test_step_returns_partial_until_finish(fake_asr: _FakeAsr) -> None:
    """Calls to ``step()`` before ``finish()`` return ``is_partial=True``."""
    stream = BufferedAsrStream(fake_asr, min_chunk_size_s=0.5)
    stream.push_audio(np.zeros(8000, dtype=np.float32))

    snap = stream.step()
    assert snap is not None
    assert snap.is_partial is True
    assert snap.text == "hello"
    assert stream.is_endpoint is False


def test_step_after_finish_is_final(fake_asr: _FakeAsr) -> None:
    """``finish()`` then ``step()`` produces a final, committed snapshot."""
    stream = BufferedAsrStream(fake_asr, min_chunk_size_s=0.5)
    stream.push_audio(np.zeros(8000, dtype=np.float32))
    stream.step()  # partial
    stream.finish()
    final = stream.step()
    assert final is not None
    assert final.is_partial is False
    assert stream.is_endpoint is True


def test_step_returns_none_when_not_ready(fake_asr: _FakeAsr) -> None:
    """``step()`` is a no-op poll when the buffer is below the threshold."""
    stream = BufferedAsrStream(fake_asr, min_chunk_size_s=1.0)
    stream.push_audio(np.zeros(100, dtype=np.float32))
    assert stream.step() is None
    assert fake_asr.call_count == 0


def test_reset_clears_state_and_bumps_segment(fake_asr: _FakeAsr) -> None:
    """``reset()`` zeros the buffer and increments segment id."""
    stream = BufferedAsrStream(fake_asr, min_chunk_size_s=0.5)
    stream.push_audio(np.zeros(8000, dtype=np.float32))
    stream.finish()
    first = stream.step()
    assert first is not None
    assert first.segment_id == 0

    stream.reset()
    assert stream.is_endpoint is False
    assert stream.is_ready() is False
    assert stream.buffered_samples == 0

    stream.push_audio(np.zeros(8000, dtype=np.float32))
    stream.finish()
    second = stream.step()
    assert second is not None
    assert second.segment_id == 1


def test_reset_keep_audio(fake_asr: _FakeAsr) -> None:
    """``reset(keep_audio=True)`` preserves buffered samples."""
    stream = BufferedAsrStream(fake_asr, min_chunk_size_s=0.5)
    stream.push_audio(np.zeros(8000, dtype=np.float32))
    stream.reset(keep_audio=True)
    assert stream.buffered_samples == 8000
    assert stream.is_ready() is True


def test_rejects_sample_rate_mismatch(fake_asr: _FakeAsr) -> None:
    """Sample rate must match the constructor argument."""
    stream = BufferedAsrStream(fake_asr, sample_rate=16_000)
    with pytest.raises(ValueError, match="sample_rate mismatch"):
        stream.push_audio(np.zeros(100, dtype=np.float32), sample_rate=8000)


def test_buffer_grows_across_pushes(fake_asr: _FakeAsr) -> None:
    """Repeated pushes accumulate in the buffer (re-decode sees the full audio)."""
    stream = BufferedAsrStream(fake_asr, min_chunk_size_s=0.5)
    stream.push_audio(np.zeros(4000, dtype=np.float32))
    stream.push_audio(np.zeros(4000, dtype=np.float32))
    stream.push_audio(np.zeros(4000, dtype=np.float32))
    stream.step()
    assert fake_asr.last_waveform_size == 12_000


def test_create_buffered_stream_helper(fake_asr: _FakeAsr) -> None:
    """The convenience constructor returns the same class."""
    stream = create_buffered_stream(fake_asr, min_chunk_size_s=0.5)
    assert isinstance(stream, BufferedAsrStream)


def test_recognize_kwargs_are_forwarded() -> None:
    """``recognize_kwargs`` flow through to the underlying ``recognize_batch``."""
    received: list[dict[str, object]] = []

    class _AsrThatCaptures:
        capabilities = ModelCapabilities()

        @staticmethod
        def _get_sample_rate() -> int:
            return 16_000

        def recognize_batch(
            self, waveforms: np.ndarray, waveforms_len: np.ndarray, /, **kwargs: object | None
        ) -> Iterator[TimestampedResult]:
            del waveforms, waveforms_len
            received.append(dict(kwargs))
            yield TimestampedResult(text="ok")

        def create_stream(self) -> AsrStream:
            raise NotImplementedError

    capturing_asr: Asr = _AsrThatCaptures()
    stream = BufferedAsrStream(capturing_asr, min_chunk_size_s=0.1, recognize_kwargs={"language": "en", "beam_size": 5})
    stream.push_audio(np.zeros(2000, dtype=np.float32))
    stream.step()

    assert received == [{"language": "en", "beam_size": 5}]


def test_streaming_result_shape(fake_asr: _FakeAsr) -> None:
    """Snapshots include text, tokens, and the segment id."""
    stream = BufferedAsrStream(fake_asr, min_chunk_size_s=0.5)
    stream.push_audio(np.zeros(8000, dtype=np.float32))
    snap = stream.step()
    assert snap is not None
    assert isinstance(snap, StreamingResult)
    assert snap.text == "hello"
    assert snap.tokens == ["hello"]
    assert snap.segment_id == 0


def test_real_model_integration_smoke() -> None:
    """End-to-end smoke against a small real model — pushed in two chunks."""
    model = onnx_asr.load_model("whisper-base", quantization="int8")
    stream = create_buffered_stream(model.asr, min_chunk_size_s=0.5, recognize_kwargs={"language": "en"})

    rng = np.random.default_rng(0)
    waveform = rng.random(16_000, dtype=np.float32)
    stream.push_audio(waveform[:8000])
    partial = stream.step()
    stream.push_audio(waveform[8000:])
    stream.finish()
    final = stream.step()

    assert partial is None or isinstance(partial, StreamingResult)
    assert final is not None
    assert isinstance(final, StreamingResult)
    assert final.is_partial is False
    assert isinstance(final.text, str)
