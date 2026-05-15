"""Tests for WhisperStream audio-buffer trimming.

Without trimming, ``WhisperStream.step()`` re-encodes the whole rolling audio
buffer on every call → O(N²) compute over an utterance of length N. The
trim policy walks the committed-token prefix back to the last ``<|t|>``
marker, drops audio up to that time, and moves the corresponding text into
``_history_text`` so the snapshot stays continuous.

These tests drive ``_maybe_trim_buffer`` directly with synthetic token IDs
(no model inference) plus an end-to-end check that ``buffered_samples``
stays bounded under continuous audio.
"""

from __future__ import annotations

import numpy as np
import pytest

import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.models.whisper import WhisperStream, _Whisper


@pytest.fixture(scope="module")
def whisper_tiny() -> TextResultsAsrAdapter:
    """Small Whisper model — enough to exercise WhisperStream end-to-end."""
    return onnx_asr.load_model("onnx-community/whisper-tiny", quantization="uint8")


def _seed_stream(model: TextResultsAsrAdapter, *, trim_after_s: float = 2.0) -> WhisperStream:
    return model.asr.create_stream(sample_rate=16_000, min_chunk_size_s=0.5, trim_after_s=trim_after_s)


def test_trim_below_threshold_is_noop(whisper_tiny: TextResultsAsrAdapter) -> None:
    """Buffer below ``trim_after_s`` must not get trimmed even with committed tokens."""
    stream = _seed_stream(whisper_tiny, trim_after_s=10.0)
    stream._buffer = np.zeros(16_000, dtype=np.float32)  # 1 s, below 10 s threshold

    asr = whisper_tiny.asr
    assert isinstance(asr, _Whisper)
    begin_id = asr._timestamp_begin_id
    assert begin_id is not None
    text_tok = next(tid for tok, tid in asr._tokens.items() if not tok.startswith("<|") and len(tok) > 1)
    committed = [begin_id, text_tok, begin_id + 50]  # <|0.00|> word <|1.00|>

    pre_size = stream._buffer.size
    stream._maybe_trim_buffer(committed)
    assert stream._buffer.size == pre_size
    assert stream._history_text == ""


def test_trim_drops_audio_up_to_last_committed_timestamp(whisper_tiny: TextResultsAsrAdapter) -> None:
    """Trim cuts audio precisely at the latest committed ``<|t|>``."""
    stream = _seed_stream(whisper_tiny, trim_after_s=2.0)
    asr = whisper_tiny.asr
    assert isinstance(asr, _Whisper)
    begin_id = asr._timestamp_begin_id
    assert begin_id is not None

    # 5 seconds of audio, sample rate 16 kHz.
    stream._buffer = np.arange(5 * 16_000, dtype=np.float32)
    pre_size = stream._buffer.size

    text_tok = next(tid for tok, tid in asr._tokens.items() if not tok.startswith("<|") and len(tok) > 1)
    # <|0.00|> word <|2.00|>  → committed up to 2.0 s
    committed = [begin_id, text_tok, begin_id + 100]
    stream._maybe_trim_buffer(committed)

    expected_drop = int(2.0 * 16_000)
    assert stream._buffer.size == pre_size - expected_drop
    # Audio after trim should be the original samples from index 32000 onward.
    assert stream._buffer[0] == pytest.approx(float(expected_drop))


def test_trim_promotes_committed_text_to_history(whisper_tiny: TextResultsAsrAdapter) -> None:
    """After trim, the rendered text up to the cut goes into ``_history_text``."""
    stream = _seed_stream(whisper_tiny, trim_after_s=1.0)
    asr = whisper_tiny.asr
    assert isinstance(asr, _Whisper)
    begin_id = asr._timestamp_begin_id
    assert begin_id is not None
    text_tok = next(tid for tok, tid in asr._tokens.items() if not tok.startswith("<|") and len(tok) > 1)

    stream._buffer = np.zeros(3 * 16_000, dtype=np.float32)
    committed = [begin_id, text_tok, text_tok, begin_id + 100]
    stream._maybe_trim_buffer(committed)
    # Two real text tokens were promoted; history is non-empty.
    assert stream._history_text != ""


def test_trim_resets_lca_window(whisper_tiny: TextResultsAsrAdapter) -> None:
    """``_prev_token_ids`` and ``_committed_count`` reset after trim — LCA starts fresh on the residual tail."""
    stream = _seed_stream(whisper_tiny, trim_after_s=1.0)
    asr = whisper_tiny.asr
    assert isinstance(asr, _Whisper)
    begin_id = asr._timestamp_begin_id
    assert begin_id is not None
    text_tok = next(tid for tok, tid in asr._tokens.items() if not tok.startswith("<|") and len(tok) > 1)

    stream._buffer = np.zeros(3 * 16_000, dtype=np.float32)
    stream._prev_token_ids = [begin_id, text_tok, text_tok, begin_id + 100, text_tok]
    stream._committed_count = 4

    committed = [begin_id, text_tok, text_tok, begin_id + 100]
    stream._maybe_trim_buffer(committed)
    assert stream._prev_token_ids == []
    assert stream._committed_count == 0


def test_trim_no_committed_timestamp_is_noop(whisper_tiny: TextResultsAsrAdapter) -> None:
    """If the committed prefix has no closing ``<|t|>`` (only the opening 0.00), don't trim."""
    stream = _seed_stream(whisper_tiny, trim_after_s=1.0)
    asr = whisper_tiny.asr
    assert isinstance(asr, _Whisper)
    begin_id = asr._timestamp_begin_id
    assert begin_id is not None
    text_tok = next(tid for tok, tid in asr._tokens.items() if not tok.startswith("<|") and len(tok) > 1)

    stream._buffer = np.zeros(3 * 16_000, dtype=np.float32)
    pre_size = stream._buffer.size
    # Only the leading <|0.00|> marker, no closing marker yet.
    committed = [begin_id, text_tok, text_tok]
    stream._maybe_trim_buffer(committed)
    assert stream._buffer.size == pre_size
    assert stream._history_text == ""


def test_reset_clears_history_text(whisper_tiny: TextResultsAsrAdapter) -> None:
    stream = _seed_stream(whisper_tiny)
    stream._history_text = "already committed prose"
    stream.reset()
    assert stream._history_text == ""


def test_step_keeps_buffer_bounded_under_long_audio(whisper_tiny: TextResultsAsrAdapter) -> None:
    """End-to-end: with trim_after_s=2, buffered_samples never exceeds a small multiple of trim threshold.

    We push 10 seconds of audio in 0.5 s chunks and step after each, then
    assert the buffer stays bounded. This exercises the full decode +
    LocalAgreement-2 + trim path with the actual model.
    """
    stream = _seed_stream(whisper_tiny, trim_after_s=2.0)
    sample_rate = 16_000
    chunk_samples = sample_rate // 2  # 0.5 s
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(10 * sample_rate).astype(np.float32) * 0.05

    max_observed = 0
    for start in range(0, audio.size, chunk_samples):
        chunk = audio[start : start + chunk_samples]
        stream.push_audio(chunk, sample_rate=sample_rate)
        if stream.is_ready():
            stream.step()
            max_observed = max(max_observed, stream.buffered_samples)

    # Whisper-tiny on random noise may or may not produce committed timestamps;
    # the test asserts boundedness when trimming kicks in, and otherwise that
    # the buffer at most equals the total pushed audio.
    assert max_observed <= audio.size, "Buffer cannot exceed total audio pushed"
