"""Tests for the FakeAsr / FakeResampler test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.asr import TimestampedResult
from tests.fakes import FakeAsr, FakeResampler, make_fake_text_adapter


def test_fake_asr_returns_canned_transcript() -> None:
    asr = FakeAsr("hello there")
    rng = np.random.default_rng(0)
    waveforms = rng.random((2, 16_000), dtype=np.float32)
    waveforms_len = np.array([16_000, 16_000], dtype=np.int64)

    results = list(asr.recognize_batch(waveforms, waveforms_len))
    assert len(results) == 2
    assert all(isinstance(r, TimestampedResult) for r in results)
    assert results[0].text == "hello there"
    assert results[1].text == "hello there"


def test_fake_asr_cycles_through_transcripts() -> None:
    asr = FakeAsr(["alpha", "beta", "gamma"])
    rng = np.random.default_rng(0)
    waveforms = rng.random((5, 16_000), dtype=np.float32)
    waveforms_len = np.full(5, 16_000, dtype=np.int64)

    results = [r.text for r in asr.recognize_batch(waveforms, waveforms_len)]
    assert results == ["alpha", "beta", "gamma", "alpha", "beta"]


def test_fake_asr_captures_kwargs() -> None:
    asr = FakeAsr("foo")
    waveforms = np.zeros((1, 16_000), dtype=np.float32)
    waveforms_len = np.array([16_000], dtype=np.int64)

    list(asr.recognize_batch(waveforms, waveforms_len, language="en", task="transcribe"))
    assert len(asr.calls) == 1
    assert asr.calls[0]["language"] == "en"
    assert asr.calls[0]["task"] == "transcribe"


def test_fake_resampler_passes_through_at_target_rate() -> None:
    resampler = FakeResampler(sample_rate=16_000)
    waveforms = np.ones((1, 16_000), dtype=np.float32)
    waveforms_len = np.array([16_000], dtype=np.int64)

    out_wave, out_lens = resampler(waveforms, waveforms_len, 16_000)
    assert out_wave is waveforms
    assert out_lens is waveforms_len


def test_fake_resampler_ignores_sample_rate_mismatch() -> None:
    """Unlike the real Resampler, the fake does not resample — sample_rate is ignored."""
    resampler = FakeResampler(sample_rate=16_000)
    waveforms = np.ones((1, 22_050), dtype=np.float32)
    waveforms_len = np.array([22_050], dtype=np.int64)

    out_wave, out_lens = resampler(waveforms, waveforms_len, 22_050)
    assert out_wave.shape == waveforms.shape
    assert out_lens[0] == 22_050


def test_make_fake_text_adapter_returns_real_adapter() -> None:
    adapter = make_fake_text_adapter("canned")
    assert isinstance(adapter, TextResultsAsrAdapter)


def test_fake_adapter_recognize_single() -> None:
    adapter = make_fake_text_adapter("the quick brown fox")
    waveform = np.zeros(16_000, dtype=np.float32)
    result = adapter.recognize(waveform)
    assert result == "the quick brown fox"


def test_fake_adapter_recognize_batch() -> None:
    from pathlib import Path  # noqa: PLC0415

    adapter = make_fake_text_adapter(["one", "two", "three"])
    waveforms: list[str | Path | np.ndarray] = [np.zeros(16_000, dtype=np.float32) for _ in range(3)]
    results = adapter.recognize(waveforms)
    assert results == ["one", "two", "three"]


def test_fake_adapter_recognize_empty_batch() -> None:
    adapter = make_fake_text_adapter("nope")
    assert adapter.recognize([]) == []


def test_fake_adapter_with_timestamps() -> None:
    adapter = make_fake_text_adapter("hi").with_timestamps()
    result = adapter.recognize(np.zeros(16_000, dtype=np.float32))
    assert isinstance(result, TimestampedResult)
    assert result.text == "hi"


def test_fake_asr_empty_transcripts_falls_back_to_empty_string() -> None:
    """Passing an empty list never crashes — falls back to a single empty string."""
    asr = FakeAsr([])
    waveforms = np.zeros((1, 16_000), dtype=np.float32)
    waveforms_len = np.array([16_000], dtype=np.int64)
    results = list(asr.recognize_batch(waveforms, waveforms_len))
    assert results[0].text == ""


def test_fake_asr_get_sample_rate() -> None:
    assert FakeAsr(sample_rate=16_000)._get_sample_rate() == 16_000
    assert FakeAsr(sample_rate=8_000)._get_sample_rate() == 8_000


def test_fake_avoids_imports_of_huggingface_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fakes must not transitively trigger network code paths."""
    # Replace huggingface_hub.hf_hub_download with a sentinel that explodes if hit.
    import huggingface_hub  # noqa: PLC0415

    msg = "FakeAsr should not call into huggingface_hub"

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(msg)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", explode)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", explode)

    adapter = make_fake_text_adapter("safe")
    assert adapter.recognize(np.zeros(16_000, dtype=np.float32)) == "safe"
