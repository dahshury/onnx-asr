"""Stub implementations of :class:`onnx_asr.asr.Asr` and :class:`Resampler`."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import numpy as np
import numpy.typing as npt

from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.asr import TimestampedResult
from onnx_asr.preprocessors.resampler import Resampler
from onnx_asr.utils import SampleRates


class FakeResampler(Resampler):
    """No-op resampler that returns its input unchanged regardless of sample rate.

    Bypasses the real :class:`Resampler.__init__` (which loads ONNX resampling
    sessions from disk) so tests don't pay that cost.
    """

    def __init__(self, sample_rate: Literal[8_000, 16_000] = 16_000) -> None:
        """Create the no-op resampler without loading any ONNX session."""
        self._target_sample_rate = sample_rate
        self._preprocessors = {}

    def __call__(
        self,
        waveforms: npt.NDArray[np.float32],
        waveforms_lens: npt.NDArray[np.int64],
        sample_rate: SampleRates = 16_000,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Return the waveform untouched; sample rate is ignored."""
        return waveforms, waveforms_lens


class FakeAsr:
    """Canned :class:`Asr` implementation that yields preset transcripts per call.

    Args:
        transcripts: Text to return for each batched waveform. If a single
            string is given, the same text is returned for every waveform.
            If a list is given, items are cycled with ``itertools.cycle``-like
            wrap-around once the batch exceeds ``len(transcripts)``.
        sample_rate: Reported sample rate (matches ``Asr._get_sample_rate``).
    """

    def __init__(
        self,
        transcripts: str | list[str] = "hello world",
        *,
        sample_rate: Literal[8_000, 16_000] = 16_000,
    ) -> None:
        self._transcripts: list[str] = [transcripts] if isinstance(transcripts, str) else list(transcripts)
        if not self._transcripts:
            self._transcripts = [""]
        self._sample_rate = sample_rate
        self.calls: list[dict[str, object | None]] = []
        """Per-call kwargs, captured for assertion in tests."""

    def _get_sample_rate(self) -> Literal[8_000, 16_000]:
        return self._sample_rate

    def recognize_batch(
        self,
        waveforms: npt.NDArray[np.float32],
        waveforms_len: npt.NDArray[np.int64],
        /,
        **kwargs: object | None,
    ) -> Iterator[TimestampedResult]:
        """Yield one canned ``TimestampedResult`` per row in ``waveforms``."""
        self.calls.append(kwargs)
        batch = waveforms.shape[0]
        for i in range(batch):
            text = self._transcripts[i % len(self._transcripts)]
            yield TimestampedResult(text=text)


def make_fake_text_adapter(
    transcripts: str | list[str] = "hello world",
    *,
    sample_rate: Literal[8_000, 16_000] = 16_000,
) -> TextResultsAsrAdapter:
    """Compose a real :class:`TextResultsAsrAdapter` over fake ASR + resampler.

    Returns:
        An adapter whose ``recognize`` method behaves like a real model but
        never touches the network, disk, or ONNX runtime.
    """
    return TextResultsAsrAdapter(
        FakeAsr(transcripts, sample_rate=sample_rate),  # type: ignore[arg-type]
        FakeResampler(sample_rate=sample_rate),
    )
