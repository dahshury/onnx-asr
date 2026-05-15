"""Base wake-word detector classes.

Defines the :class:`WakeWord` protocol and :class:`WakeWordResult` dataclass —
the analogue of :class:`onnx_asr.vad.Vad` / :class:`onnx_asr.vad.SegmentResult`
for keyword-spotting / wake-word detection models.

A wake-word detector consumes a stream of mono float32 PCM audio at the
detector's native sample rate (usually 16 kHz) and produces, for each input
chunk, a per-wake-word probability score plus a final "detected which word"
decision based on per-word thresholds.

Protocol surface:

* :meth:`WakeWord.detect_batch` — batch detection over fixed-length audio.
* :meth:`WakeWord.scores` — per-wake-word probability stream over a single
  rolling audio array (the streaming use case).

Concrete implementations live under ``onnx_asr.models`` (see
:class:`~onnx_asr.models.openwakeword.OpenWakeWord`).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from onnx_asr.model_base import _ModelImplementation


@dataclass(frozen=True)
class WakeWordResult:
    """Result of one wake-word detection pass over a fixed audio window.

    ``scores`` maps each loaded wake-word name to its probability in ``[0, 1]``
    for this window. ``detected`` is the name of the wake-word whose score
    crossed its threshold first, or ``None`` if no wake word was detected.
    """

    scores: dict[str, float]
    """Per-wake-word probability in ``[0, 1]`` for the analyzed window."""
    detected: str | None
    """Name of the triggered wake word (highest-scoring above threshold), or ``None``."""


class WakeWord(_ModelImplementation, Protocol):
    """Wake-word detector protocol.

    Inherits :class:`_ModelImplementation`'s static infrastructure surface so
    the resolver can route HF repos / load files; adds the runtime detection
    API for keyword-spotting models like OpenWakeWord / Porcupine.
    """

    @staticmethod
    def _get_sample_rate() -> Literal[16_000]:
        return 16_000

    def detect_batch(
        self,
        waveforms: npt.NDArray[np.float32],
        waveforms_len: npt.NDArray[np.int64],
        *,
        threshold: float = 0.5,
    ) -> Iterator[WakeWordResult]:
        """Run wake-word detection on a batch of fixed-length audio clips.

        Args:
            waveforms: ``(batch, samples)`` float32 PCM mono at the detector's sample rate.
            waveforms_len: ``(batch,)`` int64 per-clip valid sample counts.
            threshold: Probability threshold for declaring a wake-word detected.

        Yields:
            One :class:`WakeWordResult` per clip in batch order.

        """
        ...

    @property
    def wake_words(self) -> tuple[str, ...]:
        """Tuple of wake-word names this detector recognizes (in stable order)."""
        ...
