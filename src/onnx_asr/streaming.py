"""Buffered streaming wrapper for any offline ASR model.

Implements the :class:`~onnx_asr.asr.AsrStream` protocol by re-decoding a
growing audio buffer on each ``step()`` call. Model-agnostic — works with
any :class:`~onnx_asr.asr.Asr` (Whisper, NeMo Conformer / Parakeet TDT,
GigaAM, etc.).

For latency-sensitive paths where the model itself supports stateful
streaming (cache-aware Conformer encoder + persistent RNN-T predictor),
a model-specific implementation that overrides ``Asr.create_stream()``
would be more efficient. This wrapper trades that performance for
universal compatibility — useful for adapters that want a uniform
streaming surface across heterogeneous backends, and as the engine-side
half of a server-orchestrated LocalAgreement-2 / watermark-commit policy
(see ``docs/plans/01-onnx-asr-fork-strategy.md`` §4).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from onnx_asr.asr import Asr, StreamingResult, TimestampedResult


class BufferedAsrStream:
    """Re-decode-on-each-step buffered streaming wrapper.

    Lifecycle:

    1. ``push_audio(samples)`` appends PCM to the input buffer.
    2. ``is_ready()`` returns True once enough audio is buffered.
    3. ``step()`` re-decodes the whole buffer through the wrapped ASR and
       returns a snapshot of the current transcript.
    4. ``finish()`` marks the input closed; the next ``step()`` returns
       ``is_partial=False`` and sets ``is_endpoint``.
    5. ``reset()`` zeros the buffer / endpoint flag and bumps the segment id.
    """

    def __init__(
        self,
        asr: Asr,
        *,
        sample_rate: int = 16_000,
        min_chunk_size_s: float = 0.5,
        recognize_kwargs: dict[str, object] | None = None,
    ) -> None:
        """Create a buffered stream over ``asr``.

        Args:
            asr: Any ASR implementing the ``Asr`` protocol.
            sample_rate: Expected sample rate of pushed audio (must match for every push).
            min_chunk_size_s: Minimum buffered duration before ``is_ready()`` returns True.
            recognize_kwargs: Forwarded to the underlying ``recognize_batch()`` on each step
                (e.g. ``language="en"``, ``temperature=0.0``).

        """
        self._asr = asr
        self._sample_rate = sample_rate
        self._min_chunk_samples = max(1, int(min_chunk_size_s * sample_rate))
        self._recognize_kwargs = recognize_kwargs or {}
        self._buffer: npt.NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._finished = False
        self._endpoint = False
        self._segment_id = 0

    def push_audio(self, samples: npt.NDArray[np.float32], sample_rate: int = 16_000) -> None:
        """Append PCM samples to the stream's buffer."""
        if sample_rate != self._sample_rate:
            msg = f"BufferedAsrStream sample_rate mismatch: expected {self._sample_rate}, got {sample_rate}"
            raise ValueError(msg)
        flat = np.asarray(samples, dtype=np.float32).ravel()
        self._buffer = np.concatenate([self._buffer, flat])

    def finish(self) -> None:
        """Mark the input as complete. The next ``step()`` produces a final snapshot."""
        self._finished = True

    def is_ready(self) -> bool:
        """Return True when ``step()`` would yield a non-None result."""
        if self._buffer.size == 0:
            return False
        if self._finished:
            return True
        return self._buffer.size >= self._min_chunk_samples

    def step(self) -> StreamingResult | None:
        """Re-decode the buffer and return a snapshot. Returns ``None`` if not ready."""
        if not self.is_ready():
            return None

        waveform = self._buffer[None, :]
        lengths = np.array([self._buffer.size], dtype=np.int64)
        results = list(self._asr.recognize_batch(waveform, lengths, **self._recognize_kwargs))
        snapshot = results[0] if results else TimestampedResult(text="")

        is_final = self._finished
        if is_final:
            self._endpoint = True

        tokens: Sequence[str] = snapshot.tokens or []
        timestamps_list = list(snapshot.timestamps) if snapshot.timestamps else None
        return StreamingResult(
            text=snapshot.text,
            tokens=list(tokens),
            timestamps=timestamps_list,
            is_partial=not is_final,
            segment_id=self._segment_id,
        )

    def reset(self, *, keep_audio: bool = False) -> None:
        """Zero buffer and endpoint state. Bumps segment id."""
        if not keep_audio:
            self._buffer = np.empty(0, dtype=np.float32)
        self._finished = False
        self._endpoint = False
        self._segment_id += 1

    @property
    def is_endpoint(self) -> bool:
        """Whether the last ``step()`` produced the final/committed snapshot."""
        return self._endpoint

    @property
    def buffered_samples(self) -> int:
        """Current buffer size in samples."""
        return int(self._buffer.size)


def create_buffered_stream(
    asr: Asr,
    *,
    sample_rate: int = 16_000,
    min_chunk_size_s: float = 0.5,
    recognize_kwargs: dict[str, object] | None = None,
) -> BufferedAsrStream:
    """Build a :class:`BufferedAsrStream` over ``asr`` without subclassing.

    Convenience wrapper. Lets callers compose streaming over any ASR:
    ``stream = create_buffered_stream(model.asr, language="en")``.
    """
    return BufferedAsrStream(
        asr,
        sample_rate=sample_rate,
        min_chunk_size_s=min_chunk_size_s,
        recognize_kwargs=recognize_kwargs,
    )
