"""LocalAgreement-2 streaming wrapper for Whisper models.

Hosts :class:`WhisperStream` — the canonical UFAL ``whisper_streaming``
policy implemented on top of any :class:`~onnx_asr.models.whisper._base._Whisper`
subclass. See the class docstring for the per-step algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from onnx_asr.asr import StreamingResult

if TYPE_CHECKING:
    from onnx_asr.models.whisper._base import _Whisper


class WhisperStream:
    """LocalAgreement-2 streaming wrapper around a Whisper model.

    The classic UFAL ``whisper_streaming`` policy, ONNX-friendly. Each ``step()``:

    1. Re-decode the whole rolling audio buffer with the underlying greedy
       decoder, with Whisper timestamp tokens enabled so output carries
       ``<|t|>`` segment markers.
    2. Compute the longest common prefix (token-level, integer IDs) of the new
       token sequence against the previous step's — those tokens are committed.
    3. Render the committed prefix as text (timestamp tokens are stripped by
       the byte-decoder filter); render the full snapshot as the live preview.
    4. If the audio buffer exceeds ``trim_after_s`` AND we have at least one
       committed ``<|t|>`` boundary, drop the audio prefix up to that boundary
       and move the corresponding text into a persistent history string.
       The next decode runs on a shorter buffer; LocalAgreement-2 starts
       fresh on the residual tail.

    On ``finish()``, the next ``step()`` commits everything (no LCA needed)
    and the snapshot has ``is_partial=False`` / ``is_endpoint=True``.

    The trim policy bounds per-step compute to O(trim_after_s) regardless of
    utterance length — without it, decoding a 30 s utterance does 30 full
    encoder passes on a growing buffer (quadratic).
    """

    def __init__(
        self,
        asr: _Whisper,
        *,
        sample_rate: int = 16_000,
        min_chunk_size_s: float = 1.0,
        language: str | None = None,
        trim_after_s: float = 8.0,
    ) -> None:
        """Bind the stream to ``asr``. See :meth:`_Whisper.create_stream` for kwargs."""
        self._asr = asr
        self._sample_rate = sample_rate
        self._min_chunk_samples = max(1, int(min_chunk_size_s * sample_rate))
        self._language = language
        self._trim_after_samples = max(self._min_chunk_samples * 2, int(trim_after_s * sample_rate))
        self._buffer: npt.NDArray[np.float32] = np.empty(0, dtype=np.float32)
        # LocalAgreement-2 state for the *current* (post-trim) window.
        self._prev_token_ids: list[int] = []
        self._committed_count = 0
        # Text committed in earlier trim windows — concatenated with current
        # window's committed text on every snapshot.
        self._history_text = ""
        self._finished = False
        self._endpoint = False
        self._segment_id = 0

    def push_audio(self, samples: npt.NDArray[np.float32], sample_rate: int = 16_000) -> None:
        """Append PCM samples to the input buffer."""
        if sample_rate != self._sample_rate:
            msg = f"WhisperStream sample_rate mismatch: expected {self._sample_rate}, got {sample_rate}"
            raise ValueError(msg)
        flat = np.asarray(samples, dtype=np.float32).ravel()
        self._buffer = np.concatenate([self._buffer, flat])

    def finish(self) -> None:
        """Mark the input as closed. Next ``step()`` commits everything."""
        self._finished = True

    def is_ready(self) -> bool:
        """Return True once buffer has enough audio for a meaningful decode."""
        if self._buffer.size == 0:
            return False
        if self._finished:
            return True
        return self._buffer.size >= self._min_chunk_samples

    def step(self) -> StreamingResult | None:
        """Re-decode, update commit prefix via LocalAgreement-2, return snapshot."""
        if not self.is_ready():
            return None

        token_ids, _ = self._asr._transcribe_single(self._buffer, language=self._language, with_timestamps=True)
        token_ids_list = [int(t) for t in token_ids]

        if self._finished:
            self._committed_count = len(token_ids_list)
            self._endpoint = True
        else:
            lcp = self._longest_common_prefix(token_ids_list, self._prev_token_ids)
            # Committed count grows monotonically — never shrink even if a later
            # decode briefly disagrees with itself at the boundary.
            self._committed_count = max(self._committed_count, min(lcp, len(token_ids_list)))

        committed_ids = token_ids_list[: self._committed_count]
        current_window_text = (
            self._asr._decode_tokens(np.asarray(committed_ids, dtype=np.int64)).text if committed_ids else ""
        )
        # Render the *full* snapshot (committed + speculative tail) as plain
        # text for live-preview consumers. ``_decode_text`` strips ``<|...|>``
        # tokens including timestamp markers.
        full_text = self._asr._decode_text(token_ids_list)

        # Trim audio buffer up to the latest fully-committed ``<|t|>`` boundary
        # once it's worth doing — keeps the next decode bounded.
        if (
            not self._finished
            and self._buffer.size >= self._trim_after_samples
            and self._asr._timestamp_begin_id is not None
        ):
            self._maybe_trim_buffer(committed_ids)

        self._prev_token_ids = token_ids_list

        snapshot_text = (self._history_text + " " + full_text).strip() if self._history_text else full_text
        snapshot_committed = (
            (self._history_text + " " + current_window_text).strip() if self._history_text else current_window_text
        )
        token_strs = [
            self._asr._vocab[tid] for tid in token_ids_list if not self._asr._vocab.get(tid, "").startswith("<|")
        ]

        return StreamingResult(
            text=snapshot_text,
            tokens=token_strs,
            timestamps=None,
            is_partial=not self._finished,
            segment_id=self._segment_id,
            committed_text=snapshot_committed,
        )

    def _maybe_trim_buffer(self, committed_ids: list[int]) -> None:
        """Trim audio + token state up to the last ``<|t|>`` in ``committed_ids``.

        Walks the committed-token prefix backwards looking for a timestamp
        marker (id ``>= _timestamp_begin_id``). If found, drops audio up to
        that time, moves the corresponding text into :attr:`_history_text`,
        and resets the LCA window so the next decode operates on the residual
        tail. No-op if no committed timestamp marker exists yet.
        """
        begin_id = self._asr._timestamp_begin_id
        if begin_id is None:
            return
        # Find the latest committed timestamp token (excluding the very first,
        # which would just be <|0.00|> — trimming there is a no-op).
        cut_token_idx = -1
        for i in range(len(committed_ids) - 1, -1, -1):
            if committed_ids[i] >= begin_id:
                cut_token_idx = i
                break
        if cut_token_idx <= 0:
            return

        cut_time_s = (committed_ids[cut_token_idx] - begin_id) * self._asr._timestamp_step_s
        cut_samples = int(cut_time_s * self._sample_rate)
        if cut_samples <= 0 or cut_samples >= self._buffer.size:
            return

        # Render the text up to (but not including) the cut timestamp marker
        # into history. The committed_ids prefix we keep is exactly what's
        # before that marker.
        trimmed_text_ids = committed_ids[:cut_token_idx]
        trimmed_text = (
            self._asr._decode_tokens(np.asarray(trimmed_text_ids, dtype=np.int64)).text if trimmed_text_ids else ""
        )
        if trimmed_text:
            self._history_text = (
                (self._history_text + " " + trimmed_text).strip() if self._history_text else trimmed_text
            )

        # Drop the corresponding audio prefix and reset LCA window.
        self._buffer = self._buffer[cut_samples:]
        self._prev_token_ids = []
        self._committed_count = 0

    def reset(self, *, keep_audio: bool = False) -> None:
        """Zero decode state; bump segment id. Drops audio buffer unless ``keep_audio=True``."""
        if not keep_audio:
            self._buffer = np.empty(0, dtype=np.float32)
        self._prev_token_ids = []
        self._committed_count = 0
        self._history_text = ""
        self._finished = False
        self._endpoint = False
        self._segment_id += 1

    @property
    def is_endpoint(self) -> bool:
        """Whether the last ``step()`` produced the final/committed snapshot."""
        return self._endpoint

    @property
    def buffered_samples(self) -> int:
        """Current buffer size in samples (post-trim)."""
        return int(self._buffer.size)

    @staticmethod
    def _longest_common_prefix(a: list[int], b: list[int]) -> int:
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n
