"""Offline speaker diarization on top of onnx-asr.

Pipeline:

1. :class:`onnx_asr.models.pyannote.PyAnnoteVad` runs ``pyannote/segmentation-3.0``
   in powerset mode and yields ``(num_frames, 3)`` per-local-speaker probability
   tracks. "Local" means within a 10 s window — the powerset decoder reorders
   adjacent windows to keep IDs consistent across overlaps, but identity across
   the full utterance still depends on clustering.
2. For each (local speaker, contiguous activity interval) we crop the audio and
   compute a Wespeaker ResNet34 embedding via
   :class:`onnx_asr.models.wespeaker.WespeakerEmbeddings`.
3. Cosine-distance complete-linkage agglomerative clustering (pure numpy)
   re-labels local IDs into global speaker IDs.
4. Consecutive intervals with the same global ID are merged.

Everything is numpy + onnxruntime — no torch, no scipy required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from onnx_asr.adapters import SeAdapter
from onnx_asr.models.pyannote import PyAnnoteVad


@dataclass(frozen=True)
class DiarSegment:
    """One contiguous speaker turn (seconds + global speaker id)."""

    start: float
    end: float
    speaker: int


class OnlineSpeakerClustering:
    """Incremental cosine-distance speaker clustering with persistent centroids.

    Port of ``diart.blocks.clustering.OnlineSpeakerClustering`` (MIT) into
    pure numpy — same algorithm, no torch / no pyannote.SpeakerMap dependency.

    Each call to :meth:`assign` takes a batch of new speaker embeddings and
    matches them against the running set of global speaker centroids:

    1. Compute cosine distance from every input embedding to every centroid.
    2. For each input, if its closest centroid is within ``delta_new``, map
       it to that centroid; otherwise create a new centroid (subject to
       ``max_speakers``). When no slots are free, the closest existing
       centroid wins (forced reuse — may cause label aliasing).
    3. Update each matched centroid with an exponential moving average of the
       new embedding, but only for inputs whose ``active_ratio`` exceeds
       ``rho_update`` (so quiet / noisy crops don't corrupt the centroid).

    State persists across calls — repeated invocations track the same
    speakers across an entire session.
    """

    def __init__(
        self,
        *,
        delta_new: float = 0.5,
        rho_update: float = 0.3,
        max_speakers: int = 20,
        ema_alpha: float = 0.5,
    ) -> None:
        """Configure clustering.

        Args:
            delta_new: cosine-distance threshold above which a new centroid is
                created (instead of matching an existing one). Default 0.5
                matches sherpa-onnx's AHC threshold for wespeaker / 3D-Speaker
                embeddings; diart's 1.0 is too loose for these models.
            rho_update: minimum ``active_ratio`` for a matched embedding to
                update its centroid. Embeddings below this still receive a
                speaker label but don't shift the centroid.
            max_speakers: hard cap on the number of distinct global speakers
                tracked through the session.
            ema_alpha: weight for the new embedding in the centroid EMA
                update (``new = alpha * embedding + (1 - alpha) * old``).
                ``0.5`` (the default) is equivalent to a running mean over
                the most recent observations.
        """
        self._delta_new = delta_new
        self._rho_update = rho_update
        self._max_speakers = max_speakers
        self._ema_alpha = ema_alpha
        self._centers: npt.NDArray[np.float32] | None = None
        self._active: list[bool] = []

    @property
    def num_known_speakers(self) -> int:
        """Distinct global speaker IDs created so far this session."""
        return sum(self._active)

    @property
    def num_free_slots(self) -> int:
        """Centroid slots remaining before hitting ``max_speakers``."""
        return self._max_speakers - self.num_known_speakers

    def reset(self) -> None:
        """Drop all centroids — next :meth:`assign` starts a fresh session."""
        self._centers = None
        self._active = []

    def _ensure_centers(self, embedding_dim: int) -> None:
        if self._centers is None:
            self._centers = np.zeros((self._max_speakers, embedding_dim), dtype=np.float32)
            self._active = [False] * self._max_speakers

    def _add_centroid(self, embedding: npt.NDArray[np.float32]) -> int:
        assert self._centers is not None
        for i, active in enumerate(self._active):
            if not active:
                self._centers[i] = embedding
                self._active[i] = True
                return i
        msg = "no free centroid slots; raise max_speakers"
        raise RuntimeError(msg)

    def _update_centroid(self, centroid_id: int, embedding: npt.NDArray[np.float32]) -> None:
        assert self._centers is not None
        alpha = self._ema_alpha
        self._centers[centroid_id] = alpha * embedding + (1.0 - alpha) * self._centers[centroid_id]

    def assign(
        self,
        embeddings: npt.NDArray[np.float32],
        active_ratios: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.int64]:
        """Match new embeddings to global speaker IDs; return per-input IDs.

        Args:
            embeddings: shape ``(n, embedding_dim)`` float32.
            active_ratios: shape ``(n,)`` in [0, 1]; the speech-frame fraction
                per embedding. Embeddings below ``rho_update`` are assigned but
                don't update centroids. If ``None``, all are treated as 1.0.

        Returns:
            ``(n,)`` int64 speaker IDs (stable across calls).
        """
        if embeddings.ndim != 2:
            msg = f"embeddings must be 2-D, got shape {embeddings.shape}"
            raise ValueError(msg)
        n, dim = embeddings.shape
        if n == 0:
            return np.zeros((0,), dtype=np.int64)

        if active_ratios is None:
            active_ratios = np.ones((n,), dtype=np.float32)
        elif active_ratios.shape[0] != n:
            msg = (
                f"active_ratios length ({active_ratios.shape[0]}) does not match "
                f"embeddings batch ({n})"
            )
            raise ValueError(msg)

        self._ensure_centers(dim)
        assert self._centers is not None

        labels = np.zeros((n,), dtype=np.int64)
        for i in range(n):
            emb = embeddings[i].astype(np.float32, copy=False)
            ratio = float(active_ratios[i])

            if self.num_known_speakers == 0:
                if self.num_free_slots > 0:
                    labels[i] = self._add_centroid(emb)
                else:
                    labels[i] = 0
                continue

            # Cosine distance against active centroids only.
            active_idxs = np.array([j for j, a in enumerate(self._active) if a], dtype=np.int64)
            active_centers = self._centers[active_idxs]
            emb_n = emb / max(float(np.linalg.norm(emb)), 1e-12)
            ctr_n = active_centers / np.maximum(
                np.linalg.norm(active_centers, axis=1, keepdims=True), 1e-12
            )
            distances = 1.0 - ctr_n @ emb_n  # (num_active,)
            closest_local = int(np.argmin(distances))
            closest_dist = float(distances[closest_local])
            closest_global = int(active_idxs[closest_local])

            if closest_dist <= self._delta_new:
                labels[i] = closest_global
                if ratio >= self._rho_update:
                    self._update_centroid(closest_global, emb)
            elif self.num_free_slots > 0:
                labels[i] = self._add_centroid(emb)
            else:
                # No room for a new speaker; forced reuse of closest centroid.
                labels[i] = closest_global

        return labels


def _cosine_distance_matrix(embeddings: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Pairwise cosine distance after L2 norm; shape ``(n, n)``."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.maximum(norms, 1e-12)
    cos = unit @ unit.T
    return np.clip(1.0 - cos, 0.0, 2.0).astype(np.float32, copy=False)


def _ahc_complete_linkage(
    distances: npt.NDArray[np.float32],
    *,
    num_clusters: int | None,
    threshold: float,
) -> npt.NDArray[np.int64]:
    """Complete-linkage AHC; either fixed cluster count or stop above ``threshold``."""
    n = distances.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if n == 1:
        return np.zeros((1,), dtype=np.int64)

    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    distance_matrix = distances.copy()
    np.fill_diagonal(distance_matrix, np.inf)

    while len(members) > 1:
        flat = int(np.argmin(distance_matrix))
        i, j = divmod(flat, distance_matrix.shape[0])
        if i == j:
            break
        min_d = float(distance_matrix[i, j])
        if num_clusters is None and min_d > threshold:
            break
        if num_clusters is not None and len(members) <= num_clusters:
            break

        keep, drop = (i, j) if i < j else (j, i)
        members[keep] = members[keep] + members[drop]
        del members[drop]
        new_row = np.maximum(distance_matrix[keep], distance_matrix[drop])
        new_row[keep] = np.inf
        distance_matrix[keep, :] = new_row
        distance_matrix[:, keep] = new_row
        distance_matrix[drop, :] = np.inf
        distance_matrix[:, drop] = np.inf

    labels = np.zeros(n, dtype=np.int64)
    for new_id, (_, idxs) in enumerate(sorted(members.items())):
        for idx in idxs:
            labels[idx] = new_id
    return labels


def _active_intervals(
    probs_one_speaker: npt.NDArray[np.float32],
    *,
    onset: float,
    offset: float,
    min_frames: int,
    merge_frames: int,
) -> list[tuple[int, int]]:
    """Hysteresis thresholding on a 1-D probability track.

    Returns ``[(start_frame, end_frame), ...]`` half-open intervals.
    """
    intervals: list[tuple[int, int]] = []
    n = probs_one_speaker.shape[0]
    state = False
    start = 0
    for i in range(n):
        p = float(probs_one_speaker[i])
        if not state and p >= onset:
            state = True
            start = i
        elif state and p < offset:
            state = False
            intervals.append((start, i))
    if state:
        intervals.append((start, n))

    if not intervals:
        return []

    # Merge intervals separated by gaps shorter than ``merge_frames``.
    merged: list[tuple[int, int]] = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s - pe < merge_frames:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    # Drop intervals shorter than ``min_frames``.
    return [(s, e) for s, e in merged if e - s >= min_frames]


class Diarizer:
    """Offline speaker diarization.

    Combine a :class:`~onnx_asr.models.pyannote.PyAnnoteVad` segmenter and a
    Wespeaker :class:`~onnx_asr.adapters.SeAdapter`; call :meth:`diarize`.

    Both components are loaded via :func:`onnx_asr.load_diarizer`, which is the
    recommended entry point.
    """

    SAMPLE_RATE: int = 16_000

    def __init__(
        self,
        segmenter: PyAnnoteVad,
        embedder: SeAdapter,
        *,
        onset: float = 0.5,
        offset: float = 0.35,
        min_segment_duration: float = 0.5,
        merge_gap_duration: float = 0.3,
        min_embedding_duration: float = 0.5,
    ) -> None:
        """Wire up the diarizer.

        Args:
            segmenter: pyannote/segmentation-3.0 ONNX model wrapper.
            embedder: wespeaker speaker-embedding adapter.
            onset: probability above which a local speaker becomes active.
            offset: probability below which the speaker becomes inactive.
            min_segment_duration: drop segments shorter than this many seconds.
            merge_gap_duration: merge adjacent same-speaker segments split by
                gaps shorter than this many seconds.
            min_embedding_duration: ignore segments shorter than this when
                computing embeddings (too-short crops yield noisy vectors).
        """
        self._seg = segmenter
        self._emb = embedder
        self._onset = onset
        self._offset = offset
        self._min_seg = min_segment_duration
        self._merge_gap = merge_gap_duration
        self._min_emb = min_embedding_duration

    def close(self, *, empty_torch_cache: bool = True) -> int:
        """Release ORT sessions held by segmentation + embedding models."""
        from onnx_asr._session_cleanup import release_inference_sessions  # noqa: PLC0415

        return release_inference_sessions(self, empty_torch_cache=empty_torch_cache)

    def __enter__(self) -> Diarizer:  # noqa: PYI034
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        self.close()

    def diarize_with_embeddings(
        self,
        waveform: npt.NDArray[np.float32],
        *,
        sample_rate: int = 16_000,
    ) -> list[dict[str, object]]:
        """Variant of :meth:`diarize` returning *un-clustered* segments + embeddings.

        Used by :class:`SessionDiarizer` to drive session-wide online clustering
        externally — each returned dict carries ``start`` / ``end`` (seconds),
        ``embedding`` (float32 vector), and ``active_ratio`` (mean per-frame
        probability over the segment). Skips the AHC step that :meth:`diarize`
        performs internally.
        """
        if sample_rate != self.SAMPLE_RATE:
            msg = f"diarizer expects {self.SAMPLE_RATE} Hz audio, got {sample_rate}"
            raise ValueError(msg)
        if waveform.ndim != 1:
            msg = f"diarizer expects a 1-D waveform, got shape {waveform.shape}"
            raise ValueError(msg)

        audio = waveform.astype(np.float32, copy=False)[None, :]
        lens = np.array([audio.shape[1]], dtype=np.int64)
        (probs, frame_step) = next(self._seg.speaker_probs_batch(audio, lens, self.SAMPLE_RATE))
        if probs.shape[0] == 0:
            return []

        sr = self.SAMPLE_RATE
        frame_to_sec = frame_step / sr
        min_frames = max(1, int(self._min_seg / frame_to_sec))
        merge_frames = max(1, int(self._merge_gap / frame_to_sec))
        min_emb_frames = max(1, int(self._min_emb / frame_to_sec))

        segments: list[tuple[int, int, int]] = []  # (local_id, start_f, end_f)
        for local_id in range(probs.shape[1]):
            for start_f, end_f in _active_intervals(
                probs[:, local_id],
                onset=self._onset,
                offset=self._offset,
                min_frames=min_frames,
                merge_frames=merge_frames,
            ):
                if end_f - start_f >= min_emb_frames:
                    segments.append((local_id, start_f, end_f))

        if not segments:
            return []

        crops: list[npt.NDArray[np.float32]] = []
        for _local, s, e in segments:
            start_s = max(0, s * frame_step)
            end_s = min(waveform.shape[0], e * frame_step)
            crops.append(waveform[start_s:end_s].astype(np.float32, copy=False))

        max_len = max(c.shape[0] for c in crops)
        batch = np.zeros((len(crops), max_len), dtype=np.float32)
        batch_lens = np.zeros((len(crops),), dtype=np.int64)
        for i, crop in enumerate(crops):
            batch[i, : crop.shape[0]] = crop
            batch_lens[i] = crop.shape[0]
        embeddings = np.asarray(self._emb.se.embedding(batch, batch_lens), dtype=np.float32)

        return [
            {
                "start": s * frame_to_sec,
                "end": e * frame_to_sec,
                "embedding": embeddings[i],
                "active_ratio": float(np.mean(probs[s:e, local])),
            }
            for i, (local, s, e) in enumerate(segments)
        ]

    def diarize(
        self,
        waveform: npt.NDArray[np.float32],
        *,
        sample_rate: int = 16_000,
        num_speakers: int | None = None,
        threshold: float = 0.7,
    ) -> list[DiarSegment]:
        """Run diarization on a single mono waveform.

        Args:
            waveform: 1-D float32 PCM in [-1, 1].
            sample_rate: must be 16_000 (pyannote-segmentation-3.0 is 16 kHz only).
            num_speakers: if known, force exactly this many clusters; else AHC
                stops when the next merge would exceed ``threshold``.
            threshold: cosine-distance cutoff for AHC; larger = fewer speakers.

        Returns:
            Time-ordered list of :class:`DiarSegment` covering the audio.
        """
        if sample_rate != self.SAMPLE_RATE:
            msg = f"diarizer expects {self.SAMPLE_RATE} Hz audio, got {sample_rate}"
            raise ValueError(msg)
        if waveform.ndim != 1:
            msg = f"diarizer expects a 1-D waveform, got shape {waveform.shape}"
            raise ValueError(msg)

        # ---- 1. Per-frame, per-local-speaker probabilities --------------------
        audio = waveform.astype(np.float32, copy=False)[None, :]
        lens = np.array([audio.shape[1]], dtype=np.int64)
        (probs, frame_step) = next(self._seg.speaker_probs_batch(audio, lens, self.SAMPLE_RATE))
        if probs.shape[0] == 0:
            return []

        # ---- 2. Extract continuous activity intervals per local speaker -------
        sr = self.SAMPLE_RATE
        frame_to_sec = frame_step / sr
        min_frames = max(1, int(self._min_seg / frame_to_sec))
        merge_frames = max(1, int(self._merge_gap / frame_to_sec))
        min_emb_frames = max(1, int(self._min_emb / frame_to_sec))

        # ``raw_segments[i]`` = (local_speaker_idx, start_frame, end_frame)
        raw_segments: list[tuple[int, int, int]] = []
        for local_id in range(probs.shape[1]):
            for start_f, end_f in _active_intervals(
                probs[:, local_id],
                onset=self._onset,
                offset=self._offset,
                min_frames=min_frames,
                merge_frames=merge_frames,
            ):
                raw_segments.append((local_id, start_f, end_f))

        if not raw_segments:
            return []

        # ---- 3. Embed each segment that's long enough -------------------------
        embeddable = [
            (idx, local_id, s, e)
            for idx, (local_id, s, e) in enumerate(raw_segments)
            if e - s >= min_emb_frames
        ]
        if not embeddable:
            # Audio is too fragmented to embed; collapse everything to speaker 0.
            return self._build_output(raw_segments, frame_to_sec, [0] * len(raw_segments))

        # Crop each segment, pad to the longest one, then batch through embedder.
        crops: list[npt.NDArray[np.float32]] = []
        for _idx, _local, s, e in embeddable:
            start_s = max(0, s * frame_step)
            end_s = min(waveform.shape[0], e * frame_step)
            crops.append(waveform[start_s:end_s].astype(np.float32, copy=False))

        # Pad to longest, build the (batch, samples) tensor.
        max_len = max(c.shape[0] for c in crops)
        batch = np.zeros((len(crops), max_len), dtype=np.float32)
        batch_lens = np.zeros((len(crops),), dtype=np.int64)
        for i, crop in enumerate(crops):
            batch[i, : crop.shape[0]] = crop
            batch_lens[i] = crop.shape[0]

        embeddings = self._emb.se.embedding(batch, batch_lens)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        # ---- 4. Cluster embeddings into global speaker IDs --------------------
        distances = _cosine_distance_matrix(embeddings)
        global_ids_for_embedded = _ahc_complete_linkage(
            distances, num_clusters=num_speakers, threshold=threshold
        )

        # Map global IDs back over ALL raw_segments. For too-short segments we
        # assign by nearest embedded segment of the same local_id (heuristic).
        labels: list[int] = [0] * len(raw_segments)
        embedded_lookup: dict[int, int] = {
            embeddable[i][0]: int(global_ids_for_embedded[i]) for i in range(len(embeddable))
        }
        # For non-embedded segments, fall back to nearest neighbour by start time
        # among embedded segments with the same local_id.
        embedded_by_local: dict[int, list[tuple[int, int, int, int]]] = {}
        for emb_i, (raw_idx, local, s, e) in enumerate(embeddable):
            embedded_by_local.setdefault(local, []).append(
                (s, e, raw_idx, int(global_ids_for_embedded[emb_i]))
            )

        for idx, (local_id, s, _e) in enumerate(raw_segments):
            if idx in embedded_lookup:
                labels[idx] = embedded_lookup[idx]
                continue
            neighbours = embedded_by_local.get(local_id)
            if not neighbours:
                # No same-local-id reference; fall back to global speaker 0.
                labels[idx] = 0
                continue
            best = min(neighbours, key=lambda n: abs(n[0] - s))
            labels[idx] = best[3]

        return self._build_output(raw_segments, frame_to_sec, labels)

    @staticmethod
    def _build_output(
        raw_segments: list[tuple[int, int, int]],
        frame_to_sec: float,
        labels: list[int],
    ) -> list[DiarSegment]:
        """Convert (local_id, start_f, end_f) + per-segment labels into time-ordered
        :class:`DiarSegment` list, merging adjacent same-speaker runs."""
        triples = sorted(
            (
                (start_f * frame_to_sec, end_f * frame_to_sec, labels[i])
                for i, (_local, start_f, end_f) in enumerate(raw_segments)
            ),
            key=lambda t: t[0],
        )
        merged: list[DiarSegment] = []
        for start, end, spk in triples:
            if merged and merged[-1].speaker == spk and start - merged[-1].end < 0.05:
                last = merged[-1]
                merged[-1] = DiarSegment(last.start, max(last.end, end), spk)
            else:
                merged.append(DiarSegment(start, end, spk))
        return merged


class SessionDiarizer:
    """Per-utterance diarizer with session-wide speaker identity tracking.

    Wraps a :class:`Diarizer` and a persistent :class:`OnlineSpeakerClustering`
    so that speaker IDs remain stable across multiple calls — when a speaker
    who appeared in an earlier utterance returns, they get the same ID.

    Use this from a streaming server: feed each finalized utterance's audio
    through :meth:`diarize`, attach the returned segments to the transcript.
    Call :meth:`reset` between sessions (e.g. when the recorder restarts).
    """

    def __init__(
        self,
        diarizer: Diarizer,
        *,
        max_speakers: int = 20,
        delta_new: float = 0.5,
        rho_update: float = 0.3,
        ema_alpha: float = 0.5,
    ) -> None:
        """Wrap a stateless :class:`Diarizer` with persistent clustering."""
        self._diarizer = diarizer
        self._clustering = OnlineSpeakerClustering(
            delta_new=delta_new,
            rho_update=rho_update,
            max_speakers=max_speakers,
            ema_alpha=ema_alpha,
        )

    def reset(self) -> None:
        """Forget all session speakers (next utterance starts fresh IDs)."""
        self._clustering.reset()

    @property
    def num_known_speakers(self) -> int:
        """Distinct global speakers tracked so far this session."""
        return self._clustering.num_known_speakers

    def close(self, *, empty_torch_cache: bool = True) -> int:
        """Release ORT sessions held by the wrapped diarizer."""
        return self._diarizer.close(empty_torch_cache=empty_torch_cache)

    def __enter__(self) -> SessionDiarizer:  # noqa: PYI034
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        self.close()

    def diarize(
        self,
        waveform: npt.NDArray[np.float32],
        *,
        sample_rate: int = 16_000,
    ) -> list[DiarSegment]:
        """Diarize a single utterance; speaker IDs persist across calls."""
        local_segments = self._diarizer.diarize_with_embeddings(
            waveform, sample_rate=sample_rate
        )
        if not local_segments:
            return []

        embeddings = np.stack([s["embedding"] for s in local_segments], axis=0)
        ratios = np.array([s["active_ratio"] for s in local_segments], dtype=np.float32)
        global_ids = self._clustering.assign(embeddings, ratios)

        merged: list[DiarSegment] = []
        triples = sorted(
            (
                (float(s["start"]), float(s["end"]), int(global_ids[i]))
                for i, s in enumerate(local_segments)
            ),
            key=lambda t: t[0],
        )
        for start, end, spk in triples:
            if merged and merged[-1].speaker == spk and start - merged[-1].end < 0.05:
                last = merged[-1]
                merged[-1] = DiarSegment(last.start, max(last.end, end), spk)
            else:
                merged.append(DiarSegment(start, end, spk))
        return merged


def assign_speakers_to_words(
    words: list[tuple[str, float, float]],
    segments: list[DiarSegment],
    *,
    smoothing_window_s: float = 1.5,
) -> list[tuple[str, float, float, int]]:
    """Tag each ``(text, start, end)`` word tuple with the dominant speaker id.

    Per-word assignment is *overlap-weighted majority vote within a temporal
    window* — for each word we sum the time each speaker is active inside
    ``[word_mid - smoothing_window_s/2, word_mid + smoothing_window_s/2]`` and
    pick the winner. This suppresses per-word toggling in overlap regions and
    handles short function words ("a", "the", "I") whose own interval may not
    intersect any active segment cleanly.

    Pass ``smoothing_window_s=0`` to fall back to plain per-word overlap (the
    behaviour before this argument was added).
    """
    if not segments:
        return [(t, s, e, -1) for t, s, e in words]

    out: list[tuple[str, float, float, int]] = []
    half_window = max(0.0, smoothing_window_s) / 2.0

    for text, w_start, w_end in words:
        midpoint = 0.5 * (w_start + w_end)
        if half_window > 0:
            lo = midpoint - half_window
            hi = midpoint + half_window
        else:
            lo, hi = w_start, w_end

        scores: dict[int, float] = {}
        for seg in segments:
            overlap = max(0.0, min(hi, seg.end) - max(lo, seg.start))
            if overlap > 0:
                scores[seg.speaker] = scores.get(seg.speaker, 0.0) + overlap

        if scores:
            best_spk = max(scores.items(), key=lambda kv: kv[1])[0]
        else:
            # Nothing overlaps the window — fall back to nearest segment edge.
            best_spk = min(
                segments,
                key=lambda s: min(abs(midpoint - s.start), abs(midpoint - s.end)),
            ).speaker
        out.append((text, w_start, w_end, best_spk))
    return out
