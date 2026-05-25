"""PyAnnote VAD implementation."""

from collections.abc import Iterable, Iterator
from itertools import permutations
from pathlib import Path
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.onnx import OnnxSessionOptions, TensorRtOptions
from onnx_asr.utils import is_float32_array
from onnx_asr.vad import BaseVad


class PyAnnoteVad(BaseVad):
    """PyAnnote VAD implementation."""

    RECEPTIVE_FIELD = 991
    STRIDE = 270

    def __init__(self, model_files: dict[str, Path], onnx_options: OnnxSessionOptions):
        """Create PyAnnote VAD.

        Args:
            model_files: Dict with paths to model files.
            onnx_options: Options for onnxruntime InferenceSession.

        """
        self._model = rt.InferenceSession(model_files["model"], **onnx_options)
        self._num_windows = 0

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        return TensorRtOptions.get_provider_names()

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"**/model{suffix}.onnx"}

    @staticmethod
    def _sample2frame(l_in: int) -> int:
        return (l_in - PyAnnoteVad.RECEPTIVE_FIELD) // PyAnnoteVad.STRIDE + 1

    @staticmethod
    def _frame2sample(l_in: int) -> int:
        return (l_in - 1) * PyAnnoteVad.STRIDE + PyAnnoteVad.RECEPTIVE_FIELD

    def _encode(
        self, waveforms: npt.NDArray[np.float32], window_size: int, overlap: int, **kwargs: float
    ) -> Iterator[npt.NDArray[np.float32]]:
        """Encode waveforms into windowed model outputs.

        Args:
            waveforms: audio samples with shape (batch_size, num_samples)
            window_size: number of window samples
            overlap: number of overlap samples
            **kwargs: additional keyword arguments passed through to inner calls

        """
        # Sliding window needs at least one full window of audio. Short clips
        # (typical for short utterances < 10 s) get right-padded with zeros to
        # one window length so we still produce a single inference pass.
        original_len = waveforms.shape[1]
        if original_len < window_size:
            waveforms = np.pad(waveforms, ((0, 0), (0, window_size - original_len)))

        # 10s sliding window and 5s overlap
        windows = np.lib.stride_tricks.sliding_window_view(waveforms, window_size, axis=-1)[
            :, :: (window_size - overlap), :
        ]  # This will drop the last window if its length is less than the overlap (5s), be careful

        self._num_windows = windows.shape[1]
        # No partial-tail pass when the original audio fit inside one window.
        tail_remainder = original_len % (window_size - overlap) if original_len >= window_size else 0
        if tail_remainder:
            self._num_windows += 1

        def process(window: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            """Run model inference on a single window batch.

            Input:
                window: (batch_size, window_size)
            Output:
                batch of shape(batch_size, num_frames(589 for 10s), 7)
                { no-speech }, { spk1 }, { spk2 }, { spk3 }, { spk1 + spk2 }, { spk1 + spk3 }, { spk2 + spk3 }
            """
            output = np.exp(
                cast(npt.NDArray[np.float32], self._model.run(["logits"], {"input_values": window[:, None, :]})[0])
            )
            assert is_float32_array(output)
            return output

        for i in range(windows.shape[1]):
            yield process(windows[:, i, :])

        if tail_remainder:
            yield process(
                np.pad(
                    waveforms[:, -(tail_remainder + overlap) :],
                    ((0, 0), (0, (window_size - overlap - tail_remainder))),
                )
            )

    def _decode(
        self, windows: Iterator[npt.NDArray[np.float32]], window_size: int, overlap: int, **kwargs: float
    ) -> Iterator[tuple[int, npt.NDArray[np.float32]]]:
        """Decode windowed model outputs into per-sample speaker probability arrays.

        Args:
            windows: Iterator of window with shape (num_frames, num_spk) where num_spk = 7
            window_size: number of window samples
            overlap: number of overlap samples
            **kwargs: additional keyword arguments passed through to inner calls

        """

        def reorder(window: npt.NDArray[np.float32], chunk: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            """Make sure all the windows have the correct speaker order."""
            perms: list[npt.NDArray[np.float32]] = [
                np.array(perm, dtype=np.float32).T for perm in permutations(window.T)
            ]
            diffs = np.sum(np.abs(np.sum(np.array(perms)[:, :overlap_len] - chunk, axis=1)), axis=1)
            return perms[int(np.argmin(diffs))]

        def fuse(window: npt.NDArray[np.float32], chunk: npt.NDArray[np.float32]) -> None:
            """Combine overlapping chunks and take an average of their probs."""
            window[:, :] = (window[:, :] + chunk[:, :]) / 2

        def merge_spk_probs(window: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            # for each spk add all probs up
            window[:, 1] += window[:, 4] + window[:, 5]
            window[:, 2] += window[:, 4] + window[:, 6]
            window[:, 3] += window[:, 5] + window[:, 6]

            # dump other probs
            return window[:, :4]

        overlap_len = self._sample2frame(overlap)
        overlap_chunk: npt.NDArray[np.float32] = np.zeros((overlap_len, 4), dtype=np.float32)
        for i, window in enumerate(windows, 1):
            window = merge_spk_probs(window)
            if i == 1:
                overlap_chunk = window[-overlap_len:]
                yield 0, window[:-overlap_len]
            elif i > 1 and i < self._num_windows:
                fuse(window[:overlap_len], overlap_chunk)
                overlap_chunk = window[-overlap_len:]
                yield (i - 1) * (window_size - overlap), window[:-overlap_len]
            else:
                fuse(window[:overlap_len], overlap_chunk)
                yield (i - 1) * (window_size - overlap), window

    def _find_segments(
        self,
        decoding: Iterable[tuple[int, npt.NDArray[np.float32]]],
        *,
        threshold: float = 0.5,
        neg_threshold: float | None = None,
        **kwargs: float,
    ) -> Iterator[tuple[int, int]]:
        """Find speech segments from decoded probabilities.

        Args:
            decoding: a Iterable of (begin and window)
            threshold: probability threshold to enter speech state.
            neg_threshold: probability threshold to exit speech state, defaults to threshold - 0.15.
            **kwargs: additional keyword arguments passed through to inner calls

        """
        if neg_threshold is None:
            neg_threshold = threshold - 0.15

        offset = 135  # half of the stride

        state = 0
        start = 0
        last_begin = 0
        last_i = 0
        for begin, window in decoding:
            last_begin = begin
            for frame_idx, p in enumerate(window):
                last_i = frame_idx
                if state == 0 and p > threshold:
                    state = 1
                    start = begin + self._frame2sample(frame_idx) + offset
                if state == 1 and p < neg_threshold:
                    state = 0
                    yield start, begin + self._frame2sample(frame_idx) + offset

        if state == 1:
            yield start, last_begin + self._frame2sample(last_i) + offset

    def segment_batch(
        self,
        waveforms: npt.NDArray[np.float32],
        waveforms_len: npt.NDArray[np.int64],
        sample_rate: Literal[8000, 16000],
        **kwargs: float,
    ) -> Iterator[Iterator[tuple[int, int]]]:
        """Segment a batch of waveforms into speech intervals.

        Args:
            waveforms: audio samples with shape (batch_size, num_samples)
            waveforms_len: ints with shape( batch_size, 1 )
            sample_rate: Literal[16_000]
            **kwargs: additional keyword arguments forwarded to segmentation helpers

        """
        window_size = 10 * sample_rate
        overlap = 5 * sample_rate

        def segment(
            decoding: Iterable[tuple[int, npt.NDArray[np.float32]]], waveform_len: int, **kwargs: float
        ) -> Iterator[tuple[int, int]]:
            return self._merge_segments(self._find_segments(decoding, **kwargs), waveform_len, sample_rate, **kwargs)

        encoding = self._encode(waveforms, window_size, overlap, **kwargs)
        if len(waveforms) == 1:
            decoding = (
                (begin, 1 - window[:, 0])  # only put the first {no-speech} prob in. prob_speech = 1 - prob(no-speech)
                for begin, window in self._decode(
                    (batch_windows[0] for batch_windows in encoding), window_size, overlap
                )
            )
            yield segment(decoding, int(waveforms_len[0]), **kwargs)
        else:
            yield from (
                segment(
                    (
                        (begin, 1 - window[:, 0])
                        for begin, window in self._decode(
                            cast(Iterator[npt.NDArray[np.float32]], iter(undecode_windows)),
                            window_size,
                            overlap,
                        )
                    ),
                    int(waveform_len),
                    **kwargs,
                )
                for undecode_windows, waveform_len in zip(zip(*encoding, strict=True), waveforms_len, strict=False)
            )

    def speaker_probs_batch(
        self,
        waveforms: npt.NDArray[np.float32],
        waveforms_len: npt.NDArray[np.int64],
        sample_rate: Literal[8000, 16000] = 16000,
    ) -> Iterator[tuple[npt.NDArray[np.float32], int]]:
        """Yield ``(probs, frame_step_samples)`` for each waveform in the batch.

        ``probs`` is a ``(num_frames, 3)`` float32 array of per-frame per-local-speaker
        probabilities (the powerset decoder collapses to up to 3 simultaneous speakers
        within a 10 s window; the reorder step keeps local IDs consistent across
        overlapping windows). Each frame represents :attr:`STRIDE` audio samples.

        Use for downstream speaker diarization: thread these per-frame activations
        through an embedding extractor and a clusterer (see :mod:`onnx_asr.diarization`).
        """
        window_size = 10 * sample_rate
        overlap = 5 * sample_rate
        del waveforms_len  # used by segment_batch only; full prob track is yielded here

        encoding = self._encode(waveforms, window_size, overlap)
        if len(waveforms) == 1:
            chunks: list[npt.NDArray[np.float32]] = []
            offsets: list[int] = []
            for begin, window in self._decode(
                (batch_windows[0] for batch_windows in encoding), window_size, overlap
            ):
                # window has shape (num_frames, 4); drop the no-speech column.
                chunks.append(window[:, 1:4])
                offsets.append(begin)
            if not chunks:
                yield np.zeros((0, 3), dtype=np.float32), self.STRIDE
                return
            # Concatenate; offsets are in samples and chunks are non-overlapping
            # (the decoder already removed the overlap region from each except the
            # last). We rebuild a single contiguous (frames, 3) array.
            yield np.concatenate(chunks, axis=0).astype(np.float32, copy=False), self.STRIDE
        else:
            for undecode_windows in zip(*encoding, strict=True):
                chunks2: list[npt.NDArray[np.float32]] = []
                for _begin, window in self._decode(
                    cast(Iterator[npt.NDArray[np.float32]], iter(undecode_windows)),
                    window_size,
                    overlap,
                ):
                    chunks2.append(window[:, 1:4])
                if not chunks2:
                    yield np.zeros((0, 3), dtype=np.float32), self.STRIDE
                else:
                    yield np.concatenate(chunks2, axis=0).astype(np.float32, copy=False), self.STRIDE
