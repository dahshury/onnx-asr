"""OpenWakeWord ONNX adapter.

Wraps the `openWakeWord <https://github.com/dscripka/openWakeWord>`_ ONNX
artifacts: a shared front-end (melspectrogram + Google speech_embedding) and
a per-wake-word classifier head. The front-end runs once per chunk; each
wake-word classifier consumes the same embedding window and outputs a
probability.

The class itself is a thin wrapper around three ``rt.InferenceSession``
groups:

* ``self._melspec``  — int16 PCM → mel-spectrogram (32-bin)
* ``self._embedding`` — mel windows → 96-d audio embeddings
* ``self._classifiers`` — embedding window → wake-word probability (per word)

Each detection pass works on a window of audio (default 1.28 s = 20480 samples
@ 16 kHz, matching OpenWakeWord's reference). For streaming use, the caller
slides a window and calls :meth:`detect_batch` on each.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.onnx import OnnxSessionOptions, TensorRtOptions
from onnx_asr.utils import is_float32_array
from onnx_asr.wake_word import WakeWord, WakeWordResult

#: Reference window size used by OpenWakeWord's training pipeline (1.28 s at 16 kHz).
DEFAULT_WINDOW_SAMPLES = 20_480
#: Default per-wake-word threshold for declaring a detection.
DEFAULT_THRESHOLD = 0.5
#: Embedding output dimensionality from Google's speech_embedding model.
EMBEDDING_DIM = 96
#: Window of embedding frames consumed by the per-wake-word classifier head.
EMBEDDING_WINDOW_FRAMES = 16


class OpenWakeWord(WakeWord):
    """OpenWakeWord-style wake-word detector backed by three ONNX sessions.

    ``model_files`` must contain:

    * ``melspec``     — path to ``melspectrogram.onnx``
    * ``embedding``   — path to ``embedding_model.onnx``
    * ``classifier_<name>`` — one or more wake-word classifier ONNX files;
      the part after ``classifier_`` becomes the wake-word's name.
    """

    def __init__(self, model_files: dict[str, Path], onnx_options: OnnxSessionOptions) -> None:
        """Build sessions for the shared front-end and each wake-word head.

        Args:
            model_files: Dict from logical key → resolved file path.
            onnx_options: ORT session options applied uniformly to all sessions.

        """
        self._melspec = rt.InferenceSession(model_files["melspec"], **onnx_options)
        self._embedding = rt.InferenceSession(model_files["embedding"], **onnx_options)
        # Every other key that starts with ``classifier_`` becomes a wake-word.
        self._classifiers: dict[str, rt.InferenceSession] = {}
        for key, path in model_files.items():
            if key.startswith("classifier_"):
                name = key.removeprefix("classifier_")
                self._classifiers[name] = rt.InferenceSession(path, **onnx_options)
        if not self._classifiers:
            msg = "OpenWakeWord requires at least one ``classifier_<name>`` entry in model_files."
            raise ValueError(msg)
        self._wake_words = tuple(sorted(self._classifiers))

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        # OWW's melspec model uses ops TRT doesn't accelerate cleanly; the
        # embedding+classifier paths are small and run great on CPU anyway.
        return TensorRtOptions.get_provider_names()

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        # Quantization is irrelevant for OWW; the artifacts are tiny (<10 MB
        # combined) and already shipped at low precision. The resolver still
        # passes the kwarg, so we accept and ignore it.
        _ = quantization
        return {
            "melspec": "melspectrogram.onnx",
            "embedding": "embedding_model.onnx",
            # Wake-word classifiers are wildcard-matched by the resolver and
            # routed into ``classifier_<basename>`` keys below by the loader.
            # The pattern matches all ``*.onnx`` files at the repo root except
            # the two well-known front-end files; selection happens in the loader.
        }

    @property
    def wake_words(self) -> tuple[str, ...]:
        """Names of the loaded wake-word classifiers in stable sorted order."""
        return self._wake_words

    def _mel_window(self, pcm_int16: npt.NDArray[np.int16]) -> npt.NDArray[np.float32]:
        """Compute the 76 x 32 mel-spectrogram for a single audio window.

        Mirrors OpenWakeWord's reference scaling (``mel/10 + 2``) which
        aligns the ONNX export with the original TF SavedModel range.
        """
        x = pcm_int16.astype(np.float32)[None, :]
        (mel,) = self._melspec.run(None, {"input": x})
        assert is_float32_array(mel)
        return np.squeeze(mel) / 10.0 + 2.0

    def _embed_window(self, mel: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Compute the embedding feature vector for a single mel window."""
        # Embedding model expects shape (batch, frames, mel_bins, 1).
        batch = mel[None, :, :, None].astype(np.float32)
        (emb,) = self._embedding.run(None, {"input_1": batch})
        assert is_float32_array(emb)
        return np.squeeze(emb)

    def _classify(self, embedding: npt.NDArray[np.float32]) -> dict[str, float]:
        """Run every loaded wake-word classifier head against ``embedding``."""
        # Each head expects (1, window_frames, embedding_dim).
        x = embedding[None, :, :].astype(np.float32)
        scores: dict[str, float] = {}
        for name, session in self._classifiers.items():
            (out,) = session.run(None, {session.get_inputs()[0].name: x})
            scores[name] = float(np.asarray(out).squeeze())
        return scores

    def detect_batch(
        self,
        waveforms: npt.NDArray[np.float32],
        waveforms_len: npt.NDArray[np.int64],
        *,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> Iterator[WakeWordResult]:
        """Detect wake-words in each clip of ``waveforms``.

        Each clip is processed independently. Audio is converted to int16
        (the format the melspec ONNX export expects), front-end is run to
        produce an embedding window, and every classifier scores against it.
        """
        for i in range(waveforms.shape[0]):
            length = int(waveforms_len[i])
            clip = waveforms[i, :length]
            yield self._detect_single(clip, threshold=threshold)

    def _detect_single(self, waveform: npt.NDArray[np.float32], *, threshold: float) -> WakeWordResult:
        # Pad or truncate to the reference window length so the classifier
        # always sees the same shape (matches OWW's training-time slicing).
        if waveform.shape[0] < DEFAULT_WINDOW_SAMPLES:
            padded = np.zeros(DEFAULT_WINDOW_SAMPLES, dtype=np.float32)
            padded[: waveform.shape[0]] = waveform
            waveform = padded
        else:
            waveform = waveform[-DEFAULT_WINDOW_SAMPLES:]

        pcm_int16 = (waveform * 32767.0).astype(np.int16)
        mel = self._mel_window(pcm_int16)
        embedding = self._embed_window(mel)
        # Keep only the most recent EMBEDDING_WINDOW_FRAMES embedding frames
        # (drop the earlier ones; classifier head was trained on this fixed
        # window length).
        if embedding.ndim == 1:
            # Single-frame embedding (very short input); broadcast to window.
            embedding = np.tile(embedding[None, :], (EMBEDDING_WINDOW_FRAMES, 1))
        elif embedding.shape[0] >= EMBEDDING_WINDOW_FRAMES:
            embedding = embedding[-EMBEDDING_WINDOW_FRAMES:]
        else:
            pad = np.zeros((EMBEDDING_WINDOW_FRAMES - embedding.shape[0], embedding.shape[1]), dtype=np.float32)
            embedding = np.concatenate([pad, embedding], axis=0)

        scores = self._classify(embedding)
        # Pick the highest-scoring wake-word above threshold (stable name order).
        triggered = max(
            (name for name in self._wake_words if scores[name] >= threshold),
            key=lambda n: scores[n],
            default=None,
        )
        return WakeWordResult(scores=scores, detected=triggered)

    @staticmethod
    def _get_sample_rate() -> Literal[16_000]:
        return 16_000
