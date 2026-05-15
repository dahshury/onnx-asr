"""Picovoice Porcupine wake-word adapter (optional dependency).

Wraps `pvporcupine <https://github.com/Picovoice/porcupine>`_'s engine
behind the :class:`onnx_asr.wake_word.WakeWord` protocol so callers can
plug it into the same code paths as :class:`~onnx_asr.models.openwakeword.OpenWakeWord`.

This is NOT registered in :data:`onnx_asr.loader.create_wake_word_resolver`'s
default ``model_types`` map — pvporcupine is a proprietary/commercial SDK
that requires a free access key from Picovoice and isn't a hard dependency
of onnx-asr. Users opt in by importing the class directly::

    from onnx_asr.models.porcupine import PorcupineWakeWord

    detector = PorcupineWakeWord.create(
        access_key="<your-key>",
        keywords=["picovoice", "bumblebee"],
        sensitivities=[0.5, 0.5],
    )
    result = next(detector.detect_batch(waveforms, lengths))

If ``pvporcupine`` is not installed, the import succeeds (we only fail at
``create()`` time) so this module's other definitions stay usable.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from onnx_asr.wake_word import WakeWord, WakeWordResult

#: Porcupine's required input sample rate.
PORCUPINE_SAMPLE_RATE: Literal[16_000] = 16_000


class PorcupineWakeWord(WakeWord):
    """Adapter around a :mod:`pvporcupine` ``Porcupine`` instance.

    Unlike :class:`OpenWakeWord`, this wraps a third-party engine rather
    than ONNX sessions — so the usual ``model_files`` constructor pattern
    doesn't apply. Construct via :meth:`create` instead, passing your
    Picovoice access key and chosen built-in keywords.
    """

    def __init__(self, porcupine: Any, keywords: tuple[str, ...]) -> None:  # noqa: ANN401
        """Wrap an already-constructed Porcupine engine. Prefer :meth:`create`."""
        self._engine = porcupine
        self._wake_words = tuple(keywords)
        self._frame_length: int = int(porcupine.frame_length)
        self._sample_rate: int = int(porcupine.sample_rate)
        if self._sample_rate != PORCUPINE_SAMPLE_RATE:
            msg = f"Porcupine engine reports sample_rate={self._sample_rate}, expected {PORCUPINE_SAMPLE_RATE}"
            raise ValueError(msg)

    @classmethod
    def create(
        cls,
        *,
        access_key: str,
        keywords: list[str] | tuple[str, ...] | None = None,
        keyword_paths: list[str] | None = None,
        sensitivities: list[float] | None = None,
        model_path: str | None = None,
        library_path: str | None = None,
    ) -> PorcupineWakeWord:
        """Build a Porcupine engine via ``pvporcupine.create()`` and wrap it.

        Forwards all kwargs to ``pvporcupine.create`` unchanged. Either
        ``keywords`` (Picovoice's built-in set) or ``keyword_paths`` (custom
        ``.ppn`` files) must be provided.
        """
        try:
            import pvporcupine  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover — exercised when extra not installed
            msg = (
                "pvporcupine is not installed. Install with ``pip install pvporcupine`` "
                "(requires a Picovoice access key from https://console.picovoice.ai/)."
            )
            raise ImportError(msg) from exc

        if keywords is None and keyword_paths is None:
            msg = "PorcupineWakeWord.create: provide either ``keywords`` or ``keyword_paths``."
            raise ValueError(msg)

        keyword_list = list(keywords) if keywords is not None else []
        engine = pvporcupine.create(
            access_key=access_key,
            keywords=keyword_list or None,
            keyword_paths=keyword_paths,
            sensitivities=sensitivities,
            model_path=model_path,
            library_path=library_path,
        )

        # Build the canonical name list — keyword filenames when paths were
        # used, otherwise the built-in keyword strings.
        if keyword_paths:
            from pathlib import Path  # noqa: PLC0415

            names = tuple(Path(p).stem for p in keyword_paths)
        else:
            names = tuple(keyword_list)
        return cls(engine, names)

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        # Porcupine doesn't use ONNX Runtime — irrelevant for ``onnx_asr``'s
        # provider machinery, but the protocol requires the method.
        return []

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        # Porcupine ships its model with the package; nothing for the
        # resolver to download. Returning an empty mapping signals "no
        # files needed", which the resolver/loader will respect (it just
        # won't download anything — users must construct via ``create()``).
        _ = quantization
        return {}

    @staticmethod
    def _get_sample_rate() -> Literal[16_000]:
        return PORCUPINE_SAMPLE_RATE

    @property
    def wake_words(self) -> tuple[str, ...]:
        """Names of the keywords this detector is configured to recognize."""
        return self._wake_words

    @property
    def frame_length(self) -> int:
        """Native chunk size Porcupine expects on each :meth:`detect` call."""
        return self._frame_length

    def _process_frame(self, frame_int16: npt.NDArray[np.int16]) -> int:
        """Run a single Porcupine frame; returns the detected keyword index or -1."""
        return int(self._engine.process(frame_int16))

    def detect_batch(
        self,
        waveforms: npt.NDArray[np.float32],
        waveforms_len: npt.NDArray[np.int64],
        *,
        threshold: float = 0.5,  # Porcupine has internal sensitivity — kwarg ignored.
    ) -> Iterator[WakeWordResult]:
        """Run Porcupine over each clip; yield one :class:`WakeWordResult` per clip.

        Porcupine outputs a hard detection (keyword index or ``-1``), not a
        probability — so the ``scores`` map carries 1.0 for the detected
        keyword and 0.0 for the rest. ``threshold`` is accepted for protocol
        compatibility but ignored (Porcupine has its own per-keyword sensitivity).
        """
        _ = threshold
        for i in range(waveforms.shape[0]):
            length = int(waveforms_len[i])
            clip = waveforms[i, :length]
            yield self._detect_single(clip)

    def _detect_single(self, waveform: npt.NDArray[np.float32]) -> WakeWordResult:
        # Convert to int16 PCM at Porcupine's frame length.
        pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16)
        frame_length = self._frame_length
        detected_index = -1
        # Walk the clip in fixed-length frames; stop on first detection
        # (matches RealtimeSTT's behavior at audio_recorder.py:1600).
        for start in range(0, pcm.shape[0] - frame_length + 1, frame_length):
            frame = pcm[start : start + frame_length]
            idx = self._process_frame(frame)
            if idx >= 0:
                detected_index = idx
                break
        scores = {name: (1.0 if i == detected_index else 0.0) for i, name in enumerate(self._wake_words)}
        detected = self._wake_words[detected_index] if 0 <= detected_index < len(self._wake_words) else None
        return WakeWordResult(scores=scores, detected=detected)

    def cleanup(self) -> None:
        """Release the underlying Porcupine engine handle (idempotent)."""
        engine = getattr(self, "_engine", None)
        if engine is not None and hasattr(engine, "delete"):
            engine.delete()
            self._engine = None
