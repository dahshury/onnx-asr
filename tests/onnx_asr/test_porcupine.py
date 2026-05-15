"""Tests for the optional Porcupine wake-word adapter (plan item #16).

Drives the adapter with a hand-rolled fake ``Porcupine`` engine — Porcupine
itself is a proprietary SDK, so we don't import ``pvporcupine`` in the test
suite. The fake matches the public surface (``process``, ``frame_length``,
``sample_rate``, ``delete``) the adapter touches.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from onnx_asr.model_base import _ModelImplementation
from onnx_asr.models.porcupine import PORCUPINE_SAMPLE_RATE, PorcupineWakeWord
from onnx_asr.wake_word import WakeWordResult


class _FakePorcupine:
    """Stand-in for ``pvporcupine.Porcupine``.

    Returns a scripted detection index for each ``process`` call so tests can
    exercise both "detected" and "no detection" code paths deterministically.
    """

    sample_rate = PORCUPINE_SAMPLE_RATE
    frame_length = 512

    def __init__(self, scripted: list[int] | None = None) -> None:
        self._scripted = list(scripted or [])
        self.deleted = False
        self.calls = 0

    def process(self, frame: object) -> int:
        _ = frame
        self.calls += 1
        if self._scripted:
            return self._scripted.pop(0)
        return -1

    def delete(self) -> None:
        self.deleted = True


# ---------------------------------------------------------------------------
# Protocol / ABC conformance
# ---------------------------------------------------------------------------


def test_class_implements_model_implementation_protocol() -> None:
    assert isinstance(PorcupineWakeWord, _ModelImplementation)


def test_get_model_files_is_empty() -> None:
    """Porcupine ships its model files with the package — nothing for the resolver to download."""
    assert PorcupineWakeWord._get_model_files() == {}


def test_get_model_files_ignores_quantization() -> None:
    assert PorcupineWakeWord._get_model_files("int8") == {}


def test_get_excluded_providers_is_empty() -> None:
    """Porcupine doesn't use ORT, so provider exclusions don't apply."""
    assert PorcupineWakeWord._get_excluded_providers() == []


def test_static_methods_are_actually_static() -> None:
    for name in ("_get_model_files", "_get_excluded_providers", "_get_sample_rate"):
        raw = inspect.getattr_static(PorcupineWakeWord, name)
        assert isinstance(raw, staticmethod), f"{name} must be @staticmethod"


def test_sample_rate_is_16khz() -> None:
    """Porcupine only supports 16 kHz."""
    assert PorcupineWakeWord._get_sample_rate() == 16_000
    assert PORCUPINE_SAMPLE_RATE == 16_000


# ---------------------------------------------------------------------------
# Detection behavior
# ---------------------------------------------------------------------------


def test_detect_returns_none_when_no_detection() -> None:
    engine = _FakePorcupine(scripted=[-1, -1, -1])
    det = PorcupineWakeWord(engine, ("picovoice",))
    audio = np.zeros(3 * engine.frame_length, dtype=np.float32)
    (result,) = det.detect_batch(audio[None, :], np.array([audio.shape[0]], dtype=np.int64))
    assert result.detected is None
    assert result.scores == {"picovoice": 0.0}


def test_detect_returns_keyword_on_match() -> None:
    """Engine returning index 0 maps to the first keyword name."""
    engine = _FakePorcupine(scripted=[-1, -1, 0])  # match on 3rd frame
    det = PorcupineWakeWord(engine, ("picovoice", "bumblebee"))
    audio = np.zeros(3 * engine.frame_length, dtype=np.float32)
    (result,) = det.detect_batch(audio[None, :], np.array([audio.shape[0]], dtype=np.int64))
    assert result.detected == "picovoice"
    assert result.scores == {"picovoice": 1.0, "bumblebee": 0.0}


def test_detect_stops_at_first_match() -> None:
    """Once Porcupine reports a hit we don't keep scanning the rest of the clip."""
    engine = _FakePorcupine(scripted=[1, 0, 0])  # would match on first frame
    det = PorcupineWakeWord(engine, ("alpha", "beta"))
    audio = np.zeros(3 * engine.frame_length, dtype=np.float32)
    (result,) = det.detect_batch(audio[None, :], np.array([audio.shape[0]], dtype=np.int64))
    assert result.detected == "beta"  # index 1 → second keyword
    assert engine.calls == 1  # stopped immediately after the hit


def test_threshold_kwarg_is_accepted_but_ignored() -> None:
    """Porcupine has internal sensitivities — the ``threshold`` kwarg is accepted for protocol parity."""
    engine = _FakePorcupine(scripted=[0])
    det = PorcupineWakeWord(engine, ("picovoice",))
    audio = np.zeros(engine.frame_length, dtype=np.float32)
    (result,) = det.detect_batch(audio[None, :], np.array([audio.shape[0]], dtype=np.int64), threshold=0.99)
    assert result.detected == "picovoice"


def test_batch_processing_independent_per_clip() -> None:
    engine = _FakePorcupine(scripted=[0, -1])  # clip 0 matches, clip 1 doesn't
    det = PorcupineWakeWord(engine, ("picovoice",))
    audio = np.zeros((2, engine.frame_length), dtype=np.float32)
    lens = np.array([engine.frame_length, engine.frame_length], dtype=np.int64)
    results = list(det.detect_batch(audio, lens))
    assert results[0].detected == "picovoice"
    assert results[1].detected is None


def test_short_clip_below_frame_length_yields_no_detection() -> None:
    """Clips shorter than ``frame_length`` skip processing entirely (no partial frame)."""
    engine = _FakePorcupine(scripted=[0])
    det = PorcupineWakeWord(engine, ("picovoice",))
    short_audio = np.zeros(engine.frame_length // 2, dtype=np.float32)
    (result,) = det.detect_batch(short_audio[None, :], np.array([short_audio.shape[0]], dtype=np.int64))
    assert result.detected is None
    assert engine.calls == 0


def test_wake_words_property() -> None:
    engine = _FakePorcupine()
    det = PorcupineWakeWord(engine, ("hey-siri", "ok-google"))
    assert det.wake_words == ("hey-siri", "ok-google")


def test_frame_length_property() -> None:
    engine = _FakePorcupine()
    engine.frame_length = 256
    det = PorcupineWakeWord(engine, ("x",))
    assert det.frame_length == 256


def test_cleanup_calls_engine_delete() -> None:
    engine = _FakePorcupine()
    det = PorcupineWakeWord(engine, ("x",))
    det.cleanup()
    assert engine.deleted is True
    # Idempotent — second call doesn't blow up.
    det.cleanup()


def test_rejects_engine_with_wrong_sample_rate() -> None:
    bad_engine = _FakePorcupine()
    bad_engine.sample_rate = 8_000
    with pytest.raises(ValueError, match="sample_rate"):
        PorcupineWakeWord(bad_engine, ("x",))


def test_create_without_keywords_or_paths_raises() -> None:
    """At least one of keywords / keyword_paths must be supplied."""
    pytest.importorskip("pvporcupine", reason="pvporcupine not installed — skipping engine wiring test")
    with pytest.raises(ValueError, match="keywords"):
        PorcupineWakeWord.create(access_key="x")


def test_pvporcupine_import_error_message_is_helpful(monkeypatch: pytest.MonkeyPatch) -> None:
    """If pvporcupine isn't importable, ``create`` raises a clear message."""
    import sys  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "pvporcupine", None)
    with pytest.raises((ImportError, TypeError)):
        # Either pvporcupine import fails (ImportError) or sys.modules sentinel
        # raises TypeError — both are acceptable signals to the user.
        PorcupineWakeWord.create(access_key="x", keywords=["picovoice"])


def test_result_dataclass_carries_scores() -> None:
    engine = _FakePorcupine(scripted=[1])
    det = PorcupineWakeWord(engine, ("a", "b", "c"))
    audio = np.zeros(engine.frame_length, dtype=np.float32)
    (result,) = det.detect_batch(audio[None, :], np.array([audio.shape[0]], dtype=np.int64))
    assert isinstance(result, WakeWordResult)
    assert result.scores == {"a": 0.0, "b": 1.0, "c": 0.0}
    assert result.detected == "b"
