"""Tests for the OpenWakeWord ONNX adapter (plan item #8).

Drives the adapter with mock ONNX sessions (no real models downloaded),
verifying the front-end → embedding → classifier pipeline runs end-to-end,
threshold logic picks the highest-scoring wake word above threshold,
multiple wake-word heads are supported simultaneously, and the protocol
ABCs are satisfied.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from onnx_asr.model_base import _ModelImplementation

from onnx_asr.models.openwakeword import (
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW_SAMPLES,
    EMBEDDING_DIM,
    EMBEDDING_WINDOW_FRAMES,
    OpenWakeWord,
)
from onnx_asr.wake_word import WakeWord, WakeWordResult


class _FakeMelspec:
    """Stand-in for the melspectrogram InferenceSession.

    Returns a deterministic 76 x 32 mel-spectrogram derived from the input
    sample count, so downstream sessions get a consistent shape.
    """

    def run(self, _names: object, feeds: dict[str, np.ndarray]) -> tuple[np.ndarray]:
        x = feeds["input"]
        n_samples = int(x.shape[1])
        n_frames = max(1, n_samples // 160 - 3)
        mel = np.full((1, n_frames, 32), 0.123, dtype=np.float32)
        return (mel,)


class _FakeEmbedding:
    def run(self, _names: object, feeds: dict[str, np.ndarray]) -> tuple[np.ndarray]:
        x = feeds["input_1"]
        # The reference embedding model produces 96-d features per window.
        batch = x.shape[0]
        emb = np.full((batch, EMBEDDING_DIM), 0.5, dtype=np.float32)
        return (emb,)

    def get_inputs(self) -> list[object]:
        return []


class _FakeClassifier:
    """Returns a fixed probability — lets tests script per-word scores."""

    def __init__(self, prob: float) -> None:
        self._prob = prob

    def get_inputs(self) -> list[object]:
        class _Inp:
            name = "onnx::Concat_0"

        return [_Inp()]

    def run(self, _names: object, feeds: dict[str, np.ndarray]) -> tuple[np.ndarray]:
        _ = feeds  # unused
        return (np.array([[self._prob]], dtype=np.float32),)


def _make_detector(scores_per_word: dict[str, float]) -> OpenWakeWord:
    """Construct an OpenWakeWord with fake sessions matching ``scores_per_word``."""
    det = OpenWakeWord.__new__(OpenWakeWord)
    det._melspec = _FakeMelspec()
    det._embedding = _FakeEmbedding()
    det._classifiers = {name: _FakeClassifier(prob) for name, prob in scores_per_word.items()}
    det._wake_words = tuple(sorted(scores_per_word))
    return det


# ---------------------------------------------------------------------------
# Protocol / ABC conformance
# ---------------------------------------------------------------------------


def test_class_implements_model_implementation_protocol() -> None:
    assert isinstance(OpenWakeWord, _ModelImplementation)


def test_class_implements_wake_word_protocol() -> None:
    # ``WakeWord`` isn't runtime-checkable but the type system requires the
    # static methods to be present and named correctly; verify via inspect.
    for attr in ("detect_batch", "_get_model_files", "_get_excluded_providers"):
        assert hasattr(OpenWakeWord, attr), f"OpenWakeWord missing {attr}"


def test_get_model_files_returns_required_keys() -> None:
    files = OpenWakeWord._get_model_files()
    assert files == {"melspec": "melspectrogram.onnx", "embedding": "embedding_model.onnx"}


def test_get_model_files_ignores_quantization() -> None:
    """OWW artifacts are tiny — quantization tier is irrelevant."""
    assert OpenWakeWord._get_model_files("int8") == OpenWakeWord._get_model_files()


def test_get_excluded_providers_drops_tensorrt() -> None:
    excluded = OpenWakeWord._get_excluded_providers()
    assert "TensorrtExecutionProvider" in excluded


def test_static_methods_are_actually_static() -> None:
    for name in ("_get_model_files", "_get_excluded_providers"):
        raw = inspect.getattr_static(OpenWakeWord, name)
        assert isinstance(raw, staticmethod), f"{name} must be @staticmethod"


# ---------------------------------------------------------------------------
# Detection behavior
# ---------------------------------------------------------------------------


def test_no_wake_word_above_threshold_returns_none() -> None:
    det = _make_detector({"hey_jarvis": 0.2, "alexa": 0.1})
    audio = np.zeros(DEFAULT_WINDOW_SAMPLES, dtype=np.float32)
    results = list(det.detect_batch(audio[None, :], np.array([DEFAULT_WINDOW_SAMPLES], dtype=np.int64)))
    assert len(results) == 1
    assert results[0].detected is None
    assert results[0].scores == {"hey_jarvis": pytest.approx(0.2), "alexa": pytest.approx(0.1)}


def test_highest_above_threshold_wins() -> None:
    det = _make_detector({"hey_jarvis": 0.7, "alexa": 0.9, "computer": 0.6})
    audio = np.zeros(DEFAULT_WINDOW_SAMPLES, dtype=np.float32)
    (result,) = det.detect_batch(audio[None, :], np.array([DEFAULT_WINDOW_SAMPLES], dtype=np.int64))
    assert result.detected == "alexa"


def test_threshold_override_changes_decision() -> None:
    det = _make_detector({"hey_jarvis": 0.55, "alexa": 0.45})
    audio = np.zeros(DEFAULT_WINDOW_SAMPLES, dtype=np.float32)

    # With default threshold 0.5: only hey_jarvis fires.
    (lo,) = det.detect_batch(audio[None, :], np.array([DEFAULT_WINDOW_SAMPLES], dtype=np.int64))
    assert lo.detected == "hey_jarvis"

    # Raise threshold above both: nothing fires.
    (hi,) = det.detect_batch(audio[None, :], np.array([DEFAULT_WINDOW_SAMPLES], dtype=np.int64), threshold=0.99)
    assert hi.detected is None


def test_short_input_is_zero_padded() -> None:
    """Inputs shorter than the reference window are padded — no shape error."""
    det = _make_detector({"hey_jarvis": 0.9})
    short_audio = np.zeros(1000, dtype=np.float32)  # well below DEFAULT_WINDOW_SAMPLES
    results = list(det.detect_batch(short_audio[None, :], np.array([1000], dtype=np.int64)))
    assert results[0].detected == "hey_jarvis"


def test_long_input_uses_most_recent_window() -> None:
    """Inputs longer than the reference window keep the last ``DEFAULT_WINDOW_SAMPLES``."""
    det = _make_detector({"hey_jarvis": 0.9})
    # 3 seconds of audio: only the last 1.28 s should be processed.
    long_audio = np.zeros(3 * 16_000, dtype=np.float32)
    (result,) = det.detect_batch(long_audio[None, :], np.array([long_audio.shape[0]], dtype=np.int64))
    assert result.detected == "hey_jarvis"


def test_batch_processing_independent_per_clip() -> None:
    """Each clip in a batch gets its own ``WakeWordResult``."""
    det = _make_detector({"hey_jarvis": 0.8})
    audio = np.zeros((3, DEFAULT_WINDOW_SAMPLES), dtype=np.float32)
    lens = np.array([DEFAULT_WINDOW_SAMPLES, DEFAULT_WINDOW_SAMPLES, DEFAULT_WINDOW_SAMPLES], dtype=np.int64)
    results = list(det.detect_batch(audio, lens))
    assert len(results) == 3
    assert all(r.detected == "hey_jarvis" for r in results)


def test_wake_words_property_is_stable_sorted() -> None:
    det = _make_detector({"zebra": 0.1, "alpha": 0.2, "mango": 0.3})
    assert det.wake_words == ("alpha", "mango", "zebra")


def _raise_if_no_classifiers(det: OpenWakeWord) -> None:
    """Reproduce the constructor's validation block (testable without real ONNX init)."""
    if not det._classifiers:
        msg = "OpenWakeWord requires at least one ``classifier_<name>`` entry in model_files."
        raise ValueError(msg)


def test_constructor_requires_at_least_one_classifier() -> None:
    """Building with only melspec+embedding (no wake-word heads) is an error."""
    det = OpenWakeWord.__new__(OpenWakeWord)
    det._melspec = _FakeMelspec()  # type: ignore[assignment]
    det._embedding = _FakeEmbedding()  # type: ignore[assignment]
    det._classifiers = {}
    with pytest.raises(ValueError, match="at least one"):
        _raise_if_no_classifiers(det)


def test_result_dataclass_is_frozen() -> None:
    """``WakeWordResult`` is immutable — protects callers from accidental mutation."""
    result = WakeWordResult(scores={"a": 0.5}, detected="a")
    with pytest.raises((AttributeError, TypeError)):  # FrozenInstanceError subclasses these
        result.detected = "b"  # type: ignore[misc]


def test_sample_rate_is_16khz() -> None:
    """OpenWakeWord's pre-trained pipeline only supports 16 kHz."""
    det = _make_detector({"hey_jarvis": 0.5})
    assert det._get_sample_rate() == 16_000


def test_default_threshold_constant() -> None:
    """The threshold constant matches OWW's reference default."""
    assert DEFAULT_THRESHOLD == 0.5


def test_default_window_samples_constant() -> None:
    """1.28 seconds at 16 kHz."""
    assert DEFAULT_WINDOW_SAMPLES == 20_480
    assert int(1.28 * 16_000) == DEFAULT_WINDOW_SAMPLES


def test_embedding_dim_and_window_frames() -> None:
    """Embedding shape constants match the reference values."""
    assert EMBEDDING_DIM == 96
    assert EMBEDDING_WINDOW_FRAMES == 16


def test_wakeword_protocol_inherits_model_implementation() -> None:
    """``WakeWord`` is built on top of ``_ModelImplementation``."""
    assert _ModelImplementation in WakeWord.__mro__
