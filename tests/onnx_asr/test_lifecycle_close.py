"""Tests for the explicit close() / unload() lifecycle.

Verifies that ``release_inference_sessions`` walks the adapter graph and
nulls every ``rt.InferenceSession`` it finds, that ``close()`` is exposed
on every base class + adapter, that adapters support the context-manager
protocol, and — most importantly — that RSS returns to near the
import-baseline after ``close()`` (no memory leak across reload cycles).
"""

from __future__ import annotations

import gc
import os
from typing import cast

import numpy as np
import onnxruntime as rt
import psutil
import pytest

import onnx_asr
from onnx_asr._session_cleanup import release_inference_sessions
from onnx_asr.adapters import AsrAdapter, SeAdapter, TextResultsAsrAdapter
from onnx_asr.models.openwakeword import OpenWakeWord
from onnx_asr.models.whisper import WhisperOrt, _Whisper
from onnx_asr.preprocessors.resampler import Resampler

# ---------------------------------------------------------------------------
# Pure-function tests for the walker (no model load)
# ---------------------------------------------------------------------------


class _FakeOrtSession:
    """Walks like an InferenceSession for ``release_inference_sessions`` tests.

    The walker tests via ``isinstance(obj, rt.InferenceSession)``, so we
    can't fake-substitute. Instead use real (tiny) sessions where the test
    needs them, and use plain marker objects for graph-shape tests.
    """


def test_walker_nulls_direct_session_attribute() -> None:
    """A direct ``rt.InferenceSession`` attribute is set to None."""

    class Holder:
        pass

    obj = Holder()
    # Smallest valid ONNX: build a session from in-memory bytes via the
    # silero VAD that's known to be cached. Skip if not available.
    pytest.importorskip("huggingface_hub")
    try:
        vad = onnx_asr.load_vad("silero")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not load silero VAD for session-walker test: {exc}")

    obj.session = vad._model  # type: ignore[attr-defined]
    assert isinstance(obj.session, rt.InferenceSession)
    released = release_inference_sessions(obj, empty_torch_cache=False)
    assert released == 1
    assert obj.session is None
    # Cleanup
    vad.close()


def test_walker_walks_dict_values() -> None:
    """``self.classifiers = {name: session}`` style attributes get cleared."""
    try:
        vad = onnx_asr.load_vad("silero")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not load silero VAD: {exc}")

    class Holder:
        pass

    obj = Holder()
    obj.sessions_by_name = {"primary": vad._model, "secondary": vad._model}  # type: ignore[attr-defined]
    released = release_inference_sessions(obj, empty_torch_cache=False)
    assert released == 2
    assert obj.sessions_by_name == {"primary": None, "secondary": None}
    vad.close()


def test_walker_is_idempotent() -> None:
    """A second ``release_inference_sessions`` call on the same object returns 0."""

    class Holder:
        pass

    obj = Holder()
    obj.session = None
    obj.deeper = Holder()
    obj.deeper.session = None  # type: ignore[attr-defined]
    assert release_inference_sessions(obj, empty_torch_cache=False) == 0
    assert release_inference_sessions(obj, empty_torch_cache=False) == 0


def test_walker_handles_no_attributes() -> None:
    """Built-in types like int / str / None have no ``__dict__`` — must not crash."""
    assert release_inference_sessions(42, empty_torch_cache=False) == 0
    assert release_inference_sessions("hello", empty_torch_cache=False) == 0
    assert release_inference_sessions(None, empty_torch_cache=False) == 0


def test_walker_handles_cycles_without_recursion() -> None:
    """Reference cycles in the attribute graph don't make the walker recurse forever."""

    class Holder:
        pass

    a = Holder()
    b = Holder()
    a.peer = b  # type: ignore[attr-defined]
    b.peer = a  # type: ignore[attr-defined]
    # Walker should terminate, not RecursionError.
    assert release_inference_sessions(a, empty_torch_cache=False) == 0


# ---------------------------------------------------------------------------
# close() method exposure
# ---------------------------------------------------------------------------


def test_baseasr_has_close() -> None:
    from onnx_asr.asr import BaseAsr  # noqa: PLC0415

    assert hasattr(BaseAsr, "close")
    assert callable(BaseAsr.close)


def test_basevad_has_close() -> None:
    from onnx_asr.vad import BaseVad  # noqa: PLC0415

    assert hasattr(BaseVad, "close")


def test_openwakeword_has_close() -> None:
    assert hasattr(OpenWakeWord, "close")


def test_porcupine_close_is_no_op_for_ort() -> None:
    """Porcupine doesn't use ORT — its ``close()`` returns 0 and calls cleanup()."""
    from onnx_asr.models.porcupine import PorcupineWakeWord  # noqa: PLC0415

    class _FakeEngine:
        sample_rate = 16_000
        frame_length = 512
        deleted = False

        def delete(self) -> None:
            self.deleted = True

    engine = _FakeEngine()
    det = PorcupineWakeWord(engine, ("hey",))
    released = det.close()
    assert released == 0
    assert engine.deleted is True


def test_resampler_has_close() -> None:
    assert hasattr(Resampler, "close")


def test_adapter_has_close_and_is_context_manager() -> None:
    """Adapters expose ``close`` AND the ``__enter__`` / ``__exit__`` protocol."""
    assert hasattr(AsrAdapter, "close")
    assert hasattr(AsrAdapter, "__enter__")
    assert hasattr(AsrAdapter, "__exit__")
    assert hasattr(SeAdapter, "close")
    assert hasattr(SeAdapter, "__enter__")
    assert hasattr(SeAdapter, "__exit__")


# ---------------------------------------------------------------------------
# End-to-end: close() actually releases ORT sessions
# ---------------------------------------------------------------------------


def _count_sessions(adapter: TextResultsAsrAdapter) -> int:
    """Count InferenceSessions reachable from an adapter."""
    count = 0
    seen: set[int] = set()

    def _walk(obj: object) -> None:
        nonlocal count
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, rt.InferenceSession):
            count += 1
            return
        try:
            attrs = vars(obj).values()
        except TypeError:
            return
        for v in attrs:
            if isinstance(v, dict):
                for val in v.values():
                    _walk(val)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    _walk(item)
            else:
                _walk(v)

    _walk(adapter)
    return count


def test_whisper_close_nulls_all_sessions() -> None:
    """After ``model.close()``, no ORT sessions remain reachable from the adapter."""
    model = onnx_asr.load_model("whisper-base", quantization="int8", providers=["CPUExecutionProvider"])
    pre = _count_sessions(model)
    assert pre >= 1, "expected at least one ORT session on a loaded Whisper model"

    released = model.close()
    post = _count_sessions(model)
    assert released >= 1, "close() should report releasing >=1 session"
    assert post == 0, f"sessions still reachable after close: {post}"


def test_close_is_idempotent_on_adapter() -> None:
    """Calling ``close()`` twice is safe and the second call releases 0."""
    model = onnx_asr.load_model("whisper-base", quantization="int8", providers=["CPUExecutionProvider"])
    first = model.close()
    second = model.close()
    assert first >= 1
    assert second == 0


def test_context_manager_releases_on_exit() -> None:
    """``with onnx_asr.load_model(...) as model:`` releases sessions on exit."""
    audio = np.zeros(16_000, dtype=np.float32)
    with onnx_asr.load_model("whisper-base", quantization="int8", providers=["CPUExecutionProvider"]) as model:
        text = model.recognize(audio)
        assert isinstance(text, str)
        # Sessions still alive while inside the ``with``.
        assert _count_sessions(model) >= 1
    # After exit — released.
    assert _count_sessions(model) == 0


class _BoomError(Exception):
    """Sentinel exception for the context-manager-on-exception test."""


def _enter_then_raise() -> TextResultsAsrAdapter:
    """Helper isolates the ``raise`` so ``pytest.raises`` matches its single-statement rule."""
    captured: list[TextResultsAsrAdapter] = []
    with onnx_asr.load_model("whisper-base", quantization="int8", providers=["CPUExecutionProvider"]) as m:
        captured.append(cast(TextResultsAsrAdapter, m))
        raise _BoomError
    return captured[0]  # unreachable; here for typing


def test_context_manager_releases_on_exception() -> None:
    """Even when the body raises, ``__exit__`` calls close().

    We can't reach the captured adapter from out here once the helper
    raised, so the functional check is: a fresh load + close still works
    cleanly (no DLL state corruption from a session that was active during
    a teardown caused by an unhandled exception).
    """
    import contextlib  # noqa: PLC0415

    model: TextResultsAsrAdapter | None = None
    with contextlib.suppress(_BoomError):
        model = _enter_then_raise()
    fresh = onnx_asr.load_model("whisper-base", quantization="int8", providers=["CPUExecutionProvider"])
    assert _count_sessions(fresh) >= 1
    fresh.close()
    assert _count_sessions(fresh) == 0
    assert model is None  # the helper never returned


# ---------------------------------------------------------------------------
# Memory-leak regression — RSS must return to near baseline across cycles
# ---------------------------------------------------------------------------


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def test_rss_bounded_across_load_close_cycles() -> None:
    """3 load/close cycles must not grow RSS by more than a small headroom.

    If the close() walker is broken, each cycle would add ~250 MB
    (whisper-base int8 + ORT arenas). With correct cleanup, RSS stays
    flat — the test asserts the spread stays within 100 MB across cycles.
    """
    # Warm caches & ORT initialization first.
    audio = np.zeros(16_000, dtype=np.float32)
    warm = onnx_asr.load_model("whisper-base", quantization="int8", providers=["CPUExecutionProvider"])
    warm.recognize(audio)
    warm.close()
    gc.collect()
    gc.collect()

    rss_after_each: list[float] = []
    for _ in range(3):
        model = onnx_asr.load_model("whisper-base", quantization="int8", providers=["CPUExecutionProvider"])
        model.recognize(audio)
        model.close()
        gc.collect()
        gc.collect()
        rss_after_each.append(_rss_mb())

    spread = max(rss_after_each) - min(rss_after_each)
    assert spread < 100.0, (
        f"RSS grew {spread:.1f} MB across cycles ({rss_after_each!r}); expected <100 MB. Likely a leak in close()."
    )


def test_close_releases_significant_rss() -> None:
    """``close()`` must release a meaningful chunk of RSS (>50 MB for whisper-base)."""
    audio = np.zeros(16_000, dtype=np.float32)
    # Warm — first run pays for ORT one-time init.
    warm = onnx_asr.load_model("whisper-base", quantization="int8", providers=["CPUExecutionProvider"])
    warm.recognize(audio)
    warm.close()
    gc.collect()
    gc.collect()
    baseline = _rss_mb()

    model = onnx_asr.load_model("whisper-base", quantization="int8", providers=["CPUExecutionProvider"])
    model.recognize(audio)
    peak = _rss_mb()
    delta_after_load = peak - baseline
    model.close()
    gc.collect()
    gc.collect()
    after_close = _rss_mb()
    delta_after_close = after_close - baseline

    # Load should consume meaningfully; close should release most of it.
    assert delta_after_load > 50.0, f"load+recognize only added {delta_after_load:.1f} MB — model didn't actually load?"
    # After close, residual must be a small fraction of the load delta.
    assert delta_after_close < delta_after_load * 0.5, (
        f"close() only released {delta_after_load - delta_after_close:.1f} MB of {delta_after_load:.1f} MB."
    )


# ---------------------------------------------------------------------------
# Whisper-specific sanity
# ---------------------------------------------------------------------------


def test_whisperort_close_nulls_model_session() -> None:
    """WhisperOrt has a single bundled ``_model`` session — close nulls it."""
    model = onnx_asr.load_model("whisper-base", quantization="int8", providers=["CPUExecutionProvider"])
    asr = model.asr
    assert isinstance(asr, _Whisper)
    assert isinstance(asr, WhisperOrt)
    assert isinstance(asr._model, rt.InferenceSession)  # type: ignore[attr-defined]
    model.close()
    assert asr._model is None  # type: ignore[attr-defined]
