"""Helpers for releasing ORT InferenceSessions held by model objects.

The pattern: when a long-running server (or a hot-swap workflow) needs to
release the memory held by a loaded model, dropping the Python reference
is usually sufficient — Python's ref-count machinery will fire ORT's C++
destructor, which releases the model weights and allocator arenas. But:

* Reference cycles (e.g. ``WhisperStream._asr`` → ``_Whisper`` → ...) can
  delay release until the next ``gc.collect()``.
* On Windows, ORT C++ destructors firing *during* interpreter shutdown
  race against DLL unload — sometimes producing a noisy traceback on
  process exit even though everything is logically freed.
* PyTorch's CUDA pool stays allocated even after ORT frees its share.

This module exposes :func:`release_inference_sessions`, which walks an
object's attribute graph and explicitly nulls every ``rt.InferenceSession``
it finds, then forces two GC passes. Subclasses of :class:`BaseAsr` /
:class:`BaseVad` / :class:`SpeakerEmbedding` / :class:`WakeWord` inherit a
:meth:`close` method that invokes it. Adapters expose :meth:`close` plus
the context-manager protocol.

The pattern is borrowed from WinSTT-old's ``_unload_current_model`` (in
``examples/WinSTT-old/src/infrastructure/transcription/onnx_transcription_service.py``)
— battle-tested for hot-swap model switching in a desktop STT app.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def _is_inference_session(obj: object) -> bool:
    """Return True if ``obj`` is a ``rt.InferenceSession`` (lazy-import-safe)."""
    try:
        import onnxruntime as rt  # noqa: PLC0415
    except ImportError:
        return False
    return isinstance(obj, rt.InferenceSession)


def release_inference_sessions(  # noqa: C901 — single-purpose walker; splitting wouldn't aid readability
    root: object,
    *,
    empty_torch_cache: bool = True,
) -> int:
    """Walk ``root``'s attribute graph and null every ``rt.InferenceSession``.

    Args:
        root: The object whose ORT sessions to release (e.g. a loaded
              ``Asr`` / ``Vad`` / ``SpeakerEmbedding`` / ``WakeWord`` or an
              ``AsrAdapter`` / ``SeAdapter`` wrapper).
        empty_torch_cache: If True (default) and ``torch`` is importable,
              also call ``torch.cuda.empty_cache()`` so the CUDA allocator
              pool releases its share. Harmless no-op on CPU-only setups.

    Returns:
        Number of sessions nulled.

    Notes:
        * Walks via ``vars()`` — only attributes stored in ``__dict__`` are
          inspected. Sessions stashed in slots or weakref containers won't
          be touched (rare in practice).
        * Idempotent: calling twice on the same root is a no-op the second
          time (everything's already None).
        * Follows ``dict`` / ``list`` / ``tuple`` / ``set`` containers one
          level deep so e.g. ``self._classifiers = {"hey_jarvis": <session>}``
          (used by ``OpenWakeWord``) gets cleaned out.

    """
    visited: set[int] = set()
    released = 0

    def _release_value(container: Any, key: Any, value: object) -> bool:  # noqa: ANN401
        nonlocal released
        if _is_inference_session(value):
            try:
                if isinstance(container, (dict, list)):
                    container[key] = None
                else:
                    setattr(container, key, None)
                released += 1
            except (AttributeError, TypeError):
                # read-only attribute or non-mutable container — skip.
                pass
            return True
        return False

    def _walk(obj: object) -> None:  # noqa: C901 — inner walker; same readability argument as outer

        if obj is None:
            return
        oid = id(obj)
        if oid in visited:
            return
        visited.add(oid)

        # Direct attributes on object.
        try:
            attrs = list(vars(obj).items())
        except TypeError:
            attrs = []
        for name, value in attrs:
            if value is None:
                continue
            if _release_value(obj, name, value):
                continue
            if isinstance(value, Mapping):
                for k in list(value.keys()):
                    v = value[k]
                    if v is None:
                        continue
                    if _release_value(value, k, v):
                        continue
                    _walk(v)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if item is None:
                        continue
                    if _release_value(value, i, item):
                        continue
                    _walk(item)
            elif isinstance(value, (tuple, set, frozenset)):
                # Tuples / frozensets are immutable — we can't null entries
                # in place, but we still recurse so transitive sessions on
                # contained objects get released. The references in the
                # container itself stay, but they point at now-empty objects.
                for item in value:
                    _walk(item)
            else:
                _walk(value)

    _walk(root)

    if empty_torch_cache:
        try:
            import torch  # type: ignore[import-not-found]  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    gc.collect()
    gc.collect()
    return released
