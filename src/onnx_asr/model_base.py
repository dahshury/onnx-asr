"""Shared private ABC for all model classes (ASR / VAD / SE).

Plan item #14 in ``docs/plans/01-onnx-asr-fork-strategy.md``: capture the
two infrastructure-level static methods that every concrete model exposes
to the resolver/loader (downloadable file list + ORT EP exclusions) into a
single Protocol. The runtime decode/embed/segment surface stays specific
to ``Asr`` / ``Vad`` / ``SpeakerEmbedding``.

Used as the bound for ``Resolver``'s generic type parameter so the resolver
can call ``_get_model_files`` / ``_get_excluded_providers`` on whatever
model class it's instantiated with.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class _ModelImplementation(Protocol):  # noqa: PYI046 — used as TypeVar bound in resolver.py and parent of Asr/Vad/SpeakerEmbedding
    """Shared infrastructure surface for every concrete ``models/*.py`` class.

    Required static methods:

    * ``_get_excluded_providers()`` — ONNXRuntime execution providers known
      to fail (or be unsupported) for this model class. The resolver drops
      these from the provider list before creating sessions.
    * ``_get_model_files(quantization)`` — map of logical keys to file-path
      patterns (relative to the HF repo root). The resolver downloads only
      files whose pattern matches; the model class's ``__init__`` looks
      them up by key.

    Both are static so the resolver can query them BEFORE instantiating the
    model (it needs the file list to even attempt a download).
    """

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        """ORT execution providers known to fail for this model class."""
        ...

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        """Map of logical key → HF-repo-relative file pattern for download."""
        ...
