"""Tests for ``load_wake_word`` (plan item #13).

Verifies the resolver / Manager / public ``load_wake_word`` plumbing
without forcing real ONNX downloads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import onnx_asr
from onnx_asr.loader import create_wake_word_resolver
from onnx_asr.models.openwakeword import OpenWakeWord
from onnx_asr.utils import (
    InvalidModelTypeInConfigError,
    ModelNotSupportedError,
    NoModelNameOrPathSpecifiedError,
)


def test_load_wake_word_exported_from_package() -> None:
    """``load_wake_word`` is re-exported at the package root alongside ``load_vad``."""
    assert hasattr(onnx_asr, "load_wake_word")
    assert callable(onnx_asr.load_wake_word)


def test_wake_word_result_exported() -> None:
    """``WakeWordResult`` is re-exported for type-checking in user code."""
    assert hasattr(onnx_asr, "WakeWordResult")
    assert hasattr(onnx_asr, "WakeWord")


def test_resolver_routes_openwakeword_type_to_openwakeword(tmp_path: Path) -> None:
    """``model="openwakeword"`` with a local dir routes to the :class:`OpenWakeWord` class.

    No HF mirror is registered yet (no ``model_repos["openwakeword"]`` entry —
    OpenWakeWord upstream doesn't publish ONNX on the Hub), so callers must
    supply a local directory of model files when using the short alias.
    """
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    resolver = create_wake_word_resolver(model="openwakeword", local_dir=tmp_path, offline=True)
    assert resolver.model_type is OpenWakeWord


def test_resolver_with_config_json_routes_to_openwakeword(tmp_path: Path) -> None:
    """When the config.json reports model_type=openwakeword, route via config-based dispatch."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "openwakeword"}), encoding="utf-8")
    resolver = create_wake_word_resolver(model="some-user/some-oww-repo", local_dir=tmp_path, offline=True)
    assert resolver.model_type is OpenWakeWord


def test_resolver_rejects_unknown_config_model_type(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "not-a-wake-word"}), encoding="utf-8")
    with pytest.raises(InvalidModelTypeInConfigError):
        create_wake_word_resolver(model="some-user/foo", local_dir=tmp_path, offline=True)


def test_resolver_rejects_unsupported_alias() -> None:
    """Unknown short aliases that aren't HF repo paths and aren't in the type map are rejected."""
    with pytest.raises(ModelNotSupportedError):
        create_wake_word_resolver(model="picovoice-porcupine", local_dir=None, offline=False)


def test_resolver_requires_name_or_path() -> None:
    with pytest.raises(NoModelNameOrPathSpecifiedError):
        create_wake_word_resolver(model=None, local_dir=None, offline=False)


def test_wake_word_names_and_type_names_literals() -> None:
    """Public Literal types document the supported aliases."""
    from typing import get_args  # noqa: PLC0415

    from onnx_asr.loader import WakeWordNames, WakeWordTypeNames  # noqa: PLC0415

    assert "openwakeword" in get_args(WakeWordNames)
    assert "openwakeword" in get_args(WakeWordTypeNames)
