"""Routing tests for HF Optimum-style Whisper variants whose ``config.json``
``model_type`` is something other than ``"whisper"``.

When a caller passes a full HF repo path (e.g.
``onnx-community/lite-whisper-large-v3-acc-ONNX``), the resolver falls through
to reading the repo's ``config.json`` and looking up its ``model_type`` field
in the loader's ``model_types`` map. Variants with custom architectures but
the same I/O contract must be aliased to ``WhisperHf`` so they load.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.loader import create_asr_resolver
from onnx_asr.models.whisper import WhisperHf


def _write_minimal_local_dir(tmp_path: Path, model_type: str) -> Path:
    """Write a ``config.json`` with the given ``model_type`` to a temp dir."""
    config = {"model_type": model_type}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def test_lite_whisper_config_routes_to_whisper_hf(tmp_path: Path) -> None:
    """``model_type=lite-whisper`` in config.json routes to WhisperHf."""
    local_dir = _write_minimal_local_dir(tmp_path, "lite-whisper")
    resolver = create_asr_resolver(model="any-org/lite-whisper-x", local_dir=local_dir, offline=True)
    assert resolver.model_type is WhisperHf


def test_distil_whisper_config_routes_to_whisper_hf(tmp_path: Path) -> None:
    """``model_type=distil-whisper`` in config.json routes to WhisperHf."""
    local_dir = _write_minimal_local_dir(tmp_path, "distil-whisper")
    resolver = create_asr_resolver(model="any-org/distil-whisper-x", local_dir=local_dir, offline=True)
    assert resolver.model_type is WhisperHf


def test_plain_whisper_config_still_routes_to_whisper_hf(tmp_path: Path) -> None:
    """``model_type=whisper`` (pre-existing) still routes via the same path."""
    local_dir = _write_minimal_local_dir(tmp_path, "whisper")
    resolver = create_asr_resolver(model="any-org/whisper-x", local_dir=local_dir, offline=True)
    assert resolver.model_type is WhisperHf


def test_unknown_config_model_type_still_rejected(tmp_path: Path) -> None:
    """A truly unknown ``model_type`` raises ``InvalidModelTypeInConfigError``."""
    from onnx_asr.utils import InvalidModelTypeInConfigError  # noqa: PLC0415

    local_dir = _write_minimal_local_dir(tmp_path, "not-a-real-model-type")
    with pytest.raises(InvalidModelTypeInConfigError):
        create_asr_resolver(model="any-org/whatever", local_dir=local_dir, offline=True)


def test_lite_whisper_end_to_end_load_via_full_repo_path() -> None:
    """End-to-end: ``onnx-community/lite-whisper-large-v3-acc-ONNX`` loads as WhisperHf.

    Skipped if the model isn't already cached — we don't want to force a
    1.5 GB download during CI. Verifies the bug from the user's report:
    passing the full HF repo path no longer raises InvalidModelTypeInConfigError.
    """
    model_id = "onnx-community/lite-whisper-large-v3-acc-ONNX"
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"
    if not cache_root.exists():
        pytest.skip(f"{model_id} not in HF cache — skipping E2E load test")

    model = onnx_asr.load_model(model_id, quantization="int8")
    assert isinstance(model, TextResultsAsrAdapter)
    assert isinstance(model.asr, WhisperHf)
