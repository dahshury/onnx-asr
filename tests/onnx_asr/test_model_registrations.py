"""Smoke tests for new Whisper variant aliases (item #11).

Verifies that the short aliases map to the right Hugging Face repos and
load via the appropriate model class. Downloads are gated on a
``--run-network`` opt-in to avoid pulling 1+ GB on every test run.
"""

from __future__ import annotations

import pytest

from onnx_asr.loader import create_asr_resolver
from onnx_asr.models.whisper import WhisperHf, WhisperOrt
from onnx_asr.resolver import model_repos


@pytest.mark.parametrize(
    ("alias", "expected_repo"),
    [
        ("whisper-tiny", "onnx-community/whisper-tiny"),
        ("whisper-large-v3-turbo", "onnx-community/whisper-large-v3-turbo"),
        ("distil-whisper-v3.5", "onnx-community/distil-large-v3.5-ONNX"),
        ("lite-whisper-acc", "onnx-community/lite-whisper-large-v3-acc-ONNX"),
        ("whisper-base-timestamped", "onnx-community/whisper-base_timestamped"),
        ("whisper-medium-en-timestamped", "onnx-community/whisper-medium.en_timestamped"),
        ("whisper-large-v3-timestamped", "onnx-community/whisper-large-v3_timestamped"),
    ],
)
def test_alias_maps_to_expected_repo(alias: str, expected_repo: str) -> None:
    """The short alias resolves to the documented HF repo id."""
    assert model_repos[alias] == expected_repo


@pytest.mark.parametrize(
    "alias",
    [
        "whisper-tiny",
        "whisper-large-v3-turbo",
        "distil-whisper-v3.5",
        "lite-whisper-acc",
        "whisper-base-timestamped",
        "whisper-medium-en-timestamped",
        "whisper-large-v3-timestamped",
    ],
)
def test_alias_dispatches_to_whisper_hf(alias: str) -> None:
    """Every new alias is a HF-style export → handled by ``WhisperHf``."""
    resolver = create_asr_resolver(alias, offline=True)
    # ``offline=True`` with no local_dir lets us inspect ``model_type`` without downloading.
    # The Resolver picks the type during __init__ from the registered dict.
    assert resolver.model_type is WhisperHf


def test_legacy_alias_still_whisper_ort() -> None:
    """The existing ``whisper-base`` (istupakov beam-search ONNX) keeps its WhisperOrt routing."""
    resolver = create_asr_resolver("whisper-base", offline=True)
    assert resolver.model_type is WhisperOrt
