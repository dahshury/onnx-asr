"""Tests for the shared ``_ModelImplementation`` ABC (plan item #14).

Verifies that every concrete model class in ``onnx_asr.models`` satisfies the
shared protocol: it exposes ``_get_model_files`` and ``_get_excluded_providers``
as static methods. The resolver relies on these to download artifacts and
choose execution providers before instantiation, so a missing method here
would break ``load_model`` / ``load_vad`` for that model.
"""

from __future__ import annotations

import inspect

import pytest

from onnx_asr.asr import Asr
from onnx_asr.model_base import _ModelImplementation
from onnx_asr.models.gigaam import GigaamV2Ctc, GigaamV2Rnnt, GigaamV3E2eCtc, GigaamV3E2eRnnt
from onnx_asr.models.kaldi import KaldiTransducer
from onnx_asr.models.nemo import NemoConformerAED, NemoConformerCtc, NemoConformerRnnt, NemoConformerTdt
from onnx_asr.models.pyannote import PyAnnoteVad
from onnx_asr.models.silero import SileroVad
from onnx_asr.models.tone import TOneCtc
from onnx_asr.models.wespeaker import WespeakerEmbeddings
from onnx_asr.models.whisper import WhisperHf, WhisperOrt
from onnx_asr.se import SpeakerEmbedding
from onnx_asr.vad import Vad

ASR_MODELS = [
    GigaamV2Ctc,
    GigaamV2Rnnt,
    GigaamV3E2eCtc,
    GigaamV3E2eRnnt,
    KaldiTransducer,
    NemoConformerAED,
    NemoConformerCtc,
    NemoConformerRnnt,
    NemoConformerTdt,
    TOneCtc,
    WhisperHf,
    WhisperOrt,
]

VAD_MODELS = [PyAnnoteVad, SileroVad]
SE_MODELS = [WespeakerEmbeddings]

ALL_MODELS = ASR_MODELS + VAD_MODELS + SE_MODELS


@pytest.mark.parametrize("model_cls", ALL_MODELS, ids=[m.__name__ for m in ALL_MODELS])
def test_class_implements_model_implementation_protocol(model_cls: type) -> None:
    """Every concrete model class is a ``_ModelImplementation`` instance.

    ``_ModelImplementation`` is ``runtime_checkable``, so we can isinstance-check
    the *class object* against it (because the protocol only requires static
    methods, classes themselves satisfy the protocol).
    """
    assert isinstance(model_cls, _ModelImplementation), f"{model_cls.__name__} missing protocol methods"


@pytest.mark.parametrize("model_cls", ALL_MODELS, ids=[m.__name__ for m in ALL_MODELS])
def test_get_model_files_is_static(model_cls: type) -> None:
    """``_get_model_files`` must be a staticmethod (resolver queries it pre-instantiation)."""
    raw = inspect.getattr_static(model_cls, "_get_model_files")
    assert isinstance(raw, staticmethod), f"{model_cls.__name__}._get_model_files is not @staticmethod"


@pytest.mark.parametrize("model_cls", ALL_MODELS, ids=[m.__name__ for m in ALL_MODELS])
def test_get_excluded_providers_is_static(model_cls: type) -> None:
    """``_get_excluded_providers`` must be a staticmethod."""
    raw = inspect.getattr_static(model_cls, "_get_excluded_providers")
    assert isinstance(raw, staticmethod), f"{model_cls.__name__}._get_excluded_providers is not @staticmethod"


@pytest.mark.parametrize("model_cls", ALL_MODELS, ids=[m.__name__ for m in ALL_MODELS])
def test_get_model_files_returns_dict(model_cls: type) -> None:
    """Calling ``_get_model_files()`` returns a non-empty dict[str, str]."""
    files = model_cls._get_model_files()  # type: ignore[attr-defined]
    assert isinstance(files, dict)
    assert files, f"{model_cls.__name__}._get_model_files() returned empty dict"
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in files.items())


@pytest.mark.parametrize("model_cls", ALL_MODELS, ids=[m.__name__ for m in ALL_MODELS])
def test_get_excluded_providers_returns_list(model_cls: type) -> None:
    """Calling ``_get_excluded_providers()`` returns a list of provider name strings."""
    providers = model_cls._get_excluded_providers()  # type: ignore[attr-defined]
    assert isinstance(providers, list)
    assert all(isinstance(p, str) for p in providers)


def test_asr_protocol_inherits_model_implementation() -> None:
    """The ``Asr`` Protocol inherits the shared infrastructure surface."""
    assert issubclass(Asr, _ModelImplementation) or _ModelImplementation in Asr.__mro__


def test_vad_protocol_inherits_model_implementation() -> None:
    assert issubclass(Vad, _ModelImplementation) or _ModelImplementation in Vad.__mro__


def test_speaker_embedding_protocol_inherits_model_implementation() -> None:
    assert issubclass(SpeakerEmbedding, _ModelImplementation) or _ModelImplementation in SpeakerEmbedding.__mro__
