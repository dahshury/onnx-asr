"""Whisper model implementations.

Plan #7 refactor: the monolithic ``models/whisper.py`` is now a package.
The four classes live in dedicated modules:

* :mod:`._base` — ``_Whisper`` abstract base (alias ``_WhisperBase``) plus
  shared tokenizer / prompt / segment / word-alignment logic.
* :mod:`._stream` — ``WhisperStream`` LocalAgreement-2 wrapper.
* :mod:`._ort` — ``WhisperOrt`` (single-graph beam-search export).
* :mod:`._hf` — ``WhisperHf`` (Optimum encoder+decoder export).

The package re-exports the same names the old module exposed, so
``from onnx_asr.models.whisper import WhisperHf, WhisperOrt, WhisperStream, _Whisper``
keeps working unchanged.
"""

from __future__ import annotations

from onnx_asr.models.whisper._base import _Whisper, _WhisperBase, bytes_to_unicode
from onnx_asr.models.whisper._hf import WhisperHf
from onnx_asr.models.whisper._ort import WhisperOrt
from onnx_asr.models.whisper._stream import WhisperStream

__all__ = [
    "WhisperHf",
    "WhisperOrt",
    "WhisperStream",
    "_Whisper",
    "_WhisperBase",
    "bytes_to_unicode",
]
