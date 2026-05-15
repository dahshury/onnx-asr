"""A lightweight Python package for Automatic Speech Recognition using ONNX models."""

from importlib.metadata import version as _version

from .asr import TimestampedResult, WordResult
from .loader import load_model, load_vad

__version__ = _version("onnx-asr")

__all__ = ["TimestampedResult", "WordResult", "load_model", "load_vad"]
