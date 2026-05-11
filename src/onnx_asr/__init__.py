"""A lightweight Python package for Automatic Speech Recognition using ONNX models."""

from importlib.metadata import version as _version

from .loader import load_model, load_vad
from .progress import DownloadProgress, ProgressCallback

__version__ = _version("onnx-asr")

__all__ = ["DownloadProgress", "ProgressCallback", "load_model", "load_vad"]
