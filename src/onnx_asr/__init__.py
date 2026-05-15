"""A lightweight Python package for Automatic Speech Recognition using ONNX models."""

from importlib.metadata import version as _version

from .asr import AsrStream, ModelCapabilities, StreamingResult
from .loader import load_model, load_vad, load_wake_word
from .progress import DownloadProgress, ProgressCallback
from .utils import StreamingNotSupportedError
from .wake_word import WakeWord, WakeWordResult

__version__ = _version("onnx-asr")

__all__ = [
    "AsrStream",
    "DownloadProgress",
    "ModelCapabilities",
    "ProgressCallback",
    "StreamingNotSupportedError",
    "StreamingResult",
    "WakeWord",
    "WakeWordResult",
    "load_model",
    "load_vad",
    "load_wake_word",
]
