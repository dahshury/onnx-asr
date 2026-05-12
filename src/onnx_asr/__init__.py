"""A lightweight Python package for Automatic Speech Recognition using ONNX models."""

from importlib.metadata import version as _version

from .asr import AsrStream, ModelCapabilities, StreamingResult
from .loader import load_model, load_vad
from .streaming import BufferedAsrStream, create_buffered_stream
from .utils import StreamingNotSupportedError

__version__ = _version("onnx-asr")

__all__ = [
    "AsrStream",
    "BufferedAsrStream",
    "ModelCapabilities",
    "StreamingNotSupportedError",
    "StreamingResult",
    "create_buffered_stream",
    "load_model",
    "load_vad",
]
