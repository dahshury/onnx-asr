"""A lightweight Python package for Automatic Speech Recognition using ONNX models."""

from importlib.metadata import version as _version

from .asr import AsrStream, ModelCapabilities, StreamingResult, TimestampedResult, WordResult
from .diarization import (
    DiarSegment,
    Diarizer,
    OnlineSpeakerClustering,
    SessionDiarizer,
    assign_speakers_to_words,
)
from .loader import load_diarizer, load_model, load_session_diarizer, load_vad, load_wake_word
from .progress import DownloadProgress, ProgressCallback
from .utils import StreamingNotSupportedError
from .wake_word import WakeWord, WakeWordResult

__version__ = _version("onnx-asr")

__all__ = [
    "AsrStream",
    "DiarSegment",
    "Diarizer",
    "DownloadProgress",
    "ModelCapabilities",
    "OnlineSpeakerClustering",
    "ProgressCallback",
    "SessionDiarizer",
    "StreamingNotSupportedError",
    "StreamingResult",
    "TimestampedResult",
    "WakeWord",
    "WakeWordResult",
    "WordResult",
    "assign_speakers_to_words",
    "load_diarizer",
    "load_model",
    "load_session_diarizer",
    "load_vad",
    "load_wake_word",
]
