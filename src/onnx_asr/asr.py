"""Base ASR classes."""

import json
import re
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Generic, Literal, Protocol, TypedDict, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt

from onnx_asr.onnx import OnnxSessionOptions, TensorRtOptions
from onnx_asr.utils import StreamingNotSupportedError, log_softmax

S = TypeVar("S")


@dataclass
class TimestampedResult:
    """Timestamped recognition result."""

    text: str
    """Recognized text."""
    timestamps: list[float] | None = None
    """Tokens timestamp list."""
    tokens: list[str] | None = None
    """Tokens list."""
    logprobs: list[float] | None = None
    """Tokens logprob list."""


@dataclass(frozen=True)
class ModelCapabilities:
    """Per-model capability metadata.

    Used by callers (e.g. a server orchestrator) to route between models
    without inspecting concrete classes. Models override the class-level
    ``capabilities`` constant; the defaults here describe a plain offline
    batch ASR with no streaming, no timestamps, and no decode-time knobs.
    """

    streaming_native: bool = False
    """Model supports stateful streaming via ``create_stream()`` (cache-aware encoder + persistent decoder state)."""
    supports_timestamps: bool = False
    """Model emits segment timestamps in ``recognize()`` output."""
    supports_word_timestamps: bool = False
    """Model can produce word-level timestamps (typically via cross-attention DTW)."""
    supports_beam_search: bool = False
    """Model honors the ``beam_size`` decode option."""
    supports_temperature_fallback: bool = False
    """Model honors the temperature ladder + quality guards."""
    is_multilingual: bool = False
    """Model handles multiple languages (vs. English-only)."""


@dataclass(frozen=True)
class StreamingResult:
    """Snapshot of stream state after one decode step.

    Streams that use a stability policy (LocalAgreement-2 for Whisper, etc.)
    populate ``committed_text`` with the prefix that won't change in future
    snapshots. Streams without a stability policy leave it as ``""``.
    """

    text: str
    """Concatenated text emitted in this snapshot (committed + preview)."""
    tokens: list[str]
    """Tokens corresponding to ``text``."""
    timestamps: list[float] | None
    """Per-token timestamps if the model supports them."""
    is_partial: bool
    """True for live preview; False once the segment is committed/final."""
    segment_id: int
    """Increments each time the stream advances past a VAD endpoint."""
    committed_text: str = ""
    """Stable prefix that successive snapshots are guaranteed not to revise."""


@runtime_checkable
class AsrStream(Protocol):
    """Streaming ASR session protocol.

    The shape follows sherpa-onnx's ``OnlineStream`` / ``OnlineRecognizer``
    pair: producers ``push_audio()`` PCM samples and call ``finish()`` when
    the source closes; consumers poll ``is_ready()`` then call ``step()`` to
    advance one chunk. ``reset()`` zeros predictor state (and optionally the
    buffered audio) so the same stream object can serve multiple utterances.
    """

    def push_audio(self, samples: npt.NDArray[np.float32], sample_rate: int = 16_000) -> None:
        """Append PCM samples to the stream's input buffer."""
        ...

    def finish(self) -> None:
        """Mark the input as complete. After this, ``step()`` drains residual buffer."""
        ...

    def is_ready(self) -> bool:
        """Return True when at least one decode step's worth of audio is buffered."""
        ...

    def step(self) -> StreamingResult | None:
        """Consume one chunk from the buffer and return a snapshot, or None if not yet ready."""
        ...

    def reset(self, *, keep_audio: bool = False) -> None:
        """Zero decoder state. By default also drops the buffered audio."""
        ...

    @property
    def is_endpoint(self) -> bool:
        """Whether an endpoint (e.g. trailing silence) was detected on the last step."""
        ...


class AsrConfig(TypedDict, total=False):
    """Config for ASR model."""

    model_type: str
    features_size: int
    subsampling_factor: int
    max_tokens_per_step: int
    max_sequence_length: int


class Preprocessor(Protocol):
    """ASR preprocessor protocol."""

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Convert waveforms to model features."""
        ...


class Asr(Protocol):
    """ASR protocol."""

    capabilities: ClassVar[ModelCapabilities]

    @staticmethod
    def _get_sample_rate() -> Literal[8_000, 16_000]:
        return 16_000

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        """Recognize waveforms batch."""
        ...

    def create_stream(self) -> AsrStream:
        """Open a streaming session.

        Raises ``StreamingNotSupportedError`` unless the model overrides this and
        sets ``capabilities.streaming_native = True``.
        """
        ...


class BaseAsr(Asr):
    """Base ASR class."""

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities()

    def __init__(
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        """Create ASR.

        Args:
            model_files: Dict with paths to model files.
            preprocessor_factory: Factory for preprocessor creation.
            onnx_options: Options for onnxruntime InferenceSession.

        """
        if "config" in model_files:
            with model_files["config"].open("rt", encoding="utf-8") as f:
                self.config: AsrConfig = json.load(f)
        else:
            self.config = {}

        self.runtime_config = onnx_options
        self.use_low_precision = TensorRtOptions.is_low_precision(onnx_options)
        self._preprocessor = preprocessor_factory(self._preprocessor_name)

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        return []

    @staticmethod
    @abstractmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]: ...

    @property
    @abstractmethod
    def _preprocessor_name(self) -> str: ...

    def create_stream(self) -> AsrStream:
        """Open a streaming session. Default raises ``StreamingNotSupportedError``.

        Models that support streaming override this AND set
        ``capabilities = ModelCapabilities(streaming_native=True, ...)``.
        """
        raise StreamingNotSupportedError(type(self).__name__)


class _AsrWithDecoding(BaseAsr):
    DECODE_SPACE_PATTERN = re.compile(r"\A\s|\s\B|(\s)\b")
    window_step = 0.01

    def __init__(
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)

        if "vocab" in model_files:
            with Path(model_files["vocab"]).open("rt", encoding="utf-8") as f:
                self._vocab = {
                    int(id): token.replace("\u2581", " ") for token, id in (line.strip("\n").split(" ") for line in f)
                }
            self._vocab_size = len(self._vocab)
            if (blank_idx := next((id for id, token in self._vocab.items() if token == "<blk>"), None)) is not None:
                self._blank_idx = blank_idx

    @property
    @abstractmethod
    def _subsampling_factor(self) -> int: ...

    @abstractmethod
    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...

    @abstractmethod
    def _decoding(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[tuple[Iterable[int], Iterable[int] | None, Iterable[float] | None]]: ...

    def _decode_tokens(
        self, ids: Iterable[int], indices: Iterable[int] | None, logprobs: Iterable[float] | None
    ) -> TimestampedResult:
        tokens = [self._vocab[i] for i in ids]
        text = re.sub(self.DECODE_SPACE_PATTERN, lambda x: " " if x.group(1) else "", "".join(tokens))
        timestamps = (
            None if indices is None else (self.window_step * self._subsampling_factor * np.asarray(indices)).tolist()
        )
        return TimestampedResult(text, timestamps, tokens, None if logprobs is None else np.asarray(logprobs).tolist())

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        encoder_out, encoder_out_lens = self._encode(*self._preprocessor(waveforms, waveforms_len))
        return map(self._decode_tokens, *zip(*self._decoding(encoder_out, encoder_out_lens, **kwargs), strict=False))


class _AsrWithCtcDecoding(_AsrWithDecoding):
    def _decoding(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[tuple[Iterable[int], Iterable[int], Iterable[float]]]:
        assert encoder_out.shape[-1] <= self._vocab_size
        assert encoder_out.shape[1] >= max(encoder_out_lens)

        batch_tokens = encoder_out.argmax(axis=-1)
        batch_mask = np.arange(batch_tokens.shape[1])[None, :] < encoder_out_lens[:, None]
        batch_mask &= batch_tokens != self._blank_idx
        batch_logprobs = np.take_along_axis(encoder_out, batch_tokens[:, :, None], axis=-1).squeeze(axis=-1)
        batch_logprobs = np.where(batch_mask, batch_logprobs, 0.0)
        batch_mask &= np.diff(batch_tokens, axis=-1, prepend=self._blank_idx) != 0
        for i in range(encoder_out.shape[0]):
            mask = batch_mask[i]
            idx = np.flatnonzero(mask)
            yield batch_tokens[i][mask], idx, np.add.reduceat(batch_logprobs[i], idx)


class _AsrWithTransducerDecoding(_AsrWithDecoding, Generic[S]):
    @property
    @abstractmethod
    def _max_tokens_per_step(self) -> int: ...

    @abstractmethod
    def _create_state(self) -> S: ...

    @abstractmethod
    def _decode(
        self, prev_tokens: list[int], prev_state: S, encoder_out: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], int, S]: ...

    def _decoding(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[tuple[Iterable[int], Iterable[int], Iterable[float] | None]]:
        need_logprobs = kwargs.get("need_logprobs")
        if self.use_low_precision:  # TensorRT low precision models may return incorrect encoder_out_lens
            encoder_out_lens = np.minimum(encoder_out_lens, encoder_out.shape[1])

        for encodings, encodings_len in zip(encoder_out, encoder_out_lens, strict=True):
            assert encodings_len <= encodings.shape[0]
            prev_state = self._create_state()
            tokens: list[int] = []
            timestamps: list[int] = []
            logprobs: list[float] = []

            t = 0
            emitted_tokens = 0
            while t < encodings_len:
                logits, step, state = self._decode(tokens, prev_state, encodings[t])
                assert logits.shape[-1] <= self._vocab_size

                token = logits.argmax()

                if token != self._blank_idx:
                    prev_state = state
                    tokens.append(int(token))
                    timestamps.append(t)
                    emitted_tokens += 1
                    if need_logprobs:
                        logprobs.append(log_softmax(logits)[token])

                if step > 0:
                    t += step
                    emitted_tokens = 0
                elif token == self._blank_idx or emitted_tokens == self._max_tokens_per_step:
                    t += 1
                    emitted_tokens = 0

            yield tokens, timestamps, logprobs if need_logprobs else None
