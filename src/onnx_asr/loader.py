"""Loader for ASR and VAD models."""

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias

import onnxruntime as rt

from onnx_asr.adapters import SeAdapter, TextResultsAsrAdapter
from onnx_asr.asr import Asr, Preprocessor
from onnx_asr.diarization import Diarizer, SessionDiarizer
from onnx_asr.models.gigaam import GigaamV2Ctc, GigaamV2Rnnt, GigaamV3E2eCtc, GigaamV3E2eRnnt
from onnx_asr.models.kaldi import KaldiTransducer
from onnx_asr.models.moonshine import Moonshine
from onnx_asr.models.nemo import NemoConformerAED, NemoConformerCtc, NemoConformerRnnt, NemoConformerTdt
from onnx_asr.models.openwakeword import OpenWakeWord
from onnx_asr.models.pyannote import PyAnnoteVad
from onnx_asr.models.silero import SileroVad
from onnx_asr.models.tone import TOneCtc
from onnx_asr.models.wespeaker import WespeakerEmbeddings
from onnx_asr.models.whisper import WhisperHf, WhisperOrt
from onnx_asr.onnx import OnnxSessionOptions, Provider, get_onnx_providers, update_onnx_providers
from onnx_asr.preprocessors.numpy_preprocessor import (
    GigaamPreprocessorNumpy,
    KaldiPreprocessorNumpy,
    NemoPreprocessorNumpy,
    WhisperPreprocessorNumpy,
)
from onnx_asr.preprocessors.preprocessor import ConcurrentPreprocessor, IdentityPreprocessor, OnnxPreprocessor
from onnx_asr.preprocessors.resampler import Resampler
from onnx_asr.progress import ProgressCallback
from onnx_asr.resolver import Resolver
from onnx_asr.se import SpeakerEmbedding
from onnx_asr.utils import (
    ModelNotSupportedError,
)
from onnx_asr.vad import Vad
from onnx_asr.wake_word import WakeWord

AsrNames = Literal[
    "gigaam-v2-ctc",
    "gigaam-v2-rnnt",
    "gigaam-v3-ctc",
    "gigaam-v3-rnnt",
    "gigaam-v3-e2e-ctc",
    "gigaam-v3-e2e-rnnt",
    "nemo-fastconformer-ru-ctc",
    "nemo-fastconformer-ru-rnnt",
    "nemo-parakeet-ctc-0.6b",
    "nemo-parakeet-rnnt-0.6b",
    "nemo-parakeet-tdt-0.6b-v2",
    "nemo-parakeet-tdt-0.6b-v3",
    "nemo-canary-1b-v2",
    "alphacep/vosk-model-ru",
    "alphacep/vosk-model-small-ru",
    "t-tech/t-one",
    "whisper-base",
    "moonshine-tiny",
    "moonshine-base",
    "moonshine-tiny-zh",
    "moonshine-tiny-ja",
    "moonshine-tiny-ko",
    "moonshine-tiny-ar",
    "moonshine-tiny-vi",
    "moonshine-base-zh",
    "moonshine-base-ja",
    "moonshine-base-ko",
]
"""Supported ASR model names (can be automatically downloaded from the Hugging Face)."""

AsrTypeNames = Literal[
    "kaldi-rnnt",
    "nemo-conformer-ctc",
    "nemo-conformer-rnnt",
    "nemo-conformer-tdt",
    "nemo-conformer-aed",
    "t-one-ctc",
    "vosk",
    "whisper-ort",
    "whisper",
    "moonshine",
]
"""Supported ASR model types."""

VadNames = Literal["silero", "onnx-community/pyannote-segmentation-3.0"]
"""Supported VAD model names (can be automatically downloaded from the Hugging Face)."""

VadTypeNames = Literal["pyannote"]
"""Supported VAD model types."""

WakeWordNames = Literal["openwakeword"]
"""Supported wake-word model names (can be automatically downloaded from the Hugging Face)."""

WakeWordTypeNames = Literal["openwakeword"]
"""Supported wake-word model types."""

WakeWordTypes: TypeAlias = OpenWakeWord

AsrTypes: TypeAlias = (
    GigaamV2Ctc
    | GigaamV2Rnnt
    | KaldiTransducer
    | Moonshine
    | NemoConformerCtc
    | NemoConformerRnnt
    | NemoConformerAED
    | TOneCtc
    | WhisperHf
    | WhisperOrt
)


def create_asr_resolver(
    model: str | None = None,
    local_dir: str | Path | None = None,
    *,
    offline: bool | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Resolver[AsrTypes]:
    """Create resolver for ASR models."""
    model_types: dict[str, type[AsrTypes]] = {
        "gigaam-v2-ctc": GigaamV2Ctc,
        "gigaam-v2-rnnt": GigaamV2Rnnt,
        "gigaam-v3-ctc": GigaamV2Ctc,
        "gigaam-v3-rnnt": GigaamV2Rnnt,
        "gigaam-v3-e2e-ctc": GigaamV3E2eCtc,
        "gigaam-v3-e2e-rnnt": GigaamV3E2eRnnt,
        "nemo-fastconformer-ru-ctc": NemoConformerCtc,
        "nemo-fastconformer-ru-rnnt": NemoConformerRnnt,
        "nemo-parakeet-ctc-0.6b": NemoConformerCtc,
        "nemo-parakeet-rnnt-0.6b": NemoConformerRnnt,
        "nemo-parakeet-tdt-0.6b-v2": NemoConformerTdt,
        "nemo-parakeet-tdt-0.6b-v3": NemoConformerTdt,
        "nemo-canary-1b-v2": NemoConformerAED,
        "whisper-base": WhisperOrt,
        "kaldi-rnnt": KaldiTransducer,
        "nemo-conformer-ctc": NemoConformerCtc,
        "nemo-conformer-rnnt": NemoConformerRnnt,
        "nemo-conformer-tdt": NemoConformerTdt,
        "nemo-conformer-aed": NemoConformerAED,
        "t-one-ctc": TOneCtc,
        "vosk": KaldiTransducer,
        "whisper-ort": WhisperOrt,
        "whisper": WhisperHf,
        # ``model_type`` values reported in ``config.json`` for HF Optimum-style
        # exports that are structurally Whisper-compatible (encoder_model.onnx +
        # decoder_model_merged.onnx + vocab.json + added_tokens.json). The
        # encoder internals differ (Lite-Whisper factorizes linear layers,
        # Distil-Whisper distills decoder layers, timestamped variants alter
        # the suppression mask) but the I/O contract is the same.
        "lite-whisper": WhisperHf,
        "distil-whisper": WhisperHf,
        # Moonshine — bare aliases (resolver hits the dict before parsing
        # config.json) plus the ``model_type`` reported by every variant.
        "moonshine": Moonshine,
        "moonshine-tiny": Moonshine,
        "moonshine-base": Moonshine,
        "moonshine-tiny-zh": Moonshine,
        "moonshine-tiny-ja": Moonshine,
        "moonshine-tiny-ko": Moonshine,
        "moonshine-tiny-ar": Moonshine,
        "moonshine-tiny-vi": Moonshine,
        "moonshine-base-zh": Moonshine,
        "moonshine-base-ja": Moonshine,
        "moonshine-base-ko": Moonshine,
        "alphacep/vosk-model-ru": KaldiTransducer,
        "alphacep/vosk-model-small-ru": KaldiTransducer,
        "t-tech/t-one": TOneCtc,
    }
    return Resolver(model_types, model, local_dir, offline=offline, progress_callback=progress_callback)


VadTypes: TypeAlias = SileroVad | PyAnnoteVad


def create_vad_resolver(
    model: str | None = None,
    local_dir: str | Path | None = None,
    *,
    offline: bool | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Resolver[VadTypes]:
    """Create resolver for VAD models."""
    model_types: dict[str, type[VadTypes]] = {
        "silero": SileroVad,
        "pyannote": PyAnnoteVad,
    }
    return Resolver(model_types, model, local_dir, offline=offline, progress_callback=progress_callback)


def create_wake_word_resolver(
    model: str | None = None, local_dir: str | Path | None = None, *, offline: bool | None = None
) -> Resolver[WakeWordTypes]:
    """Create resolver for wake-word models."""
    model_types: dict[str, type[WakeWordTypes]] = {
        "openwakeword": OpenWakeWord,
    }
    return Resolver(model_types, model, local_dir, offline=offline)


def create_se_resolver(
    model: str | None = None,
    local_dir: str | Path | None = None,
    *,
    offline: bool | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Resolver[WespeakerEmbeddings]:
    """Create resolver for SE models."""
    return Resolver(WespeakerEmbeddings, model, local_dir, offline=offline, progress_callback=progress_callback)


class PreprocessorRuntimeConfig(OnnxSessionOptions, total=False):
    """Preprocessor runtime config."""

    max_concurrent_workers: int | None
    """Max parallel preprocessing threads (None - auto, 1 - without parallel processing)."""

    use_numpy_preprocessors: bool | None
    """Use NumPy preprocessors backend instead of ONNX."""


class Manager:
    """Manager for models creation."""

    def __init__(
        self,
        sess_options: rt.SessionOptions | None = None,
        providers: Sequence[str | Provider | tuple[str | Provider, dict[Any, Any]]] | None = None,
        provider_options: Sequence[dict[Any, Any]] | None = None,
        preprocessor_config: PreprocessorRuntimeConfig | None = None,
        resampler_config: OnnxSessionOptions | None = None,
    ) -> None:
        """Create model manager."""
        default_providers: list[str] = rt.get_available_providers()
        if default_providers[0] == "AzureExecutionProvider":
            default_providers.pop(0)

        self.default_onnx_config: OnnxSessionOptions = {
            "sess_options": sess_options,
            "providers": providers or default_providers,
            "provider_options": provider_options,
        }

        if preprocessor_config is None:
            self.preprocessor_config = update_onnx_providers(
                self.default_onnx_config,
                new_options={"TensorrtExecutionProvider": {"trt_fp16_enable": False}},
                excluded_providers=OnnxPreprocessor._get_excluded_providers(),
            )
            self.preprocessor_max_workers: int | None = 1
            self.use_numpy_preprocessors = None
        else:
            self.preprocessor_max_workers = preprocessor_config.pop("max_concurrent_workers", 1)
            self.use_numpy_preprocessors = preprocessor_config.pop("use_numpy_preprocessors")
            self.preprocessor_config = preprocessor_config

        providers = get_onnx_providers(self.preprocessor_config)
        if self.use_numpy_preprocessors is None:
            self.use_numpy_preprocessors = not providers or providers[0] == "CPUExecutionProvider"

        if resampler_config is None:
            resampler_config = update_onnx_providers(
                self.default_onnx_config, excluded_providers=Resampler._get_excluded_providers()
            )
        self.resampler_config = resampler_config

    def _create_preprocessor(self, name: str) -> Preprocessor:
        if name == "identity":
            return IdentityPreprocessor()

        preprocessor: Preprocessor
        if self.use_numpy_preprocessors:
            if name.startswith("gigaam"):
                preprocessor = GigaamPreprocessorNumpy(name)
            elif name in ("kaldi", "wespeaker"):
                preprocessor = KaldiPreprocessorNumpy(name)
            elif name.startswith("nemo"):
                preprocessor = NemoPreprocessorNumpy(name)
            elif name.startswith("whisper"):
                preprocessor = WhisperPreprocessorNumpy(name)
            else:
                raise ModelNotSupportedError(name)
        else:
            preprocessor = OnnxPreprocessor(name, self.preprocessor_config)

        if self.preprocessor_max_workers == 1:
            return preprocessor
        return ConcurrentPreprocessor(preprocessor, self.preprocessor_max_workers)

    def _create_resampler(self, sample_rate: Literal[8000, 16000]) -> Resampler:
        return Resampler(sample_rate, self.resampler_config)

    def _create_asr_adapter(self, asr: Asr) -> TextResultsAsrAdapter:
        return TextResultsAsrAdapter(asr, self._create_resampler(asr._get_sample_rate()))

    def _create_se_adapter(self, se: SpeakerEmbedding) -> SeAdapter:
        return SeAdapter(se, self._create_resampler(se._get_sample_rate()))

    def create_asr(
        self,
        model: str | AsrNames | AsrTypeNames | None = None,
        local_dir: str | Path | None = None,
        *,
        quantization: str | None = None,
        offline: bool | None = None,
        config: OnnxSessionOptions | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> TextResultsAsrAdapter:
        """Create ASR model."""
        resolver = create_asr_resolver(model, local_dir, offline=offline, progress_callback=progress_callback)
        if config is None:
            config = update_onnx_providers(
                self.default_onnx_config, excluded_providers=resolver.model_type._get_excluded_providers()
            )
        return self._create_asr_adapter(
            resolver.model_type(resolver.resolve_model(quantization=quantization), self._create_preprocessor, config)
        )

    def create_vad(
        self,
        model: str | VadNames | VadTypeNames | None = None,
        local_dir: str | Path | None = None,
        *,
        quantization: str | None = None,
        offline: bool | None = None,
        config: OnnxSessionOptions | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> Vad:
        """Create VAD model."""
        resolver = create_vad_resolver(model, local_dir, offline=offline, progress_callback=progress_callback)
        if config is None:
            config = update_onnx_providers(
                self.default_onnx_config, excluded_providers=resolver.model_type._get_excluded_providers()
            )
        return resolver.model_type(resolver.resolve_model(quantization=quantization), config)

    def create_se(
        self,
        model: str | None = None,
        local_dir: str | Path | None = None,
        *,
        quantization: str | None = None,
        offline: bool | None = None,
        config: OnnxSessionOptions | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> SeAdapter:
        """Create SE model."""
        resolver = create_se_resolver(model, local_dir, offline=offline, progress_callback=progress_callback)
        if config is None:
            config = update_onnx_providers(
                self.default_onnx_config, excluded_providers=resolver.model_type._get_excluded_providers()
            )
        return self._create_se_adapter(
            resolver.model_type(resolver.resolve_model(quantization=quantization), self._create_preprocessor, config)
        )

    def create_diarizer(
        self,
        *,
        segmentation_model: str = "onnx-community/pyannote-segmentation-3.0",
        embedding_model: str = "wespeaker-voxceleb-resnet34-LM",
        segmentation_path: str | Path | None = None,
        embedding_path: str | Path | None = None,
        quantization: str | None = None,
        offline: bool | None = None,
        config: OnnxSessionOptions | None = None,
        progress_callback: ProgressCallback | None = None,
        **diarizer_kwargs: float,
    ) -> Diarizer:
        """Create a speaker diarizer.

        Wires a PyAnnote VAD (used in segmentation-only mode here) and a
        Wespeaker embedding adapter into :class:`~onnx_asr.diarization.Diarizer`.

        Args:
            segmentation_model: HF repo id for pyannote-segmentation-3.0 ONNX.
            embedding_model: HF repo id or alias for the speaker embedder.
            segmentation_path: optional local dir override for the segmentation model.
            embedding_path: optional local dir override for the embedding model.
            quantization: ORT quantization tier passed to both models.
            offline: forbid network downloads on both models.
            config: ONNX session config applied to both models.
            progress_callback: download progress hook.
            **diarizer_kwargs: forwarded to :class:`~onnx_asr.diarization.Diarizer`'s
                constructor (``onset``, ``offset``, ``min_segment_duration`` …).
        """
        seg_resolver = create_vad_resolver(
            segmentation_model, segmentation_path, offline=offline, progress_callback=progress_callback
        )
        seg_config = config or update_onnx_providers(
            self.default_onnx_config, excluded_providers=seg_resolver.model_type._get_excluded_providers()
        )
        segmenter = seg_resolver.model_type(seg_resolver.resolve_model(quantization=quantization), seg_config)
        # The diarizer needs the PyAnnoteVad concrete class for speaker_probs_batch().
        from onnx_asr.models.pyannote import PyAnnoteVad  # noqa: PLC0415

        if not isinstance(segmenter, PyAnnoteVad):
            msg = f"diarizer requires a pyannote segmentation model, got {type(segmenter).__name__}"
            raise TypeError(msg)

        embed_resolver = create_se_resolver(
            embedding_model, embedding_path, offline=offline, progress_callback=progress_callback
        )
        emb_config = config or update_onnx_providers(
            self.default_onnx_config, excluded_providers=embed_resolver.model_type._get_excluded_providers()
        )
        embedder_model = embed_resolver.model_type(
            embed_resolver.resolve_model(quantization=quantization), self._create_preprocessor, emb_config
        )
        embedder_adapter = self._create_se_adapter(embedder_model)
        return Diarizer(segmenter, embedder_adapter, **diarizer_kwargs)

    def create_wake_word(
        self,
        model: str | WakeWordNames | WakeWordTypeNames | None = None,
        local_dir: str | Path | None = None,
        *,
        quantization: str | None = None,
        offline: bool | None = None,
        config: OnnxSessionOptions | None = None,
    ) -> WakeWord:
        """Create wake-word model."""
        resolver = create_wake_word_resolver(model, local_dir, offline=offline)
        if config is None:
            config = update_onnx_providers(
                self.default_onnx_config, excluded_providers=resolver.model_type._get_excluded_providers()
            )
        return resolver.model_type(resolver.resolve_model(quantization=quantization), config)


def load_model(
    model: str | AsrNames | AsrTypeNames,
    path: str | Path | None = None,
    *,
    quantization: str | None = None,
    sess_options: rt.SessionOptions | None = None,
    providers: Sequence[str | Provider | tuple[str | Provider, dict[Any, Any]]] | None = None,
    provider_options: Sequence[dict[Any, Any]] | None = None,
    cpu_preprocessing: bool | None = None,
    asr_config: OnnxSessionOptions | None = None,
    preprocessor_config: PreprocessorRuntimeConfig | None = None,
    resampler_config: OnnxSessionOptions | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TextResultsAsrAdapter:
    """Load ASR model.

    Args:
        model: Model name or type (download from Hugging Face supported if full model name is provided):

                GigaAM v2 (`gigaam-v2-ctc` | `gigaam-v2-rnnt`)
                GigaAM v3 (`gigaam-v3-ctc` | `gigaam-v3-rnnt` |
                           `gigaam-v3-e2e-ctc` | `gigaam-v3-e2e-rnnt`)
                Kaldi Transducer (`kaldi-rnnt`)
                NeMo Conformer (`nemo-conformer-ctc` | `nemo-conformer-rnnt` | `nemo-conformer-tdt` |
                                `nemo-conformer-aed`)
                NeMo FastConformer Hybrid Large Ru P&C (`nemo-fastconformer-ru-ctc` |
                                                        `nemo-fastconformer-ru-rnnt`)
                NeMo Parakeet 0.6B En (`nemo-parakeet-ctc-0.6b` | `nemo-parakeet-rnnt-0.6b` |
                                       `nemo-parakeet-tdt-0.6b-v2`)
                NeMo Parakeet 0.6B Multilingual (`nemo-parakeet-tdt-0.6b-v3`)
                NeMo Canary (`nemo-canary-1b-v2`)
                T-One (`t-one-ctc` | `t-tech/t-one`)
                Vosk (`vosk` | `alphacep/vosk-model-ru` | `alphacep/vosk-model-small-ru`)
                Whisper Base exported with onnxruntime (`whisper-ort` | `whisper-base-ort`)
                Whisper from onnx-community (`whisper` | `onnx-community/whisper-large-v3-turbo` |
                                             `onnx-community/*whisper*`)
        path: Path to directory with model files.
        quantization: Model quantization (`None` | `int8` | ... ).
        sess_options: Default SessionOptions for onnxruntime.
        providers: Default providers for onnxruntime.
        provider_options: Default provider_options for onnxruntime.
        cpu_preprocessing: Deprecated and ignored, use `preprocessor_config` and `resampler_config` instead.
        asr_config: ASR ONNX config.
        preprocessor_config: Preprocessor ONNX and concurrency config.
        resampler_config: Resampler ONNX config.
        progress_callback: Optional callback receiving a :class:`~onnx_asr.progress.DownloadProgress`
            event for each chunk of bytes downloaded from Hugging Face. When provided,
            ``tqdm``'s own terminal output is suppressed. Has no effect when all required
            files are already present locally (no downloads happen).

    Returns:
        ASR model class.

    Raises:
        utils.ModelLoadingError: Model loading error (onnx-asr specific).

    """
    if cpu_preprocessing is not None:
        warnings.warn(
            "The cpu_preprocessing argument is deprecated and ignored (use preprocessor_config and resampler_config).",
            stacklevel=2,
        )

    manager = Manager(sess_options, providers, provider_options, preprocessor_config, resampler_config)
    return manager.create_asr(
        model, path, quantization=quantization, config=asr_config, progress_callback=progress_callback
    )


def load_vad(
    model: str | VadNames | VadTypeNames = "silero",
    path: str | Path | None = None,
    *,
    quantization: str | None = None,
    sess_options: rt.SessionOptions | None = None,
    providers: Sequence[str | Provider | tuple[str | Provider, dict[Any, Any]]] | None = None,
    provider_options: Sequence[dict[Any, Any]] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Vad:
    """Load VAD model.

    Args:
        model: VAD model name or type (download from Hugging Face supported if full model name is provided).
        path: Path to directory with model files.
        quantization: Model quantization (`None` | `int8` | ... ).
        sess_options: Optional SessionOptions for onnxruntime.
        providers: Optional providers for onnxruntime.
        provider_options: Optional provider_options for onnxruntime.
        progress_callback: Optional callback receiving a :class:`~onnx_asr.progress.DownloadProgress`
            event for each chunk of bytes downloaded from Hugging Face. Has no effect when
            all required files are already present locally.

    Returns:
        VAD model class.

    Raises:
        utils.ModelLoadingError: Model loading error (onnx-asr specific).

    """
    manager = Manager()
    config: OnnxSessionOptions = {
        "sess_options": sess_options,
        "providers": providers,
        "provider_options": provider_options,
    }

    return manager.create_vad(
        model,
        path,
        quantization=quantization,
        config=config if any(value is not None for value in config.values()) else None,
        progress_callback=progress_callback,
    )


def load_diarizer(
    *,
    segmentation_model: str = "onnx-community/pyannote-segmentation-3.0",
    embedding_model: str = "wespeaker-voxceleb-resnet34-LM",
    segmentation_path: str | Path | None = None,
    embedding_path: str | Path | None = None,
    quantization: str | None = None,
    sess_options: rt.SessionOptions | None = None,
    providers: Sequence[str | Provider | tuple[str | Provider, dict[Any, Any]]] | None = None,
    provider_options: Sequence[dict[Any, Any]] | None = None,
    progress_callback: ProgressCallback | None = None,
    **diarizer_kwargs: float,
) -> Diarizer:
    """Load a speaker diarizer (pyannote segmentation + wespeaker embedding).

    The default model pair — ``onnx-community/pyannote-segmentation-3.0`` (MIT,
    ~6 MB) + ``Wespeaker/wespeaker-voxceleb-resnet34-LM`` (CC-BY-4.0, ~26 MB) —
    adds ~32 MB to the bundle and runs entirely on ONNX Runtime, no torch.

    Args:
        segmentation_model: HF repo id for pyannote-segmentation-3.0 ONNX.
        embedding_model: HF repo id or alias for the speaker embedder.
        segmentation_path: optional local dir override for the segmentation model.
        embedding_path: optional local dir override for the embedding model.
        quantization: ORT quantization tier (``None`` | ``int8`` | ...).
        sess_options: optional ORT SessionOptions.
        providers: optional ORT execution providers.
        provider_options: optional provider_options.
        progress_callback: download progress hook.
        **diarizer_kwargs: forwarded to :class:`~onnx_asr.diarization.Diarizer`
            (``onset``, ``offset``, ``min_segment_duration`` …).

    Returns:
        A ready-to-use :class:`~onnx_asr.diarization.Diarizer`.

    Example:
        >>> import onnx_asr
        >>> diar = onnx_asr.load_diarizer()
        >>> segments = diar.diarize(audio_float32, sample_rate=16_000)
        >>> for seg in segments:
        ...     print(seg.start, seg.end, seg.speaker)
    """
    manager = Manager(sess_options, providers, provider_options)
    return manager.create_diarizer(
        segmentation_model=segmentation_model,
        embedding_model=embedding_model,
        segmentation_path=segmentation_path,
        embedding_path=embedding_path,
        quantization=quantization,
        progress_callback=progress_callback,
        **diarizer_kwargs,
    )


def load_session_diarizer(
    *,
    segmentation_model: str = "onnx-community/pyannote-segmentation-3.0",
    embedding_model: str = "wespeaker-voxceleb-resnet34-LM",
    segmentation_path: str | Path | None = None,
    embedding_path: str | Path | None = None,
    quantization: str | None = None,
    sess_options: rt.SessionOptions | None = None,
    providers: Sequence[str | Provider | tuple[str | Provider, dict[Any, Any]]] | None = None,
    provider_options: Sequence[dict[Any, Any]] | None = None,
    progress_callback: ProgressCallback | None = None,
    max_speakers: int = 20,
    delta_new: float = 0.5,
    rho_update: float = 0.3,
    ema_alpha: float = 0.5,
    **diarizer_kwargs: float,
) -> SessionDiarizer:
    """Load a session diarizer with persistent across-utterance speaker tracking.

    Convenience wrapper: builds a stateless :class:`~onnx_asr.diarization.Diarizer`
    via :func:`load_diarizer` and wraps it in a :class:`~onnx_asr.diarization.SessionDiarizer`
    so successive :meth:`~onnx_asr.diarization.SessionDiarizer.diarize` calls
    share an :class:`~onnx_asr.diarization.OnlineSpeakerClustering` state.

    See :class:`~onnx_asr.diarization.OnlineSpeakerClustering` for ``max_speakers``,
    ``delta_new``, ``rho_update``, ``ema_alpha``. Remaining kwargs flow to
    :func:`load_diarizer`.
    """
    diarizer = load_diarizer(
        segmentation_model=segmentation_model,
        embedding_model=embedding_model,
        segmentation_path=segmentation_path,
        embedding_path=embedding_path,
        quantization=quantization,
        sess_options=sess_options,
        providers=providers,
        provider_options=provider_options,
        progress_callback=progress_callback,
        **diarizer_kwargs,
    )
    return SessionDiarizer(
        diarizer,
        max_speakers=max_speakers,
        delta_new=delta_new,
        rho_update=rho_update,
        ema_alpha=ema_alpha,
    )


def load_wake_word(
    model: str | WakeWordNames | WakeWordTypeNames = "openwakeword",
    path: str | Path | None = None,
    *,
    quantization: str | None = None,
    sess_options: rt.SessionOptions | None = None,
    providers: Sequence[str | Provider | tuple[str | Provider, dict[Any, Any]]] | None = None,
    provider_options: Sequence[dict[Any, Any]] | None = None,
) -> WakeWord:
    """Load a wake-word detection model.

    Mirrors :func:`load_vad`: pass an alias / HF repo id / type name, optionally
    a local directory of pre-downloaded artifacts, optional quantization tier
    and ORT session knobs. Returns a :class:`~onnx_asr.wake_word.WakeWord`
    instance ready for :meth:`~onnx_asr.wake_word.WakeWord.detect_batch`.

    Args:
        model: Wake-word model name / type / HF repo id. Default: ``openwakeword``.
        path: Local directory with pre-downloaded model files (skips download).
        quantization: ORT quantization tier (``None`` | ``int8`` | ...). Usually irrelevant for OWW.
        sess_options: Optional ORT SessionOptions.
        providers: Optional ORT execution providers.
        provider_options: Optional provider_options.

    Returns:
        Concrete :class:`~onnx_asr.wake_word.WakeWord` implementation.

    Raises:
        utils.ModelLoadingError: Model loading error (onnx-asr specific).

    """
    manager = Manager()
    config: OnnxSessionOptions = {
        "sess_options": sess_options,
        "providers": providers,
        "provider_options": provider_options,
    }

    return manager.create_wake_word(
        model,
        path,
        quantization=quantization,
        config=config if any(value is not None for value in config.values()) else None,
    )
