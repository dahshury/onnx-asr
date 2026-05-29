"""Resolver for ASR and VAD models."""

import contextlib
import json
from pathlib import Path, PurePosixPath
from typing import Generic, TypeVar

from onnx_asr.model_base import _ModelImplementation
from onnx_asr.progress import ProgressCallback, hf_progress_patch
from onnx_asr.utils import (
    InvalidModelTypeInConfigError,
    ModelFileNotFoundError,
    ModelNotSupportedError,
    ModelPathNotDirectoryError,
    MoreThanOneModelFileFoundError,
    NoModelNameOrPathSpecifiedError,
)

model_repos = {
    "gigaam-v2-ctc": "istupakov/gigaam-v2-onnx",
    "gigaam-v2-rnnt": "istupakov/gigaam-v2-onnx",
    "gigaam-v3-ctc": "istupakov/gigaam-v3-onnx",
    "gigaam-v3-rnnt": "istupakov/gigaam-v3-onnx",
    "gigaam-v3-e2e-ctc": "istupakov/gigaam-v3-onnx",
    "gigaam-v3-e2e-rnnt": "istupakov/gigaam-v3-onnx",
    "nemo-fastconformer-ru-ctc": "istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx",
    "nemo-fastconformer-ru-rnnt": "istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx",
    "nemo-parakeet-ctc-0.6b": "istupakov/parakeet-ctc-0.6b-onnx",
    "nemo-parakeet-rnnt-0.6b": "istupakov/parakeet-rnnt-0.6b-onnx",
    "nemo-parakeet-tdt-0.6b-v2": "istupakov/parakeet-tdt-0.6b-v2-onnx",
    "nemo-parakeet-tdt-0.6b-v3": "istupakov/parakeet-tdt-0.6b-v3-onnx",
    "nemo-canary-1b-v2": "istupakov/canary-1b-v2-onnx",
    "whisper-base": "istupakov/whisper-base-onnx",
    # Moonshine ASR (Useful Sensors) — tiny / base + per-language variants. Each
    # checkpoint is already language-specialized at training time, so we route
    # each short alias to its own repo.
    "moonshine-tiny": "onnx-community/moonshine-tiny-ONNX",
    "moonshine-base": "onnx-community/moonshine-base-ONNX",
    "moonshine-tiny-zh": "onnx-community/moonshine-tiny-zh-ONNX",
    "moonshine-tiny-ja": "onnx-community/moonshine-tiny-ja-ONNX",
    "moonshine-tiny-ko": "onnx-community/moonshine-tiny-ko-ONNX",
    "moonshine-tiny-ar": "onnx-community/moonshine-tiny-ar-ONNX",
    "moonshine-tiny-vi": "onnx-community/moonshine-tiny-vi-ONNX",
    "moonshine-base-zh": "onnx-community/moonshine-base-zh-ONNX",
    "moonshine-base-ja": "onnx-community/moonshine-base-ja-ONNX",
    "moonshine-base-ko": "onnx-community/moonshine-base-ko-ONNX",
    # transformers>=4.57 re-exports (Apr 2026) — declare attention-mask inputs
    # the Moonshine adapter now feeds. Only uk + fr are ONNX-converted so far
    # (es exists upstream only as safetensors).
    "moonshine-tiny-uk": "onnx-community/moonshine-tiny-uk-ONNX",
    "moonshine-tiny-fr": "onnx-community/moonshine-tiny-fr-ONNX",
    "cohere-transcribe": "onnx-community/cohere-transcribe-03-2026-ONNX",
    # IBM Granite Speech — Conformer audio encoder + Q-Former projector +
    # Granite LLM decoder. The 4.0-1b release lands at this onnx-community
    # repo; future granite_speech variants get their own aliases here.
    "granite-4.0-1b-speech": "onnx-community/granite-4.0-1b-speech-ONNX",
    # DataoceanAI Dolphin (sherpa-onnx CTC export) — int8 repos are the default
    # (the int8 repo ships only ``model.int8.onnx``, so catalog entries select
    # quantization="int8").
    "dolphin-base-ctc": "csukuangfj/sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02",
    "dolphin-small-ctc": "csukuangfj/sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02",
    # icefall / sherpa-onnx offline Zipformer transducer packs.
    "zipformer-en": "csukuangfj/sherpa-onnx-zipformer-en-2023-06-26",
    "silero": "istupakov/silero-vad-onnx",
    # Speaker-embedding model used by the diarizer (CC-BY-4.0; VoxCeleb-trained).
    "wespeaker-voxceleb-resnet34-LM": "Wespeaker/wespeaker-voxceleb-resnet34-LM",
}


T = TypeVar("T", bound=_ModelImplementation)


class Resolver(Generic[T]):
    """Model resolver."""

    offline: bool = False
    local_dir: Path | None = None
    repo_id: str | None = None

    def __init__(  # noqa: C901
        self,
        model_type: type[T] | dict[str, type[T]],
        model: str | None = None,
        local_dir: str | Path | None = None,
        *,
        offline: bool | None = None,
        progress_callback: ProgressCallback | None = None,
    ):
        """Create model loader."""
        self.progress_callback = progress_callback

        if model is not None:
            if "/" in model:
                self.repo_id = model
            elif model in model_repos:
                self.repo_id = model_repos[model]
            elif not (isinstance(model_type, dict) and model in model_type):
                raise ModelNotSupportedError(model)

        if local_dir is not None:
            self.local_dir = Path(local_dir)
            if self.local_dir.exists():
                self.offline = True
                if not self.local_dir.is_dir():
                    raise ModelPathNotDirectoryError(self.local_dir)

        if offline is not None:
            self.offline = offline

        if not (self.repo_id or (self.offline and self.local_dir)):
            raise NoModelNameOrPathSpecifiedError

        if isinstance(model_type, type):
            self.model_type = model_type
        elif model in model_type:
            self.model_type = model_type[model]
        else:
            with self.resolve_config().open("rt", encoding="utf-8") as f:
                config = json.load(f)

            config_model_type: str = config.get("model_type")
            if "/" in config_model_type or config_model_type not in model_type:
                raise InvalidModelTypeInConfigError(config_model_type)
            self.model_type = model_type[config_model_type]

    def _download_config(self, *, local_files_only: bool) -> Path:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        assert self.repo_id is not None
        return Path(
            hf_hub_download(self.repo_id, "config.json", local_dir=self.local_dir, local_files_only=local_files_only)  # nosec
        )

    def _download_model(self, quantization: str | None, *, local_files_only: bool) -> Path:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        files = list(self.model_type._get_model_files(quantization).values())
        files = [
            *files,
            *(file.removeprefix("**/") for file in files if file.startswith("**/")),
        ]
        files = [
            "config.json",
            "config.yaml",
            *files,
            # Windows-safe: use ``PurePosixPath`` so the generated pattern keeps
            # forward slashes. HF's ``snapshot_download`` allow_patterns are
            # ``fnmatch``-ed against POSIX-style repo paths; constructing the
            # pattern via the host's ``Path`` on Windows produces backslashes
            # that never match (silently skips the ``.onnx_data`` sidecar
            # downloads for any model with >2 GB ONNX weights, e.g.
            # Xenova/whisper-large-v3, which then fails at load with
            # ``External data path validation failed`` from ORT).
            *(str(path.with_suffix(".onnx?data")) for file in files if (path := PurePosixPath(file)).suffix == ".onnx"),
        ]

        assert self.repo_id is not None
        patch_ctx = (
            hf_progress_patch(self.progress_callback)
            if self.progress_callback is not None
            else contextlib.nullcontext()
        )
        with patch_ctx:
            return Path(
                snapshot_download(  # nosec
                    self.repo_id,
                    local_dir=self.local_dir,
                    local_files_only=local_files_only,
                    allow_patterns=files,
                )
            )

    def _resolve_model_files(self, path: Path, quantization: str | None) -> dict[str, Path]:
        files = self.model_type._get_model_files(quantization)
        if Path(path, "config.json").exists():
            files |= {"config": "config.json"}

        def find(filename: str) -> Path:
            files = list(path.glob(filename))
            if len(files) > 1:
                raise MoreThanOneModelFileFoundError(filename, path)
            if len(files) == 0:
                orig_path = Path(filename)
                if orig_path.suffix == ".onnx":
                    files = list(path.glob(str(orig_path.with_suffix(".ort"))))
                    if len(files) == 1 and files[0].is_file():
                        return files[0]
                raise ModelFileNotFoundError(filename, path)
            if not files[0].is_file():
                raise ModelFileNotFoundError(filename, path)
            return files[0]

        return {key: find(filename) for key, filename in files.items()}

    def resolve_config(self) -> Path:
        """Resolve path to model config."""
        try:
            if self.local_dir:
                config_path = Path(self.local_dir, "config.json")
                if not config_path.is_file():
                    raise ModelFileNotFoundError(config_path.name, self.local_dir)
                return config_path

            return self._download_config(local_files_only=True)
        except FileNotFoundError:
            if self.offline:
                raise
            return self._download_config(local_files_only=False)

    def resolve_model(self, *, quantization: str | None = None) -> dict[str, Path]:
        """Resolve paths to model files."""
        try:
            return self._resolve_model_files(
                self.local_dir or self._download_model(quantization, local_files_only=True), quantization
            )
        except FileNotFoundError:
            if self.offline:
                raise
            return self._resolve_model_files(self._download_model(quantization, local_files_only=False), quantization)
