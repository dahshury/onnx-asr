"""Whisper exported via ``onnxruntime`` beam-search — single-ONNX export.

The ORT-side beam-search export bundles encoder+decoder+beam-search into a
single ``.onnx`` graph that takes ``input_features`` + decoding hyperparams
and outputs ``sequences``. Faster startup and a smaller artifact but no
access to per-step state — so streaming / word timestamps aren't available
on this backend (subclass inherits the base's no-op for those).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt
from onnxruntime import OrtValue

from onnx_asr.asr import Preprocessor
from onnx_asr.models.whisper._base import _Whisper
from onnx_asr.onnx import OnnxSessionOptions
from onnx_asr.utils import is_int32_array


class WhisperOrt(_Whisper):
    """Whisper (exported via onnxruntime) model implementation."""

    def __init__(
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        self._model = rt.InferenceSession(model_files["model"], **onnx_options)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"whisper-*_beamsearch{suffix}.onnx"} | _Whisper._get_model_files(quantization)

    @property
    def _preprocessor_name(self) -> str:
        return f"whisper{self.config.get('features_size', 80)}"

    def _decoding(
        self, input_features: OrtValue, tokens: npt.NDArray[np.int64], max_length: int = 448
    ) -> npt.NDArray[np.int64]:
        (sequences,) = self._model.run(
            ["sequences"],
            {
                "input_features": input_features,
                "max_length": [max_length],
                "min_length": [0],
                "num_beams": [1],
                "num_return_sequences": [1],
                "length_penalty": [1.0],
                "repetition_penalty": [1.0],
                "decoder_input_ids": tokens.astype(np.int32),
            },
        )
        assert is_int32_array(sequences)
        return sequences[:, 0, :].astype(np.int64)
