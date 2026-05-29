"""DataoceanAI Dolphin multilingual CTC ASR (sherpa-onnx export).

Dolphin is a joint CTC/Attention E-Branchformer ASR covering ~40 Eastern
languages (East / South / South-East Asia + Middle East — including Arabic,
Hindi and Chinese) plus 22 Chinese dialects. We use the **CTC head only**:
a single self-contained ONNX graph (``model.onnx`` / ``model.int8.onnx``)
exported by the k2-fsa ``sherpa-onnx`` project.

Two things make it unlike the other NeMo/Kaldi CTC adapters:

* **No ``config.json``.** The per-mel-bin CMVN statistics (``mean`` and
  ``invstd``, 80 floats each) ride in the ONNX ``custom_metadata_map``. The
  shared kaldi fbank preprocessor does *not* apply them, so we do —
  ``x = (fbank - mean) * invstd`` — right before the encoder.
* **Blank is ``<blank>`` (id 0)**, not the ``<blk>`` the base vocab loader
  looks for, and the log-prob output tensor is (mis)named ``lob_probs`` in
  the published export. We resolve both by id/shape rather than by name.

Input ``x`` ``(N, T, 80)`` time-major kaldi fbank + ``x_len`` ``(N,)``;
output log-probs ``(N, T', vocab=40002)`` + ``log_probs_len``. Greedy CTC
collapse is handled by :class:`~onnx_asr.asr._AsrWithCtcDecoding`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import ModelCapabilities, Preprocessor, _AsrWithCtcDecoding
from onnx_asr.onnx import OnnxSessionOptions
from onnx_asr.utils import is_float32_array, is_int64_array


class DolphinCtc(_AsrWithCtcDecoding):
    """DataoceanAI Dolphin CTC model (single-graph sherpa-onnx export)."""

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(is_multilingual=True)

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        self._model = rt.InferenceSession(model_files["model"], **onnx_options)

        # Dolphin's blank token is ``<blank>`` (id 0); the base ``_AsrWithDecoding``
        # vocab loader only auto-detects ``<blk>``, so set it explicitly.
        self._blank_idx = 0

        # Per-mel-bin CMVN stats live in the ONNX metadata (no config.json is
        # shipped). Parse the comma-separated 80-float vectors once.
        meta = self._model.get_modelmeta().custom_metadata_map
        self._mean = np.fromstring(meta["mean"], sep=",", dtype=np.float32)
        self._invstd = np.fromstring(meta["invstd"], sep=",", dtype=np.float32)

        # Output tensor names resolved by rank — the log-prob tensor is 3-D
        # (and literally named ``lob_probs`` in the export), the length is 1-D.
        outputs = self._model.get_outputs()
        self._logits_name = next(o.name for o in outputs if len(o.shape) == 3)
        self._logits_len_name = next(o.name for o in outputs if len(o.shape) == 1)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        # The published repos ship a flat root with ``model.onnx`` /
        # ``model.int8.onnx``. ``?`` matches the ``.`` (or ``_``) before the
        # quant suffix, mirroring the Moonshine/Kaldi globs.
        suffix = "?" + quantization if quantization else ""
        return {
            "model": f"model{suffix}.onnx",
            "vocab": "tokens.txt",
        }

    @property
    def _preprocessor_name(self) -> str:
        # 80-dim kaldi fbank (same extractor the Kaldi/Vosk transducers use);
        # CMVN is applied here in :meth:`_encode`, not by the preprocessor.
        return "kaldi"

    @property
    def _subsampling_factor(self) -> int:
        # Used only for timestamp spacing; Dolphin downsamples ~4x.
        return self.config.get("subsampling_factor", 4)

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        x = ((features - self._mean) * self._invstd).astype(np.float32)
        log_probs, log_probs_len = self._model.run(
            [self._logits_name, self._logits_len_name], {"x": x, "x_len": features_lens}
        )
        assert is_float32_array(log_probs)
        assert is_int64_array(log_probs_len)
        return log_probs, log_probs_len
