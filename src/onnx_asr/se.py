"""Base Speaker Embedding classes."""

from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from onnx_asr.model_base import _ModelImplementation


class SpeakerEmbedding(_ModelImplementation, Protocol):
    """Speaker Embedding protocol.

    Inherits :class:`_ModelImplementation`'s static infrastructure surface
    (``_get_model_files`` / ``_get_excluded_providers``); adds the embedding
    runtime API.
    """

    @staticmethod
    def _get_sample_rate() -> Literal[8_000, 16_000]:
        return 16_000

    def embedding(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float32]:
        """Compute speaker embedding."""
        ...
