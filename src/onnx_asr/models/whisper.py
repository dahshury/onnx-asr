"""Whisper model implementations."""

import json
import typing
from abc import abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt
from onnxruntime import OrtValue

from onnx_asr.asr import BaseAsr, Preprocessor, TimestampedResult
from onnx_asr.onnx import OnnxSessionOptions, TensorRtOptions, get_onnx_device
from onnx_asr.utils import is_float32_array, is_int32_array


@typing.no_type_check
def bytes_to_unicode() -> dict[int, str]:
    """Magic func copied from transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))  # noqa: B905


class _Whisper(BaseAsr):
    def __init__(
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)

        with model_files["vocab"].open("rt", encoding="utf-8") as f:
            self._tokens: dict[str, int] = json.load(f)

        with model_files["added_tokens"].open("rt", encoding="utf-8") as f:
            self._tokens |= json.load(f)

        self._vocab = {id: token for token, id in self._tokens.items()}
        self._bos_token_id = self._tokens["<|startoftranscript|>"]
        self._eos_token_id = self._tokens["<|endoftext|>"]
        self._byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
        self._transcribe_input = np.array(
            [
                [
                    self._bos_token_id,
                    self._eos_token_id,
                    self._tokens["<|transcribe|>"],
                    self._tokens["<|notimestamps|>"],
                ]
            ],
            dtype=np.int64,
        )
        self._detect_lang_input = np.array([[self._bos_token_id]], dtype=np.int64)

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        return TensorRtOptions.get_provider_names()

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        return {"vocab": "vocab.json", "added_tokens": "added_tokens.json"}

    def _encode(self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]) -> OrtValue:
        input_features, _ = self._preprocessor(waveforms, waveforms_len)
        return OrtValue.ortvalue_from_numpy(input_features)

    @abstractmethod
    def _decoding(
        self,
        input_features: OrtValue,
        tokens: npt.NDArray[np.int64],
        max_length: int = 448,
        *,
        num_beams: int = 1,
        length_penalty: float = 1.0,
    ) -> npt.NDArray[np.int64]: ...

    def _decode_tokens(self, tokens: npt.NDArray[np.int64]) -> TimestampedResult:
        text = "".join(token for id in tokens if (token := self._vocab[id]) and not token.startswith("<|"))
        return TimestampedResult(
            bytearray([self._byte_decoder[c] for c in text]).decode("utf-8", errors="replace").removeprefix(" ")
        )

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        input_encoding = self._encode(waveforms, waveforms_len)
        input_tokens = np.repeat(self._transcribe_input, len(waveforms), axis=0)

        beam_size_raw = kwargs.get("beam_size")
        num_beams = max(1, int(beam_size_raw)) if isinstance(beam_size_raw, int) else 1
        length_penalty_raw = kwargs.get("length_penalty")
        length_penalty = float(length_penalty_raw) if isinstance(length_penalty_raw, (int, float)) else 1.0

        language = kwargs.get("language")
        if language:
            input_tokens[:, 1] = self._tokens[f"<|{language}|>"]
        else:
            # Language detection is always greedy — max_length=3 produces a single decoder step.
            input_tokens_detect_lang = np.repeat(self._detect_lang_input, len(waveforms), axis=0)
            input_tokens[:, 1] = self._decoding(input_encoding, input_tokens_detect_lang, 3)[:, 1]

        return map(
            self._decode_tokens,
            self._decoding(input_encoding, input_tokens, num_beams=num_beams, length_penalty=length_penalty),
        )


class WhisperOrt(_Whisper):
    """Whisper (exported via onnxruntime) model implementation."""

    def __init__(  # noqa: D107
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
        self,
        input_features: OrtValue,
        tokens: npt.NDArray[np.int64],
        max_length: int = 448,
        *,
        num_beams: int = 1,
        length_penalty: float = 1.0,
    ) -> npt.NDArray[np.int64]:
        (sequences,) = self._model.run(
            ["sequences"],
            {
                "input_features": input_features,
                "max_length": [max_length],
                "min_length": [0],
                "num_beams": [num_beams],
                "num_return_sequences": [1],
                "length_penalty": [length_penalty],
                "repetition_penalty": [1.0],
                "decoder_input_ids": tokens.astype(np.int32),
            },
        )
        assert is_int32_array(sequences)
        return sequences[:, 0, :].astype(np.int64)


class WhisperHf(_Whisper):
    """Whisper (exported via optimum) model implementation."""

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        self._encoder = rt.InferenceSession(model_files["encoder"], **onnx_options)
        self._decoder = rt.InferenceSession(model_files["decoder"], **onnx_options)
        self._device_type, self._device_id = get_onnx_device(self._encoder)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"**/encoder_model{suffix}.onnx",
            "decoder": f"**/decoder_model_merged{suffix}.onnx",
        } | _Whisper._get_model_files(suffix)

    @property
    def _preprocessor_name(self) -> str:
        return f"whisper{self.config.get('num_mel_bins', 80)}"

    def _encode(self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]) -> OrtValue:
        input_features = super()._encode(waveforms, waveforms_len)
        binding = self._encoder.io_binding()
        binding.bind_ortvalue_input("input_features", input_features)
        binding.bind_output("last_hidden_state", self._device_type, self._device_id)
        self._encoder.run_with_iobinding(binding)
        last_hidden_state: OrtValue = binding.get_outputs()[0]
        return last_hidden_state

    def _create_state(self) -> dict[str, OrtValue]:
        return {
            x.name: OrtValue.ortvalue_from_numpy(np.zeros((0, x.shape[1], 0, x.shape[3]), dtype=np.float32))
            for x in self._decoder.get_inputs()
            if x.name.startswith("past_key_values.")
        }

    def _decode(
        self,
        tokens: npt.NDArray[np.int64],
        prev_state: dict[str, OrtValue],
        encoder_out: OrtValue,
    ) -> tuple[npt.NDArray[np.float32], dict[str, OrtValue]]:
        use_cache = any(x.shape()[0] for x in prev_state.values())

        binding = self._decoder.io_binding()
        binding.bind_cpu_input("input_ids", tokens[:, -1:] if use_cache else tokens)
        binding.bind_ortvalue_input("encoder_hidden_states", encoder_out)
        binding.bind_output("logits")
        if prev_state:
            binding.bind_cpu_input("use_cache_branch", np.array([use_cache]))
            for key, value in prev_state.items():
                binding.bind_ortvalue_input(key, value)
                binding.bind_output(key.replace("past_key_values.", "present."), self._device_type, self._device_id)

        self._decoder.run_with_iobinding(binding)
        outputs = binding.get_outputs()
        logits = outputs[0].numpy()
        assert is_float32_array(logits)
        return logits, {
            key: next_value if next_value.shape()[0] else prev_value
            for (key, prev_value), next_value in zip(prev_state.items(), outputs[1:], strict=True)
        }

    def _decoding(
        self,
        input_features: OrtValue,
        tokens: npt.NDArray[np.int64],
        max_length: int = 448,
        *,
        num_beams: int = 1,
        length_penalty: float = 1.0,
    ) -> npt.NDArray[np.int64]:
        if num_beams <= 1:
            return self._greedy_decode(input_features, tokens, max_length)

        if tokens.shape[0] == 1:
            return self._beam_search(input_features, tokens, max_length, num_beams, length_penalty)

        # Batched beam search: process each audio item independently and pad results.
        enc_np = input_features.numpy()
        per_item: list[npt.NDArray[np.int64]] = []
        for i in range(tokens.shape[0]):
            single_enc = OrtValue.ortvalue_from_numpy(enc_np[i : i + 1])
            single_tokens = tokens[i : i + 1]
            per_item.append(self._beam_search(single_enc, single_tokens, max_length, num_beams, length_penalty))

        common_len = max(item.shape[1] for item in per_item)
        out = np.full((tokens.shape[0], common_len), self._eos_token_id, dtype=np.int64)
        for i, item in enumerate(per_item):
            out[i, : item.shape[1]] = item[0]
        return out

    def _greedy_decode(
        self, input_features: OrtValue, tokens: npt.NDArray[np.int64], max_length: int
    ) -> npt.NDArray[np.int64]:
        state = self._create_state()
        for _ in range(tokens.shape[-1], max_length):
            logits, state = self._decode(tokens, state, input_features)
            next_tokens = logits[:, -1].argmax(axis=-1)
            next_tokens[tokens[:, -1] == self._eos_token_id] = self._eos_token_id
            tokens = np.hstack((tokens, next_tokens[:, None]))
            if (tokens[:, -1] == self._eos_token_id).all():
                break

        return tokens

    @staticmethod
    def _reorder_state(state: dict[str, OrtValue], indices: npt.NDArray[np.int64]) -> dict[str, OrtValue]:
        """Reorder KV-cache entries along the batch (beam) axis."""
        return {key: OrtValue.ortvalue_from_numpy(value.numpy()[indices]) for key, value in state.items()}

    def _beam_search(  # noqa: C901
        self,
        input_features: OrtValue,
        tokens: npt.NDArray[np.int64],
        max_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> npt.NDArray[np.int64]:
        """Run standard beam search over the merged decoder.

        Replicates the prompt to ``num_beams``, at each step takes the top
        ``num_beams`` continuations across all (beam, token) pairs, reorders
        the KV cache to follow the surviving beams, and finalizes beams that
        emit EOS using the Wu et al. (2016) length-penalty norm.

        Operates on a single audio item (``tokens.shape[0] == 1``); the
        public ``_decoding`` wrapper loops over a batch.
        """
        assert tokens.shape[0] == 1, "_beam_search expects a single audio item"
        eos = self._eos_token_id
        prompt_len = tokens.shape[1]
        n_new_max = max_length - prompt_len
        if n_new_max <= 0:
            return tokens

        # Replicate encoder hidden state and prompt to num_beams.
        enc_np = input_features.numpy()
        enc_replicated = OrtValue.ortvalue_from_numpy(np.repeat(enc_np, num_beams, axis=0))
        beam_tokens = np.repeat(tokens, num_beams, axis=0)

        # Seed cumulative log-probs: beam 0 carries the prompt, others -inf so
        # the first step picks num_beams distinct first continuations.
        cum_logprobs = np.full(num_beams, -np.inf, dtype=np.float32)
        cum_logprobs[0] = 0.0

        state = self._create_state()
        finished: list[tuple[float, npt.NDArray[np.int64]]] = []

        def length_norm(seq_len_excl_prompt: int) -> float:
            # Wu et al. 2016 length penalty.
            return float(((5.0 + max(seq_len_excl_prompt, 1)) / 6.0) ** length_penalty)

        for _ in range(n_new_max):
            logits, state = self._decode(beam_tokens, state, enc_replicated)
            last_logits = logits[:, -1, :]

            # Numerically stable log-softmax along vocab axis.
            shifted = last_logits - last_logits.max(axis=-1, keepdims=True)
            log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

            combined = cum_logprobs[:, None] + log_probs
            vocab_size = combined.shape[1]
            flat = combined.reshape(-1)

            # Pull 2*num_beams candidates to ensure enough non-EOS survivors.
            k = min(2 * num_beams, flat.size)
            partition = np.argpartition(-flat, k - 1)[:k]
            order = partition[np.argsort(-flat[partition])]

            source = (order // vocab_size).astype(np.int64)
            picks = (order % vocab_size).astype(np.int64)

            new_sources: list[int] = []
            new_tokens: list[int] = []
            new_scores: list[float] = []

            for src, tok, abs_idx in zip(source, picks, order, strict=True):
                score = float(flat[abs_idx])
                if int(tok) == eos:
                    seq = np.concatenate([beam_tokens[int(src)], np.array([eos], dtype=np.int64)])
                    norm = length_norm(seq.size - prompt_len)
                    finished.append((score / norm, seq))
                elif len(new_sources) < num_beams:
                    new_sources.append(int(src))
                    new_tokens.append(int(tok))
                    new_scores.append(score)
                if len(new_sources) >= num_beams and len(finished) >= num_beams:
                    break

            if len(finished) >= num_beams or not new_sources:
                break

            survivor_idx = np.array(new_sources, dtype=np.int64)
            state = self._reorder_state(state, survivor_idx)
            beam_tokens = np.column_stack(
                [beam_tokens[survivor_idx], np.array(new_tokens, dtype=np.int64)]
            )
            cum_logprobs = np.array(new_scores, dtype=np.float32)

        # Finalize any in-flight beams.
        for i in range(beam_tokens.shape[0]):
            seq = beam_tokens[i]
            norm = length_norm(seq.size - prompt_len)
            finished.append((float(cum_logprobs[i]) / norm, seq))

        if not finished:
            return beam_tokens[:1]

        finished.sort(key=lambda item: -item[0])
        best = finished[0][1]
        return best[None, :]
