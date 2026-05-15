"""Word-level timestamps for Whisper via cross-attention DTW (plan #17).

Algorithm port of ``openai-whisper/whisper/timing.py`` (MIT) into pure NumPy
so onnx-asr can produce word boundaries from the cross-attention tensors
that the ``onnx-community/whisper-*_timestamped`` exports expose.

Pipeline:

1. Collect per-step cross-attention from selected ``alignment_heads`` while
   the autoregressive decoder generates ``text_tokens``.
2. Stack into a ``(num_heads, num_tokens, num_encoder_frames)`` tensor and
   crop the encoder dim to ``num_audio_frames // 2`` (Whisper downsamples).
3. Soft-normalize across the token axis (subtract mean, divide by std).
4. Median-filter along the encoder-time axis with a width-7 window.
5. Mean over heads → 2D ``(tokens, time)`` alignment matrix.
6. DTW on the negated matrix → monotonic ``(text_idx, time_idx)`` path.
7. Group tokens into words via the GPT-2 byte-decoder (split on the ``Ġ``
   space marker — same logic as openai-whisper's ``split_tokens_on_spaces``).
8. For each word, the start/end times are the time indices where the path
   first enters / leaves the word's token range, divided by
   ``TOKENS_PER_SECOND`` (50, since each "audio token" = 20 ms).

The :data:`_ALIGNMENT_HEADS` table is copied verbatim from openai-whisper.
"""

from __future__ import annotations

import base64
import gzip
import string
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

#: Whisper's audio frontend: 50 audio frames per second post-encoder downsample.
TOKENS_PER_SECOND = 50

#: Alignment-heads tables copied verbatim from ``openai-whisper/whisper/__init__.py``
#: — each entry is a base85-encoded gzip of a flat bool array over
#: ``(num_decoder_layers, num_decoder_attention_heads)``. The heads marked
#: ``True`` correlate most strongly with word-level alignment.
_ALIGNMENT_HEADS: dict[str, bytes] = {
    "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
    "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
    "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
    "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
    "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
    "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
    "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
    "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
    "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
    "large-v2": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
    "large-v3": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
    "large-v3-turbo": b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`",
    "turbo": b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`",
}

#: ``(num_decoder_layers, num_decoder_attention_heads)`` for each Whisper
#: model size. Used to pick an alignment_heads entry by config.
_MODEL_SIZE_BY_DIMS: dict[tuple[int, int], str] = {
    (4, 6): "tiny",
    (6, 8): "base",
    (12, 12): "small",
    (24, 16): "medium",
    (32, 20): "large-v3",
    (4, 20): "large-v3-turbo",
}


def decode_alignment_heads(dump: bytes, num_layers: int, num_heads: int) -> npt.NDArray[np.bool_]:
    """Decode a base85-gzipped flat bool array into ``(num_layers, num_heads)``.

    Mirrors :meth:`Whisper.set_alignment_heads` in ``openai-whisper``.
    """
    raw = gzip.decompress(base64.b85decode(dump))
    arr: npt.NDArray[np.bool_] = np.frombuffer(raw, dtype=bool).reshape(num_layers, num_heads).copy()
    return arr


def lookup_alignment_heads(num_layers: int, num_heads: int, vocab_size: int) -> npt.NDArray[np.bool_]:
    """Pick an alignment_heads mask by Whisper model size.

    Falls back to "all heads in the second half of layers" (the default
    :meth:`Whisper.alignment_heads` would build if no override was set) when
    the dimensions don't match any known model.

    Args:
        num_layers: ``config.decoder_layers``.
        num_heads:  ``config.decoder_attention_heads``.
        vocab_size: ``config.vocab_size`` — distinguishes ``-en`` (51 864)
                    from multilingual (51 865 / 51 866) variants.

    """
    size = _MODEL_SIZE_BY_DIMS.get((num_layers, num_heads))
    if size:
        # English-only variants of tiny/base/small/medium have a separate entry.
        english_only = vocab_size == 51_864
        key = f"{size}.en" if english_only and f"{size}.en" in _ALIGNMENT_HEADS else size
        if key in _ALIGNMENT_HEADS:
            return decode_alignment_heads(_ALIGNMENT_HEADS[key], num_layers, num_heads)
    # Fallback: heads in the upper half of layers.
    mask = np.zeros((num_layers, num_heads), dtype=bool)
    mask[num_layers // 2 :] = True
    return mask


def median_filter_1d(x: npt.NDArray[np.float32], filter_width: int) -> npt.NDArray[np.float32]:
    """Median filter along the last axis with reflect-padding.

    Pure-numpy equivalent of openai-whisper's torch + triton implementation
    (``timing.py:19-54``). ``filter_width`` must be odd.
    """
    if filter_width <= 1:
        return x
    if x.shape[-1] <= filter_width // 2:
        return x
    if filter_width % 2 == 0:
        msg = f"filter_width must be odd, got {filter_width}"
        raise ValueError(msg)

    pad = filter_width // 2
    # Reflect-pad along last axis only.
    padding = [(0, 0)] * (x.ndim - 1) + [(pad, pad)]
    padded = np.pad(x, padding, mode="reflect")
    # Sliding window view: shape (..., new_length, filter_width).
    windows = np.lib.stride_tricks.sliding_window_view(padded, filter_width, axis=-1)
    # Sort along the window dim then pick the median (matches openai-whisper).
    sorted_windows = np.sort(windows, axis=-1)
    return sorted_windows[..., filter_width // 2]


def dtw(cost_input: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Dynamic-time-warp a 2D cost matrix; return ``(text_indices, time_indices)``.

    Port of ``openai-whisper/whisper/timing.py:82-105`` ``dtw_cpu``. We walk
    a ``(N+1, M+1)`` cost lattice computing the minimum-cost path from
    ``(0,0)`` to ``(N,M)`` using moves diag / down / right, then backtrace
    to recover the monotonic index pairs.
    """
    n_text, n_time = cost_input.shape
    cost = np.full((n_text + 1, n_time + 1), np.inf, dtype=np.float64)
    trace = -np.ones((n_text + 1, n_time + 1), dtype=np.int8)
    cost[0, 0] = 0.0

    for j in range(1, n_time + 1):
        for i in range(1, n_text + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]
            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2
            cost[i, j] = cost_input[i - 1, j - 1] + c
            trace[i, j] = t

    # Backtrace (matches openai-whisper/timing.py:57-79).
    trace[0, :] = 2
    trace[:, 0] = 1
    i = n_text
    j = n_time
    text_indices: list[int] = []
    time_indices: list[int] = []
    while i > 0 or j > 0:
        text_indices.append(i - 1)
        time_indices.append(j - 1)
        step = int(trace[i, j])
        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        elif step == 2:
            j -= 1
        else:
            break
    return np.asarray(text_indices[::-1], dtype=np.int64), np.asarray(time_indices[::-1], dtype=np.int64)


def split_tokens_into_words(
    tokens: list[int],
    decode_one: Callable[[list[int]], str],
    *,
    eot_id: int,
    language: str | None = None,
) -> tuple[list[str], list[list[int]]]:
    """Group token IDs into words using the same byte-decoder logic as Whisper.

    ``decode_one`` is a callback that renders a list of token IDs into
    (possibly partial / replacement-char) text — passing the model's
    own byte-decoder. Tokens are first grouped at unicode boundaries
    (handling multi-byte characters), then merged into words on space /
    punctuation rules (matching ``Tokenizer.split_tokens_on_spaces``).

    For CJK languages (``zh``, ``ja``, ...) we stop at unicode boundaries
    since spaces aren't word delimiters there.
    """
    # Stage 1: split on unicode boundaries (handles multi-byte tokens that
    # only render to a valid char when combined).
    decoded_full = decode_one(tokens)
    replacement = "�"
    subwords: list[str] = []
    subword_tokens: list[list[int]] = []
    current: list[int] = []
    offset = 0
    for tok in tokens:
        current.append(tok)
        decoded = decode_one(current)
        if (replacement not in decoded) or (
            offset + decoded.index(replacement) < len(decoded_full)
            and decoded_full[offset + decoded.index(replacement)] == replacement
        ):
            subwords.append(decoded)
            subword_tokens.append(current)
            current = []
            offset += len(decoded)

    if language in {"zh", "ja", "th", "lo", "my", "yue"}:
        return subwords, subword_tokens

    # Stage 2: collapse subwords into space-delimited words.
    words: list[str] = []
    word_tokens: list[list[int]] = []
    for subword, ids in zip(subwords, subword_tokens, strict=True):
        is_special = ids[0] >= eot_id
        is_space_prefixed = subword.startswith(" ")
        is_punct = subword.strip() in string.punctuation
        if is_special or is_space_prefixed or is_punct or not words:
            words.append(subword)
            word_tokens.append(list(ids))
        else:
            words[-1] = words[-1] + subword
            word_tokens[-1].extend(ids)
    return words, word_tokens


@dataclass(frozen=True)
class WordTiming:
    """Per-word alignment result: rendered text + start/end seconds + token IDs."""

    word: str
    start: float
    end: float
    tokens: tuple[int, ...]


def align_words(
    cross_attentions: npt.NDArray[np.float32],
    alignment_heads: npt.NDArray[np.bool_],
    *,
    text_tokens: list[int],
    decode_one: Callable[[list[int]], str],
    eot_id: int,
    prompt_length: int,
    num_audio_frames: int,
    language: str | None = None,
    medfilt_width: int = 7,
    qk_scale: float = 1.0,
) -> list[WordTiming]:
    """Run the full word-alignment pipeline on collected cross-attentions.

    Args:
        cross_attentions: Stacked per-layer/per-head cross-attention from
            the decoder. Shape ``(num_layers, num_heads, num_decoder_tokens,
            num_encoder_frames)``. Element ``[l, h, i, j]`` is layer ``l``,
            head ``h``'s attention weight from decoder token ``i`` to
            encoder frame ``j``.
        alignment_heads: Bool mask over ``(num_layers, num_heads)`` selecting
            which heads to average. See :func:`lookup_alignment_heads`.
        text_tokens: The generated text token IDs (excluding prompt prefix
            but including the trailing EOT).
        decode_one: Callback rendering a list of token IDs as text. Passed
            to :func:`split_tokens_into_words`.
        eot_id: Whisper's ``<|endoftext|>`` token id.
        prompt_length: Number of prompt tokens at the start of the decoder
            input (e.g. ``[SOT, lang, transcribe, notimestamps]`` → 4). These
            tokens occupy the leading rows of ``cross_attentions``; we strip
            them before DTW.
        num_audio_frames: ``num_samples // HOP_LENGTH`` from the original
            audio (before encoder 2x downsample).
        language: Whisper language code; affects word-splitting policy.
        medfilt_width: Width of the median filter along the time axis.
        qk_scale: Pre-softmax scaling factor (defaults to 1.0).

    """
    if len(text_tokens) == 0:
        return []

    # Select heads → shape (num_selected_heads, num_tokens, num_frames).
    layers, heads = np.where(alignment_heads)
    if layers.size == 0:
        return []
    weights = np.stack([cross_attentions[lyr, hd] for lyr, hd in zip(layers, heads, strict=True)])

    # Crop to half the audio frame count (encoder downsamples by 2).
    weights = weights[:, :, : num_audio_frames // 2]

    # Softmax across the time axis.
    scaled = weights.astype(np.float32) * float(qk_scale)
    scaled -= scaled.max(axis=-1, keepdims=True)  # numerical stability
    exp = np.exp(scaled)
    weights = exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)

    # Normalize across the token axis: subtract mean, divide by std.
    mean = weights.mean(axis=-2, keepdims=True)
    std = weights.std(axis=-2, keepdims=True)
    weights = (weights - mean) / np.maximum(std, 1e-9)

    # Median filter along the time axis.
    weights = median_filter_1d(weights, medfilt_width)

    # Mean across heads → 2D (num_tokens, num_frames) alignment cost.
    matrix = weights.mean(axis=0)

    # Strip everything before ``prompt_length - 1`` and the trailing EOT row.
    # Including row ``prompt_length - 1`` (the ``<|notimestamps|>`` slot)
    # gives DTW an extra leading anchor — matches openai-whisper's
    # ``matrix[len(sot_sequence):-1]`` slice where ``sot_sequence`` is the
    # 3-token ``[SOT, lang, transcribe]`` (not including ``<|notimestamps|>``).
    anchor = max(0, prompt_length - 1)
    matrix = matrix[anchor : prompt_length + len(text_tokens) - 1]
    if matrix.shape[0] == 0:
        return []

    text_indices, time_indices = dtw(-matrix.astype(np.float64))

    # Group tokens into words using the byte-decoder.
    words, word_tokens = split_tokens_into_words(text_tokens, decode_one, eot_id=eot_id, language=language)
    if len(word_tokens) <= 1:
        return []

    # ``word_boundaries[k]`` = cumulative token count up to start of word k.
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0)).astype(np.int64)

    # Identify the DTW path positions where a NEW text-token transition occurs;
    # the time index at those positions is the word-onset / -offset frame.
    jumps_mask = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times_frames = time_indices[jumps_mask]
    jump_times_s = jump_times_frames.astype(np.float64) / TOKENS_PER_SECOND

    # Defensive: jump_times length must cover all word boundaries.
    if len(jump_times_s) <= int(word_boundaries[-1]):
        return []

    start_times = jump_times_s[word_boundaries[:-1]]
    end_times = jump_times_s[word_boundaries[1:]]
    # ``words`` / ``word_tokens`` include the trailing EOT entry, but
    # ``start_times`` / ``end_times`` only cover the real words (one less,
    # since the EOT row was excluded from DTW). Truncate to align.
    real_word_count = len(start_times)
    timings: list[WordTiming] = []
    for word, ids, start, end in zip(
        words[:real_word_count],
        word_tokens[:real_word_count],
        start_times,
        end_times,
        strict=True,
    ):
        if ids and ids[0] == eot_id:
            continue
        timings.append(WordTiming(word=word, start=float(start), end=float(end), tokens=tuple(ids)))
    return timings
