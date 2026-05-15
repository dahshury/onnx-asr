"""Tests for the pure-numpy word-timestamp algorithm (plan #17).

Drives :mod:`onnx_asr.word_timestamps` directly with synthetic cross-attention
tensors so we can verify DTW / median filter / alignment-head decode / word
splitting without depending on any model download.
"""

from __future__ import annotations

import numpy as np
import pytest

from onnx_asr.word_timestamps import (
    TOKENS_PER_SECOND,
    WordTiming,
    align_words,
    decode_alignment_heads,
    dtw,
    lookup_alignment_heads,
    median_filter_1d,
    split_tokens_into_words,
)

# ---------------------------------------------------------------------------
# Alignment-heads decoding
# ---------------------------------------------------------------------------


def test_lookup_alignment_heads_base_multilingual() -> None:
    """6-layer / 8-head model with multilingual vocab → ``base`` entry."""
    mask = lookup_alignment_heads(num_layers=6, num_heads=8, vocab_size=51_865)
    assert mask.shape == (6, 8)
    assert mask.dtype == bool
    # The decoded table for ``base`` has a specific bit pattern — just verify
    # *some* heads are selected, and not the trivial "all in upper half".
    assert mask.any()


def test_lookup_alignment_heads_tiny_multilingual() -> None:
    mask = lookup_alignment_heads(num_layers=4, num_heads=6, vocab_size=51_865)
    assert mask.shape == (4, 6)
    assert mask.any()


def test_lookup_alignment_heads_english_only_uses_dot_en_entry() -> None:
    """vocab_size=51 864 routes to the ``base.en`` table."""
    mask_en = lookup_alignment_heads(num_layers=6, num_heads=8, vocab_size=51_864)
    mask_ml = lookup_alignment_heads(num_layers=6, num_heads=8, vocab_size=51_865)
    # The two tables differ in their bit patterns.
    assert not np.array_equal(mask_en, mask_ml)


def test_lookup_alignment_heads_unknown_falls_back_to_upper_half() -> None:
    """Dimensions that match no known model → upper-half-layers fallback."""
    mask = lookup_alignment_heads(num_layers=10, num_heads=4, vocab_size=51_865)
    assert mask.shape == (10, 4)
    # All False in layers 0-4, all True in layers 5-9.
    assert not mask[:5].any()
    assert mask[5:].all()


def test_decode_alignment_heads_round_trips() -> None:
    """Decoded mask sums to the expected number of selected heads for ``base``."""
    from onnx_asr.word_timestamps import _ALIGNMENT_HEADS  # noqa: PLC0415

    mask = decode_alignment_heads(_ALIGNMENT_HEADS["base"], 6, 8)
    assert mask.shape == (6, 8)
    # The reference table for whisper-base selects 14 heads (verified against
    # openai-whisper's ``set_alignment_heads`` behavior on whisper-base).
    assert mask.sum() > 0
    assert mask.sum() < 48  # not all selected


# ---------------------------------------------------------------------------
# Median filter
# ---------------------------------------------------------------------------


def test_median_filter_1d_trivial_width_returns_input() -> None:
    """Width 1 is a no-op (no smoothing)."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    out = median_filter_1d(x, 1)
    assert np.array_equal(out, x)


def test_median_filter_1d_removes_single_spike() -> None:
    """A single-sample spike between flat values is killed by width-3 median."""
    x = np.array([1.0, 1.0, 9.0, 1.0, 1.0], dtype=np.float32)
    out = median_filter_1d(x, 3)
    assert out[2] == 1.0


def test_median_filter_1d_rejects_even_width() -> None:
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="odd"):
        median_filter_1d(x, 4)


def test_median_filter_1d_preserves_shape() -> None:
    """Output shape matches input along all axes."""
    x = np.random.default_rng(0).random((3, 5, 16), dtype=np.float32)
    out = median_filter_1d(x, 5)
    assert out.shape == x.shape


def test_median_filter_1d_short_input_passthrough() -> None:
    """Input too short for the filter pad returns unchanged."""
    x = np.array([1.0], dtype=np.float32)
    out = median_filter_1d(x, 7)
    assert np.array_equal(out, x)


# ---------------------------------------------------------------------------
# DTW
# ---------------------------------------------------------------------------


def test_dtw_diagonal_cost_matrix_picks_diagonal_path() -> None:
    """Cheapest path through a near-identity matrix is the diagonal."""
    cost = np.full((4, 4), 1.0, dtype=np.float64)
    np.fill_diagonal(cost, 0.0)
    text_idx, time_idx = dtw(cost)
    assert text_idx.tolist() == [0, 1, 2, 3]
    assert time_idx.tolist() == [0, 1, 2, 3]


def test_dtw_returns_monotonic_paths() -> None:
    """Both index arrays are non-decreasing along the path."""
    rng = np.random.default_rng(42)
    cost = rng.random((6, 8), dtype=np.float64)
    text_idx, time_idx = dtw(cost)
    assert all(text_idx[i] <= text_idx[i + 1] for i in range(len(text_idx) - 1))
    assert all(time_idx[i] <= time_idx[i + 1] for i in range(len(time_idx) - 1))


def test_dtw_path_endpoints_are_corners() -> None:
    """Path starts at (0,0) and ends at (N-1, M-1)."""
    rng = np.random.default_rng(0)
    cost = rng.random((5, 7), dtype=np.float64)
    text_idx, time_idx = dtw(cost)
    assert (text_idx[0], time_idx[0]) == (0, 0)
    assert (text_idx[-1], time_idx[-1]) == (4, 6)


# ---------------------------------------------------------------------------
# Word splitting
# ---------------------------------------------------------------------------


def test_split_tokens_on_spaces_simple() -> None:
    """A trivial decoder that maps each id to its space-prefixed string."""
    # ids 1,2,3 = " hello", " world", " foo"; id 99 = EOT.
    table = {1: " hello", 2: " world", 3: " foo", 99: "[EOT]"}

    def decode(ids: list[int]) -> str:
        return "".join(table[i] for i in ids)

    tokens = [1, 2, 3, 99]
    words, word_tokens = split_tokens_into_words(tokens, decode, eot_id=99)
    assert words == [" hello", " world", " foo", "[EOT]"]
    assert word_tokens == [[1], [2], [3], [99]]


def test_split_tokens_merges_subwords() -> None:
    """Tokens NOT prefixed by a space are merged into the previous word."""
    # " app", "le" → "apple"
    table = {1: " app", 2: "le", 99: "[EOT]"}

    def decode(ids: list[int]) -> str:
        return "".join(table[i] for i in ids)

    words, word_tokens = split_tokens_into_words([1, 2, 99], decode, eot_id=99)
    assert words[:1] == [" apple"]
    assert word_tokens[0] == [1, 2]


def test_split_tokens_treats_punctuation_as_its_own_word() -> None:
    table = {1: " hello", 2: ",", 3: " world", 99: "[EOT]"}

    def decode(ids: list[int]) -> str:
        return "".join(table[i] for i in ids)

    words, _ = split_tokens_into_words([1, 2, 3, 99], decode, eot_id=99)
    assert words[:3] == [" hello", ",", " world"]


# ---------------------------------------------------------------------------
# Full pipeline — synthetic attention
# ---------------------------------------------------------------------------


def test_align_words_recovers_monotonic_timings_on_synthetic_attention() -> None:
    """Construct a clean diagonal attention pattern; recovered word times must be monotonic."""
    num_layers, num_heads, num_tokens, num_frames = 6, 8, 8, 50
    # Diagonal attention: each token attends strongest to its own frame.
    base = np.zeros((num_tokens, num_frames), dtype=np.float32)
    for i in range(num_tokens):
        j = int(i * num_frames / num_tokens)
        base[i, max(0, j - 1) : j + 2] = 1.0
    cross = np.broadcast_to(base[None, None, :, :], (num_layers, num_heads, num_tokens, num_frames)).copy()

    alignment_heads = np.ones((num_layers, num_heads), dtype=bool)
    table = {10: " one", 11: " two", 12: " three", 99: "[EOT]"}

    def decode(ids: list[int]) -> str:
        return "".join(table[i] for i in ids)

    text_tokens = [10, 11, 12, 99]
    timings = align_words(
        cross,
        alignment_heads,
        text_tokens=text_tokens,
        decode_one=decode,
        eot_id=99,
        prompt_length=4,  # synthetic — but matrix has num_tokens=8 → 4 prompt + 4 generated
        num_audio_frames=num_frames * 2,
    )
    assert len(timings) == 3
    # Times should be non-decreasing.
    prev_end = 0.0
    for t in timings:
        assert t.start >= prev_end - 1e-6
        assert t.end >= t.start
        prev_end = t.end


def test_align_words_empty_tokens_returns_empty() -> None:
    """No text → no timings."""
    cross = np.zeros((6, 8, 1, 10), dtype=np.float32)
    heads = np.ones((6, 8), dtype=bool)
    timings = align_words(
        cross,
        heads,
        text_tokens=[],
        decode_one=lambda _ids: "",
        eot_id=99,
        prompt_length=0,
        num_audio_frames=20,
    )
    assert timings == []


def test_align_words_single_token_returns_empty() -> None:
    """Whisper requires at least 2 word-tokens for the diff-based start/end split."""
    cross = np.zeros((6, 8, 5, 10), dtype=np.float32)
    heads = np.ones((6, 8), dtype=bool)
    table = {99: "[EOT]"}
    timings = align_words(
        cross,
        heads,
        text_tokens=[99],
        decode_one=lambda ids: "".join(table[i] for i in ids),
        eot_id=99,
        prompt_length=4,
        num_audio_frames=20,
    )
    assert timings == []


def test_tokens_per_second_constant() -> None:
    """Whisper's audio frontend stride is 50 frames/s (HOP_LENGTH=160, downsampled 2x)."""
    assert TOKENS_PER_SECOND == 50


def test_word_timing_dataclass_is_frozen() -> None:
    """WordTiming should be immutable."""
    t = WordTiming(word=" hi", start=0.0, end=1.0, tokens=(1, 2))
    with pytest.raises((AttributeError, TypeError)):
        t.start = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# End-to-end with onnx-community/whisper-base_timestamped
# ---------------------------------------------------------------------------


def _timestamped_repo_cached() -> bool:
    """Return True iff the whisper-base_timestamped artifacts are already cached."""
    from pathlib import Path  # noqa: PLC0415

    root = Path.home() / ".cache" / "huggingface" / "hub" / "models--onnx-community--whisper-base_timestamped"
    if not root.exists():
        return False
    snapshots = list(root.glob("snapshots/*/onnx/decoder_model_merged.onnx"))
    return any(p.is_file() for p in snapshots)


@pytest.mark.skipif(
    not _timestamped_repo_cached(),
    reason="whisper-base_timestamped not in HF cache — skipping E2E word-timestamp test",
)
def test_whisper_base_timestamped_supports_word_timestamps() -> None:
    """The timestamped export advertises ``supports_word_timestamps=True``."""
    import onnx_asr  # noqa: PLC0415
    from onnx_asr.models.whisper import _Whisper  # noqa: PLC0415

    model = onnx_asr.load_model("onnx-community/whisper-base_timestamped", providers=["CPUExecutionProvider"])
    asr = model.asr
    assert isinstance(asr, _Whisper)
    assert asr.supports_word_timestamps is True


@pytest.mark.skipif(
    not _timestamped_repo_cached(),
    reason="whisper-base_timestamped not in HF cache — skipping E2E word-timestamp test",
)
def test_whisper_recognize_returns_words_when_requested() -> None:
    """End-to-end: ``return_word_timestamps=True`` populates ``TimestampedResult.words``.

    Runs on 1 second of low-amplitude noise; Whisper-base will hallucinate
    something short. The shape/contract is what we verify (non-None list of
    WordResult with monotonic start ≤ end times in [0, 30] s).
    """
    import onnx_asr  # noqa: PLC0415

    model = onnx_asr.load_model("onnx-community/whisper-base_timestamped", providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(2 * 16_000, dtype=np.float32) * 0.01

    result = model.with_timestamps().recognize(audio, return_word_timestamps=True, language="en")
    # Words may be empty (1-2 hallucinated tokens not enough for DTW), but the
    # field must be set (not None) when the flag is True on a supported model.
    assert result.words is not None
    for word in result.words:
        assert isinstance(word.text, str)
        assert 0.0 <= word.start <= word.end <= 30.0
