"""End-to-end verification for the Cohere Transcribe ASR integration.

Run from the WinSTT server venv (which has onnxruntime installed):

    cd E:/DL/Projects/WinSTT/server
    uv run --no-sync python ../../onnx-asr-cohere/scripts/verify_cohere.py

Steps:
1. Download a known English clip (Xenova/transformers.js JFK sample).
2. Load the Cohere Transcribe q4 ONNX export.
3. Transcribe → must contain "country" / "fellow Americans".
4. Confirm silence yields empty / near-empty output.
5. Try the model's own English demo clip (cohere_asr-en.wav) for a
   more confident sanity check on the expected sentence.
6. Try the French demo clip with ``language='fr'``.
"""

from __future__ import annotations

import os
import sys
import time
import tracemalloc
import urllib.request
import wave
from pathlib import Path

import numpy as np

import onnx_asr

# Some Windows consoles default to cp1252, which mangles transcription output
# (e.g. French accented characters). Force UTF-8 if the terminal supports it.
for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "verify_cohere"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _fetch(url: str, name: str) -> Path:
    """Cache a URL to disk and return the local path."""
    dst = CACHE_DIR / name
    if not dst.exists() or dst.stat().st_size == 0:
        print(f"  downloading {url} -> {dst}", flush=True)
        with urllib.request.urlopen(url) as resp, dst.open("wb") as f:  # noqa: S310
            f.write(resp.read())
    return dst


def _wav_info(path: Path) -> str:
    """Return ``"channels @ Hz, N frames"`` for a WAV file, or ``"<not PCM>"``."""
    try:
        with wave.open(str(path), "rb") as w:
            return f"{w.getnchannels()}ch @ {w.getframerate()} Hz, {w.getnframes()} frames"
    except wave.Error as exc:
        return f"<non-PCM wav: {exc}>"


def _load_wav_anyformat(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file even if it's IEEE_FLOAT (format=3).

    Standard ``wave`` and onnx-asr's ``read_wav`` only handle PCM int formats.
    The HF demo clips are float32 RIFF, so we parse the header manually and
    feed the model an in-memory numpy array.
    """
    with path.open("rb") as f:
        header = f.read(44)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            msg = f"{path.name}: not a RIFF/WAVE file"
            raise RuntimeError(msg)
        # fmt chunk starts at offset 12 (4 bytes "fmt ") + 4 bytes size + ...
        # Just walk chunks for robustness.
        f.seek(12)
        fmt = None
        n_channels = 0
        sample_rate = 0
        bits = 0
        data_chunk: bytes | None = None
        while True:
            chunk_id = f.read(4)
            if not chunk_id:
                break
            chunk_size = int.from_bytes(f.read(4), "little")
            chunk = f.read(chunk_size)
            if chunk_id == b"fmt ":
                fmt = int.from_bytes(chunk[0:2], "little")
                n_channels = int.from_bytes(chunk[2:4], "little")
                sample_rate = int.from_bytes(chunk[4:8], "little")
                bits = int.from_bytes(chunk[14:16], "little")
            elif chunk_id == b"data":
                data_chunk = chunk
                break
            if chunk_size % 2:  # RIFF chunks are word-aligned
                f.read(1)

    if data_chunk is None or fmt is None:
        msg = f"{path.name}: missing fmt/data chunks"
        raise RuntimeError(msg)

    if fmt == 1:  # PCM int
        dtype = {8: np.int8, 16: np.int16, 32: np.int32}.get(bits)
        if dtype is None:
            msg = f"{path.name}: unsupported PCM bit depth {bits}"
            raise RuntimeError(msg)
        max_val = float(2 ** (bits - 1))
        samples = np.frombuffer(data_chunk, dtype=dtype).astype(np.float32) / max_val
    elif fmt == 3:  # IEEE_FLOAT
        dtype = np.float32 if bits == 32 else np.float64
        samples = np.frombuffer(data_chunk, dtype=dtype).astype(np.float32, copy=True)
    else:
        msg = f"{path.name}: unsupported WAV format tag {fmt}"
        raise RuntimeError(msg)

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples.astype(np.float32, copy=False), sample_rate


def main() -> int:
    quantization = os.environ.get("COHERE_QUANT", "q4")
    print(f"== Cohere Transcribe verification (quant={quantization}) ==", flush=True)

    tracemalloc.start()
    t0 = time.perf_counter()
    model = onnx_asr.load_model(
        "onnx-community/cohere-transcribe-03-2026-ONNX",
        quantization=quantization,
        providers=["CPUExecutionProvider"],
    )
    load_secs = time.perf_counter() - t0
    cur, peak = tracemalloc.get_traced_memory()
    print(
        f"  loaded in {load_secs:.2f}s; tracemalloc current={cur / 1e6:.1f} MB peak={peak / 1e6:.1f} MB",
        flush=True,
    )

    # 1) JFK clip (well-known English sample). The adapter resamples the WAV
    # internally — just hand it the path.
    jfk = _fetch(
        "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav",
        "jfk.wav",
    )
    print(f"  jfk.wav: {_wav_info(jfk)}", flush=True)
    t1 = time.perf_counter()
    text_jfk = model.recognize(jfk, language="en", channel="mean")
    print(f"\n[JFK / en] ({time.perf_counter() - t1:.2f}s)\n  {text_jfk!r}", flush=True)
    text_jfk_lc = text_jfk.lower()
    assert "country" in text_jfk_lc or "fellow americans" in text_jfk_lc, (
        f"JFK transcription failed: {text_jfk!r}"
    )

    # 2) Silence — should be empty / near-empty.
    audio_silence = np.zeros(16_000, dtype=np.float32)
    t2 = time.perf_counter()
    text_silence = model.recognize(audio_silence, language="en")
    print(f"\n[silence / en] ({time.perf_counter() - t2:.2f}s)\n  {text_silence!r}", flush=True)
    # The README warns the model is "eager to transcribe even non-speech sounds",
    # so we only assert it doesn't produce a long hallucination.
    assert len(text_silence) < 80, (
        f"silence produced suspiciously long output: {text_silence!r}"
    )

    # 3) Cohere's own English demo clip — same sentence the model card cites.
    en_demo = _fetch(
        "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cohere_asr-en.wav",
        "cohere_asr-en.wav",
    )
    print(f"  cohere_asr-en.wav: {_wav_info(en_demo)}", flush=True)
    en_arr, en_sr = _load_wav_anyformat(en_demo)
    t3 = time.perf_counter()
    text_en = model.recognize(en_arr, language="en", sample_rate=en_sr)
    print(f"\n[cohere_asr-en / en] ({time.perf_counter() - t3:.2f}s)\n  {text_en!r}", flush=True)
    assert "insects" in text_en.lower(), (
        f"English demo transcription failed: {text_en!r}"
    )

    # 4) Cohere's own French demo clip with language='fr'.
    fr_demo = _fetch(
        "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cohere_asr-fr.wav",
        "cohere_asr-fr.wav",
    )
    print(f"  cohere_asr-fr.wav: {_wav_info(fr_demo)}", flush=True)
    fr_arr, fr_sr = _load_wav_anyformat(fr_demo)
    t4 = time.perf_counter()
    text_fr = model.recognize(fr_arr, language="fr", sample_rate=fr_sr)
    print(f"\n[cohere_asr-fr / fr] ({time.perf_counter() - t4:.2f}s)\n  {text_fr!r}", flush=True)
    assert "insectes" in text_fr.lower(), (
        f"French demo transcription failed: {text_fr!r}"
    )

    cur, peak = tracemalloc.get_traced_memory()
    print(
        f"\n== ALL CHECKS PASSED ==\n  final tracemalloc current={cur / 1e6:.1f} MB peak={peak / 1e6:.1f} MB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
