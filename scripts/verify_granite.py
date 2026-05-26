"""End-to-end verification for the IBM Granite Speech ASR integration.

Run from the WinSTT server venv (which has onnxruntime installed)::

    cd E:/DL/Projects/WinSTT/server
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe \\
        ../../onnx-asr-granite/scripts/verify_granite.py

Steps:
1. Download the well-known JFK English sample (cached locally).
2. Load ``onnx-community/granite-4.0-1b-speech-ONNX`` at ``quantization="q4"``
   (the smallest variant — fp32 weights are ~4 GB which OOMs most laptops).
3. Transcribe → assert ``"fellow americans"`` AND ``"country"`` appear in the
   text. JFK clip is "And so my fellow Americans, ask not what your country
   can do for you...".
4. Run silence (zeros) and print the result — no assertion; Granite may
   hallucinate on noise, same as Cohere/Whisper.
"""

from __future__ import annotations

import os
import sys
import time
import tracemalloc
import urllib.request
from pathlib import Path

import numpy as np

import onnx_asr

# Some Windows consoles default to cp1252, which mangles transcription output.
# Force UTF-8 if the terminal supports it.
for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "verify_granite"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _fetch(url: str, name: str) -> Path:
    """Cache a URL to disk and return the local path."""
    dst = CACHE_DIR / name
    if not dst.exists() or dst.stat().st_size == 0:
        print(f"  downloading {url} -> {dst}", flush=True)
        with urllib.request.urlopen(url) as resp, dst.open("wb") as f:  # noqa: S310
            f.write(resp.read())
    return dst


def main() -> int:
    quantization = os.environ.get("GRANITE_QUANT", "q4")
    print(f"== Granite Speech verification (quant={quantization}) ==", flush=True)

    tracemalloc.start()
    t0 = time.perf_counter()
    model = onnx_asr.load_model(
        "onnx-community/granite-4.0-1b-speech-ONNX",
        quantization=quantization,
        providers=["CPUExecutionProvider"],
    )
    load_secs = time.perf_counter() - t0
    cur, peak = tracemalloc.get_traced_memory()
    print(
        f"  loaded in {load_secs:.2f}s; tracemalloc current={cur / 1e6:.1f} MB peak={peak / 1e6:.1f} MB",
        flush=True,
    )

    # 1) JFK clip — well-known English sample shared across the onnx-community demos.
    jfk = _fetch(
        "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav",
        "jfk.wav",
    )
    t1 = time.perf_counter()
    text_jfk = model.recognize(jfk, channel="mean")
    print(f"\n[JFK] ({time.perf_counter() - t1:.2f}s)\n  {text_jfk!r}", flush=True)
    text_jfk_lc = text_jfk.lower()
    assert "country" in text_jfk_lc, f"JFK transcription missing 'country': {text_jfk!r}"
    assert "fellow americans" in text_jfk_lc, (
        f"JFK transcription missing 'fellow americans': {text_jfk!r}"
    )

    # 2) Silence — print but don't assert. Some LLM-based ASR models will
    # hallucinate text on pure-zero input; this is informational only.
    audio_silence = np.zeros(16_000, dtype=np.float32)
    t2 = time.perf_counter()
    text_silence = model.recognize(audio_silence)
    print(f"\n[silence / 1 s of zeros] ({time.perf_counter() - t2:.2f}s)\n  {text_silence!r}", flush=True)

    cur, peak = tracemalloc.get_traced_memory()
    print(
        f"\n== ALL CHECKS PASSED ==\n  final tracemalloc current={cur / 1e6:.1f} MB peak={peak / 1e6:.1f} MB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
