# ruff: noqa: T201, INP001
"""End-to-end smoke test for the :class:`onnx_asr.models.moonshine.Moonshine` class.

Downloads (and caches) the JFK reference clip used by HF Transformers.js examples,
loads ``moonshine-tiny-ONNX`` and ``moonshine-base-ONNX`` from the onnx-community
HF org, runs ``recognize`` on both real speech and digital silence, prints the
output and asserts the transcript contains a recognisable phrase from the clip.

Run with::

    cd <repo-root>/WinSTT/server
    uv run --no-sync python ../../onnx-asr-moonshine/scripts/verify_moonshine.py
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np


def _project_root() -> Path:
    """Return the onnx-asr fork's top-level dir (where ``src/`` and ``tests/`` live)."""
    return Path(__file__).resolve().parent.parent


def _ensure_jfk_wav() -> Path:
    """Fetch the Transformers.js JFK demo clip once and cache it under ``tests/data/``."""
    target = _project_root() / "tests" / "data" / "jfk.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav"
    print(f"[verify] downloading JFK fixture: {url}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - hardcoded public dataset URL
            data = resp.read()
    except urllib.error.URLError as exc:
        msg = f"failed to download {url}: {exc}"
        raise RuntimeError(msg) from exc
    target.write_bytes(data)
    print(f"[verify] cached {len(data):,} bytes at {target}", flush=True)
    return target


def _read_wav_info(path: Path) -> tuple[int, int, int]:
    """Return ``(framerate_hz, n_channels, sampwidth_bytes)`` for the WAV at ``path``."""
    with wave.open(str(path), "rb") as wf:
        return wf.getframerate(), wf.getnchannels(), wf.getsampwidth()


def _load_model(model_id: str):  # noqa: ANN202 - returns TextResultsAsrAdapter
    """Load a Moonshine model with CPU-only providers (quiet, deterministic)."""
    import onnx_asr  # noqa: PLC0415

    print(f"[verify] loading {model_id} ...", flush=True)
    return onnx_asr.load_model(model_id, providers=["CPUExecutionProvider"])


def _run_one_path(model_id: str, audio_path: Path) -> str:
    """Recognize via the path-based ``recognize()`` overload (handles resampling).

    Passes ``channel="mean"`` because the JFK reference clip ships stereo.
    """
    model = _load_model(model_id)
    try:
        text = model.recognize(str(audio_path), channel="mean")
    finally:
        model.close()
    return text


def _run_one_array(model_id: str, audio: np.ndarray) -> str:
    """Recognize a pre-loaded 16-kHz float32 numpy array — used for the silence check."""
    model = _load_model(model_id)
    try:
        text = model.recognize(audio, sample_rate=16_000)
    finally:
        model.close()
    return text


def main() -> int:
    """Run the verification flow and report. Returns process exit code (0 = pass)."""
    audio_path = _ensure_jfk_wav()
    framerate, n_channels, sampwidth = _read_wav_info(audio_path)
    print(
        f"[verify] jfk.wav: framerate={framerate}Hz channels={n_channels} sampwidth={sampwidth}B",
        flush=True,
    )

    silence = np.zeros(16_000, dtype=np.float32)

    failures: list[str] = []

    for model_id, expected_substrings in [
        ("onnx-community/moonshine-tiny-ONNX", ("fellow", "country")),
        ("onnx-community/moonshine-base-ONNX", ("fellow", "country")),
    ]:
        text = _run_one_path(model_id, audio_path)
        lower = text.lower()
        print(f"\n[verify] {model_id} -&gt; {text!r}", flush=True)
        if not any(sub in lower for sub in expected_substrings):
            failures.append(f"{model_id}: expected one of {expected_substrings} in {text!r}")

        # Silence sanity-check — should be empty or near-empty.
        silence_text = _run_one_array(model_id, silence)
        print(f"[verify] {model_id} on 1s silence -&gt; {silence_text!r}", flush=True)
        if len(silence_text) > 30:
            failures.append(f"{model_id}: silence produced verbose output {silence_text!r}")

    if failures:
        print("\n[verify] FAILURES:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        return 1

    print("\n[verify] OK - Moonshine integration verified end-to-end.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
