from typing import Any

import pytest

from onnx_asr import DownloadProgress, load_model
from onnx_asr.progress import make_tqdm_class
from onnx_asr.resolver import Resolver


def test_make_tqdm_class_invokes_callback() -> None:
    events: list[DownloadProgress] = []
    tqdm_class = make_tqdm_class(events.append)

    bar = tqdm_class(desc="encoder_model.onnx", total=1000, initial=0)
    bar.update(250)
    bar.update(750)

    assert len(events) == 2
    assert events[0] == DownloadProgress(filename="encoder_model.onnx", downloaded=250, total=1000)
    assert events[1] == DownloadProgress(filename="encoder_model.onnx", downloaded=1000, total=1000)


def test_make_tqdm_class_handles_unknown_total() -> None:
    events: list[DownloadProgress] = []
    tqdm_class = make_tqdm_class(events.append)

    bar = tqdm_class(desc="file.onnx", total=None, initial=0)
    bar.update(42)

    assert events == [DownloadProgress(filename="file.onnx", downloaded=42, total=None)]


def test_make_tqdm_class_respects_initial() -> None:
    events: list[DownloadProgress] = []
    tqdm_class = make_tqdm_class(events.append)

    bar = tqdm_class(desc="file.onnx", total=200, initial=100)
    bar.update(50)

    assert events == [DownloadProgress(filename="file.onnx", downloaded=150, total=200)]


def test_resolver_patches_file_download_tqdm_when_callback_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub  # noqa: PLC0415
    import huggingface_hub.file_download as fd  # noqa: PLC0415

    from onnx_asr.models.whisper import WhisperHf  # noqa: PLC0415

    observed: list[type] = []

    def fake_snapshot_download(*_args: Any, **_kwargs: Any) -> str:
        observed.append(fd.tqdm)  # type: ignore[attr-defined]
        return "/tmp/fake_snapshot"  # noqa: S108

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    original = fd.tqdm  # type: ignore[attr-defined]
    resolver = Resolver({"whisper": WhisperHf}, "onnx-community/whisper-tiny", progress_callback=lambda _e: None)
    resolver._download_model(quantization="uint8", local_files_only=False)

    assert len(observed) == 1
    assert observed[0] is not original  # tqdm symbol was swapped during the call
    assert fd.tqdm is original  # type: ignore[attr-defined]  # ...and restored after


def test_resolver_does_not_patch_without_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub  # noqa: PLC0415
    import huggingface_hub.file_download as fd  # noqa: PLC0415

    from onnx_asr.models.whisper import WhisperHf  # noqa: PLC0415

    observed: list[type] = []

    def fake_snapshot_download(*_args: Any, **_kwargs: Any) -> str:
        observed.append(fd.tqdm)  # type: ignore[attr-defined]
        return "/tmp/fake_snapshot"  # noqa: S108

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    original = fd.tqdm  # type: ignore[attr-defined]
    resolver = Resolver({"whisper": WhisperHf}, "onnx-community/whisper-tiny")
    resolver._download_model(quantization="uint8", local_files_only=False)

    assert len(observed) == 1
    assert observed[0] is original  # tqdm symbol untouched


def test_load_model_threads_callback() -> None:
    events: list[DownloadProgress] = []
    # Use a tiny pre-cached model from the existing test suite. If the HF cache is warm
    # no chunks transfer and we just assert the call succeeds without error.
    load_model("onnx-community/whisper-tiny", quantization="uint8", progress_callback=events.append)

    for event in events:
        assert isinstance(event, DownloadProgress)
        assert event.downloaded >= 0
        assert event.total is None or event.total > 0
