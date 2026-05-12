"""Download progress callbacks for model loading."""

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias


@dataclass(frozen=True)
class DownloadProgress:
    """Progress event emitted while downloading model files from Hugging Face.

    A new event is emitted for every chunk of bytes received. When multiple files
    are downloaded in parallel (e.g. via ``snapshot_download``), events for
    different files may interleave and be delivered from worker threads.

    Attributes:
        filename: Description of the file currently being downloaded, as
            reported by ``huggingface_hub`` (typically the file name).
        downloaded: Bytes downloaded so far for this file.
        total: Total bytes for this file, or ``None`` if unknown.

    """

    filename: str
    downloaded: int
    total: int | None


ProgressCallback: TypeAlias = Callable[[DownloadProgress], None]
"""Callback invoked with a :class:`DownloadProgress` event per download chunk."""


def make_tqdm_class(callback: ProgressCallback) -> type:
    """Build a ``tqdm``-compatible class that forwards updates to ``callback``.

    ``huggingface_hub`` accepts a ``tqdm_class`` argument on ``hf_hub_download``
    and ``snapshot_download``. The returned class suppresses ``tqdm``'s own
    terminal output and invokes ``callback`` on each progress update instead.

    Args:
        callback: Function invoked for every progress event.

    Returns:
        A ``tqdm.tqdm`` subclass suitable for use as ``tqdm_class``.

    """
    from tqdm import tqdm  # noqa: PLC0415

    class _CallbackTqdm(tqdm):  # type: ignore[misc]
        _progress_callback: ClassVar[ProgressCallback] = staticmethod(callback)

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            # ``tqdm`` skips setting many attributes when ``disable=True``, so we capture
            # the values we need from the constructor kwargs before forcing it.
            self._cb_desc: str = str(kwargs.get("desc") or "")
            total = kwargs.get("total")
            self._cb_total: int | None = int(total) if isinstance(total, (int, float)) and total else None
            self._downloaded: int = int(kwargs.get("initial") or 0)
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def update(self, n: float | None = 1) -> bool | None:
            self._downloaded += int(n or 0)
            type(self)._progress_callback(
                DownloadProgress(filename=self._cb_desc, downloaded=self._downloaded, total=self._cb_total)
            )
            return None

    return _CallbackTqdm


@contextlib.contextmanager
def hf_progress_patch(callback: ProgressCallback) -> Iterator[None]:
    """Route ``huggingface_hub`` per-file byte progress through ``callback``.

    ``huggingface_hub.snapshot_download`` accepts a ``tqdm_class`` parameter, but
    that only controls the outer "files completed" bar. Per-byte progress for
    individual files is created by ``_get_progress_bar_context`` in
    ``huggingface_hub.utils.tqdm`` via the module-level ``tqdm`` symbol, with no
    public injection point. This context manager temporarily swaps that symbol
    for a ``tqdm`` subclass that forwards updates to ``callback`` while
    suppressing the default terminal output. ``file_download.tqdm`` is also
    swapped because ``http_get`` annotations reference it.

    Args:
        callback: Function invoked for every progress event.

    """
    import sys  # noqa: PLC0415

    import huggingface_hub.utils.tqdm  # noqa: F401, PLC0415  # ensure submodule is loaded
    from huggingface_hub import file_download  # noqa: PLC0415

    hf_tqdm_mod = sys.modules["huggingface_hub.utils.tqdm"]
    cb_tqdm = make_tqdm_class(callback)
    original_utils = hf_tqdm_mod.tqdm
    original_fd = file_download.tqdm  # type: ignore[attr-defined]
    hf_tqdm_mod.tqdm = cb_tqdm  # type: ignore[attr-defined]
    file_download.tqdm = cb_tqdm  # type: ignore[attr-defined, assignment]
    try:
        yield
    finally:
        hf_tqdm_mod.tqdm = original_utils  # type: ignore[attr-defined]
        file_download.tqdm = original_fd  # type: ignore[attr-defined]
