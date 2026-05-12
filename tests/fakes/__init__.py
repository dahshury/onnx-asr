"""Test fakes for onnx-asr.

Provides drop-in stubs of the ``Asr`` protocol and the ``Resampler`` class that
return canned data without downloading models or initializing ONNX sessions.
Use these to write integration tests that exercise the adapter / VAD / streaming
layers without paying the 100+ MB download cost on every CI run.

See :func:`make_fake_text_adapter` for the common entry point.
"""

from __future__ import annotations

from tests.fakes.fake_model import FakeAsr, FakeResampler, make_fake_text_adapter

__all__ = ["FakeAsr", "FakeResampler", "make_fake_text_adapter"]
