"""Core audio super-resolution wiring on TTSVoice (engine-agnostic).

Covers the lazy engine loader and the per-chunk upscale helper directly, without
constructing a full ONNX voice. A gated end-to-end test exercises the real
audiosronnx pipeline when the package (and its weights) are available.
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from phoonnx.config import SynthesisConfig
from phoonnx.voice import TTSVoice


def _bare_voice():
    """A TTSVoice instance with only the SR-cache fields set (no ONNX session)."""
    v = TTSVoice.__new__(TTSVoice)
    v._sr_engine = None
    v._sr_loaded = False
    return v


class TestLoadSuperResolution(unittest.TestCase):
    def test_disabled_when_unset(self):
        self.assertIsNone(_bare_voice()._load_super_resolution(SynthesisConfig()))

    def test_defaults_to_disabled(self):
        """The runtime knob is off unless a caller explicitly turns it on."""
        cfg = SynthesisConfig()
        self.assertIs(cfg.super_resolution, False)
        self.assertIsNone(cfg.super_resolution_model)

    def test_explicitly_disabled(self):
        cfg = SynthesisConfig(super_resolution=False)
        self.assertIsNone(_bare_voice()._load_super_resolution(cfg))

    def test_disabled_path_never_imports_audiosronnx(self):
        """The default path must not touch the optional dependency at all."""
        cfg = SynthesisConfig()
        with patch.dict(sys.modules):
            sys.modules.pop("audiosronnx", None)
            self.assertIsNone(_bare_voice()._load_super_resolution(cfg))
            self.assertNotIn("audiosronnx", sys.modules)

    def test_enabled_but_not_installed_raises_actionable_import_error(self):
        cfg = SynthesisConfig(super_resolution=True)
        with patch.dict(sys.modules, {"audiosronnx": None}):
            with self.assertRaises(ImportError) as ctx:
                _bare_voice()._load_super_resolution(cfg)
        self.assertIn("phoonnx[audiosr]", str(ctx.exception))

    def _fake_audiosronnx(self):
        fake_sr = MagicMock(name="sr_engine")
        module = types.ModuleType("audiosronnx")
        module.load_sr = MagicMock(name="load_sr", return_value=fake_sr)
        patcher = patch.dict(sys.modules, {"audiosronnx": module})
        patcher.start()
        self.addCleanup(patcher.stop)
        return module, fake_sr

    def test_enabled_loads_named_engine(self):
        cfg = SynthesisConfig(super_resolution=True,
                              super_resolution_model="lavasr")
        module, fake_sr = self._fake_audiosronnx()
        got = _bare_voice()._load_super_resolution(cfg)
        self.assertIs(got, fake_sr)
        module.load_sr.assert_called_once_with(engine="lavasr")

    def test_defaults_to_novasr(self):
        cfg = SynthesisConfig(super_resolution=True)
        module, _ = self._fake_audiosronnx()
        _bare_voice()._load_super_resolution(cfg)
        module.load_sr.assert_called_once_with(engine="novasr")

    def test_loaded_once_and_cached(self):
        cfg = SynthesisConfig(super_resolution=True)
        module, fake_sr = self._fake_audiosronnx()
        v = _bare_voice()
        self.assertIs(v._load_super_resolution(cfg), fake_sr)
        self.assertIs(v._load_super_resolution(cfg), fake_sr)
        module.load_sr.assert_called_once()

    def test_per_call_disable_wins_over_a_loaded_engine(self):
        """super_resolution is a per-synthesis runtime knob: a voice that already
        loaded an engine still yields native audio when the caller's config says
        off."""
        module, fake_sr = self._fake_audiosronnx()
        v = _bare_voice()
        self.assertIs(v._load_super_resolution(SynthesisConfig(super_resolution=True)),
                      fake_sr)
        self.assertIsNone(v._load_super_resolution(SynthesisConfig()))

    def test_load_failure_propagates(self):
        cfg = SynthesisConfig(super_resolution=True)
        module, _ = self._fake_audiosronnx()
        module.load_sr.side_effect = RuntimeError("bad weights")
        with self.assertRaises(RuntimeError):
            _bare_voice()._load_super_resolution(cfg)


class TestMaybeUpscale(unittest.TestCase):
    def test_none_engine_is_passthrough(self):
        audio = np.zeros(10, dtype=np.float32)
        out, sr = TTSVoice._maybe_upscale(audio, None, 22050)
        self.assertIs(out, audio)
        self.assertEqual(sr, 22050)

    def test_upscales_and_returns_new_sr(self):
        engine = MagicMock()
        engine.upscale.return_value = (np.zeros(480, dtype=np.float64), 48000)
        audio = np.linspace(-0.5, 0.5, 220, dtype=np.float32)
        out, sr = TTSVoice._maybe_upscale(audio, engine, 22050)
        self.assertEqual(sr, 48000)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape[0], 480)
        self.assertEqual(engine.upscale.call_args.args[1], 22050)

    def test_engine_failure_falls_back_to_native(self):
        engine = MagicMock()
        engine.upscale.side_effect = RuntimeError("boom")
        audio = np.ones(5, dtype=np.float32)
        out, sr = TTSVoice._maybe_upscale(audio, engine, 16000)
        self.assertIs(out, audio)
        self.assertEqual(sr, 16000)


class TestChunkAndWavSampleRate(unittest.TestCase):
    """Upscaled audio must be advertised at the upscaled rate, everywhere."""

    def _voice(self, sr_engine, native_sr=22050):
        v = TTSVoice.__new__(TTSVoice)
        v._sr_engine = sr_engine
        v._sr_loaded = sr_engine is not None
        return v

    def test_chunk_reports_engine_sample_rate(self):
        engine = MagicMock()
        engine.upscale.return_value = (np.zeros(480, dtype=np.float32), 48000)
        _, sr = TTSVoice._maybe_upscale(
            np.zeros(220, dtype=np.float32), engine, 22050)
        self.assertEqual(sr, 48000)

    def test_synthesize_wav_header_follows_the_chunk(self):
        """synthesize_wav takes the format from the first chunk, so an upscaled
        chunk sets an upscaled WAV header (no chipmunk/slow playback)."""
        from phoonnx.voice import AudioChunk

        v = TTSVoice.__new__(TTSVoice)
        v.config = MagicMock(sample_rate=22050)
        chunk = AudioChunk(sample_rate=48000, sample_width=2, sample_channels=1,
                           audio_float_array=np.zeros(48, dtype=np.float32))
        with patch.object(TTSVoice, "synthesize", return_value=iter([chunk])):
            wav = MagicMock()
            TTSVoice.synthesize_wav(v, "hi", wav)
        wav.setframerate.assert_called_with(48000)
        wav.setsampwidth.assert_called_with(2)
        wav.setnchannels.assert_called_with(1)


class TestSuperResolutionE2E(unittest.TestCase):
    """Runs only when audiosronnx and its weights are actually available."""

    def test_real_upscale_to_48k(self):
        try:
            from audiosronnx import load_sr
        except ImportError:
            self.skipTest("audiosronnx not installed")
        try:
            engine = load_sr(engine="novasr")
        except Exception as exc:  # weights download / runtime unavailable
            self.skipTest(f"super-resolution engine unavailable: {exc}")

        t = np.linspace(0, 0.25, 4000, endpoint=False, dtype=np.float32)
        audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        out, sr = TTSVoice._maybe_upscale(audio, engine, 16000)
        self.assertEqual(sr, 48000)
        self.assertGreater(out.shape[0], audio.shape[0])
        self.assertEqual(out.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
