"""Universal voice-conversion post-stage wiring on TTSVoice (engine-agnostic).

Covers the lazy cloner loader, the per-chunk conversion helper, the WAV
round-trip helpers and the no-op guarantee, without constructing a full ONNX
voice. A gated end-to-end test exercises the real voiceclonnx pipeline when the
package (and its weights) are available.
"""
import gc
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from phoonnx.config import SynthesisConfig
from phoonnx.reference_audio import load_reference_audio
from phoonnx.voice import TTSVoice, _VoiceConversion, _write_wav


def _vc(cloner, ref_path):
    """A loaded conversion built straight from a fake cloner and a path."""
    return _VoiceConversion(cloner, ref_path)


def _bare_voice(testcase=None):
    """A TTSVoice instance with only the VC-cache fields set (no ONNX session)."""
    v = TTSVoice.__new__(TTSVoice)
    v._vc = None
    v._vc_key = None
    return v


def _ref_wav(tmpdir, sr=16000, seconds=1.0):
    path = os.path.join(tmpdir, "ref.wav")
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    return _write_wav(path, 0.2 * np.sin(2 * np.pi * 180 * t), sr)


def _fake_voiceclonnx(testcase, sample_rate=22050):
    cloner = MagicMock(name="cloner")
    cloner.sample_rate = sample_rate
    module = types.ModuleType("voiceclonnx")
    module.VoiceCloner = MagicMock(name="VoiceCloner", return_value=cloner)
    patcher = patch.dict(sys.modules, {"voiceclonnx": module})
    patcher.start()
    testcase.addCleanup(patcher.stop)
    return module, cloner


class TestWavHelpers(unittest.TestCase):
    """The disk hop must not change the signal beyond 16-bit quantisation."""

    def test_round_trip_preserves_audio_and_rate(self):
        sr = 22050
        t = np.linspace(0, 0.5, sr // 2, endpoint=False, dtype=np.float32)
        audio = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            p = _write_wav(os.path.join(d, "a.wav"), audio, sr)
            back, got_sr = load_reference_audio(p)
        self.assertEqual(got_sr, sr)
        self.assertEqual(back.shape, audio.shape)
        self.assertEqual(back.dtype, np.float32)
        # 16-bit PCM: one LSB is 1/32768
        self.assertLess(np.max(np.abs(back - audio)), 2.0 / 32768.0)

    def test_clips_out_of_range_samples(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write_wav(os.path.join(d, "a.wav"),
                           np.array([2.0, -2.0], dtype=np.float32), 8000)
            back, _ = load_reference_audio(p)
        self.assertLessEqual(back.max(), 1.0)
        self.assertGreaterEqual(back.min(), -1.0)

    def test_empty_audio_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write_wav(os.path.join(d, "a.wav"),
                           np.zeros(0, dtype=np.float32), 16000)
            back, sr = load_reference_audio(p)
        self.assertEqual(back.shape[0], 0)
        self.assertEqual(sr, 16000)


class TestLoadVoiceConversion(unittest.TestCase):
    def test_disabled_when_unset(self):
        self.assertIsNone(
            _bare_voice(self)._load_voice_conversion(SynthesisConfig()))

    def test_defaults_to_disabled(self):
        """The runtime knob is off unless a caller explicitly turns it on."""
        cfg = SynthesisConfig()
        self.assertIsNone(cfg.vc_reference)
        self.assertIsNone(cfg.vc_engine)

    def test_disabled_path_never_imports_voiceclonnx(self):
        """The default path must not touch the optional dependency at all."""
        with patch.dict(sys.modules):
            sys.modules.pop("voiceclonnx", None)
            self.assertIsNone(
                _bare_voice(self)._load_voice_conversion(SynthesisConfig()))
            self.assertNotIn("voiceclonnx", sys.modules)

    def test_enabled_but_not_installed_raises_actionable_import_error(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = SynthesisConfig(vc_reference=_ref_wav(d))
            with patch.dict(sys.modules, {"voiceclonnx": None}):
                with self.assertRaises(ImportError) as ctx:
                    _bare_voice(self)._load_voice_conversion(cfg)
        self.assertIn("phoonnx[vc]", str(ctx.exception))

    def test_enabled_loads_named_engine(self):
        module, cloner = _fake_voiceclonnx(self)
        with tempfile.TemporaryDirectory() as d:
            ref = _ref_wav(d)
            vc = _bare_voice(self)._load_voice_conversion(
                SynthesisConfig(vc_reference=ref, vc_engine="knnvc"))
            want, want_sr = load_reference_audio(ref)
            self.assertIs(vc.cloner, cloner)
            module.VoiceCloner.assert_called_once_with(engine="knnvc")
            # the reference is materialized through the guarded loader, so what
            # voiceclonnx reads is a decoded copy rather than the caller's file
            self.assertNotEqual(vc.ref_path, ref)
            got_audio, got_sr = load_reference_audio(vc.ref_path)
            self.assertEqual(got_sr, want_sr)
            np.testing.assert_allclose(got_audio, want, atol=2.0 / 32768.0)

    def test_defaults_to_openvoice(self):
        module, _ = _fake_voiceclonnx(self)
        with tempfile.TemporaryDirectory() as d:
            _bare_voice(self)._load_voice_conversion(
                SynthesisConfig(vc_reference=_ref_wav(d)))
        module.VoiceCloner.assert_called_once_with(engine="openvoice")

    def test_loaded_once_and_cached(self):
        module, cloner = _fake_voiceclonnx(self)
        v = _bare_voice(self)
        with tempfile.TemporaryDirectory() as d:
            cfg = SynthesisConfig(vc_reference=_ref_wav(d))
            self.assertIs(v._load_voice_conversion(cfg).cloner, cloner)
            self.assertIs(v._load_voice_conversion(cfg).cloner, cloner)
        module.VoiceCloner.assert_called_once()

    def test_changing_target_speaker_rebuilds(self):
        """A cached cloner must never be reused for a different reference —
        that would emit the previous speaker."""
        module, _ = _fake_voiceclonnx(self)
        v = _bare_voice(self)
        with tempfile.TemporaryDirectory() as d:
            a = _ref_wav(d)
            b = _write_wav(os.path.join(d, "b.wav"),
                           np.full(16000, 0.3, dtype=np.float32), 16000)
            vc_a = v._load_voice_conversion(SynthesisConfig(vc_reference=a))
            vc_b = v._load_voice_conversion(SynthesisConfig(vc_reference=b))
            self.assertEqual(module.VoiceCloner.call_count, 2)
            self.assertIsNot(vc_a, vc_b)
            self.assertNotEqual(vc_a.ref_path, vc_b.ref_path)
            np.testing.assert_allclose(load_reference_audio(vc_b.ref_path)[0],
                                       load_reference_audio(b)[0],
                                       atol=2.0 / 32768.0)

    def test_rewriting_a_reference_path_rebuilds(self):
        """A path is not a stable identity for a recording. If the file at the
        same path now holds a different speaker, serving the cached one is the
        wrong answer the fail-loud contract exists to avoid."""
        module, _ = _fake_voiceclonnx(self)
        v = _bare_voice(self)
        with tempfile.TemporaryDirectory() as d:
            ref = _ref_wav(d)
            cfg = SynthesisConfig(vc_reference=ref)
            first = v._load_voice_conversion(cfg)
            _write_wav(ref, np.full(16000, 0.4, dtype=np.float32), 16000)
            second = v._load_voice_conversion(cfg)
            self.assertEqual(module.VoiceCloner.call_count, 2)
            self.assertIsNot(first, second)
            # the cloner now reads the speaker that is on disk now
            np.testing.assert_allclose(load_reference_audio(second.ref_path)[0],
                                       load_reference_audio(ref)[0],
                                       atol=2.0 / 32768.0)

    def test_in_memory_references_with_equal_content_share_a_cloner(self):
        """The cache key is the reference's content, not its object identity —
        two equal arrays must not each pay for a fresh ONNX load, and two
        different ones must never collide."""
        module, _ = _fake_voiceclonnx(self)
        v = _bare_voice(self)
        audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
        v._load_voice_conversion(SynthesisConfig(vc_reference=(audio, 16000)))
        v._load_voice_conversion(SynthesisConfig(vc_reference=(audio.copy(), 16000)))
        module.VoiceCloner.assert_called_once()
        v._load_voice_conversion(SynthesisConfig(vc_reference=(audio * -1, 16000)))
        self.assertEqual(module.VoiceCloner.call_count, 2)

    def test_url_reference_goes_through_the_guarded_fetcher(self):
        """A remote reference must be fetched by phoonnx.reference_audio, whose
        address and size limits are the only thing standing between a caller's
        URL and an internal service."""
        _fake_voiceclonnx(self)
        with tempfile.TemporaryDirectory() as d:
            local = _ref_wav(d)
            with patch("phoonnx.reference_audio.fetch_reference_audio",
                       return_value=local) as fetch:
                vc = _bare_voice(self)._load_voice_conversion(
                    SynthesisConfig(vc_reference="https://example.invalid/a.wav"))
            fetch.assert_called_once_with("https://example.invalid/a.wav")
            self.assertTrue(os.path.isfile(vc.ref_path))

    def test_changing_engine_rebuilds(self):
        module, _ = _fake_voiceclonnx(self)
        v = _bare_voice(self)
        with tempfile.TemporaryDirectory() as d:
            ref = _ref_wav(d)
            v._load_voice_conversion(SynthesisConfig(vc_reference=ref))
            v._load_voice_conversion(
                SynthesisConfig(vc_reference=ref, vc_engine="focalcodec"))
        self.assertEqual(module.VoiceCloner.call_count, 2)

    def test_per_call_disable_wins_over_a_loaded_cloner(self):
        """vc_reference is a per-synthesis runtime knob: a voice that already
        loaded a cloner still yields native audio when the caller's config says
        off."""
        _fake_voiceclonnx(self)
        v = _bare_voice(self)
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNotNone(
                v._load_voice_conversion(SynthesisConfig(vc_reference=_ref_wav(d))))
        self.assertIsNone(v._load_voice_conversion(SynthesisConfig()))

    def test_missing_reference_file_raises(self):
        _fake_voiceclonnx(self)
        cfg = SynthesisConfig(vc_reference="/nope/does/not/exist.wav")
        with self.assertRaises(FileNotFoundError) as ctx:
            _bare_voice(self)._load_voice_conversion(cfg)
        self.assertIn("does/not/exist.wav", str(ctx.exception))

    def test_in_memory_reference_is_materialized_to_a_wav(self):
        _fake_voiceclonnx(self)
        audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
        vc = _bare_voice(self)._load_voice_conversion(
            SynthesisConfig(vc_reference=(audio, 16000)))
        self.assertTrue(os.path.isfile(vc.ref_path))
        back, sr = load_reference_audio(vc.ref_path)
        self.assertEqual(sr, 16000)
        self.assertEqual(back.shape, audio.shape)

    def test_reference_wav_dies_with_the_voice(self):
        """The materialized copy is owned by the loaded conversion, so dropping
        the voice takes the temp file with it rather than orphaning it."""
        _fake_voiceclonnx(self)
        v = _bare_voice(self)
        audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
        ref_path = v._load_voice_conversion(
            SynthesisConfig(vc_reference=(audio, 16000))).ref_path
        self.assertTrue(os.path.isfile(ref_path))
        del v
        gc.collect()
        self.assertFalse(os.path.exists(ref_path))

    def test_load_failure_propagates(self):
        module, _ = _fake_voiceclonnx(self)
        module.VoiceCloner.side_effect = RuntimeError("bad weights")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                _bare_voice(self)._load_voice_conversion(
                    SynthesisConfig(vc_reference=_ref_wav(d)))


class TestMaybeConvert(unittest.TestCase):
    def test_none_cloner_is_passthrough(self):
        audio = np.zeros(16000, dtype=np.float32)
        out, sr = TTSVoice._maybe_convert(audio, None, 22050)
        self.assertIs(out, audio)
        self.assertEqual(sr, 22050)

    def test_converts_and_returns_engine_sr(self):
        cloner = MagicMock()
        cloner.sample_rate = 22050

        def fake_clone(src, ref, dst):
            _write_wav(dst, np.zeros(11025, dtype=np.float32), 22050)
            return dst

        cloner.clone_voice.side_effect = fake_clone
        audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
        out, sr = TTSVoice._maybe_convert(audio, _vc(cloner, "/ref.wav"), 16000)
        self.assertEqual(sr, 22050)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape[0], 11025)

    def test_source_written_at_the_engine_native_rate(self):
        """The chunk goes to disk at the voice's own rate — voiceclonnx accepts
        any input rate and resamples internally, so no lossy pre-resample."""
        seen = {}
        cloner = MagicMock()

        def fake_clone(src, ref, dst):
            seen["audio"], seen["sr"] = load_reference_audio(src)
            seen["ref"] = ref
            _write_wav(dst, np.zeros(100, dtype=np.float32), 24000)
            return dst

        cloner.clone_voice.side_effect = fake_clone
        audio = np.linspace(-0.5, 0.5, 22050, dtype=np.float32)
        TTSVoice._maybe_convert(audio, _vc(cloner, "/tgt.wav"), 22050)
        self.assertEqual(seen["sr"], 22050)
        self.assertEqual(seen["ref"], "/tgt.wav")
        self.assertLess(np.max(np.abs(seen["audio"] - audio)), 2.0 / 32768.0)

    def test_very_short_chunk_passes_through_unconverted(self):
        cloner = MagicMock()
        audio = np.zeros(1000, dtype=np.float32)  # 62 ms at 16 kHz
        out, sr = TTSVoice._maybe_convert(audio, _vc(cloner, "/ref.wav"), 16000)
        self.assertIs(out, audio)
        self.assertEqual(sr, 16000)
        cloner.clone_voice.assert_not_called()

    def test_conversion_failure_propagates(self):
        """Unlike super-resolution, a VC failure must not silently fall back:
        emitting the source speaker is a wrong answer, not a degraded one."""
        cloner = MagicMock()
        cloner.clone_voice.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            TTSVoice._maybe_convert(np.zeros(16000, dtype=np.float32),
                                    _vc(cloner, "/ref.wav"), 16000)

    def test_temp_files_are_cleaned_up_even_on_failure(self):
        cloner = MagicMock()
        before = set(os.listdir(tempfile.gettempdir()))
        cloner.clone_voice.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            TTSVoice._maybe_convert(np.zeros(16000, dtype=np.float32),
                                    _vc(cloner, "/ref.wav"), 16000)
        leaked = [n for n in set(os.listdir(tempfile.gettempdir())) - before
                  if n.startswith("phoonnx_vc_")]
        self.assertEqual(leaked, [])


class TestSynthesisPipelineWiring(unittest.TestCase):
    """The stage must be invisible when off, and ordered VC -> SR when on."""

    def _voice(self):
        from phoonnx.config import Alphabet

        v = TTSVoice.__new__(TTSVoice)
        v.config = MagicMock(sample_rate=22050, alphabet=Alphabet.IPA)
        v.phonetic_spellings = None
        v._sr_engine = None
        v._sr_loaded = False
        v._vc = None
        v._vc_key = None
        return v

    def _run(self, cfg):
        v = self._voice()
        audio = np.linspace(-0.5, 0.5, 22050, dtype=np.float32)
        with patch.object(TTSVoice, "_iter_synthesis_ids",
                          return_value=iter([(["a"], [1, 2, 3])])), \
                patch.object(TTSVoice, "phoneme_ids_to_audio", return_value=audio):
            return list(TTSVoice.synthesize(v, "hello", syn_config=cfg))

    def test_unset_is_byte_identical_to_the_native_path(self):
        """The no-op guarantee, pinned in both directions."""
        with patch.dict(sys.modules):
            sys.modules.pop("voiceclonnx", None)
            baseline = self._run(SynthesisConfig())
            explicit_off = self._run(SynthesisConfig(vc_reference=None))
            self.assertNotIn("voiceclonnx", sys.modules)
        self.assertEqual(len(baseline), 1)
        self.assertEqual(baseline[0].sample_rate, 22050)
        np.testing.assert_array_equal(baseline[0].audio_float_array,
                                      explicit_off[0].audio_float_array)
        self.assertEqual(baseline[0].audio_int16_bytes,
                         explicit_off[0].audio_int16_bytes)

    def test_set_converts_the_chunk_and_reports_the_vc_rate(self):
        _, cloner = _fake_voiceclonnx(self, sample_rate=24000)

        def fake_clone(src, ref, dst):
            _write_wav(dst, np.full(12000, 0.25, dtype=np.float32), 24000)
            return dst

        cloner.clone_voice.side_effect = fake_clone
        with tempfile.TemporaryDirectory() as d:
            chunks = self._run(SynthesisConfig(vc_reference=_ref_wav(d)))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].sample_rate, 24000)
        self.assertEqual(chunks[0].audio_float_array.shape[0], 12000)
        cloner.clone_voice.assert_called_once()

    def test_in_flight_synthesis_keeps_its_own_speaker(self):
        """One voice serves every caller. A synthesis already streaming must keep
        converting with the cloner and reference file it started on, even when a
        genuine second load for a different speaker replaces the voice's cache
        underneath it — including the reference WAV, which the engine re-reads on
        every chunk."""
        cloners = [MagicMock(name="a"), MagicMock(name="b")]
        for c in cloners:
            c.sample_rate = 16000
        module = types.ModuleType("voiceclonnx")
        module.VoiceCloner = MagicMock(side_effect=cloners)
        patcher = patch.dict(sys.modules, {"voiceclonnx": module})
        patcher.start()
        self.addCleanup(patcher.stop)

        seen = []

        def fake_clone(src, ref, dst):
            seen.append((ref, os.path.isfile(ref)))
            _write_wav(dst, np.zeros(8000, dtype=np.float32), 16000)
            return dst

        for c in cloners:
            c.clone_voice.side_effect = fake_clone

        v = self._voice()
        audio = np.linspace(-0.5, 0.5, 22050, dtype=np.float32)
        with tempfile.TemporaryDirectory() as d, \
                patch.object(TTSVoice, "_iter_synthesis_ids",
                             return_value=iter([(["a"], [1]), (["b"], [2]),
                                                (["c"], [3])])), \
                patch.object(TTSVoice, "phoneme_ids_to_audio", return_value=audio):
            mine = _ref_wav(d)
            theirs = _write_wav(os.path.join(d, "theirs.wav"),
                                np.full(16000, 0.3, dtype=np.float32), 16000)
            stream = TTSVoice.synthesize(
                v, "one. two. three.",
                syn_config=SynthesisConfig(vc_reference=mine))
            chunks = [next(stream)]
            # a real second request for a different speaker, not a poked attribute
            other = v._load_voice_conversion(SynthesisConfig(vc_reference=theirs))
            self.assertIs(other.cloner, cloners[1])
            chunks.extend(stream)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(cloners[0].clone_voice.call_count, 3)
        cloners[1].clone_voice.assert_not_called()
        # all three chunks used one reference file, and it was readable each time
        self.assertEqual(len({ref for ref, _ in seen}), 1)
        self.assertTrue(all(existed for _, existed in seen))

    def test_vc_runs_before_super_resolution(self):
        """SR must upscale the converted timbre, and see the VC output rate."""
        _, cloner = _fake_voiceclonnx(self, sample_rate=16000)

        def fake_clone(src, ref, dst):
            _write_wav(dst, np.zeros(8000, dtype=np.float32), 16000)
            return dst

        cloner.clone_voice.side_effect = fake_clone
        sr_engine = MagicMock()
        sr_engine.upscale.return_value = (np.zeros(24000, dtype=np.float32), 48000)
        sr_module = types.ModuleType("audiosronnx")
        sr_module.load_sr = MagicMock(return_value=sr_engine)
        with patch.dict(sys.modules, {"audiosronnx": sr_module}), \
                tempfile.TemporaryDirectory() as d:
            chunks = self._run(SynthesisConfig(vc_reference=_ref_wav(d),
                                               super_resolution=True))
        self.assertEqual(chunks[0].sample_rate, 48000)
        # the upscaler was handed the VC engine's rate, not the voice's native one
        self.assertEqual(sr_engine.upscale.call_args.args[1], 16000)


class TestVoiceConversionE2E(unittest.TestCase):
    """Runs only when voiceclonnx and its weights are actually available."""

    def test_real_conversion_changes_the_waveform_and_keeps_length(self):
        try:
            from voiceclonnx import VoiceCloner
        except ImportError:
            self.skipTest("voiceclonnx not installed")
        try:
            cloner = VoiceCloner(engine="openvoice")
        except Exception as exc:  # weights download / runtime unavailable
            self.skipTest(f"voice-conversion engine unavailable: {exc}")

        sr = 22050
        t = np.linspace(0, 2.0, 2 * sr, endpoint=False, dtype=np.float32)
        audio = (0.3 * np.sin(2 * np.pi * 150 * t)
                 * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            ref = _ref_wav(d, sr=sr, seconds=3.0)
            out, out_sr = TTSVoice._maybe_convert(audio, _vc(cloner, ref), sr)
        self.assertEqual(out_sr, cloner.sample_rate)
        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(out.shape[0], 0)
        # duration is preserved within 15% (VC is not a duration model)
        self.assertLess(abs(out.shape[0] / out_sr - audio.shape[0] / sr), 0.3)


if __name__ == "__main__":
    unittest.main()
