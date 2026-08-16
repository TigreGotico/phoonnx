"""Tests for the OpenVoiceOS TTS plugin integration (``phoonnx.opm``).

These exercise the plugin's wiring - voice selection, default fallback,
refresh-on-miss, and synthesis-parameter mapping - without any network
access or real ONNX models. The model manager, voices and ``wave`` are
mocked so the tests stay fast and hermetic.
"""
import unittest
from unittest.mock import MagicMock, patch

import phoonnx.opm as opm
from phoonnx.voice import SynthesisConfig


class _FakeVoiceConfig:
    """Stand-in for ``VoiceConfig`` carrying the synthesis defaults."""

    def __init__(self, speaker_id_map=None):
        self.noise_scale = 0.667
        self.length_scale = 1.0
        self.noise_w_scale = 0.8
        self.add_diacritics = False
        self.speaker_id_map = speaker_id_map or {}


class _FakeVoiceInfo:
    """Stand-in for ``TTSModelInfo``; ``load()`` yields a mock ``TTSVoice``."""

    def __init__(self, voice_id, lang="en-US", speaker_id_map=None):
        self.voice_id = voice_id
        self.lang = lang
        self.config = _FakeVoiceConfig(speaker_id_map=speaker_id_map)
        self.tts_voice = MagicMock(name=f"TTSVoice[{voice_id}]")
        self.load_providers = "unset"

    def load(self, providers=None):
        self.load_providers = providers
        return self.tts_voice


class _FakeManager:
    """Stand-in for ``TTSModelManager``.

    ``lazy`` voices are only merged into ``voices`` when
    ``merge_default_voices`` is called - this models a voice that is unknown
    until a refresh fetches it.
    """

    def __init__(self):
        self.voices = {}
        self.lazy = {}
        self.load_calls = 0
        self.merge_calls = 0

    def load(self):
        self.load_calls += 1

    def merge_default_voices(self, store=False):
        self.merge_calls += 1
        self.voices.update(self.lazy)

    def get_lang_voices(self, lang):
        # tests don't depend on lang matching; return everything registered
        return list(self.voices.values())


def _make_plugin(config=None, voices=(), lazy=()):
    """Build a plugin with a pre-populated fake manager (no I/O)."""
    config = dict(config or {})
    config.setdefault("lang", "en-US")
    manager = _FakeManager()
    for v in voices:
        manager.voices[v.voice_id] = v
    for v in lazy:
        manager.lazy[v.voice_id] = v
    with patch.object(opm, "TTSModelManager", return_value=manager):
        plugin = opm.PhoonnxTTSPlugin(config=config)
    # the OVOS base merges the global mycroft.conf into self.config; pin it to
    # the test's exact config so get_tts param-mapping is deterministic
    plugin.config = config
    return plugin, manager


class TestPluginEntryPoint(unittest.TestCase):
    def test_entry_point_resolves_to_plugin(self):
        """The declared opm.tts entry point loads the plugin class."""
        from importlib.metadata import entry_points

        eps = [e for e in entry_points(group="opm.tts")
               if e.name == "ovos-tts-plugin-phoonnx"]
        self.assertTrue(eps, "ovos-tts-plugin-phoonnx entry point not registered")
        self.assertIs(eps[0].load(), opm.PhoonnxTTSPlugin)

    def test_no_legacy_entry_point_group(self):
        """The plugin is not declared under the deprecated mycroft.plugin.tts
        group, which ovos-plugin-manager only scans for backward compatibility
        and flags with a warning on every discovery pass."""
        from importlib.metadata import entry_points

        legacy = [e.name for e in entry_points(group="mycroft.plugin.tts")
                  if e.name == "ovos-tts-plugin-phoonnx"]
        self.assertEqual(legacy, [])

    def test_is_a_tts_subclass(self):
        from ovos_plugin_manager.templates.tts import TTS
        self.assertTrue(issubclass(opm.PhoonnxTTSPlugin, TTS))


class TestInit(unittest.TestCase):
    def test_init_resolves_default_voice_when_unconfigured(self):
        v = _FakeVoiceInfo("OpenVoiceOS/en_default")
        plugin, mgr = _make_plugin(voices=[v])
        self.assertEqual(mgr.load_calls, 1)
        self.assertEqual(plugin.voice_info.voice_id, v.voice_id)
        # resolved, not fetched: loading is deferred to the first synthesis
        self.assertEqual(plugin.voice_cache.voices, {})

    def test_init_resolves_configured_voice(self):
        a = _FakeVoiceInfo("OpenVoiceOS/a")
        b = _FakeVoiceInfo("OpenVoiceOS/b")
        plugin, _ = _make_plugin(config={"voice": "OpenVoiceOS/b"}, voices=[a, b])
        self.assertEqual(plugin.voice_info.voice_id, "OpenVoiceOS/b")
        self.assertEqual(plugin.voice_cache.voices, {})


class TestVoiceResolution(unittest.TestCase):
    def test_get_default_voice_refreshes_when_missing(self):
        lazy = _FakeVoiceInfo("OpenVoiceOS/lazy")
        # no eager voices: init refresh (lazy merged) gives a default
        plugin, mgr = _make_plugin(lazy=[lazy])
        # merge happened during init; default voice resolved from lazy
        self.assertEqual(plugin.voice_info.voice_id, "OpenVoiceOS/lazy")

    def test_get_default_voice_raises_when_none(self):
        with self.assertRaises(ValueError):
            _make_plugin()  # no voices anywhere

    def test_get_model_unknown_voice_raises(self):
        v = _FakeVoiceInfo("OpenVoiceOS/known")
        plugin, _ = _make_plugin(voices=[v])
        with self.assertRaises(Exception):
            plugin.get_model("OpenVoiceOS/nope")

    def test_configured_providers_reach_the_voice_loader(self):
        v = _FakeVoiceInfo("OpenVoiceOS/known")
        plugin, _ = _make_plugin(
            config={"onnx_providers": ["ROCMExecutionProvider", "CPUExecutionProvider"]},
            voices=[v])
        plugin.get_model(v.voice_id)
        self.assertEqual(v.load_providers,
                         ["ROCMExecutionProvider", "CPUExecutionProvider"])

    def test_single_provider_string_is_wrapped_in_a_list(self):
        v = _FakeVoiceInfo("OpenVoiceOS/known")
        plugin, _ = _make_plugin(config={"providers": "ROCMExecutionProvider"},
                                 voices=[v])
        plugin.get_model(v.voice_id)
        self.assertEqual(v.load_providers, ["ROCMExecutionProvider"])

    def test_unconfigured_providers_are_left_to_autodetection(self):
        v = _FakeVoiceInfo("OpenVoiceOS/known")
        plugin, _ = _make_plugin(voices=[v])
        plugin.get_model(v.voice_id)
        self.assertIsNone(v.load_providers)

    def test_get_model_caches(self):
        v = _FakeVoiceInfo("OpenVoiceOS/known")
        plugin, _ = _make_plugin(voices=[v])
        first = plugin.get_model(v.voice_id)
        second = plugin.get_model(v.voice_id)
        self.assertIs(first, second)


class TestGetTts(unittest.TestCase):
    def _synth_config_from_call(self, tts_voice):
        """Return the SynthesisConfig passed to synthesize_wav."""
        self.assertTrue(tts_voice.synthesize_wav.called)
        args, kwargs = tts_voice.synthesize_wav.call_args
        for cand in (*args, *kwargs.values()):
            if isinstance(cand, SynthesisConfig):
                return cand
        self.fail("no SynthesisConfig passed to synthesize_wav")

    def test_get_tts_returns_wavfile_and_none(self):
        v = _FakeVoiceInfo("OpenVoiceOS/v")
        plugin, _ = _make_plugin(voices=[v])
        with patch.object(opm, "wave"):
            out = plugin.get_tts("hello world", "/tmp/out.wav")
        self.assertEqual(out, ("/tmp/out.wav", None))

    def test_get_tts_uses_voice_config_defaults(self):
        v = _FakeVoiceInfo("OpenVoiceOS/v")
        plugin, _ = _make_plugin(voices=[v])
        with patch.object(opm, "wave"):
            plugin.get_tts("hi", "/tmp/out.wav")
        cfg = self._synth_config_from_call(v.tts_voice)
        self.assertAlmostEqual(cfg.noise_scale, 0.667)
        self.assertAlmostEqual(cfg.length_scale, 1.0)
        self.assertAlmostEqual(cfg.noise_w_scale, 0.8)

    def test_documented_underscore_config_keys_are_honoured(self):
        """Regression: documented keys (noise_scale, length_scale, noise_w,
        enable_phonetic_spellings) used to be ignored due to key drift."""
        v = _FakeVoiceInfo("OpenVoiceOS/v")
        plugin, _ = _make_plugin(config={
            "noise_scale": 0.1,
            "length_scale": 2.0,
            "noise_w": 0.3,
            "enable_phonetic_spellings": False,
        }, voices=[v])
        with patch.object(opm, "wave"):
            plugin.get_tts("hi", "/tmp/out.wav")
        cfg = self._synth_config_from_call(v.tts_voice)
        self.assertAlmostEqual(cfg.noise_scale, 0.1)
        self.assertAlmostEqual(cfg.length_scale, 2.0)
        self.assertAlmostEqual(cfg.noise_w_scale, 0.3)
        self.assertFalse(cfg.enable_phonetic_spellings)

    def test_legacy_hyphen_config_keys_still_work(self):
        """Backwards compat: old hyphenated keys remain accepted as fallbacks."""
        v = _FakeVoiceInfo("OpenVoiceOS/v")
        plugin, _ = _make_plugin(config={
            "noise-scale": 0.2,
            "length-scale": 3.0,
            "noise-w": 0.4,
        }, voices=[v])
        with patch.object(opm, "wave"):
            plugin.get_tts("hi", "/tmp/out.wav")
        cfg = self._synth_config_from_call(v.tts_voice)
        self.assertAlmostEqual(cfg.noise_scale, 0.2)
        self.assertAlmostEqual(cfg.length_scale, 3.0)
        self.assertAlmostEqual(cfg.noise_w_scale, 0.4)

    def test_explicit_voice_not_preloaded_no_keyerror(self):
        """Regression: a configured/explicit voice that needs a refresh used to
        KeyError because the model info was read before get_model refreshed."""
        default = _FakeVoiceInfo("OpenVoiceOS/default")
        lazy = _FakeVoiceInfo("OpenVoiceOS/lazy")
        plugin, mgr = _make_plugin(voices=[default])
        # 'lazy' is only discoverable via a refresh, not yet registered
        mgr.lazy = {lazy.voice_id: lazy}
        self.assertNotIn(lazy.voice_id, mgr.voices)
        with patch.object(opm, "wave"):
            out = plugin.get_tts("hi", "/tmp/out.wav", voice=lazy.voice_id)
        self.assertEqual(out, ("/tmp/out.wav", None))
        self.assertTrue(lazy.tts_voice.synthesize_wav.called)


class TestSpeakerSelection(unittest.TestCase):
    """``speaker_id`` / ``speaker`` config -> SynthesisConfig.speaker_id."""

    # the Catalan multiaccent matxa model: accent/name -> id
    CAT_MAP = {"quim": 0, "olga": 1, "grau": 2, "elia": 3,
               "pere": 4, "emma": 5, "lluc": 6, "gina": 7}

    def _speaker_id_for(self, config, speaker_id_map=None):
        v = _FakeVoiceInfo("OpenVoiceOS/v", speaker_id_map=speaker_id_map)
        plugin, _ = _make_plugin(config=config, voices=[v])
        with patch.object(opm, "wave"):
            plugin.get_tts("hi", "/tmp/out.wav")
        args, kwargs = v.tts_voice.synthesize_wav.call_args
        for cand in (*args, *kwargs.values()):
            if isinstance(cand, SynthesisConfig):
                return cand.speaker_id
        self.fail("no SynthesisConfig passed to synthesize_wav")

    def test_no_speaker_config_is_none(self):
        self.assertIsNone(self._speaker_id_for({}))

    def test_integer_speaker_id_passthrough(self):
        self.assertEqual(self._speaker_id_for({"speaker_id": 3}), 3)

    def test_digit_string_speaker_id(self):
        self.assertEqual(self._speaker_id_for({"speaker_id": "5"}), 5)

    def test_speaker_name_resolved_via_map(self):
        self.assertEqual(
            self._speaker_id_for({"speaker": "elia"}, self.CAT_MAP), 3)

    def test_accent_qualified_speaker_name(self):
        self.assertEqual(
            self._speaker_id_for({"speaker": "central/elia"}, self.CAT_MAP), 3)

    def test_unknown_speaker_name_is_none(self):
        self.assertIsNone(
            self._speaker_id_for({"speaker": "nope"}, self.CAT_MAP))

    def test_speaker_id_takes_priority_over_name(self):
        self.assertEqual(
            self._speaker_id_for({"speaker_id": 6, "speaker": "elia"},
                                 self.CAT_MAP), 6)


class TestSuperResolutionConfigThreading(unittest.TestCase):
    """The plugin maps the ``super_resolution`` / ``super_resolution_model`` keys
    from a mycroft.conf TTS block onto the SynthesisConfig it hands to the core
    voice; the upscaling itself lives in TTSVoice.synthesize (covered in
    test_super_resolution.py)."""

    def _synth_config(self, tts_voice):
        args, kwargs = tts_voice.synthesize_wav.call_args
        for cand in (*args, *kwargs.values()):
            if isinstance(cand, SynthesisConfig):
                return cand
        self.fail("no SynthesisConfig passed to synthesize_wav")

    def _synthesize(self, config=None):
        v = _FakeVoiceInfo("OpenVoiceOS/v")
        plugin, _ = _make_plugin(config=config, voices=[v])
        with patch.object(opm, "wave"):
            out = plugin.get_tts("hi", "/tmp/out.wav")
        self.assertEqual(out, ("/tmp/out.wav", None))
        return self._synth_config(v.tts_voice)

    def test_no_config_is_disabled(self):
        cfg = self._synthesize()
        self.assertIs(cfg.super_resolution, False)
        self.assertIsNone(cfg.super_resolution_model)

    def test_mycroft_conf_block_reaches_synthesis_config(self):
        """Exactly the keys a user would write under
        ``tts/ovos-tts-plugin-phoonnx`` in mycroft.conf."""
        cfg = self._synthesize({"voice": "OpenVoiceOS/v",
                                "super_resolution": True,
                                "super_resolution_model": "novasr"})
        self.assertIs(cfg.super_resolution, True)
        self.assertEqual(cfg.super_resolution_model, "novasr")

    def test_enabled_without_model_defers_to_engine_default(self):
        cfg = self._synthesize({"super_resolution": True})
        self.assertIs(cfg.super_resolution, True)
        self.assertIsNone(cfg.super_resolution_model)


if __name__ == "__main__":
    unittest.main()


class TestVoiceResolvedWithoutLoading(unittest.TestCase):
    """Boot resolves the configured voice but does not fetch it.

    A voice that was named explicitly and does not exist is a configuration
    error and must still raise — substituting a different voice would silently
    speak in the wrong one. An unset voice (or "default") resolves to the
    language's default. Loading is deferred so that fetching a model is never
    what decides whether the TTS service can start.
    """

    def test_unknown_named_voice_raises(self):
        with self.assertRaises(Exception) as ctx:
            opm.PhoonnxTTSPlugin(config={"voice": "totally-unknown-voice"})
        self.assertIn("totally-unknown-voice", str(ctx.exception))

    def test_unknown_lang_raises(self):
        with self.assertRaises(ValueError):
            opm.PhoonnxTTSPlugin(config={"lang": "zzz"})

    def test_default_voice_resolves_without_loading(self):
        for voice in ("default", None):
            with self.subTest(voice=voice):
                cfg = {"lang": "en-US"}
                if voice is not None:
                    cfg["voice"] = voice
                plugin = opm.PhoonnxTTSPlugin(config=cfg)
                self.assertIsNotNone(plugin.voice_info)
                self.assertEqual(plugin.voice_cache.voices, {},
                                 "boot must not load (download) the model")
