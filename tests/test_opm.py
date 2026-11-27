"""Tests for the OpenVoiceOS TTS plugin integration (``phoonnx.opm``).

These exercise the plugin's wiring - voice selection, default fallback,
refresh-on-miss, and synthesis-parameter mapping - without any network
access or real ONNX models. The model manager, voices and ``wave`` are
mocked so the tests stay fast and hermetic.
"""
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from linguatag import default_region

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
        self.load_threads = "unset"
        self.precision = None
        self.variants = None

    def load(self, providers=None, precision=None, threads=None):
        self.load_providers = providers
        self.load_threads = threads
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

    def get_voice(self, voice_id):
        return self.voices.get(voice_id)

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

    def test_configured_threads_reach_the_voice_loader(self):
        v = _FakeVoiceInfo("OpenVoiceOS/known")
        plugin, _ = _make_plugin(config={"onnx_threads": 2}, voices=[v])
        plugin.get_model(v.voice_id)
        self.assertEqual(v.load_threads, 2)

    def test_unconfigured_threads_are_left_to_the_environment(self):
        v = _FakeVoiceInfo("OpenVoiceOS/known")
        plugin, _ = _make_plugin(voices=[v])
        plugin.get_model(v.voice_id)
        self.assertIsNone(v.load_threads)

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

    def test_get_tts_leaves_unset_scales_to_the_loaded_voice(self):
        # The index entry's scales can differ from the loaded voice's (x-vits takes
        # its scales from runtime.json at load), so the plugin passes none of them.
        v = _FakeVoiceInfo("OpenVoiceOS/v")
        v.config.length_scale, v.config.noise_scale, v.config.noise_w_scale = 9.0, 9.0, 9.0
        plugin, _ = _make_plugin(voices=[v])
        with patch.object(opm, "wave"):
            plugin.get_tts("hi", "/tmp/out.wav")
        cfg = self._synth_config_from_call(v.tts_voice)
        self.assertEqual((cfg.noise_scale, cfg.length_scale, cfg.noise_w_scale), (None, None, None))

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
        self.assertFalse(cfg.phonetic_spellings)

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

    def test_get_tts_without_a_voice_speaks_the_configured_voice(self):
        default = _FakeVoiceInfo("OpenVoiceOS/default")
        configured = _FakeVoiceInfo("OpenVoiceOS/configured")
        plugin, _ = _make_plugin(config={"voice": configured.voice_id}, voices=[default, configured])
        with patch.object(opm, "wave"):
            plugin.get_tts("hi", "/tmp/out.wav")
            plugin.get_tts("hi", "/tmp/out.wav", lang="en-us")
            plugin.get_tts("hi", "/tmp/out.wav", voice="default")
        self.assertEqual(configured.tts_voice.synthesize_wav.call_count, 3)
        self.assertFalse(default.tts_voice.synthesize_wav.called)

    def test_get_tts_in_another_language_takes_that_language_default(self):
        default = _FakeVoiceInfo("OpenVoiceOS/default", lang="pt-PT")
        configured = _FakeVoiceInfo("OpenVoiceOS/configured")
        plugin, _ = _make_plugin(config={"voice": configured.voice_id}, voices=[default, configured])
        with patch.object(opm, "wave"):
            plugin.get_tts("olá", "/tmp/out.wav", lang="pt-PT")
        self.assertTrue(default.tts_voice.synthesize_wav.called)
        self.assertFalse(configured.tts_voice.synthesize_wav.called)

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
    """``speaker_id`` / ``speaker`` config -> SynthesisConfig.speaker."""

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
                return cand.speaker
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
        self.assertIs(cfg.audio.super_resolution, False)
        self.assertIsNone(cfg.audio.super_resolution_model)

    def test_stretch_pace_is_off_unless_the_config_turns_it_on(self):
        self.assertIs(self._synthesize().audio.stretch_pace, False)
        self.assertIs(self._synthesize({"stretch_pace": True}).audio.stretch_pace, True)

    def test_mycroft_conf_block_reaches_synthesis_config(self):
        """Exactly the keys a user would write under
        ``tts/ovos-tts-plugin-phoonnx`` in mycroft.conf."""
        cfg = self._synthesize({"voice": "OpenVoiceOS/v",
                                "super_resolution": True,
                                "super_resolution_model": "novasr"})
        self.assertIs(cfg.audio.super_resolution, True)
        self.assertEqual(cfg.audio.super_resolution_model, "novasr")

    def test_enabled_without_model_defers_to_engine_default(self):
        cfg = self._synthesize({"super_resolution": True})
        self.assertIs(cfg.audio.super_resolution, True)
        self.assertIsNone(cfg.audio.super_resolution_model)

    def test_the_chunk_key_reaches_synthesis_config(self):
        self.assertEqual(self._synthesize({"chunk": "clause"}).stages.chunk, "clause")

    def test_no_chunk_key_reads_one_sentence_per_call(self):
        self.assertIsNone(self._synthesize().stages.chunk)


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


class TestLangDefaultVoiceOverrides(unittest.TestCase):
    """``lang2voice`` config and ``PHOONNX_DEFAULT_VOICE_<LANG>`` env vars pick
    the per-language default voice, so one container serves every language."""

    def test_lang2voice_selects_configured_voice(self):
        chosen = _FakeVoiceInfo("proxectonos/celtia", lang="gl-ES")
        other = _FakeVoiceInfo("some/other-voice", lang="gl-ES")
        plugin, _ = _make_plugin(
            config={"lang": "gl-ES", "lang2voice": {"gl": "proxectonos/celtia"}},
            voices=(other, chosen))
        self.assertEqual(plugin.get_default_voice("gl-ES").voice_id,
                         "proxectonos/celtia")

    def test_full_tag_beats_primary_subtag(self):
        br = _FakeVoiceInfo("voice/br", lang="pt-BR")
        pt = _FakeVoiceInfo("voice/pt", lang="pt-PT")
        plugin, _ = _make_plugin(
            config={"lang": "pt-PT",
                    "lang2voice": {"pt-br": "voice/br", "pt": "voice/pt"}},
            voices=(br, pt))
        self.assertEqual(plugin.get_default_voice("pt-BR").voice_id, "voice/br")
        self.assertEqual(plugin.get_default_voice("pt-PT").voice_id, "voice/pt")

    def test_env_var_sets_default_voice(self):
        chosen = _FakeVoiceInfo("env/voice", lang="ru-RU")
        other = _FakeVoiceInfo("some/other", lang="ru-RU")
        with patch.dict("os.environ", {"PHOONNX_DEFAULT_VOICE_RU": "env/voice"}):
            plugin, _ = _make_plugin(config={"lang": "ru-RU"},
                                     voices=(other, chosen))
            self.assertEqual(plugin.get_default_voice("ru-RU").voice_id,
                             "env/voice")

    def test_env_var_full_bcp47_suffix(self):
        br = _FakeVoiceInfo("env/br", lang="pt-BR")
        other = _FakeVoiceInfo("some/other", lang="pt-BR")
        with patch.dict("os.environ", {"PHOONNX_DEFAULT_VOICE_PT_BR": "env/br"}):
            plugin, _ = _make_plugin(config={"lang": "pt-BR"},
                                     voices=(other, br))
            self.assertEqual(plugin.get_default_voice("pt-BR").voice_id, "env/br")

    def test_config_beats_env(self):
        cfg = _FakeVoiceInfo("cfg/voice", lang="gl-ES")
        env = _FakeVoiceInfo("env/voice", lang="gl-ES")
        with patch.dict("os.environ", {"PHOONNX_DEFAULT_VOICE_GL": "env/voice"}):
            # env's voice is registered first (index-default voices[0]) and
            # cfg's second, so a resolver that silently fell through to the
            # index default instead of honouring lang2voice/env would return
            # "env/voice" here too - it must not.
            plugin, _ = _make_plugin(
                config={"lang": "gl-ES", "lang2voice": {"gl": "cfg/voice"}},
                voices=(env, cfg))
            self.assertEqual(plugin.get_default_voice("gl-ES").voice_id,
                             "cfg/voice")

    def test_unset_lang_falls_back_to_voice_index(self):
        first = _FakeVoiceInfo("index/first", lang="en-US")
        plugin, _ = _make_plugin(config={"lang": "en-US"}, voices=(first,))
        self.assertEqual(plugin.get_default_voice("en-US").voice_id,
                         "index/first")

    def test_env_ignores_unrelated_vars(self):
        with patch.dict("os.environ", {"PHOONNX_DEFAULT_VOICE_EU": "eu/voice",
                                       "UNRELATED_VAR": "nope"}):
            langs = opm.PhoonnxTTSPlugin.env_lang_voices()
        self.assertEqual(langs.get("eu"), "eu/voice")
        self.assertNotIn("unrelated-var", langs)

    def test_full_tag_config_key_matches_a_bare_request(self):
        # an admin who wrote the config key as the full tag ("gl-ES") must
        # still be matched by a caller that only ever sends the bare
        # primary subtag ("gl") - normalize_lang folds both to one canonical
        # form for a language whose primary subtag has a single default region
        chosen = _FakeVoiceInfo("proxectonos/celtia", lang="gl-ES")
        other = _FakeVoiceInfo("some/other", lang="gl-ES")
        plugin, _ = _make_plugin(
            config={"lang": "gl-ES", "lang2voice": {"gl-ES": "proxectonos/celtia"}},
            voices=(other, chosen))
        self.assertEqual(plugin.get_default_voice("gl").voice_id,
                         "proxectonos/celtia")

    def test_default_region_config_key_matches_a_bare_request(self):
        # a bare request names the default region of its language, so a key
        # written with that region is the configured voice for the bare tag
        chosen = _FakeVoiceInfo("voice/gb", lang="en-GB")
        other = _FakeVoiceInfo("some/other", lang="en-GB")
        plugin, _ = _make_plugin(
            config={"lang": "en-GB",
                    "lang2voice": {f"en-{default_region('en')}": "voice/gb"}},
            voices=(other, chosen))
        self.assertEqual(plugin.get_default_voice("en").voice_id, "voice/gb")

    def test_bare_config_key_serves_every_region(self):
        # a bare key is the primary subtag that every region falls back to,
        # including regions other than the language's default one
        chosen = _FakeVoiceInfo("voice/en", lang="en-GB")
        other = _FakeVoiceInfo("some/other", lang="en-US")
        plugin, _ = _make_plugin(
            config={"lang": "en-US", "lang2voice": {"en": "voice/en"}},
            voices=(other, chosen))
        for lang in ("en", "en-US", "en-AU", f"en-{default_region('en')}"):
            with self.subTest(lang=lang):
                self.assertEqual(plugin.get_default_voice(lang).voice_id, "voice/en")

    def test_underscore_config_key_matches_a_dashed_request(self):
        chosen = _FakeVoiceInfo("voice/br", lang="pt-BR")
        other = _FakeVoiceInfo("some/other", lang="pt-BR")
        plugin, _ = _make_plugin(
            config={"lang": "pt-BR", "lang2voice": {"pt_BR": "voice/br"}},
            voices=(other, chosen))
        self.assertEqual(plugin.get_default_voice("pt-BR").voice_id, "voice/br")

    def test_underscore_request_tag_still_matches(self):
        # a caller using locale-style "pt_BR" (underscore) rather than BCP-47
        # "pt-BR" must not silently disable the whole feature
        chosen = _FakeVoiceInfo("voice/br", lang="pt-BR")
        other = _FakeVoiceInfo("some/other", lang="pt-BR")
        plugin, _ = _make_plugin(
            config={"lang": "pt-BR", "lang2voice": {"pt-br": "voice/br"}},
            voices=(other, chosen))
        self.assertEqual(plugin.get_default_voice("pt_BR").voice_id, "voice/br")

    def test_unknown_configured_voice_falls_back_instead_of_raising(self):
        # a typo'd or since-removed lang2voice id is a deployment mistake, not
        # a reason to refuse to serve the language - and must not abort
        # plugin construction, which calls get_default_voice at boot
        fallback = _FakeVoiceInfo("index/fallback", lang="gl-ES")
        plugin, _ = _make_plugin(
            config={"lang": "gl-ES", "lang2voice": {"gl": "does/not-exist"}},
            voices=(fallback,))
        self.assertEqual(plugin.get_default_voice("gl-ES").voice_id,
                         "index/fallback")

    def test_unknown_configured_voice_via_env_falls_back_too(self):
        fallback = _FakeVoiceInfo("index/fallback", lang="ru-RU")
        with patch.dict("os.environ", {"PHOONNX_DEFAULT_VOICE_RU": "does/not-exist"}):
            plugin, _ = _make_plugin(config={"lang": "ru-RU"}, voices=(fallback,))
            self.assertEqual(plugin.get_default_voice("ru-RU").voice_id,
                             "index/fallback")

class TestPiperVoiceNames(unittest.TestCase):
    """A Piper client says ``lessac-medium`` or ``en_US-lessac-medium``; the catalog says
    ``piper/en_US-lessac-medium``."""

    def _plugin(self, lang="en-US"):
        ids = ["piper/en_US-lessac-medium", "piper/en_GB-alan-medium", "piper/fr_FR-mls-medium",
               "piper/de_DE-mls-medium", "piper/nl_NL-mls-medium", "supertonic/F1/en"]
        plugin, _ = _make_plugin({"lang": lang}, voices=[_FakeVoiceInfo(v) for v in ids])
        return plugin

    def test_the_locale_name_and_the_short_name_resolve(self):
        plugin = self._plugin()
        for name in ("piper/en_US-lessac-medium", "en_US-lessac-medium", "lessac-medium"):
            with self.subTest(name=name):
                self.assertEqual(plugin.get_voice_info(name).voice_id, "piper/en_US-lessac-medium")

    def test_a_piper_speaker_suffix_is_ignored(self):
        self.assertEqual(self._plugin().get_voice_info("lessac-medium#3").voice_id,
                         "piper/en_US-lessac-medium")

    def test_a_short_name_other_languages_share_goes_to_the_requested_language(self):
        plugin = self._plugin()
        for lang, want in (("fr-FR", "fr_FR"), ("de-DE", "de_DE"), ("nl-NL", "nl_NL")):
            with self.subTest(lang=lang):
                self.assertEqual(plugin.get_voice_info("mls-medium", lang).voice_id,
                                 f"piper/{want}-mls-medium")

    def test_the_plugin_language_settles_it_when_the_call_names_none(self):
        self.assertEqual(self._plugin("fr-FR").get_voice_info("mls-medium").voice_id,
                         "piper/fr_FR-mls-medium")

    def test_an_ambiguous_short_name_is_an_error_not_a_guess(self):
        with self.assertRaises(Exception) as ctx:
            self._plugin("en-US").get_voice_info("mls-medium")
        self.assertIn("Unknown voice", str(ctx.exception))

    def test_a_single_voice_in_another_language_is_still_found(self):
        self.assertEqual(self._plugin("en-US").get_voice_info("alan-medium").voice_id,
                         "piper/en_GB-alan-medium")

    def test_a_catalog_id_that_is_not_piper_is_left_alone(self):
        self.assertEqual(self._plugin().get_voice_info("supertonic/F1/en").voice_id,
                         "supertonic/F1/en")
        with self.assertRaises(Exception):
            self._plugin().get_voice_info("F1")

    def test_two_spellings_load_one_model(self):
        plugin = self._plugin()
        info = plugin.get_voice_info("lessac-medium")
        info.load = MagicMock(return_value=info.tts_voice)
        with patch.object(opm, "wave"):
            for name in ("lessac-medium", "en_US-lessac-medium", "piper/en_US-lessac-medium"):
                plugin.get_tts("hello", "/tmp/out.wav", lang="en-US", voice=name)
        self.assertEqual(info.load.call_count, 1)
        self.assertEqual(info.tts_voice.synthesize_wav.call_count, 3)

    # --- a catalog id with a Piper speaker suffix ---------------------------------------------

    def test_a_speaker_suffix_on_a_catalog_id_is_ignored(self):
        self.assertEqual(
            self._plugin().get_voice_info("piper/en_US-lessac-medium#3").voice_id,
            "piper/en_US-lessac-medium")

    def test_a_speaker_suffix_on_a_catalog_id_that_is_not_piper_is_still_unknown(self):
        with self.assertRaises(Exception) as ctx:
            self._plugin().get_voice_info("supertonic/F1/en#3")
        self.assertIn("Unknown voice", str(ctx.exception))

    def test_a_suffixed_catalog_id_synthesizes_with_the_requested_speaker(self):
        # Every spelling, suffixed or not, loads one model, and the speaker of the request
        # reaches synthesis the same way for each of them.
        plugin = self._plugin()
        info = plugin.get_voice_info("lessac-medium")
        info.load = MagicMock(return_value=info.tts_voice)
        names = ("lessac-medium#3", "en_US-lessac-medium#3", "piper/en_US-lessac-medium#3")
        with patch.object(opm, "wave"):
            for name in names:
                plugin.get_tts("hello", "/tmp/out.wav", lang="en-US", voice=name, speaker_id=2)
        self.assertEqual(info.load.call_count, 1)
        calls = info.tts_voice.synthesize_wav.call_args_list
        self.assertEqual(len(calls), len(names))
        speakers = [next(c.speaker for c in (*call.args, *call.kwargs.values())
                         if isinstance(c, SynthesisConfig)) for call in calls]
        self.assertEqual(speakers, [2, 2, 2])

    # --- a pinned voice named the way a Piper client names it ---------------------------------

    MB = 10 ** 6

    def _counted_plugin(self, **config):
        """A plugin whose voices count their loads and have a size and an artifact key."""
        ids = ["piper/en_US-lessac-medium", "piper/en_GB-alan-medium", "piper/fr_FR-mls-medium",
               "piper/de_DE-mls-medium", "piper/nl_NL-mls-medium", "supertonic/F1/en"]
        voices = {}
        for vid in ids:
            info = _FakeVoiceInfo(vid)
            info.load = MagicMock(return_value=info.tts_voice)
            info.disk_size = MagicMock(return_value=300 * self.MB)
            info.artifact_key = MagicMock(return_value=f"files:{vid}")
            voices[vid] = info
        config.setdefault("lang", "en-US")
        plugin, _ = _make_plugin(config, voices=list(voices.values()))
        return plugin, voices

    def test_a_pinned_piper_name_and_a_request_by_catalog_id_share_one_cache_entry(self):
        plugin, voices = self._counted_plugin(pinned_voices=["lessac-medium"],
                                              max_loaded_voices=1, max_loaded_bytes="1GB")
        lessac = voices["piper/en_US-lessac-medium"]
        self.assertEqual(lessac.load.call_count, 1, "the pin loads the voice at start")
        with patch.object(opm, "wave"):
            for name in ("piper/en_US-lessac-medium", "en_US-lessac-medium", "lessac-medium"):
                plugin.get_tts("hello", "/tmp/out.wav", lang="en-US", voice=name)
        self.assertEqual(lessac.load.call_count, 1, "a request finds the pinned entry")
        self.assertEqual(list(plugin.voice_cache.voices), ["piper/en_US-lessac-medium"])
        self.assertEqual(plugin.voice_cache.pinned_voices, ["piper/en_US-lessac-medium"])
        self.assertEqual(plugin.voice_cache.resident_bytes(), 300 * self.MB,
                         "the byte budget counts the voice once")

    def test_two_spellings_of_one_pin_are_one_pin(self):
        plugin, voices = self._counted_plugin(
            pinned_voices=["lessac-medium", "piper/en_US-lessac-medium"], max_loaded_voices=1)
        self.assertEqual(plugin.voice_cache.pinned_voices, ["piper/en_US-lessac-medium"])
        self.assertEqual(plugin.voice_cache.max_loaded_voices, 1)
        self.assertEqual(voices["piper/en_US-lessac-medium"].load.call_count, 1)

    def test_a_pinned_short_name_resolves_with_the_plugin_language(self):
        plugin, voices = self._counted_plugin(lang="fr-FR", pinned_voices="mls-medium")
        self.assertEqual(plugin.voice_cache.pinned_voices, ["piper/fr_FR-mls-medium"])
        self.assertEqual(list(plugin.voice_cache.voices), ["piper/fr_FR-mls-medium"])

    def test_a_pin_that_does_not_resolve_is_reported_as_a_failed_pin(self):
        # "mls-medium" is ambiguous for an English plugin: no voice fits, so nothing is pinned
        # under that name, and the operator is told so the way a pin that fails to load is.
        with patch.object(opm, "LOG") as log:
            plugin, voices = self._counted_plugin(pinned_voices=["mls-medium", "lessac-medium"])
        self.assertEqual(plugin.voice_cache.pinned_voices, ["piper/en_US-lessac-medium"])
        self.assertEqual(list(plugin.voice_cache.voices), ["piper/en_US-lessac-medium"])
        errors = [str(c.args[0]) for c in log.error.call_args_list]
        self.assertTrue(any("pinned voice 'mls-medium' failed to load" in e for e in errors),
                        errors)
