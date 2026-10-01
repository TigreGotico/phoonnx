"""Tests for the typed, eager language-support check at voice load.

Before this check existed, a voice whose language no scriptconv phonemizer
serves loaded fine and only failed once synthesis started, with an opaque
``ValueError: unsupported language code: ...`` raised
deep inside scriptconv. ``check_lang_supported`` runs the same registry
lookup ``TTSVoice.__post_init__`` and ``TTSModelInfo.load`` use, without
downloading anything or instantiating a phonemizer backend, so the failure
surfaces at load time instead -- and, for ``TTSModelInfo.load``, before the
(possibly multi-MB) model weights are even fetched.
"""
import glob
import json
import unittest

import onnxruntime

from phoonnx.config import (
    Alphabet,
    Engine,
    PhonemeType,
    UnsupportedVoiceLanguage,
    VoiceConfig,
    check_lang_supported,
)
from phoonnx.util import normalize_lang
from phoonnx.voice import TTSVoice

from tests.test_model_manager import HUB, HubTestCase


class TestCheckLangSupported(unittest.TestCase):
    def test_unserved_language_raises_typed_error_naming_voice_and_lang(self):
        with self.assertRaises(UnsupportedVoiceLanguage) as ctx:
            check_lang_supported("zzz_joao", "zzz", PhonemeType.ESPEAK)
        err = ctx.exception
        self.assertEqual(err.voice, "zzz_joao")
        self.assertEqual(err.lang_code, "zzz")
        self.assertIn("zzz_joao", str(err))
        self.assertIn("zzz", str(err))

    def test_supported_language_does_not_raise(self):
        # Basque is served by scriptconv's espeak backend.
        check_lang_supported("hitz-eu", "eu", PhonemeType.ESPEAK)

    def test_grapheme_voice_with_exotic_lang_is_not_rejected(self):
        # Grapheme/unicode phonemizers have no language restriction (no
        # ``get_lang`` classmethod) -- any lang_code, however exotic, passes.
        check_lang_supported("some-graphemes-voice", "zzz", PhonemeType.GRAPHEMES)
        check_lang_supported("some-unicode-voice", "zzz", PhonemeType.UNICODE)

    def test_missing_lang_code_is_skipped(self):
        check_lang_supported("no-lang-voice", None, PhonemeType.ESPEAK)

    def test_missing_optional_backend_package_is_skipped_not_rejected(self):
        # euskaphone's get_lang imports its optional backing package to
        # answer; when that package is absent it raises ImportError, which
        # must be treated as "can't tell" (skip), never as a load-time
        # rejection just because an extra wasn't installed.
        check_lang_supported("some-eu-voice", "eu", PhonemeType.EUSKAPHONE)


class TestCheckLangSupportedOnTheVoiceIndex(unittest.TestCase):
    """The strongest guard this feature can have: sweep every bundled voice
    and confirm the check accepts all of them. Every entry's effective
    language -- ``phonemizer_lang`` when set (the same override TTSModelInfo
    applies before load, see ``phonemizer_lang`` on TTSModelInfo), else
    ``lang`` -- must resolve to a supported phonemizer language. A voice
    landing in the index with an unresolvable language is a catalog/phonemizer
    inventory mismatch, not something this test should special-case away."""

    def test_sweep_bundled_voice_index_finds_no_rejections(self):
        rejections = []
        total = 0
        for path in sorted(glob.glob("phoonnx/voice_index/*.json")):
            with open(path) as f:
                data = json.load(f)
            for voice_id, entry in data.items():
                total += 1
                lang = entry.get("phonemizer_lang") or entry.get("lang")
                phoneme_type = entry.get("phoneme_type")
                if not lang or not phoneme_type:
                    continue
                try:
                    phoneme_type = PhonemeType(phoneme_type)
                except ValueError:
                    continue
                try:
                    check_lang_supported(voice_id, normalize_lang(lang), phoneme_type)
                except UnsupportedVoiceLanguage:
                    rejections.append(voice_id)

        # Sanity floor: this must actually be sweeping the real, sizeable
        # index, not an empty/truncated one.
        self.assertGreater(total, 1000)
        self.assertEqual(rejections, [])


class TestTTSVoicePostInitRejectsUnservedLanguage(unittest.TestCase):
    """Exercises the actual __post_init__ wiring, not just the helper
    function -- deleting the check_lang_supported call in voice.py must
    turn this red."""

    def test_ttsvoice_construction_raises_for_unserved_language(self):
        config = VoiceConfig(num_symbols=4, num_speakers=1, num_langs=1, sample_rate=16000,
                             lang_code="zzz", phoneme_type=PhonemeType.ESPEAK,
                             alphabet=Alphabet.IPA, phonemizer_model=None,
                             engine=Engine.PIPER)
        with self.assertRaises(UnsupportedVoiceLanguage) as ctx:
            TTSVoice(session=None, config=config, model_path="/cache/models--x--y/snapshots/abc123/zzz_joao.onnx")
        # naming the voice, not the 130-char HF snapshot cache path (review N1)
        self.assertIn("zzz_joao", str(ctx.exception))
        self.assertNotIn("snapshots", str(ctx.exception))

    def test_ttsvoice_construction_does_not_raise_for_served_language(self):
        config = VoiceConfig(num_symbols=4, num_speakers=1, num_langs=1, sample_rate=16000,
                             lang_code="eu", phoneme_type=PhonemeType.ESPEAK,
                             alphabet=Alphabet.IPA, phonemizer_model=None,
                             engine=Engine.PIPER)
        voice = TTSVoice(session=None, config=config, model_path="/cache/eu-voice.onnx")
        self.assertIsNotNone(voice.phonemizer)


class TestTTSModelInfoLoadRejectsBeforeDownloadingTheModel(HubTestCase):
    """Exercises the TTSModelInfo.load wiring: the language check must run
    (and raise) before the multi-MB model weights are downloaded."""

    def test_load_raises_before_downloading_model_weights(self):
        self.hub.stage(f"{HUB}/config.json", {"phoneme_type": "espeak", "alphabet": "ipa"})
        # deliberately NOT staged: model.onnx -- if the code tried to
        # download it before rejecting, this test would fail with
        # EntryNotFoundError instead of UnsupportedVoiceLanguage.
        info = self.make_info(voice_id="piper_community/raphaelmerx/zzz_joao",
                              lang="zzz",
                              config_url=f"{HUB}/config.json",
                              phoneme_type=PhonemeType.ESPEAK)
        with self.assertRaises(UnsupportedVoiceLanguage) as ctx:
            info.load()
        self.assertIn("zzz_joao", str(ctx.exception))
        downloaded_files = {filename for _, filename, _ in self.hub.downloads}
        self.assertNotIn("model.onnx", downloaded_files,
                         "model weights must not be fetched before the language check")


if __name__ == "__main__":
    unittest.main()
