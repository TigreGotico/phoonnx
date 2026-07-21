import unittest

from phoonnx.config import Alphabet, Engine, PhonemeType, VoiceConfig
from phoonnx.engines.vits import VitsAdapter
from phoonnx.phonemizers import GraphemePhonemizer
from phoonnx.voice import TTSVoice


class TestPiperLangCodeMayBeAbsent(unittest.TestCase):
    """config.py ~L387-389: piper's Arabic-diacritics check must not crash
    when neither the config nor the caller supplies a language code."""

    def _piper_cfg(self, **extra):
        cfg = {"piper_version": "1.0.0", "phoneme_type": "espeak",
               "phoneme_id_map": {"a": [0], "b": [1]}}
        cfg.update(extra)
        return cfg

    def test_missing_language_and_espeak_keys_does_not_raise(self):
        # no config.language, no config.espeak.voice, no explicit lang_code kwarg
        vc = VoiceConfig.from_dict(self._piper_cfg())
        self.assertFalse(vc.add_diacritics)

    def test_explicit_none_lang_code_does_not_raise(self):
        vc = VoiceConfig.from_dict(self._piper_cfg(), lang_code=None)
        self.assertFalse(vc.add_diacritics)

    def test_empty_language_dict_does_not_raise(self):
        cfg = self._piper_cfg(language={})
        vc = VoiceConfig.from_dict(cfg)
        self.assertFalse(vc.add_diacritics)

    def test_empty_espeak_dict_does_not_raise(self):
        cfg = self._piper_cfg(espeak={})
        vc = VoiceConfig.from_dict(cfg)
        self.assertFalse(vc.add_diacritics)

    def test_arabic_still_detected_when_lang_code_present(self):
        cfg = self._piper_cfg(espeak={"voice": "ar"})
        vc = VoiceConfig.from_dict(cfg)
        self.assertTrue(vc.add_diacritics)


class TestTransformersVocabBranchMissingTokenizerConfigKeys(unittest.TestCase):
    """config.py ~L446-454: the transformers/vocab branch must not KeyError
    when tokenizer_config is present but missing individual keys, or absent
    entirely with no "blank" key anywhere in config."""

    def test_no_tokenizer_config_and_no_blank_key_does_not_raise(self):
        vocab = {"_": 0, "a": 1, "b": 2}
        vc = VoiceConfig.from_dict({}, vocab=vocab, phoneme_type="graphemes",
                                    alphabet="unicode", lang_code="en")
        self.assertIsNotNone(vc.tokenizer)

    def test_tokenizer_config_missing_pad_token_key_does_not_raise(self):
        vocab = {"_": 0, "a": 1, "b": 2}
        tok_cfg = {"add_blank": False, "language": "pt"}  # no pad_token
        vc = VoiceConfig.from_dict({}, vocab=vocab, tokenizer_config=tok_cfg,
                                    phoneme_type="graphemes", alphabet="unicode")
        self.assertIsNotNone(vc.tokenizer)
        self.assertEqual(vc.lang_code, "pt")

    def test_tokenizer_config_missing_add_blank_key_does_not_raise(self):
        vocab = {"_": 0, "a": 1, "b": 2}
        tok_cfg = {"language": "de", "pad_token": "_"}  # no add_blank
        vc = VoiceConfig.from_dict({}, vocab=vocab, tokenizer_config=tok_cfg,
                                    phoneme_type="graphemes", alphabet="unicode")
        self.assertIsNotNone(vc.tokenizer)

    def test_empty_tokenizer_config_does_not_raise(self):
        vocab = {"_": 0, "a": 1, "b": 2}
        vc = VoiceConfig.from_dict({}, vocab=vocab, tokenizer_config={},
                                    phoneme_type="graphemes", alphabet="unicode",
                                    lang_code="en")
        self.assertIsNotNone(vc.tokenizer)


class TestVoiceAdapterResolutionForMissingEngines(unittest.TestCase):
    """voice.py ~L227-238: Engine.PHOONNX and Engine.TRANSFORMERS have no
    dedicated entry in the adapter registry. Resolution must not silently
    fall through to session-only auto-detection (which loses the already
    loaded config) — it must land on the shared VITS-family adapter, same
    as piper/mimic3/coqui."""

    def _config(self, engine):
        return VoiceConfig(num_symbols=4, num_speakers=1, num_langs=1, sample_rate=16000,
                            lang_code="en", phoneme_type=PhonemeType.GRAPHEMES,
                            alphabet=Alphabet.UNICODE, phonemizer_model=None, engine=engine)

    def test_phoonnx_engine_resolves_to_vits_adapter(self):
        voice = TTSVoice(session=None, config=self._config(Engine.PHOONNX),
                          phonemizer=GraphemePhonemizer())
        self.assertIsInstance(voice.adapter, VitsAdapter)

    def test_transformers_engine_resolves_to_vits_adapter(self):
        voice = TTSVoice(session=None, config=self._config(Engine.TRANSFORMERS),
                          phonemizer=GraphemePhonemizer())
        self.assertIsInstance(voice.adapter, VitsAdapter)

    def test_piper_engine_still_resolves_to_vits_adapter(self):
        # regression guard: the existing piper/mimic3/coqui aliasing must
        # keep working after adding phoonnx/transformers to the same branch
        voice = TTSVoice(session=None, config=self._config(Engine.PIPER),
                          phonemizer=GraphemePhonemizer())
        self.assertIsInstance(voice.adapter, VitsAdapter)


if __name__ == "__main__":
    unittest.main()
