import os
import tempfile
import unittest

from phoonnx.config import Alphabet, Engine, PhonemeType, VoiceConfig


class TestIsMimic3(unittest.TestCase):
    def test_missing_phonemizer_key_is_false(self):
        self.assertFalse(VoiceConfig.is_mimic3({"phonemes": {}}))

    def test_non_string_phonemizer_is_false(self):
        self.assertFalse(VoiceConfig.is_mimic3({"phonemizer": 123, "phonemes": {}}))

    def test_missing_phonemes_key_is_false(self):
        self.assertFalse(VoiceConfig.is_mimic3({"phonemizer": "gruut"}))

    def test_phonemes_not_a_dict_is_false(self):
        self.assertFalse(VoiceConfig.is_mimic3({"phonemizer": "gruut", "phonemes": ["a", "b"]}))

    def test_unknown_phonemizer_string_returns_false_not_raise(self):
        # detection must degrade gracefully on an unrecognised value, never raise
        self.assertFalse(VoiceConfig.is_mimic3({"phonemizer": "not_a_real_phonemizer", "phonemes": {}}))

    def test_each_known_phonemizer_value_is_true(self):
        for val in ("symbols", "gruut", "espeak", "epitran"):
            with self.subTest(val=val):
                self.assertTrue(VoiceConfig.is_mimic3({"phonemizer": val, "phonemes": {}}))


class TestIsPiper(unittest.TestCase):
    def test_piper_version_key_short_circuits_true(self):
        self.assertTrue(VoiceConfig.is_piper({"piper_version": "1.0.0"}))

    def test_declared_non_piper_engine_wins_over_shape_sniffing(self):
        # a canonical coqui/phoonnx config with an espeak-shaped phoneme_id_map
        # must not be misdetected as piper just because it phonemizes with espeak
        cfg = {"engine": "coqui", "phoneme_type": "espeak",
               "phoneme_id_map": {"a": [0], "b": [1]}}
        self.assertFalse(VoiceConfig.is_piper(cfg))

    def test_missing_phoneme_type_is_false(self):
        self.assertFalse(VoiceConfig.is_piper({"phoneme_id_map": {"a": [0]}}))

    def test_non_string_phoneme_type_is_false(self):
        self.assertFalse(VoiceConfig.is_piper({"phoneme_type": 5, "phoneme_id_map": {"a": [0]}}))

    def test_flat_phoneme_id_map_is_not_piper(self):
        # piper's phoneme_id_map values are lists; a flat phoneme->int map is the
        # canonical phoonnx/coqui shape, not piper
        cfg = {"phoneme_type": "espeak", "phoneme_id_map": {"a": 0, "b": 1}}
        self.assertFalse(VoiceConfig.is_piper(cfg))

    def test_empty_phoneme_id_map_is_false(self):
        cfg = {"phoneme_type": "espeak", "phoneme_id_map": {}}
        self.assertFalse(VoiceConfig.is_piper(cfg))

    def test_missing_phoneme_id_map_is_false(self):
        self.assertFalse(VoiceConfig.is_piper({"phoneme_type": "espeak"}))

    def test_unknown_phonemizer_type_is_false(self):
        cfg = {"phoneme_type": "not_espeak_or_text", "phoneme_id_map": {"a": [0]}}
        self.assertFalse(VoiceConfig.is_piper(cfg))

    def test_valid_piper_shape_is_true(self):
        cfg = {"phoneme_type": "espeak", "phoneme_id_map": {"a": [0], "b": [1]}}
        self.assertTrue(VoiceConfig.is_piper(cfg))


class TestIsCoquiVits(unittest.TestCase):
    def test_missing_characters_key_is_false(self):
        self.assertFalse(VoiceConfig.is_coqui_vits({}))

    def test_characters_not_a_dict_is_false(self):
        self.assertFalse(VoiceConfig.is_coqui_vits({"characters": "nope"}))

    def test_unrecognised_characters_class_is_false(self):
        cfg = {"characters": {"characters_class": "some.other.Class"}}
        self.assertFalse(VoiceConfig.is_coqui_vits(cfg))

    def test_recognised_vits_characters_class_is_true(self):
        cfg = {"characters": {"characters_class": "TTS.tts.models.vits.VitsCharacters"}}
        self.assertTrue(VoiceConfig.is_coqui_vits(cfg))

    def test_recognised_graphemes_class_is_true(self):
        cfg = {"characters": {"characters_class": "TTS.tts.utils.text.characters.Graphemes"}}
        self.assertTrue(VoiceConfig.is_coqui_vits(cfg))


class TestIsPhoonnx(unittest.TestCase):
    def test_missing_version_key_is_false(self):
        self.assertFalse(VoiceConfig.is_phoonnx({}))

    def test_version_key_present_is_true(self):
        self.assertTrue(VoiceConfig.is_phoonnx({"phoonnx_version": "1.0"}))


class TestFromDictChatterbox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer

        raw = Tokenizer(BPE(unk_token="[UNK]"))
        raw.train_from_iterator(["hello world", "foo bar baz"],
                                 trainer=BpeTrainer(special_tokens=["[UNK]"]))
        cls.tmpdir = tempfile.mkdtemp()
        cls.bpe_path = os.path.join(cls.tmpdir, "tokenizer.json")
        raw.save(cls.bpe_path)

    def test_missing_bpe_tokenizer_json_raises_value_error_with_clear_message(self):
        with self.assertRaises(ValueError) as ctx:
            VoiceConfig.from_dict({"engine": "chatterbox"})
        self.assertIn("tokenizer.json", str(ctx.exception))
        self.assertIn("bpe_tokenizer_json", str(ctx.exception))

    def test_chatterbox_detected_via_engine_string_literal(self):
        vc = VoiceConfig.from_dict({}, engine="chatterbox", bpe_tokenizer_json=self.bpe_path)
        self.assertEqual(vc.engine, Engine.CHATTERBOX)

    def test_chatterbox_detected_via_config_engine_key(self):
        vc = VoiceConfig.from_dict({"engine": "chatterbox"}, bpe_tokenizer_json=self.bpe_path)
        self.assertEqual(vc.engine, Engine.CHATTERBOX)

    def test_chatterbox_defaults_phoneme_type_and_alphabet_to_unicode(self):
        vc = VoiceConfig.from_dict({"engine": "chatterbox"}, bpe_tokenizer_json=self.bpe_path)
        self.assertEqual(vc.phoneme_type, PhonemeType.UNICODE)
        self.assertEqual(vc.alphabet, Alphabet.UNICODE)

    def test_chatterbox_enables_diacritics_for_arabic_and_hebrew(self):
        vc_ar = VoiceConfig.from_dict({"engine": "chatterbox"}, bpe_tokenizer_json=self.bpe_path, lang_code="ar")
        vc_he = VoiceConfig.from_dict({"engine": "chatterbox"}, bpe_tokenizer_json=self.bpe_path, lang_code="he")
        vc_en = VoiceConfig.from_dict({"engine": "chatterbox"}, bpe_tokenizer_json=self.bpe_path, lang_code="en")
        self.assertTrue(vc_ar.add_diacritics)
        self.assertTrue(vc_he.add_diacritics)
        self.assertFalse(vc_en.add_diacritics)

    def test_chatterbox_defaults_sample_rate_to_24000(self):
        vc = VoiceConfig.from_dict({"engine": "chatterbox"}, bpe_tokenizer_json=self.bpe_path)
        self.assertEqual(vc.sample_rate, 24000)

    def test_chatterbox_num_symbols_derived_from_tokenizer_vocab_size(self):
        vc = VoiceConfig.from_dict({"engine": "chatterbox"}, bpe_tokenizer_json=self.bpe_path)
        self.assertEqual(vc.num_symbols, vc.tokenizer._tok.get_vocab_size())


class TestFromDictMimic3PhonemeTypeBranches(unittest.TestCase):
    def _mimic3_cfg(self, phonemizer):
        return {"phonemizer": phonemizer, "phonemes": {}, "text_language": "en"}

    def test_symbols_phonemizer_yields_graphemes_and_unicode_alphabet(self):
        vc = VoiceConfig.from_dict(self._mimic3_cfg("symbols"), tokens_txt="0 _\n1 a\n2 b\n")
        self.assertEqual(vc.phoneme_type, PhonemeType.GRAPHEMES)
        self.assertEqual(vc.alphabet, Alphabet.UNICODE)
        self.assertEqual(vc.engine, Engine.MIMIC3)

    def test_gruut_phonemizer_yields_ipa_alphabet(self):
        vc = VoiceConfig.from_dict(self._mimic3_cfg("gruut"), tokens_txt="0 _\n1 a\n2 b\n")
        self.assertEqual(vc.phoneme_type, PhonemeType.GRUUT)
        self.assertEqual(vc.alphabet, Alphabet.IPA)

    def test_missing_tokens_txt_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            VoiceConfig.from_dict(self._mimic3_cfg("gruut"))
        self.assertIn("phonemes.txt", str(ctx.exception))


class TestFromDictCoqui(unittest.TestCase):
    def test_coqui_defaults_graphemes_and_unicode(self):
        cfg = {"characters": {"characters_class": "TTS.tts.utils.text.characters.Graphemes",
                              "characters": "ab", "punctuations": ""}}
        vc = VoiceConfig.from_dict(cfg)
        self.assertEqual(vc.engine, Engine.COQUI)
        self.assertEqual(vc.phoneme_type, PhonemeType.GRAPHEMES)
        self.assertEqual(vc.alphabet, Alphabet.UNICODE)

    def test_coqui_lang_code_pulled_from_datasets_when_absent(self):
        cfg = {"characters": {"characters_class": "TTS.tts.utils.text.characters.Graphemes",
                              "characters": "ab", "punctuations": ""},
               "datasets": [{"language": "pt-pt"}]}
        vc = VoiceConfig.from_dict(cfg)
        self.assertEqual(vc.lang_code, "pt-PT")

    def test_coqui_explicit_lang_code_wins_over_datasets(self):
        cfg = {"characters": {"characters_class": "TTS.tts.utils.text.characters.Graphemes",
                              "characters": "ab", "punctuations": ""},
               "datasets": [{"language": "pt-pt"}]}
        vc = VoiceConfig.from_dict(cfg, lang_code="en")
        self.assertEqual(vc.lang_code, "en")


class TestFromDictPiperBranches(unittest.TestCase):
    def _piper_cfg(self, phoneme_type="espeak", **extra):
        cfg = {"piper_version": "1.0.0", "phoneme_type": phoneme_type,
               "phoneme_id_map": {"a": [0], "b": [1]},
               "espeak": {"voice": "en-us"}}
        cfg.update(extra)
        return cfg

    def test_text_phoneme_type_maps_to_unicode_grapheme_model(self):
        vc = VoiceConfig.from_dict(self._piper_cfg("text"))
        self.assertEqual(vc.phoneme_type, PhonemeType.UNICODE)
        self.assertEqual(vc.alphabet, Alphabet.UNICODE)

    def test_pygoruut_phoneme_type_maps_to_goruut_ipa(self):
        vc = VoiceConfig.from_dict(self._piper_cfg("pygoruut"))
        self.assertEqual(vc.phoneme_type, PhonemeType.GORUUT)
        self.assertEqual(vc.alphabet, Alphabet.IPA)

    def test_espeak_phoneme_type_defaults_to_ipa_alphabet(self):
        vc = VoiceConfig.from_dict(self._piper_cfg("espeak"))
        self.assertEqual(vc.alphabet, Alphabet.IPA)

    def test_lang_code_falls_back_to_espeak_voice(self):
        vc = VoiceConfig.from_dict(self._piper_cfg())
        self.assertEqual(vc.lang_code, "en-US")

    def test_arabic_lang_code_enables_diacritics(self):
        cfg = self._piper_cfg()
        cfg["espeak"] = {"voice": "ar"}
        vc = VoiceConfig.from_dict(cfg)
        self.assertTrue(vc.add_diacritics)


class TestFromDictTransformersVocabBranch(unittest.TestCase):
    def test_vocab_without_tokenizer_config_defaults_add_blank_true(self):
        vocab = {"_": 0, "a": 1, "b": 2}
        vc = VoiceConfig.from_dict({"blank": "_"}, vocab=vocab, phoneme_type="graphemes",
                                    alphabet="unicode", lang_code="en")
        self.assertTrue(vc.tokenizer.add_blank_char)
        self.assertTrue(vc.tokenizer.blank_at_start)
        self.assertTrue(vc.tokenizer.blank_at_end)
        self.assertFalse(vc.tokenizer.use_eos_bos)

    def test_vocab_with_tokenizer_config_overrides_lang_and_blank(self):
        vocab = {"_": 0, "a": 1, "b": 2}
        tok_cfg = {"add_blank": False, "language": "pt", "pad_token": "_"}
        vc = VoiceConfig.from_dict({}, vocab=vocab, tokenizer_config=tok_cfg,
                                    phoneme_type="graphemes", alphabet="unicode")
        self.assertFalse(vc.tokenizer.add_blank_char)
        self.assertEqual(vc.lang_code, "pt")


class TestFromDictSherpaTokensTxtBranch(unittest.TestCase):
    def test_txt_extension_builds_tokenizer_with_eos_bos(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("_ 0\na 1\nb 2\n")
            path = f.name
        try:
            vc = VoiceConfig.from_dict({}, tokens_txt=path, phoneme_type="graphemes",
                                        alphabet="unicode", lang_code="en")
            self.assertTrue(vc.tokenizer.use_eos_bos)
            self.assertTrue(vc.tokenizer.add_blank_char)
        finally:
            os.unlink(path)


class TestGetPhonemizerDispatch(unittest.TestCase):
    def test_unsupported_phoneme_type_raises_value_error(self):
        from phoonnx.config import get_phonemizer
        with self.assertRaises(ValueError):
            get_phonemizer("not-a-real-phoneme-type")

    def test_espeak_phoneme_type_returns_espeak_phonemizer(self):
        from phoonnx.config import get_phonemizer
        from phoonnx.phonemizers import EspeakPhonemizer
        phonemizer = get_phonemizer(PhonemeType.ESPEAK)
        self.assertIsInstance(phonemizer, EspeakPhonemizer)

    def test_graphemes_phoneme_type_returns_grapheme_phonemizer(self):
        from phoonnx.config import get_phonemizer
        from phoonnx.phonemizers import GraphemePhonemizer
        phonemizer = get_phonemizer(PhonemeType.GRAPHEMES)
        self.assertIsInstance(phonemizer, GraphemePhonemizer)


class TestDiacritizerModelRoundTrip(unittest.TestCase):
    def test_native_config_round_trips_custom_diacritizer_model(self):
        cfg = {"phoonnx_version": "1.0", "phoneme_id_map": {"_": 0, "^": 1, "$": 2, "a": 3},
               "inference": {"diacritizer_model": "custom-diacritizer"}}
        vc = VoiceConfig.from_dict(cfg)
        self.assertEqual(vc.diacritizer_model, "custom-diacritizer")
        native = vc.to_native_dict()
        self.assertEqual(native["inference"]["diacritizer_model"], "custom-diacritizer")
        vc2 = VoiceConfig.from_dict(native)
        self.assertEqual(vc2.diacritizer_model, "custom-diacritizer")

    def test_default_diacritizer_model_used_when_absent(self):
        cfg = {"phoonnx_version": "1.0", "phoneme_id_map": {"_": 0, "^": 1, "$": 2, "a": 3}}
        vc = VoiceConfig.from_dict(cfg)
        self.assertEqual(vc.diacritizer_model, "rawi-ensemble")

    def test_non_phoonnx_config_also_carries_diacritizer_model_through_inference(self):
        cfg = {"inference": {"diacritizer_model": "other-model"}}
        vc = VoiceConfig.from_dict(cfg)
        self.assertEqual(vc.diacritizer_model, "other-model")


class TestPostInit(unittest.TestCase):
    def test_arabic_lang_code_defaults_add_diacritics_true(self):
        vc = VoiceConfig(num_symbols=1, num_speakers=1, num_langs=1, sample_rate=16000,
                          lang_code="ar", phoneme_type=PhonemeType.ESPEAK, alphabet=Alphabet.IPA,
                          phonemizer_model=None)
        self.assertTrue(vc.add_diacritics)

    def test_non_arabic_lang_code_defaults_add_diacritics_false(self):
        vc = VoiceConfig(num_symbols=1, num_speakers=1, num_langs=1, sample_rate=16000,
                          lang_code="en", phoneme_type=PhonemeType.ESPEAK, alphabet=Alphabet.IPA,
                          phonemizer_model=None)
        self.assertFalse(vc.add_diacritics)

    def test_explicit_add_diacritics_is_not_overridden(self):
        vc = VoiceConfig(num_symbols=1, num_speakers=1, num_langs=1, sample_rate=16000,
                          lang_code="ar", phoneme_type=PhonemeType.ESPEAK, alphabet=Alphabet.IPA,
                          phonemizer_model=None, add_diacritics=False)
        self.assertFalse(vc.add_diacritics)

    def test_missing_lang_code_defaults_to_und(self):
        vc = VoiceConfig(num_symbols=1, num_speakers=1, num_langs=1, sample_rate=16000,
                          lang_code=None, phoneme_type=PhonemeType.ESPEAK, alphabet=Alphabet.IPA,
                          phonemizer_model=None)
        self.assertEqual(vc.lang_code, "und")

    def test_string_engine_alphabet_phoneme_type_are_cast_to_enums(self):
        vc = VoiceConfig(num_symbols=1, num_speakers=1, num_langs=1, sample_rate=16000,
                          lang_code="en", phoneme_type="espeak", alphabet="ipa",
                          phonemizer_model=None, engine="piper")
        self.assertIsInstance(vc.engine, Engine)
        self.assertIsInstance(vc.alphabet, Alphabet)
        self.assertIsInstance(vc.phoneme_type, PhonemeType)
        self.assertEqual(vc.engine, Engine.PIPER)


if __name__ == "__main__":
    unittest.main()
