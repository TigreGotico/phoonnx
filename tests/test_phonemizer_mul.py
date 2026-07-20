"""Adversarial tests for phoonnx.phonemizers.mul — the multilingual phonemizer
dispatch/fallback layer (EspeakPhonemizer, GruutPhonemizer, GoruutPhonemizer,
EpitranPhonemizer, MisakiPhonemizer family, CharsiuPhonemizer).

All heavy backends (espeak-ng binary/espyak, gruut, pygoruut, epitran, misaki,
onnxruntime models) are mocked so these run without the optional extras.
"""
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from phoonnx.config import Alphabet
from phoonnx.phonemizers.mul import (
    EspeakPhonemizer,
    EspeakError,
    GruutPhonemizer,
    GoruutPhonemizer,
    EpitranPhonemizer,
    MisakiPhonemizer,
    CharsiuPhonemizer,
    ByT5Phonemizer,
)


class TestEspeakGetLang(unittest.TestCase):
    def test_en_gb_maps_to_rp_variant(self):
        self.assertEqual(EspeakPhonemizer.get_lang("en-gb"), "en-gb-x-rp")
        self.assertEqual(EspeakPhonemizer.get_lang("EN-GB"), "en-gb-x-rp")

    def test_exact_match_passthrough(self):
        self.assertEqual(EspeakPhonemizer.get_lang("pt-br"), "pt-br")

    def test_unusual_but_valid_regional_code_falls_back_to_base_lang(self):
        # pt-BR isn't itself in ESPEAK_LANGS with this casing, but its base
        # "pt" is -- get_lang lowercases and strips the region.
        self.assertEqual(EspeakPhonemizer.get_lang("pt-BR"), "pt")

    def test_zh_tw_has_no_close_match_raises_valueerror_not_keyerror(self):
        # zh-TW has no exact/base entry ("zh" isn't in ESPEAK_LANGS either,
        # only "cmn"), and langcodes' tag_distance finds nothing close enough
        # -- this must raise the controlled ValueError from match_lang,
        # never a raw KeyError from a dict/table lookup.
        with self.assertRaises(ValueError):
            EspeakPhonemizer.get_lang("zh-TW")

    def test_completely_unknown_lang_raises_valueerror_not_keyerror(self):
        with self.assertRaises(ValueError):
            EspeakPhonemizer.get_lang("xx-nonexistent-lang")


class TestEspeakPhonemizeStringDispatch(unittest.TestCase):
    def test_prefers_binary_when_available_and_not_forced_to_espyak(self):
        inst = EspeakPhonemizer(prefer_espyak=False)
        with patch.object(EspeakPhonemizer, "_has_binary", return_value=True), \
             patch.object(EspeakPhonemizer, "_run_espeak_command", return_value="h ə l oʊ") as mock_run:
            result = inst.phonemize_string("hello", "en-us")
        mock_run.assert_called_once()
        self.assertEqual(result, "h ə l oʊ")

    def test_falls_back_to_espyak_when_binary_missing(self):
        inst = EspeakPhonemizer(prefer_espyak=False)
        fake_espyak_mod = MagicMock()
        fake_g2p_instance = MagicMock()
        fake_g2p_instance.phonemize.return_value = "h ə l oʊ"
        fake_espyak_mod.G2P.return_value = fake_g2p_instance

        with patch.object(EspeakPhonemizer, "_has_binary", return_value=False), \
             patch.dict(sys.modules, {"espyak": fake_espyak_mod}):
            result = inst.phonemize_string("hello", "en-us")

        self.assertEqual(result, "h ə l oʊ")

    def test_prefer_espyak_true_uses_espyak_even_when_binary_present(self):
        inst = EspeakPhonemizer(prefer_espyak=True)
        fake_espyak_mod = MagicMock()
        fake_g2p_instance = MagicMock()
        fake_g2p_instance.phonemize.return_value = "custom"
        fake_espyak_mod.G2P.return_value = fake_g2p_instance

        with patch.object(EspeakPhonemizer, "_has_binary", return_value=True) as mock_has_bin, \
             patch.object(EspeakPhonemizer, "_run_espeak_command") as mock_run, \
             patch.dict(sys.modules, {"espyak": fake_espyak_mod}):
            result = inst.phonemize_string("hello", "en-us")

        mock_run.assert_not_called()
        self.assertEqual(result, "custom")

    def test_espyak_missing_and_no_binary_raises_espeak_error_not_crash(self):
        inst = EspeakPhonemizer(prefer_espyak=True)
        with patch.object(EspeakPhonemizer, "_has_binary", return_value=False), \
             patch.dict(sys.modules, {"espyak": None}):
            with self.assertRaises(EspeakError):
                inst.phonemize_string("hello", "en-us")

    def test_espyak_g2p_instances_are_cached_per_lang(self):
        inst = EspeakPhonemizer(prefer_espyak=True)
        fake_espyak_mod = MagicMock()
        fake_g2p_instance = MagicMock()
        fake_g2p_instance.phonemize.return_value = "x"
        fake_espyak_mod.G2P.return_value = fake_g2p_instance

        with patch.dict(sys.modules, {"espyak": fake_espyak_mod}):
            inst._espyak_phonemize("a", "en-us")
            inst._espyak_phonemize("b", "en-us")

        fake_espyak_mod.G2P.assert_called_once_with("en-us")

    def test_empty_string_input_does_not_crash(self):
        inst = EspeakPhonemizer(prefer_espyak=False)
        with patch.object(EspeakPhonemizer, "_has_binary", return_value=True), \
             patch.object(EspeakPhonemizer, "_run_espeak_command", return_value="") as mock_run:
            result = inst.phonemize_string("", "en-us")
        self.assertEqual(result, "")


class TestEspeakRunCommandErrors(unittest.TestCase):
    def test_binary_not_found_raises_espeak_error(self):
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            with self.assertRaises(EspeakError):
                EspeakPhonemizer._run_espeak_command(["-q"], input_text="hi")

    def test_nonzero_exit_raises_espeak_error_with_details(self):
        err = subprocess.CalledProcessError(1, ["espeak-ng"], output="", stderr="boom")
        with patch("subprocess.run", side_effect=err):
            with self.assertRaises(EspeakError) as ctx:
                EspeakPhonemizer._run_espeak_command(["-q"], input_text="hi")
        self.assertIn("boom", str(ctx.exception))

    def test_unexpected_exception_is_wrapped_in_espeak_error(self):
        with patch("subprocess.run", side_effect=RuntimeError("weird")):
            with self.assertRaises(EspeakError):
                EspeakPhonemizer._run_espeak_command(["-q"], input_text="hi")


class TestGruutGetLangAndDispatch(unittest.TestCase):
    def test_get_lang_exact_and_unsupported(self):
        self.assertEqual(GruutPhonemizer.get_lang("en"), "en")
        with self.assertRaises(ValueError):
            GruutPhonemizer.get_lang("xx-bogus")

    def _make_sentence(self, text, word_phonemes):
        sentence = MagicMock()
        sentence.text = text
        sentence.__iter__.return_value = iter([
            MagicMock(phonemes=ph) for ph in word_phonemes
        ])
        sentence.__bool__.return_value = True
        return sentence

    def test_question_mark_forces_final_token(self):
        inst = GruutPhonemizer()
        sentence = self._make_sentence("hi?", [["h", "i"]])
        fake_gruut = MagicMock()
        fake_gruut.sentences.return_value = [sentence]
        with patch.dict(sys.modules, {"gruut": fake_gruut}):
            result = inst.phonemize_string("hi?", "en")
        self.assertTrue(result.endswith("?"))

    def test_exclamation_forces_final_token(self):
        inst = GruutPhonemizer()
        sentence = self._make_sentence("wow!", [["w", "aʊ"]])
        fake_gruut = MagicMock()
        fake_gruut.sentences.return_value = [sentence]
        with patch.dict(sys.modules, {"gruut": fake_gruut}):
            result = inst.phonemize_string("wow!", "en")
        self.assertTrue(result.endswith("!"))

    def test_period_forces_final_token(self):
        inst = GruutPhonemizer()
        sentence = self._make_sentence("done.", [["d", "ʌ", "n"]])
        fake_gruut = MagicMock()
        fake_gruut.sentences.return_value = [sentence]
        with patch.dict(sys.modules, {"gruut": fake_gruut}):
            result = inst.phonemize_string("done.", "en")
        self.assertTrue(result.endswith("."))

    def test_empty_word_phonemes_raises_runtime_error_hinting_missing_lang_pack(self):
        """Gruut sentence with no phonemized words signals a missing
        language pack; this must raise a clear RuntimeError, not silently
        drop the sentence or crash with an IndexError."""
        inst = GruutPhonemizer()
        sentence = MagicMock()
        sentence.text = "hello"
        sentence.__iter__.return_value = iter([MagicMock(phonemes=[])])
        sentence.__bool__.return_value = True
        fake_gruut = MagicMock()
        fake_gruut.sentences.return_value = [sentence]
        with patch.dict(sys.modules, {"gruut": fake_gruut}):
            with self.assertRaises(RuntimeError):
                inst.phonemize_string("hello", "en")

    def test_empty_input_text_yields_empty_output(self):
        inst = GruutPhonemizer()
        fake_gruut = MagicMock()
        fake_gruut.sentences.return_value = []
        with patch.dict(sys.modules, {"gruut": fake_gruut}):
            result = inst.phonemize_string("", "en")
        self.assertEqual(result, "")


class TestGoruutGetLang(unittest.TestCase):
    def test_non_std_lang_passthrough(self):
        self.assertEqual(GoruutPhonemizer.get_lang("EnglishAmerican"), "EnglishAmerican")

    def test_en_us_special_case(self):
        self.assertEqual(GoruutPhonemizer.get_lang("en-us"), "EnglishAmerican")

    def test_en_gb_and_en_uk_special_case(self):
        self.assertEqual(GoruutPhonemizer.get_lang("en-gb"), "EnglishBritish")
        self.assertEqual(GoruutPhonemizer.get_lang("en-uk"), "EnglishBritish")

    def test_iso639_lookup_resolves_two_letter_code(self):
        self.assertEqual(GoruutPhonemizer.get_lang("fr"), "French")

    def test_pt_br_resolves_via_iso639_base_match(self):
        result = GoruutPhonemizer.get_lang("pt-BR")
        self.assertEqual(result, "Portuguese")

    def test_unsupported_code_raises_valueerror_not_keyerror(self):
        with self.assertRaises(ValueError):
            GoruutPhonemizer.get_lang("xx-totally-invalid")


class TestGoruutPhonemizeString(unittest.TestCase):
    def test_phonemize_string_delegates_with_resolved_lang(self):
        inst = GoruutPhonemizer.__new__(GoruutPhonemizer)
        inst.pygoruut = MagicMock()
        inst.pygoruut.phonemize.return_value = "fake_result"
        result = inst.phonemize_string("bonjour", "fr")
        inst.pygoruut.phonemize.assert_called_once_with(language="French", sentence="bonjour")
        self.assertEqual(result, "fake_result")


class TestEpitranGetLangAndDispatch(unittest.TestCase):
    def test_get_lang_exact_match(self):
        self.assertEqual(EpitranPhonemizer.get_lang("eng-Latn"), "eng-Latn")

    def test_get_lang_unsupported_raises(self):
        with self.assertRaises(ValueError):
            EpitranPhonemizer.get_lang("xx-nope")

    def test_phonemize_string_lazily_builds_and_caches_epi_instance(self):
        inst = EpitranPhonemizer.__new__(EpitranPhonemizer)
        fake_epitran_module = MagicMock()
        fake_epi_instance = MagicMock()
        fake_epi_instance.transliterate.return_value = "phonemes"
        fake_epitran_module.Epitran.return_value = fake_epi_instance
        inst.epitran = fake_epitran_module
        inst._epis = {}

        result1 = inst.phonemize_string("hello", "eng-Latn")
        result2 = inst.phonemize_string("world", "eng-Latn")

        fake_epitran_module.Epitran.assert_called_once_with("eng-Latn")
        self.assertEqual(result1, "phonemes")
        self.assertEqual(result2, "phonemes")

    def test_phonemize_string_empty_input(self):
        inst = EpitranPhonemizer.__new__(EpitranPhonemizer)
        fake_epitran_module = MagicMock()
        fake_epi_instance = MagicMock()
        fake_epi_instance.transliterate.return_value = ""
        fake_epitran_module.Epitran.return_value = fake_epi_instance
        inst.epitran = fake_epitran_module
        inst._epis = {}

        result = inst.phonemize_string("", "eng-Latn")
        self.assertEqual(result, "")


class TestMisakiDispatchTable(unittest.TestCase):
    """MisakiPhonemizer._get_phonemizer dispatches by resolved lang code to
    one of five per-language backends (zh/ko/vi/ja/en), lazily importing and
    caching each. Every branch is exercised with a mocked backend module."""

    def _make_instance(self, alphabet=Alphabet.IPA):
        inst = MisakiPhonemizer.__new__(MisakiPhonemizer)
        inst.alphabet = alphabet
        inst.g2p_en = inst.g2p_zh = inst.g2p_ko = inst.g2p_vi = inst.g2p_ja = None
        return inst

    def test_zh_branch_uses_zh_version_from_alphabet(self):
        inst = self._make_instance(alphabet=Alphabet.BOPOMOFO)
        fake_zh_mod = MagicMock()
        fake_zh_g2p = MagicMock()
        fake_zh_mod.ZHG2P.return_value = fake_zh_g2p
        with patch.dict(sys.modules, {"misaki.zh": fake_zh_mod}):
            result = inst._get_phonemizer("zh")
        fake_zh_mod.ZHG2P.assert_called_once_with(version="1.1")
        self.assertIs(result, fake_zh_g2p)
        self.assertIs(inst.g2p_zh, fake_zh_g2p)

    def test_zh_branch_ipa_version_default(self):
        inst = self._make_instance(alphabet=Alphabet.IPA)
        fake_zh_mod = MagicMock()
        with patch.dict(sys.modules, {"misaki.zh": fake_zh_mod}):
            inst._get_phonemizer("zh")
        fake_zh_mod.ZHG2P.assert_called_once_with(version="1.0")

    def test_ko_branch(self):
        inst = self._make_instance()
        fake_ko_mod = MagicMock()
        fake_ko_g2p = MagicMock()
        fake_ko_mod.KOG2P.return_value = fake_ko_g2p
        with patch.dict(sys.modules, {"misaki.ko": fake_ko_mod}):
            result = inst._get_phonemizer("ko")
        self.assertIs(result, fake_ko_g2p)

    def test_vi_branch(self):
        inst = self._make_instance()
        fake_vi_mod = MagicMock()
        fake_vi_g2p = MagicMock()
        fake_vi_mod.VIG2P.return_value = fake_vi_g2p
        with patch.dict(sys.modules, {"misaki.vi": fake_vi_mod}):
            result = inst._get_phonemizer("vi")
        self.assertIs(result, fake_vi_g2p)

    def test_ja_branch(self):
        inst = self._make_instance()
        fake_ja_mod = MagicMock()
        fake_ja_g2p = MagicMock()
        fake_ja_mod.JAG2P.return_value = fake_ja_g2p
        with patch.dict(sys.modules, {"misaki.ja": fake_ja_mod}):
            result = inst._get_phonemizer("ja")
        self.assertIs(result, fake_ja_g2p)

    def test_en_us_branch_sets_british_false(self):
        inst = self._make_instance()
        fake_misaki_pkg = MagicMock()
        fake_en_mod = MagicMock()
        fake_en_g2p = MagicMock()
        fake_en_mod.G2P.return_value = fake_en_g2p
        with patch.dict(sys.modules, {"misaki": fake_misaki_pkg, "misaki.en": fake_en_mod}), \
             patch("misaki.en", fake_en_mod, create=True):
            result = inst._get_phonemizer("en-US")
        self.assertFalse(fake_en_g2p.british)

    def test_en_gb_branch_sets_british_true(self):
        inst = self._make_instance()
        fake_misaki_pkg = MagicMock()
        fake_en_mod = MagicMock()
        fake_en_g2p = MagicMock()
        fake_en_mod.G2P.return_value = fake_en_g2p
        with patch.dict(sys.modules, {"misaki": fake_misaki_pkg, "misaki.en": fake_en_mod}), \
             patch("misaki.en", fake_en_mod, create=True):
            result = inst._get_phonemizer("en-GB")
        self.assertTrue(fake_en_g2p.british)

    def test_en_backend_is_cached_across_calls(self):
        inst = self._make_instance()
        fake_misaki_pkg = MagicMock()
        fake_en_mod = MagicMock()
        fake_en_g2p = MagicMock()
        fake_en_mod.G2P.return_value = fake_en_g2p
        with patch.dict(sys.modules, {"misaki": fake_misaki_pkg, "misaki.en": fake_en_mod}), \
             patch("misaki.en", fake_en_mod, create=True):
            inst._get_phonemizer("en-US")
            inst._get_phonemizer("en-GB")
        fake_en_mod.G2P.assert_called_once()

    def test_unsupported_lang_raises_valueerror_not_keyerror(self):
        inst = self._make_instance()
        with self.assertRaises(ValueError):
            inst._get_phonemizer("xx-nonexistent")

    def test_phonemize_string_returns_phonemes_discards_tokens(self):
        inst = self._make_instance()
        fake_backend = MagicMock(return_value=("f oʊ n iː m z", ["tok1", "tok2"]))
        with patch.object(inst, "_get_phonemizer", return_value=fake_backend):
            result = inst.phonemize_string("phonemes", "en-US")
        self.assertEqual(result, "f oʊ n iː m z")


class TestCharsiuWordByWordDispatch(unittest.TestCase):
    def test_phonemize_string_splits_and_joins_per_word(self):
        inst = CharsiuPhonemizer.__new__(CharsiuPhonemizer)
        calls = []

        def fake_infer(word, lang):
            calls.append((word, lang))
            return f"[{word}]"

        inst._infer_onnx = fake_infer
        result = inst.phonemize_string("hello world again", "eng-uk")

        self.assertEqual(result, "[hello] [world] [again]")
        self.assertEqual(calls, [("hello", "eng-uk"), ("world", "eng-uk"), ("again", "eng-uk")])

    def test_phonemize_string_empty_input_yields_empty_output(self):
        inst = CharsiuPhonemizer.__new__(CharsiuPhonemizer)
        inst._infer_onnx = MagicMock(return_value="")
        result = inst.phonemize_string("", "eng-uk")
        self.assertEqual(result, "")
        inst._infer_onnx.assert_not_called()

    def test_get_lang_unusual_lang_code_not_keyerror(self):
        with self.assertRaises(ValueError):
            CharsiuPhonemizer.get_lang("zz-doesnotexist")

    def test_get_lang_valid_code(self):
        self.assertEqual(CharsiuPhonemizer.get_lang("eng-uk"), "eng-uk")


class TestByT5DecodePhones(unittest.TestCase):
    def test_decode_phones_skips_added_tokens(self):
        inst = ByT5Phonemizer.__new__(ByT5Phonemizer)
        inst.tokens = {"1": "special"}
        # token 3 -> byte 0 -> b'\x00' decodes to a control char; verify the
        # added-token id (1) is excluded but ordinary ones are decoded.
        preds = [1, 104 + 3, 105 + 3]  # skip "1", decode "h", "i"
        result = inst._decode_phones(preds)
        self.assertEqual(result, "hi")

    def test_decode_phones_empty_predictions(self):
        inst = ByT5Phonemizer.__new__(ByT5Phonemizer)
        inst.tokens = {}
        self.assertEqual(inst._decode_phones([]), "")

    def test_infer_onnx_empty_text_returns_empty_without_running_session(self):
        inst = ByT5Phonemizer.__new__(ByT5Phonemizer)
        inst.session = MagicMock()
        result = inst._infer_onnx("   ", "en-US")
        self.assertEqual(result, "")
        inst.session.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
