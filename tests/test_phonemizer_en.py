"""Adversarial tests for scriptconv.phonemizers.en (DeepPhonemizer, OpenPhonemizer,
G2PEnPhonemizer).

Heavyweight backends (dp/torch, openphonemizer, nltk/g2p_en) are mocked via
sys.modules patching so these tests run without the optional extras installed.
"""
import unittest
from unittest.mock import MagicMock, patch
import sys

from phoonnx.config import Alphabet


def _fake_torch_and_dp_modules():
    """Build minimal fake `dp` and `torch` modules sufficient for the
    add_safe_globals dance every backend in en.py performs on import."""
    fake_torch = MagicMock()
    fake_torch.serialization.add_safe_globals = MagicMock()

    fake_dp_preprocessing_text = MagicMock()
    fake_dp_preprocessing_text.Preprocessor = MagicMock()
    fake_dp_preprocessing_text.LanguageTokenizer = MagicMock()
    fake_dp_preprocessing_text.SequenceTokenizer = MagicMock()

    fake_dp_preprocessing = MagicMock()
    fake_dp_preprocessing.text = fake_dp_preprocessing_text

    fake_dp = MagicMock()
    fake_dp.preprocessing = fake_dp_preprocessing

    return fake_torch, fake_dp, fake_dp_preprocessing, fake_dp_preprocessing_text


class TestDeepPhonemizerConstruction(unittest.TestCase):
    def test_ipa_model_name_sets_ipa_alphabet(self):
        from scriptconv.phonemizers.en import DeepPhonemizer

        fake_torch, fake_dp, _, _ = _fake_torch_and_dp_modules()
        fake_phonemizer_mod = MagicMock()
        mock_backend = MagicMock()
        fake_phonemizer_mod.Phonemizer.from_checkpoint.return_value = mock_backend

        with patch.dict(sys.modules, {
            "dp": fake_dp,
            "dp.preprocessing": fake_dp.preprocessing,
            "dp.preprocessing.text": fake_dp.preprocessing.text,
            "dp.phonemizer": fake_phonemizer_mod,
            "torch": fake_torch,
        }), patch("os.path.isfile", return_value=True):
            inst = DeepPhonemizer(model="latin_ipa_forward.pt")

        self.assertEqual(inst.alphabet, Alphabet.IPA)
        fake_phonemizer_mod.Phonemizer.from_checkpoint.assert_called_once_with("latin_ipa_forward.pt")

    def test_non_ipa_model_name_sets_arpa_alphabet(self):
        from scriptconv.phonemizers.en import DeepPhonemizer

        fake_torch, fake_dp, _, _ = _fake_torch_and_dp_modules()
        fake_phonemizer_mod = MagicMock()

        with patch.dict(sys.modules, {
            "dp": fake_dp,
            "dp.preprocessing": fake_dp.preprocessing,
            "dp.preprocessing.text": fake_dp.preprocessing.text,
            "dp.phonemizer": fake_phonemizer_mod,
            "torch": fake_torch,
        }), patch("os.path.isfile", return_value=True):
            inst = DeepPhonemizer(model="en_us_cmudict_forward.pt")

        self.assertEqual(inst.alphabet, Alphabet.ARPA)

    def test_unknown_local_model_path_raises_value_error(self):
        """A model name that isn't a real file AND isn't in MODELS must raise,
        never silently proceed to construct a Phonemizer from garbage."""
        from scriptconv.phonemizers.en import DeepPhonemizer

        fake_torch, fake_dp, _, _ = _fake_torch_and_dp_modules()
        fake_phonemizer_mod = MagicMock()

        with patch.dict(sys.modules, {
            "dp": fake_dp,
            "dp.preprocessing": fake_dp.preprocessing,
            "dp.preprocessing.text": fake_dp.preprocessing.text,
            "dp.phonemizer": fake_phonemizer_mod,
            "torch": fake_torch,
        }), patch("os.path.isfile", return_value=False):
            with self.assertRaises(ValueError):
                DeepPhonemizer(model="totally_unknown_model.pt")


class TestDeepPhonemizerGetLang(unittest.TestCase):
    def test_valid_lang_de_and_en_us(self):
        from scriptconv.phonemizers.en import DeepPhonemizer
        self.assertEqual(DeepPhonemizer.get_lang("de"), "de")
        self.assertEqual(DeepPhonemizer.get_lang("en_us"), "en_us")

    def test_unsupported_lang_raises(self):
        from scriptconv.phonemizers.en import DeepPhonemizer
        with self.assertRaises(ValueError):
            DeepPhonemizer.get_lang("zz-totally-bogus")


class TestDeepPhonemizerPhonemizeString(unittest.TestCase):
    def _make_instance(self):
        from scriptconv.phonemizers.en import DeepPhonemizer
        inst = DeepPhonemizer.__new__(DeepPhonemizer)
        inst.alphabet = Alphabet.IPA
        inst.phonemizer = MagicMock(return_value="h eh l ow")
        return inst

    def test_phonemize_string_delegates_to_backend_with_resolved_lang(self):
        inst = self._make_instance()
        result = inst.phonemize_string("hello", "de")
        inst.phonemizer.assert_called_once_with("hello", "de")
        self.assertEqual(result, "h eh l ow")

    def test_phonemize_string_empty_input(self):
        inst = self._make_instance()
        inst.phonemizer.return_value = ""
        result = inst.phonemize_string("", "de")
        self.assertEqual(result, "")

    def test_phonemize_string_invalid_lang_raises(self):
        inst = self._make_instance()
        with self.assertRaises(ValueError):
            inst.phonemize_string("hello", "xx-nope")


class TestOpenPhonemizer(unittest.TestCase):
    def test_get_lang_only_supports_en(self):
        from scriptconv.phonemizers.en import OpenPhonemizer
        self.assertEqual(OpenPhonemizer.get_lang("en"), "en")
        self.assertEqual(OpenPhonemizer.get_lang("en-US"), "en")
        with self.assertRaises(ValueError):
            OpenPhonemizer.get_lang("de")

    def test_construction_wires_backend_and_alphabet(self):
        fake_openphonemizer_mod = MagicMock()
        mock_backend_cls = MagicMock()
        fake_openphonemizer_mod.OpenPhonemizer = mock_backend_cls
        fake_torch, fake_dp, _, _ = _fake_torch_and_dp_modules()

        with patch.dict(sys.modules, {
            "openphonemizer": fake_openphonemizer_mod,
            "torch": fake_torch,
            "dp": fake_dp,
            "dp.preprocessing": fake_dp.preprocessing,
            "dp.preprocessing.text": fake_dp.preprocessing.text,
        }):
            from scriptconv.phonemizers.en import OpenPhonemizer
            inst = OpenPhonemizer()

        self.assertEqual(inst.alphabet, Alphabet.IPA)
        mock_backend_cls.assert_called_once()

    def test_phonemize_string_delegates_and_ignores_lang_arg_in_call(self):
        from scriptconv.phonemizers.en import OpenPhonemizer
        inst = OpenPhonemizer.__new__(OpenPhonemizer)
        inst.alphabet = Alphabet.IPA
        inst.phonemizer = MagicMock(return_value="h ə l oʊ")

        result = inst.phonemize_string("hello", "en-GB")
        inst.phonemizer.assert_called_once_with("hello")
        self.assertEqual(result, "h ə l oʊ")

    def test_phonemize_string_unsupported_lang_raises(self):
        from scriptconv.phonemizers.en import OpenPhonemizer
        inst = OpenPhonemizer.__new__(OpenPhonemizer)
        inst.alphabet = Alphabet.IPA
        inst.phonemizer = MagicMock()
        with self.assertRaises(ValueError):
            inst.phonemize_string("hello", "fr")


class TestG2PEnPhonemizer(unittest.TestCase):
    def test_rejects_unsupported_alphabet(self):
        from scriptconv.phonemizers.en import G2PEnPhonemizer
        with self.assertRaises(AssertionError):
            with patch.dict(sys.modules, {
                "nltk": MagicMock(),
                "g2p_en": MagicMock(),
            }):
                G2PEnPhonemizer(alphabet=Alphabet.BUCKWALTER)

    def test_construction_downloads_nltk_resources_and_builds_g2p(self):
        fake_nltk = MagicMock()
        fake_g2p_en_mod = MagicMock()
        fake_g2p_cls = MagicMock()
        fake_g2p_en_mod.G2p = fake_g2p_cls

        with patch.dict(sys.modules, {"nltk": fake_nltk, "g2p_en": fake_g2p_en_mod}):
            from scriptconv.phonemizers.en import G2PEnPhonemizer
            inst = G2PEnPhonemizer()

        fake_nltk.download.assert_any_call('averaged_perceptron_tagger_eng')
        fake_nltk.download.assert_any_call('cmudict')
        fake_g2p_cls.assert_called_once()
        self.assertEqual(inst.alphabet, Alphabet.IPA)

    def test_get_lang_only_supports_en(self):
        from scriptconv.phonemizers.en import G2PEnPhonemizer
        self.assertEqual(G2PEnPhonemizer.get_lang("en-US"), "en")
        with self.assertRaises(ValueError):
            G2PEnPhonemizer.get_lang("ja")

    def test_phonemize_string_arpa_alphabet_returns_raw_arpa(self):
        from scriptconv.phonemizers.en import G2PEnPhonemizer
        inst = G2PEnPhonemizer.__new__(G2PEnPhonemizer)
        inst.alphabet = Alphabet.ARPA
        inst.g2p = MagicMock(return_value=["HH", "AH0", "L", "OW1"])

        result = inst.phonemize_string("hello", "en")
        self.assertEqual(result, ["HH", "AH0", "L", "OW1"])

    def test_phonemize_string_ipa_alphabet_maps_arpa_to_ipa(self):
        from scriptconv.phonemizers.en import G2PEnPhonemizer
        inst = G2PEnPhonemizer.__new__(G2PEnPhonemizer)
        inst.alphabet = Alphabet.IPA
        inst.g2p = MagicMock(return_value=["HH", "AH0"])

        with patch("scriptconv.phonemizers.en.arpa_to_ipa_lookup", {"HH": "h", "AH0": "ə"}):
            result = inst.phonemize_string("hi", "en")
        self.assertEqual(result, "hə")

    def test_phonemize_string_oov_arpa_symbol_falls_back_to_symbol_itself(self):
        """An ARPA symbol absent from the lookup table must be passed through
        unchanged (via dict.get default), never raise a KeyError."""
        from scriptconv.phonemizers.en import G2PEnPhonemizer
        inst = G2PEnPhonemizer.__new__(G2PEnPhonemizer)
        inst.alphabet = Alphabet.IPA
        inst.g2p = MagicMock(return_value=["ZZZ_UNKNOWN"])

        with patch("scriptconv.phonemizers.en.arpa_to_ipa_lookup", {}):
            result = inst.phonemize_string("weirdword", "en")
        self.assertEqual(result, "ZZZ_UNKNOWN")

    def test_phonemize_string_empty_g2p_output(self):
        from scriptconv.phonemizers.en import G2PEnPhonemizer
        inst = G2PEnPhonemizer.__new__(G2PEnPhonemizer)
        inst.alphabet = Alphabet.IPA
        inst.g2p = MagicMock(return_value=[])

        with patch("scriptconv.phonemizers.en.arpa_to_ipa_lookup", {}):
            result = inst.phonemize_string("", "en")
        self.assertEqual(result, "")

    def test_phonemize_string_invalid_lang_raises(self):
        from scriptconv.phonemizers.en import G2PEnPhonemizer
        inst = G2PEnPhonemizer.__new__(G2PEnPhonemizer)
        inst.alphabet = Alphabet.IPA
        inst.g2p = MagicMock()
        with self.assertRaises(ValueError):
            inst.phonemize_string("x", "de")


if __name__ == "__main__":
    unittest.main()
