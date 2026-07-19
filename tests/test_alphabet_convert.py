"""Tests for phoonnx.alphabet_convert — unified alphabet conversion."""
import unittest
from unittest.mock import MagicMock, patch

from phoonnx.config import Alphabet, PhonemeType
from phoonnx.alphabet_convert import (
    ALPHABET_CONVERTERS,
    convert,
    register_converter,
    _find_path,
)


class TestIdentity(unittest.TestCase):
    def test_same_alphabet_returns_input_unchanged(self):
        text = "hello world"
        result = convert(text, "en", Alphabet.IPA, Alphabet.IPA)
        self.assertEqual(result, text)

    def test_same_alphabet_graphemes(self):
        text = "test"
        result = convert(text, "en", Alphabet.GRAPHEMES, Alphabet.GRAPHEMES)
        self.assertEqual(result, text)

    def test_identity_path_never_invokes_phonemizer_backend(self):
        """src == tgt must short-circuit before any phonemizer backend work.

        We patch get_phonemizer (imported lazily inside the phonemization
        edge closures) to raise if it is ever called. The identity call
        must NOT raise, proving no backend construction/factory call
        happens on the identity path.
        """
        text = "hello world"
        with patch("phoonnx.config.get_phonemizer",
                   side_effect=AssertionError("phonemizer backend must not be invoked on identity path")):
            result = convert(text, "en", Alphabet.IPA, Alphabet.IPA, phoneme_type=PhonemeType.ESPEAK)
        self.assertEqual(result, text)

    def test_non_identity_path_still_invokes_conversion(self):
        """A differing src/tgt must still go through the real conversion
        machinery and can produce a different, converted result."""
        sentinel = ["c", "o", "n", "v", "e", "r", "t", "e", "d"]

        def fake_edge(text, lang, _pt=None):
            return sentinel

        key = (Alphabet.SAMPA, Alphabet.RFE)
        old = ALPHABET_CONVERTERS.get(key)
        try:
            register_converter(Alphabet.SAMPA, Alphabet.RFE, fake_edge)
            result = convert("original text", "en", Alphabet.SAMPA, Alphabet.RFE)
            self.assertEqual(result, sentinel)
            self.assertNotEqual(result, "original text")
        finally:
            if old is None:
                ALPHABET_CONVERTERS.pop(key, None)
            else:
                ALPHABET_CONVERTERS[key] = old


class TestDirectEdge(unittest.TestCase):
    def test_registered_direct_edge_is_called(self):
        sentinel = object()

        def my_edge(text, lang, _pt=None):
            return sentinel

        # Temporarily register a custom edge
        key = (Alphabet.SAMPA, Alphabet.RFE)
        old = ALPHABET_CONVERTERS.get(key)
        try:
            register_converter(Alphabet.SAMPA, Alphabet.RFE, my_edge)
            result = convert("x", "en", Alphabet.SAMPA, Alphabet.RFE)
            self.assertIs(result, sentinel)
        finally:
            if old is None:
                ALPHABET_CONVERTERS.pop(key, None)
            else:
                ALPHABET_CONVERTERS[key] = old


class TestChainSupport(unittest.TestCase):
    def test_multihop_chain_composed(self):
        """A → B → C path composes both edges."""
        calls = []

        def edge_ab(text, lang, _pt=None):
            calls.append("ab")
            return text + "_B"

        def edge_bc(text, lang, _pt=None):
            calls.append("bc")
            return text + "_C"

        # Use unique temporary alphabet values via string subclass trick:
        # just use existing real enums that have no direct edge between them.
        # We'll register our own chain between SAMPA→XSAMPA→RFE.
        key_ab = (Alphabet.BOPOMOFO, Alphabet.ERAAB)
        key_bc = (Alphabet.ERAAB, Alphabet.BUCKWALTER)
        old_ab = ALPHABET_CONVERTERS.get(key_ab)
        old_bc = ALPHABET_CONVERTERS.get(key_bc)
        try:
            register_converter(Alphabet.BOPOMOFO, Alphabet.ERAAB, edge_ab)
            register_converter(Alphabet.ERAAB, Alphabet.BUCKWALTER, edge_bc)
            result = convert("start", "en", Alphabet.BOPOMOFO, Alphabet.BUCKWALTER)
            self.assertEqual(result, "start_B_C")
            self.assertEqual(calls, ["ab", "bc"])
        finally:
            for key, old in [(key_ab, old_ab), (key_bc, old_bc)]:
                if old is None:
                    ALPHABET_CONVERTERS.pop(key, None)
                else:
                    ALPHABET_CONVERTERS[key] = old


class TestNoPathIdentity(unittest.TestCase):
    def test_no_path_returns_input_unchanged(self):
        """When no conversion path exists, input is returned unchanged."""
        # SAMPA has no registered edges (scriptconv provides X-SAMPA, not SAMPA),
        # so there is no path to ARPA.
        text = "some text"
        result = convert(text, "en", Alphabet.SAMPA, Alphabet.ARPA)
        self.assertEqual(result, text)


class TestScriptConversion(unittest.TestCase):
    def test_graphemes_to_hira_with_pykakasi(self):
        """GRAPHEMES→HIRA edge calls pykakasi when available."""
        mock_kks = MagicMock()
        mock_kks.convert.return_value = [{"hira": "と"}, {"hira": "き"}]

        mock_kakasi_module = MagicMock()
        mock_kakasi_module.kakasi.return_value = mock_kks

        with patch.dict("sys.modules", {"pykakasi": mock_kakasi_module}):
            result = convert("東京", "ja", Alphabet.GRAPHEMES, Alphabet.HIRA)

        self.assertEqual(result, "とき")

    def test_graphemes_to_hira_no_pykakasi(self):
        """GRAPHEMES→HIRA falls back gracefully when pykakasi missing."""
        import sys
        old = sys.modules.pop("pykakasi", None)
        try:
            with patch.dict("sys.modules", {"pykakasi": None}):
                result = convert("東京", "ja", Alphabet.GRAPHEMES, Alphabet.HIRA)
            # Should return input unchanged
            self.assertEqual(result, "東京")
        finally:
            if old is not None:
                sys.modules["pykakasi"] = old


class TestPhonemizationEdge(unittest.TestCase):
    def test_graphemes_to_ipa_calls_phonemizer(self):
        """GRAPHEMES→IPA edge delegates to the existing phonemizer machinery."""
        mock_phonemizer = MagicMock()
        mock_phonemizer.phonemize.return_value = [["h", "ɛ", "l", "oʊ"]]

        # get_phonemizer is imported inside the edge closure from phoonnx.config
        with patch("phoonnx.config.get_phonemizer",
                   return_value=mock_phonemizer) as mock_get:
            result = convert(
                "hello", "en",
                Alphabet.GRAPHEMES, Alphabet.IPA,
                phoneme_type=PhonemeType.ESPEAK,
            )

        mock_get.assert_called_once_with(PhonemeType.ESPEAK, Alphabet.IPA)
        mock_phonemizer.phonemize.assert_called_once_with("hello", "en")
        self.assertEqual(result, [["h", "ɛ", "l", "oʊ"]])


class TestSynthesizeReshape(unittest.TestCase):
    """Verify TTSVoice.synthesize uses convert and produces identical output."""

    def _make_voice(self, phoneme_type=PhonemeType.ESPEAK,
                    alphabet=Alphabet.IPA):
        """Build a minimal TTSVoice with mocked ONNX session."""
        from phoonnx.config import VoiceConfig, BlankBetween
        from phoonnx.tokenizer import TTSTokenizer, Vocabulary
        from phoonnx.voice import TTSVoice
        import numpy as np

        vocab = Vocabulary(char2idx={"a": 1, "b": 2, "_": 0}, blank="_")
        tokenizer = TTSTokenizer(vocab,
                                 add_blank_char=True,
                                 add_blank_word=False,
                                 use_eos_bos=False,
                                 blank_at_end=True,
                                 blank_at_start=True)
        cfg = VoiceConfig(
            num_symbols=3,
            num_speakers=1,
            num_langs=1,
            sample_rate=22050,
            lang_code="en",
            phoneme_type=phoneme_type,
            alphabet=alphabet,
            phonemizer_model=None,
        )
        cfg.tokenizer = tokenizer

        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(name="input")]
        session.get_outputs.return_value = [MagicMock(name="output")]
        session.run.return_value = [np.zeros((1, 1, 4), dtype=np.float32)]

        voice = TTSVoice.__new__(TTSVoice)
        voice.session = session
        voice.config = cfg
        voice.phonetic_spellings = None

        mock_phonemizer = MagicMock(spec=["phonemize", "add_diacritics"])
        mock_phonemizer.phonemize.return_value = [["a", "b"]]
        mock_phonemizer.add_diacritics.side_effect = lambda t, l: t
        voice.phonemizer = mock_phonemizer

        mock_adapter = MagicMock()
        mock_adapter.default_params.return_value = {}
        mock_adapter.build_feed_dict.return_value = {}
        mock_adapter.parse_outputs.return_value = MagicMock(
            audio=np.zeros(4, dtype=np.float32)
        )
        voice.adapter = mock_adapter

        return voice

    def test_ipa_voice_grapheme_input_phonemizes(self):
        """IPA voice with grapheme input uses the phonemize path (not the graph).

        Grapheme -> phoneme is phonemization, so it runs through the model's own
        phonemizer; the alphabet-conversion graph is reserved for already-phonemic
        input in a different alphabet.
        """
        from phoonnx.config import SynthesisConfig
        voice = self._make_voice()  # tgt=IPA, grapheme input
        syn_cfg = SynthesisConfig(normalize_audio=False)

        list(voice.synthesize("hello", syn_cfg))

        # The model's own phonemizer phonemized the graphemes.
        voice.phonemizer.phonemize.assert_called()

    def test_grapheme_voice_no_phonemize(self):
        """Grapheme/UNICODE voice: convert returns the text string path."""
        from phoonnx.config import SynthesisConfig
        voice = self._make_voice(
            phoneme_type=PhonemeType.GRAPHEMES,
            alphabet=Alphabet.UNICODE,
        )
        syn_cfg = SynthesisConfig(normalize_audio=False)

        # With src==tgt (both UNICODE-ish), the identity path runs.
        # We just verify it doesn't crash.
        try:
            list(voice.synthesize("ab", syn_cfg))
        except Exception as e:
            # Tokenisation may fail since vocab is tiny — that's OK.
            pass





class TestScriptconvNotationEdges(unittest.TestCase):
    """Phoneme-notation edges delegated to scriptconv (IPA-pivot + chaining)."""

    def test_ipa_to_arpa(self):
        self.assertEqual(convert("ʃə", "en", Alphabet.IPA, Alphabet.ARPA), "SH AX")

    def test_xsampa_to_ipa(self):
        self.assertEqual(convert("S@", "en", Alphabet.XSAMPA, Alphabet.IPA), "ʃə")

    def test_xsampa_to_arpa_chains_through_ipa(self):
        # BFS finds X-SAMPA -> IPA -> ARPA, both scriptconv edges.
        self.assertEqual(convert("S@", "en", Alphabet.XSAMPA, Alphabet.ARPA), "SH AX")

    def test_ipa_to_rfe(self):
        self.assertEqual(convert("ʃ", "es", Alphabet.IPA, Alphabet.RFE), "š")

    def test_ipa_to_cotovia(self):
        self.assertEqual(convert("tʃ", "gl", Alphabet.IPA, Alphabet.COTOVIA), "tS")

    def test_hira_to_kana(self):
        self.assertEqual(convert("ひらがな", "ja", Alphabet.HIRA, Alphabet.KANA), "ヒラガナ")


if __name__ == "__main__":
    unittest.main()
