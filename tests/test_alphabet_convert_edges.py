"""Adversarial edge-case coverage for phoonnx.alphabet_convert.

Complements tests/test_alphabet_convert.py — deliberately avoids duplicating
scenarios already covered there (identity, direct edge, multi-hop chaining,
scriptconv notation edges, pykakasi hira happy/missing paths).
"""
import unittest
from unittest.mock import MagicMock, patch
import sys

from phoonnx.config import Alphabet, PhonemeType
from phoonnx.alphabet_convert import (
    ALPHABET_CONVERTERS,
    convert,
    register_converter,
    _find_path,
    _call_edge,
)


class TestFindPathNoPath(unittest.TestCase):
    def test_returns_none_when_target_unreachable(self):
        """SAMPA has no registered edges at all, so BFS must exhaust the
        queue and return None rather than raising or looping forever."""
        result = _find_path(Alphabet.SAMPA, Alphabet.ARPA)
        self.assertIsNone(result)

    def test_returns_empty_list_for_identical_nodes(self):
        result = _find_path(Alphabet.IPA, Alphabet.IPA)
        self.assertEqual(result, [])

    def test_no_path_from_isolated_node_even_as_source(self):
        """An alphabet that is never a source in the registry yields no path
        even when other alphabets *do* have outgoing edges from it as a
        target only."""
        result = _find_path(Alphabet.SAMPA, Alphabet.HANGUL)
        self.assertIsNone(result)


class TestCallEdgeArity(unittest.TestCase):
    def test_two_arg_edge_is_called_with_two_args(self):
        """An edge function that only accepts (text, lang) must be called
        with exactly those two arguments — arity is resolved from the
        function's own signature, not discovered by a trial call."""
        calls = []

        def two_arg_edge(text, lang):
            calls.append((text, lang))
            return text.upper()

        result = _call_edge(two_arg_edge, "hi", "en", PhonemeType.ESPEAK)
        self.assertEqual(result, "HI")
        self.assertEqual(calls, [("hi", "en")])

    def test_three_arg_edge_receives_phoneme_type(self):
        def three_arg_edge(text, lang, phoneme_type):
            return f"{text}:{lang}:{phoneme_type}"

        result = _call_edge(three_arg_edge, "hi", "en", PhonemeType.ESPEAK)
        self.assertEqual(result, "hi:en:PhonemeType.ESPEAK")

    def test_via_convert_end_to_end_with_two_arg_edge(self):
        """Same arity resolution, exercised through the public convert()
        entry point rather than calling _call_edge directly."""
        key = (Alphabet.SAMPA, Alphabet.RFE)
        old = ALPHABET_CONVERTERS.get(key)
        try:
            register_converter(Alphabet.SAMPA, Alphabet.RFE,
                                lambda text, lang: text + "!")
            result = convert("x", "en", Alphabet.SAMPA, Alphabet.RFE,
                              phoneme_type=PhonemeType.ESPEAK)
            self.assertEqual(result, "x!")
        finally:
            if old is None:
                ALPHABET_CONVERTERS.pop(key, None)
            else:
                ALPHABET_CONVERTERS[key] = old

    def test_via_convert_end_to_end_with_three_arg_edge(self):
        key = (Alphabet.SAMPA, Alphabet.RFE)
        old = ALPHABET_CONVERTERS.get(key)
        try:
            register_converter(Alphabet.SAMPA, Alphabet.RFE,
                                lambda text, lang, pt=None: f"{text}:{pt}")
            result = convert("x", "en", Alphabet.SAMPA, Alphabet.RFE,
                              phoneme_type=PhonemeType.ESPEAK)
            self.assertEqual(result, "x:PhonemeType.ESPEAK")
        finally:
            if old is None:
                ALPHABET_CONVERTERS.pop(key, None)
            else:
                ALPHABET_CONVERTERS[key] = old

    def test_typeerror_raised_inside_edge_propagates_and_runs_once(self):
        """A genuine TypeError raised inside a 3-arg edge's own body must
        not be swallowed and silently retried with 2 args — it must
        propagate, and the edge must have executed exactly once."""
        calls = []

        def three_arg_edge(text, lang, phoneme_type):
            calls.append((text, lang, phoneme_type))
            raise TypeError("bad internal cast")

        with self.assertRaises(TypeError):
            _call_edge(three_arg_edge, "hi", "en", PhonemeType.ESPEAK)
        self.assertEqual(len(calls), 1)

    def test_typeerror_raised_inside_two_arg_edge_propagates_and_runs_once(self):
        calls = []

        def two_arg_edge(text, lang):
            calls.append((text, lang))
            raise TypeError("bad internal cast")

        with self.assertRaises(TypeError):
            _call_edge(two_arg_edge, "hi", "en", PhonemeType.ESPEAK)
        self.assertEqual(len(calls), 1)


class TestRegisterConverterOverwrite(unittest.TestCase):
    def test_registering_same_edge_twice_overwrites(self):
        key = (Alphabet.SAMPA, Alphabet.RFE)
        old = ALPHABET_CONVERTERS.get(key)
        try:
            register_converter(Alphabet.SAMPA, Alphabet.RFE, lambda t, l, _=None: "first")
            register_converter(Alphabet.SAMPA, Alphabet.RFE, lambda t, l, _=None: "second")
            result = convert("x", "en", Alphabet.SAMPA, Alphabet.RFE)
            self.assertEqual(result, "second")
        finally:
            if old is None:
                ALPHABET_CONVERTERS.pop(key, None)
            else:
                ALPHABET_CONVERTERS[key] = old


class TestConvertNoOpSameAlphabet(unittest.TestCase):
    def test_convert_short_circuits_before_registry_lookup(self):
        """src == tgt must never even consult ALPHABET_CONVERTERS."""
        key = (Alphabet.BUCKWALTER, Alphabet.BUCKWALTER)
        self.assertNotIn(key, ALPHABET_CONVERTERS)
        result = convert("same", "ar", Alphabet.BUCKWALTER, Alphabet.BUCKWALTER)
        self.assertEqual(result, "same")

    def test_convert_no_op_on_empty_string(self):
        result = convert("", "en", Alphabet.IPA, Alphabet.IPA)
        self.assertEqual(result, "")


class TestCangjiePassthrough(unittest.TestCase):
    def test_pkuseg_import_error_falls_back_to_passthrough(self):
        import sys as _sys
        old = _sys.modules.pop("pkuseg", None)
        try:
            with patch.dict("sys.modules", {"pkuseg": None}):
                result = convert("你好世界", "zh", Alphabet.GRAPHEMES, Alphabet.CANGJIE)
            self.assertEqual(result, "你好世界")
        finally:
            if old is not None:
                _sys.modules["pkuseg"] = old

    def test_pkuseg_generic_exception_falls_back_to_passthrough(self):
        mock_pkuseg_module = MagicMock()
        mock_pkuseg_module.pkuseg.side_effect = RuntimeError("model not downloaded")
        with patch.dict("sys.modules", {"pkuseg": mock_pkuseg_module}):
            result = convert("你好", "zh", Alphabet.GRAPHEMES, Alphabet.CANGJIE)
        self.assertEqual(result, "你好")

    def test_pkuseg_success_path_segments_words(self):
        mock_seg = MagicMock()
        mock_seg.cut.return_value = ["你好", "世界"]
        mock_pkuseg_module = MagicMock()
        mock_pkuseg_module.pkuseg.return_value = mock_seg
        with patch.dict("sys.modules", {"pkuseg": mock_pkuseg_module}):
            result = convert("你好世界", "zh", Alphabet.GRAPHEMES, Alphabet.CANGJIE)
        self.assertEqual(result, "你好 世界")


class TestKanaPassthrough(unittest.TestCase):
    def test_pykakasi_import_error_falls_back_to_passthrough_for_kana(self):
        import sys as _sys
        old = _sys.modules.pop("pykakasi", None)
        try:
            with patch.dict("sys.modules", {"pykakasi": None}):
                result = convert("東京", "ja", Alphabet.GRAPHEMES, Alphabet.KANA)
            self.assertEqual(result, "東京")
        finally:
            if old is not None:
                _sys.modules["pykakasi"] = old

    def test_pykakasi_generic_exception_falls_back_to_passthrough(self):
        mock_kakasi_module = MagicMock()
        mock_kakasi_module.kakasi.side_effect = RuntimeError("dict load failed")
        with patch.dict("sys.modules", {"pykakasi": mock_kakasi_module}):
            result = convert("東京", "ja", Alphabet.GRAPHEMES, Alphabet.KANA)
        self.assertEqual(result, "東京")

    def test_pykakasi_success_path_for_kana(self):
        mock_kks = MagicMock()
        mock_kks.convert.return_value = [{"kana": "トウ"}, {"kana": "キョウ"}]
        mock_kakasi_module = MagicMock()
        mock_kakasi_module.kakasi.return_value = mock_kks
        with patch.dict("sys.modules", {"pykakasi": mock_kakasi_module}):
            result = convert("東京", "ja", Alphabet.GRAPHEMES, Alphabet.KANA)
        self.assertEqual(result, "トウキョウ")


class TestNoHangulToHiraEdge(unittest.TestCase):
    """HANGUL→HIRA was previously wired to a Korean jamo-decomposition
    function (a copy/paste error: it emitted NFD jamo, not hiragana, and
    there is no JAMO alphabet). That edge must not exist, directly or via
    a composed path."""

    def test_no_direct_edge_registered(self):
        self.assertNotIn((Alphabet.HANGUL, Alphabet.HIRA), ALPHABET_CONVERTERS)

    def test_no_path_from_hangul_to_hira(self):
        self.assertIsNone(_find_path(Alphabet.HANGUL, Alphabet.HIRA))

    def test_convert_returns_input_unchanged(self):
        """With no path, convert() falls back to its documented no-op
        behavior rather than emitting incorrect jamo output."""
        text = "가"
        result = convert(text, "ko", Alphabet.HANGUL, Alphabet.HIRA)
        self.assertEqual(result, text)


class TestHangulIdentityEdge(unittest.TestCase):
    def test_graphemes_to_hangul_is_passthrough(self):
        result = convert("한글", "ko", Alphabet.GRAPHEMES, Alphabet.HANGUL)
        self.assertEqual(result, "한글")

    def test_hangul_to_hangul_registered_identity(self):
        result = convert("한글", "ko", Alphabet.HANGUL, Alphabet.HANGUL)
        self.assertEqual(result, "한글")


class TestMixedScriptScriptConversion(unittest.TestCase):
    def test_mixed_script_input_through_pykakasi_mock(self):
        """Latin+Kanji mixed input must not crash the hiragana edge."""
        mock_kks = MagicMock()
        mock_kks.convert.return_value = [{"hira": "abc"}, {"hira": "とうきょう"}]
        mock_kakasi_module = MagicMock()
        mock_kakasi_module.kakasi.return_value = mock_kks
        with patch.dict("sys.modules", {"pykakasi": mock_kakasi_module}):
            result = convert("abc東京", "ja", Alphabet.GRAPHEMES, Alphabet.HIRA)
        self.assertEqual(result, "abcとうきょう")


if __name__ == "__main__":
    unittest.main()
