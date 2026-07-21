import json
import os
import tempfile
import unittest
from unittest import mock

from phoonnx.tokenizer import (
    BlankBetween,
    BPETokenizer,
    ChatterboxMTLTokenizer,
    TTSTokenizer,
    Vocabulary,
    load_chatterbox_tokenizer,
    phoneme_map_seed,
    untrained_map_symbols,
)


def _tiny_tokenizer(add_blank_char, add_blank_word, use_eos_bos,
                     blank_at_start=True, blank_at_end=True):
    voc = Vocabulary(char2idx={"_": 0, "^": 1, "$": 2, " ": 3, "a": 4, "b": 5},
                      pad="_", bos="^", eos="$", blank="_", blank_word=" ")
    return TTSTokenizer(voc, add_blank_char=add_blank_char,
                        add_blank_word=add_blank_word,
                        use_eos_bos=use_eos_bos,
                        blank_at_start=blank_at_start,
                        blank_at_end=blank_at_end)


class TestTinyVocabBooleanPermutations(unittest.TestCase):
    """A wrong token ID here corrupts audio silently -- assert exact sequences."""

    def test_add_blank_char_true_use_eos_bos_true(self):
        tok = _tiny_tokenizer(add_blank_char=True, add_blank_word=False, use_eos_bos=True)
        self.assertEqual(tok.tokenize("ab"), [1, 0, 4, 0, 5, 0, 2])

    def test_add_blank_char_true_use_eos_bos_false(self):
        tok = _tiny_tokenizer(add_blank_char=True, add_blank_word=False, use_eos_bos=False)
        self.assertEqual(tok.tokenize("ab"), [0, 4, 0, 5, 0])

    def test_add_blank_char_false_use_eos_bos_true(self):
        tok = _tiny_tokenizer(add_blank_char=False, add_blank_word=False, use_eos_bos=True)
        self.assertEqual(tok.tokenize("ab"), [1, 0, 4, 5, 2])

    def test_add_blank_char_false_use_eos_bos_false(self):
        tok = _tiny_tokenizer(add_blank_char=False, add_blank_word=False, use_eos_bos=False)
        self.assertEqual(tok.tokenize("ab"), [0, 4, 5])

    def test_word_blank_vs_char_blank(self):
        # add_blank_word appends the word-separator id at encode-time (mimic3
        # compatibility); add_blank_char is off so no interspersed blanks appear.
        tok = _tiny_tokenizer(add_blank_char=False, add_blank_word=True, use_eos_bos=False,
                              blank_at_start=False, blank_at_end=True)
        self.assertEqual(tok.tokenize("a b"), [4, 3, 5, 3])

    def test_blank_at_start_false_blank_at_end_true(self):
        tok = _tiny_tokenizer(add_blank_char=True, add_blank_word=False, use_eos_bos=False,
                              blank_at_start=False, blank_at_end=True)
        self.assertEqual(tok.tokenize("ab"), [4, 0, 5, 0])

    def test_blank_at_start_true_blank_at_end_false(self):
        tok = _tiny_tokenizer(add_blank_char=True, add_blank_word=False, use_eos_bos=False,
                              blank_at_start=True, blank_at_end=False)
        self.assertEqual(tok.tokenize("ab"), [0, 4, 0, 5])

    def test_blank_at_start_false_blank_at_end_false(self):
        tok = _tiny_tokenizer(add_blank_char=True, add_blank_word=False, use_eos_bos=False,
                              blank_at_start=False, blank_at_end=False)
        self.assertEqual(tok.tokenize("ab"), [4, 0, 5])


class TestVocabularyFromPiperConfig(unittest.TestCase):
    def test_missing_phoneme_id_map_yields_empty_vocab_with_defaults(self):
        voc = Vocabulary.from_piper_config({})
        self.assertEqual(voc.char2idx, {})
        self.assertEqual(voc.pad, "_")
        self.assertEqual(voc.bos, "^")
        self.assertEqual(voc.eos, "$")
        self.assertEqual(voc.blank, "_")

    def test_extracts_first_element_of_id_lists(self):
        cfg = {"phoneme_id_map": {"a": [4, 99], "b": [5]}}
        voc = Vocabulary.from_piper_config(cfg)
        self.assertEqual(voc.char2idx, {"a": 4, "b": 5})

    def test_explicit_special_tokens_override_defaults(self):
        cfg = {"phoneme_id_map": {}, "pad": "<P>", "bos": "<B>", "eos": "<E>", "blank": "<K>"}
        voc = Vocabulary.from_piper_config(cfg)
        self.assertEqual((voc.pad, voc.bos, voc.eos, voc.blank), ("<P>", "<B>", "<E>", "<K>"))


class TestVocabularyFromMimic3Config(unittest.TestCase):
    def test_duplicate_ids_across_distinct_tokens_do_not_raise(self):
        # Two distinct phoneme strings sharing an id is a legal (if unusual)
        # tokens.txt shape; char2idx is keyed by token, so both survive, but
        # idx2char collapses to whichever entry iterates last.
        tokens_txt = "0 a\n0 b\n1 c\n"
        voc = Vocabulary.from_tokens_txt(tokens_txt, id_first=True)
        self.assertEqual(voc.char2idx, {"a": 0, "b": 0, "c": 1})
        self.assertIn(voc.idx2char[0], {"a", "b"})

    def test_out_of_range_id_is_accepted_without_validation(self):
        # from_tokens_txt performs no bounds checking on the parsed id.
        tokens_txt = "0 a\n9999 b\n-5 c\n"
        voc = Vocabulary.from_tokens_txt(tokens_txt, id_first=True)
        self.assertEqual(voc.char2idx["b"], 9999)
        self.assertEqual(voc.char2idx["c"], -5)

    def test_malformed_lines_are_skipped_not_raised(self):
        tokens_txt = "0 a\n\nnotanumber token\nsingletoken\n1 b\n"
        voc = Vocabulary.from_tokens_txt(tokens_txt, id_first=True)
        self.assertEqual(voc.char2idx, {"a": 0, "b": 1})

    def test_special_tokens_pulled_from_phonemes_section(self):
        tokens_txt = "0 _\n1 ^\n2 $\n3 a\n"
        cfg = {"phonemes": {"pad": "_", "bos": "^", "eos": "$",
                             "blank": "_", "blank_word": " "}}
        voc = Vocabulary.from_mimic3_config(cfg, tokens_txt)
        self.assertEqual(voc.pad, "_")
        self.assertEqual(voc.bos, "^")
        self.assertEqual(voc.eos, "$")
        self.assertEqual(voc.blank, "_")
        self.assertEqual(voc.blank_word, " ")

    def test_phoneme_separator_and_word_separator_fallbacks(self):
        # when "pad"/"blank_word" are absent, mimic3 configs use the older
        # "phoneme_separator" / "word_separator" key names instead.
        tokens_txt = "0 _\n1 a\n"
        cfg = {"phonemes": {"phoneme_separator": "_", "word_separator": "#"}}
        voc = Vocabulary.from_mimic3_config(cfg, tokens_txt)
        self.assertEqual(voc.pad, "_")
        self.assertEqual(voc.blank_word, "#")


class TestVocabularyFromCoquiConfig(unittest.TestCase):
    def test_unsupported_characters_class_raises(self):
        cfg = {"characters": {"characters_class": "some.other.Class"}}
        with self.assertRaises(ValueError):
            Vocabulary.from_coqui_config(cfg)

    def test_vits_characters_layout_exact_ids(self):
        cfg = {"characters": {
            "characters_class": "TTS.tts.models.vits.VitsCharacters",
            "pad": "_", "punctuations": "!,",
            "characters": "ab", "blank": "<BLNK>",
        }}
        voc = Vocabulary.from_coqui_config(cfg)
        # vocab = punctuations + characters, pad inserted at 0, blank appended
        self.assertEqual(voc.char2idx, {"_": 0, "!": 1, ",": 2, "a": 3, "b": 4, "<BLNK>": 5})

    def test_vits_characters_add_blank_defaults_blank_token(self):
        cfg = {"characters": {
            "characters_class": "TTS.tts.models.vits.VitsCharacters",
            "punctuations": "", "characters": "a",
        }, "add_blank": True}
        voc = Vocabulary.from_coqui_config(cfg)
        self.assertEqual(voc.blank, "<BLNK>")
        self.assertIn("<BLNK>", voc.char2idx)

    def test_graphemes_layout_exact_ordering(self):
        cfg = {"characters": {
            "characters_class": "TTS.tts.utils.text.characters.Graphemes",
            "pad": "_", "bos": "^", "eos": "$", "blank": "@",
            "characters": "ba", "punctuations": "!",
        }}
        voc = Vocabulary.from_coqui_config(cfg)
        # order: pad, eos, bos, blank, *characters(unsorted, as given), *punctuations
        self.assertEqual(list(voc.char2idx.keys()), ["_", "$", "^", "@", "b", "a", "!"])

    def test_graphemes_unique_and_sorted(self):
        cfg = {"characters": {
            "characters_class": "TTS.tts.utils.text.characters.Graphemes",
            "characters": "baab", "punctuations": "",
            "is_unique": True, "is_sorted": True,
        }}
        voc = Vocabulary.from_coqui_config(cfg)
        self.assertEqual(list(voc.char2idx.keys()), ["a", "b"])


class TestChatterboxTokenizerLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer

        cls.tmpdir = tempfile.mkdtemp()

        base_tok = Tokenizer(BPE(unk_token="[UNK]"))
        base_tok.train_from_iterator(["hello world", "foo bar baz"],
                                      trainer=BpeTrainer(special_tokens=["[UNK]"]))
        cls.base_path = os.path.join(cls.tmpdir, "base_tokenizer.json")
        base_tok.save(cls.base_path)

        mtl_tok = Tokenizer(BPE(unk_token="[UNK]"))
        mtl_tok.train_from_iterator(["hello world", "foo bar baz"],
                                     trainer=BpeTrainer(special_tokens=["[UNK]", "[SPACE]", "[pt]"]))
        cls.mtl_path = os.path.join(cls.tmpdir, "mtl_tokenizer.json")
        mtl_tok.save(cls.mtl_path)

    def test_no_space_token_yields_plain_bpe_tokenizer(self):
        tok = load_chatterbox_tokenizer(self.base_path)
        self.assertIsInstance(tok, BPETokenizer)
        self.assertNotIsInstance(tok, ChatterboxMTLTokenizer)

    def test_space_token_present_yields_mtl_tokenizer(self):
        tok = load_chatterbox_tokenizer(self.mtl_path)
        self.assertIsInstance(tok, ChatterboxMTLTokenizer)

    def test_bpe_tokenizer_encodes_to_int_ids(self):
        tok = load_chatterbox_tokenizer(self.base_path)
        ids = tok.tokenize("hello world")
        self.assertTrue(len(ids) > 0)
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_bpe_tokenizer_joins_list_input_before_encoding(self):
        tok = load_chatterbox_tokenizer(self.base_path)
        self.assertEqual(tok.tokenize(["h", "i"]), tok.tokenize("hi"))

    def test_mtl_tokenizer_without_language_degrades_to_plain_bpe(self):
        tok = load_chatterbox_tokenizer(self.mtl_path)
        # no language given and lang_tokens absent -> same output as base encode
        self.assertEqual(tok.tokenize("hello"), list(tok._tok.encode("hello").ids))

    def test_mtl_tokenizer_unknown_language_without_vocab_token_degrades(self):
        tok = load_chatterbox_tokenizer(self.mtl_path)
        # "xx" has no [xx] token in this tiny vocab and no lang_tokens override
        self.assertEqual(tok.tokenize("hello", language="xx"),
                         list(tok._tok.encode("hello").ids))

    def test_mtl_tokenizer_known_language_prefixes_lang_token(self):
        tok = load_chatterbox_tokenizer(self.mtl_path)
        out = tok.tokenize("hello", language="pt")
        plain = list(tok._tok.encode("hello").ids)
        self.assertNotEqual(out, plain)

    def test_mtl_tokenizer_joins_list_input(self):
        tok = load_chatterbox_tokenizer(self.mtl_path)
        self.assertEqual(tok.tokenize(["he", "llo"], language="pt"),
                         tok.tokenize("hello", language="pt"))

    def test_mtl_tokenizer_explicit_lang_tokens_override_wins(self):
        tok = load_chatterbox_tokenizer(self.mtl_path)
        # literal lang_tokens entry is honoured even for a code with no vocab entry
        out_literal = tok.tokenize("hello", language="eg-EG", lang_tokens={"eg-EG": "pt"})
        out_derived = tok.tokenize("hello", language="pt")
        self.assertEqual(out_literal, out_derived)


class TestPhonemeMapSeedAndUntrainedSymbols(unittest.TestCase):
    def test_seed_includes_default_ipa_table_when_requested(self):
        syms = phoneme_map_seed({"x_custom"}, ipa=True)
        self.assertIn("x_custom", syms)
        self.assertIn("a", syms)  # from DEFAULT_IPA_PHONEME_ID_MAP

    def test_seed_excludes_defaults_when_include_defaults_false(self):
        syms = phoneme_map_seed({"x_custom"}, ipa=True, include_defaults=False)
        self.assertEqual(syms, {"x_custom"})

    def test_seed_excludes_defaults_when_not_ipa(self):
        syms = phoneme_map_seed({"x_custom"}, ipa=False)
        self.assertEqual(syms, {"x_custom"})

    def test_untrained_symbols_excludes_special_tokens(self):
        phoneme_id_map = {"_": 0, "^": 1, "$": 2, " ": 3, "a": 4, "z": 5}
        untrained = untrained_map_symbols(phoneme_id_map, corpus_phonemes={"a"})
        # special tokens (_, ^, $, blank-word " ") are excluded even though
        # they're absent from the corpus; only "z" is a genuinely unseen symbol.
        self.assertEqual(untrained, ["z"])

    def test_untrained_symbols_empty_when_all_covered(self):
        phoneme_id_map = {"a": 0, "b": 1}
        self.assertEqual(untrained_map_symbols(phoneme_id_map, corpus_phonemes={"a", "b"}), [])


class TestTTSTokenizerFactoriesFromConfig(unittest.TestCase):
    def test_from_piper_config_defaults(self):
        cfg = {"phoneme_id_map": {"a": [4], "b": [5]}, "pad": "_", "bos": "^", "eos": "$", "blank": "_"}
        tok = TTSTokenizer.from_piper_config(cfg)
        self.assertTrue(tok.add_blank_char)
        self.assertFalse(tok.add_blank_word)
        self.assertTrue(tok.use_eos_bos)
        self.assertTrue(tok.blank_at_start)
        self.assertTrue(tok.blank_at_end)

    def test_from_mimic3_config_tokens_blank(self):
        cfg = {"phonemes": {"blank_between": "tokens", "blank_at_end": False,
                             "blank_at_start": False, "auto_bos_eos": False}}
        tok = TTSTokenizer.from_mimic3_config(cfg, "0 _\n1 a\n")
        self.assertTrue(tok.add_blank_char)  # blank_between != WORDS
        self.assertFalse(tok.add_blank_word)  # blank_between == TOKENS
        self.assertFalse(tok.blank_at_end)
        self.assertFalse(tok.blank_at_start)
        self.assertFalse(tok.use_eos_bos)

    def test_from_mimic3_config_words_blank(self):
        cfg = {"phonemes": {"blank_between": "words"}}
        tok = TTSTokenizer.from_mimic3_config(cfg, "0 _\n1 a\n")
        self.assertFalse(tok.add_blank_char)  # blank_between == WORDS
        self.assertTrue(tok.add_blank_word)

    def test_from_tokens_txt_factory_defaults(self):
        tok = TTSTokenizer.from_tokens_txt("0 _\n1 a\n2 b\n")
        self.assertTrue(tok.add_blank_char)
        self.assertFalse(tok.add_blank_word)
        self.assertTrue(tok.blank_at_start)
        self.assertTrue(tok.blank_at_end)
        self.assertFalse(tok.use_eos_bos)

    def test_from_coqui_config_add_blank_true(self):
        cfg = {"characters": {"characters_class": "TTS.tts.utils.text.characters.Graphemes",
                              "characters": "ab", "punctuations": ""},
               "add_blank": True, "enable_eos_bos_chars": True}
        tok = TTSTokenizer.from_coqui_config(cfg)
        self.assertTrue(tok.add_blank_char)
        self.assertTrue(tok.blank_at_start)
        self.assertTrue(tok.blank_at_end)
        self.assertTrue(tok.use_eos_bos)
        self.assertFalse(tok.add_blank_word)

    def test_from_coqui_config_add_blank_false_defaults(self):
        cfg = {"characters": {"characters_class": "TTS.tts.utils.text.characters.Graphemes",
                              "characters": "ab", "punctuations": ""}}
        tok = TTSTokenizer.from_coqui_config(cfg)
        self.assertFalse(tok.add_blank_char)
        self.assertFalse(tok.blank_at_start)
        self.assertFalse(tok.blank_at_end)
        self.assertFalse(tok.use_eos_bos)


class TestCompoundPhonemeEncoding(unittest.TestCase):
    """Mimic3-style tokens.txt can define multi-character (diphthong) tokens
    that must be matched greedily before falling back to single characters."""

    def test_compound_token_preferred_over_single_chars(self):
        voc = Vocabulary(char2idx={"a": 0, "i": 1, "ai": 2}, blank=None)
        tok = TTSTokenizer(voc, add_blank_char=False, add_blank_word=False,
                            use_eos_bos=False, blank_at_start=False, blank_at_end=False)
        self.assertEqual(tok.encode("ai"), [2])

    def test_unmatched_compound_falls_back_to_single_chars(self):
        voc = Vocabulary(char2idx={"a": 0, "b": 1, "ai": 2}, blank=None)
        tok = TTSTokenizer(voc, add_blank_char=False, add_blank_word=False,
                            use_eos_bos=False, blank_at_start=False, blank_at_end=False)
        self.assertEqual(tok.encode("ab"), [0, 1])

    def test_out_of_vocabulary_characters_are_dropped(self):
        voc = Vocabulary(char2idx={"a": 0}, blank=None)
        tok = TTSTokenizer(voc, add_blank_char=False, add_blank_word=False,
                            use_eos_bos=False, blank_at_start=False, blank_at_end=False)
        self.assertEqual(tok.encode("azb"), [0])


class TestBosEosSafetyPath(unittest.TestCase):
    def test_pad_with_bos_eos_without_bos_returns_input_unchanged(self):
        voc = Vocabulary(char2idx={"a": 0}, bos=None, eos="$")
        tok = TTSTokenizer(voc, add_blank_char=False, add_blank_word=False,
                            use_eos_bos=False, blank_at_start=False, blank_at_end=False)
        self.assertEqual(tok.pad_with_bos_eos([0]), [0])

    def test_intersperse_without_blank_token_returns_input_unchanged(self):
        voc = Vocabulary(char2idx={"a": 0}, blank=None)
        tok = TTSTokenizer(voc, add_blank_char=True, add_blank_word=False,
                            use_eos_bos=False, blank_at_start=True, blank_at_end=True)
        self.assertEqual(tok.intersperse_blank_char([0]), [0])


class TestVocabularyAndTokenizerFromPhoonnxConfig(unittest.TestCase):
    def test_vocabulary_from_phoonnx_config_defaults(self):
        voc = Vocabulary.from_phoonnx_config({"phoneme_id_map": {"a": 0}})
        self.assertEqual(voc.char2idx, {"a": 0})
        self.assertEqual((voc.pad, voc.bos, voc.eos, voc.blank), ("_", "^", "$", "_"))

    def test_vocabulary_from_phoonnx_config_explicit_tokens(self):
        cfg = {"phoneme_id_map": {"a": 0}, "pad": "P", "bos": "B", "eos": "E", "blank": "K"}
        voc = Vocabulary.from_phoonnx_config(cfg)
        self.assertEqual((voc.pad, voc.bos, voc.eos, voc.blank), ("P", "B", "E", "K"))

    def test_tokenizer_from_phoonnx_config_defaults(self):
        cfg = {"phoneme_id_map": {"a": 0, "_": 1, "^": 2, "$": 3}}
        tok = TTSTokenizer.from_phoonnx_config(cfg)
        self.assertTrue(tok.add_blank_char)
        self.assertTrue(tok.blank_at_start)
        self.assertTrue(tok.blank_at_end)
        self.assertTrue(tok.use_eos_bos)
        self.assertFalse(tok.add_blank_word)

    def test_tokenizer_from_phoonnx_config_explicit_flags_respected(self):
        cfg = {"phoneme_id_map": {"a": 0}, "add_blank_char": False, "blank_at_end": False,
               "blank_at_start": False, "use_eos_bos": False, "add_blank_word": True}
        tok = TTSTokenizer.from_phoonnx_config(cfg)
        self.assertFalse(tok.add_blank_char)
        self.assertFalse(tok.blank_at_end)
        self.assertFalse(tok.blank_at_start)
        self.assertFalse(tok.use_eos_bos)
        self.assertTrue(tok.add_blank_word)


class TestVocabularyProperties(unittest.TestCase):
    def test_num_chars_and_ids(self):
        voc = Vocabulary(char2idx={"_": 0, "^": 1, "$": 2, " ": 3, "a": 4},
                          pad="_", bos="^", eos="$", blank="_", blank_word=" ")
        self.assertEqual(voc.num_chars, 5)
        self.assertEqual(voc.pad_id, 0)
        self.assertEqual(voc.bos_id, 1)
        self.assertEqual(voc.eos_id, 2)
        self.assertEqual(voc.blank_id, 0)
        self.assertEqual(voc.blank_word_id, 3)

    def test_tokenizer_blank_word_id_property(self):
        voc = Vocabulary(char2idx={" ": 3}, blank_word=" ")
        tok = TTSTokenizer(voc, add_blank_char=False, add_blank_word=True,
                            use_eos_bos=False, blank_at_start=False, blank_at_end=False)
        self.assertEqual(tok.blank_word_id, 3)

    def test_ids_are_none_when_special_token_absent_from_vocab(self):
        voc = Vocabulary(char2idx={"a": 0}, pad="_", bos=None, eos=None, blank=None, blank_word=None)
        self.assertIsNone(voc.pad_id)  # "_" not in char2idx
        self.assertIsNone(voc.bos_id)
        self.assertIsNone(voc.eos_id)
        self.assertIsNone(voc.blank_id)
        self.assertIsNone(voc.blank_word_id)


class TestBPETokenizerDelegates(unittest.TestCase):
    def test_encode_delegates_to_tokenize(self):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer

        raw = Tokenizer(BPE(unk_token="[UNK]"))
        raw.train_from_iterator(["hello world"], trainer=BpeTrainer(special_tokens=["[UNK]"]))
        path = os.path.join(tempfile.mkdtemp(), "t.json")
        raw.save(path)

        tok = BPETokenizer(path)
        self.assertEqual(tok.encode("hello"), tok.tokenize("hello"))
        self.assertIsNone(tok.pad_id)
        self.assertIsNone(tok.blank_id)
        self.assertIsNone(tok.blank_word_id)


class TestOutOfVocabularyTracking(unittest.TestCase):
    """OOV phonemes must be dropped from the output but recorded and warned
    about, instead of vanishing silently."""

    def test_oov_char_is_recorded_and_warns(self):
        voc = Vocabulary(char2idx={"a": 0}, blank=None)
        tok = TTSTokenizer(voc, add_blank_char=False, add_blank_word=False,
                            use_eos_bos=False, blank_at_start=False, blank_at_end=False)
        with mock.patch("phoonnx.tokenizer.LOG.warning") as warn:
            result = tok.encode("azb")
        self.assertEqual(result, [0])
        self.assertEqual(tok.not_found_characters, {"z", "b"})
        warned_messages = [call.args[0] for call in warn.call_args_list]
        self.assertTrue(any("'z'" in msg for msg in warned_messages))
        self.assertTrue(any("'b'" in msg for msg in warned_messages))

    def test_same_oov_char_warns_only_once(self):
        voc = Vocabulary(char2idx={"a": 0}, blank=None)
        tok = TTSTokenizer(voc, add_blank_char=False, add_blank_word=False,
                            use_eos_bos=False, blank_at_start=False, blank_at_end=False)
        with mock.patch("phoonnx.tokenizer.LOG.warning") as warn:
            tok.encode("azzzaz")
        self.assertEqual(tok.not_found_characters, {"z"})
        warned_messages = [call.args[0] for call in warn.call_args_list]
        self.assertEqual(sum("'z'" in msg for msg in warned_messages), 1)

    def test_empty_string_encodes_to_empty_list(self):
        voc = Vocabulary(char2idx={"a": 0}, blank=None)
        tok = TTSTokenizer(voc, add_blank_char=False, add_blank_word=False,
                            use_eos_bos=False, blank_at_start=False, blank_at_end=False)
        self.assertEqual(tok.encode(""), [])
        self.assertEqual(tok.not_found_characters, set())


class TestVocabularyFormPreserved(unittest.TestCase):
    """The vocabulary is correct by construction (it comes from the model's own
    config), so its keys are used exactly as declared -- no unicode
    normalization of keys or input."""

    def test_combining_char_keys_kept_verbatim(self):
        # NFD-keyed map: 'a' + COMBINING TILDE as a single two-codepoint key
        key = "a\u0303"
        voc = Vocabulary(char2idx={key: 0, "b": 1}, blank=None)
        self.assertIn(key, voc.char2idx)
        tok = TTSTokenizer(voc, add_blank_char=False, add_blank_word=False,
                            use_eos_bos=False, blank_at_start=False, blank_at_end=False)
        self.assertEqual(tok.encode([key, "b"]), [0, 1])
        self.assertEqual(tok.not_found_characters, set())

    def test_compound_phonemes_match(self):
        voc = Vocabulary(char2idx={"a": 0, "i": 1, "ai": 2}, blank=None)
        tok = TTSTokenizer(voc, add_blank_char=False, add_blank_word=False,
                            use_eos_bos=False, blank_at_start=False, blank_at_end=False)
        self.assertEqual(tok.encode("ai"), [2])
        self.assertEqual(tok.not_found_characters, set())


if __name__ == "__main__":
    unittest.main()
