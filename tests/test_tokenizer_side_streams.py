"""A side stream keeps its length and its alignment with the ids when the tokenizer adds ids.

The vocabulary is ``_ ^ $ a b c`` with ids 0..5, where ``_`` is the blank,
``^`` the BOS and ``$`` the EOS. Every expected list is written out by hand from
the rule: a symbol's id keeps the symbol's value, an added id has value 0.
"""
import pytest

from phoonnx.tokenizer import TTSTokenizer, Vocabulary

ID_MAP = {"_": 0, "^": 1, "$": 2, "a": 3, "b": 4, "c": 5}


def _tokenizer(**flags):
    vocabulary = Vocabulary(char2idx=ID_MAP, pad="_", blank="_", bos="^", eos="$")
    settings = dict(add_blank_char=True, add_blank_word=False, use_eos_bos=True,
                    blank_at_start=True, blank_at_end=True)
    return TTSTokenizer(vocabulary, **{**settings, **flags})


def test_blanks_and_bos_eos_get_the_neutral_value():
    ids, streams = _tokenizer().tokenize_with_streams(["a", "b", "c"], {"language_ids": [1, 2, 1]})
    #        ^  _  a  _  b  _  c  _  $
    assert ids == [1, 0, 3, 0, 4, 0, 5, 0, 2]
    assert streams == {"language_ids": [0, 0, 1, 0, 2, 0, 1, 0, 0]}


def test_blanks_between_symbols_only():
    ids, streams = _tokenizer(blank_at_start=False, blank_at_end=False).tokenize_with_streams(
        ["a", "b", "c"], {"language_ids": [1, 2, 1]})
    #        ^  a  _  b  _  c  $
    assert ids == [1, 3, 0, 4, 0, 5, 2]
    assert streams == {"language_ids": [0, 1, 0, 2, 0, 1, 0]}


def test_blank_at_start_without_interspersed_blanks():
    ids, streams = _tokenizer(add_blank_char=False, use_eos_bos=False).tokenize_with_streams(
        ["a", "b"], {"stress": [1, 2]})
    #        _  a  b
    assert ids == [0, 3, 4]
    assert streams == {"stress": [0, 1, 2]}


def test_a_dropped_symbol_drops_its_value():
    ids, streams = _tokenizer().tokenize_with_streams(["a", "z", "b"], {"language_ids": [7, 8, 9]})
    #        ^  _  a  _  b  _  $
    assert ids == [1, 0, 3, 0, 4, 0, 2]
    assert streams == {"language_ids": [0, 0, 7, 0, 9, 0, 0]}


def test_two_streams_and_no_symbols():
    ids, streams = _tokenizer().tokenize_with_streams(["a"], {"language_ids": [2], "stress": [1]})
    #        ^  _  a  _  $
    assert ids == [1, 0, 3, 0, 2]
    assert streams == {"language_ids": [0, 0, 2, 0, 0], "stress": [0, 0, 1, 0, 0]}
    assert _tokenizer().tokenize_with_streams([], {"language_ids": []}) == ([1, 0, 2], {"language_ids": [0, 0, 0]})


@pytest.mark.parametrize("flags", [{}, {"blank_at_start": False}, {"add_blank_char": False}, {"use_eos_bos": False}])
def test_ids_match_tokenize(flags):
    tokenizer = _tokenizer(**flags)
    assert tokenizer.tokenize_with_streams(["a", "b", "c"], {})[0] == tokenizer.tokenize(["a", "b", "c"])


def test_word_blanks():
    vocabulary = Vocabulary(char2idx={**ID_MAP, "#": 6}, pad="_", blank="_", bos="^", eos="$", blank_word="#")
    tokenizer = TTSTokenizer(vocabulary, add_blank_char=True, add_blank_word=True, use_eos_bos=True,
                             blank_at_start=True, blank_at_end=True)
    # the second space follows a word blank, so it gives no id and its value 4 goes
    ids, streams = tokenizer.tokenize_with_streams(["a", " ", " ", "b"], {"language_ids": [1, 3, 4, 2]})
    #        ^  _  a  _  #  _  b  _  #  _  $
    assert ids == [1, 0, 3, 0, 6, 0, 4, 0, 6, 0, 2]
    assert streams == {"language_ids": [0, 0, 1, 0, 3, 0, 2, 0, 0, 0, 0]}


def test_folded_compound():
    vocabulary = Vocabulary(char2idx={**ID_MAP, "ab": 6}, pad="_", blank="_", bos="^", eos="$")
    tokenizer = TTSTokenizer(vocabulary, add_blank_char=True, add_blank_word=False, use_eos_bos=True,
                             blank_at_start=True, blank_at_end=True, fold_compounds=True)
    ids, streams = tokenizer.tokenize_with_streams(["a", "b", "c"], {"stress": [7, 8, 9]})
    #        ^  _ ab  _  c  _  $
    assert ids == [1, 0, 6, 0, 5, 0, 2]
    assert streams == {"stress": [0, 0, 7, 0, 9, 0, 0]}
