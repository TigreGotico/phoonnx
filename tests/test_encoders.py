"""The encoders turn frontend output into ids, and ``TTSVoice.encoder`` builds the right one.

The frontends here are stand-ins that yield chunks written by hand, so each expected id
list follows from the chunk text and the tokenizer rule given in the test.
"""
import json
import types
from unittest.mock import MagicMock

from phoonnx.config import SynthesisConfig, VoiceConfig
from phoonnx.encoders import SymbolEncoder, TextEncoder, Utterance
from phoonnx.engines import get_adapter
from phoonnx.engines.base import BaseOnnxAdapter
from phoonnx.voice import TTSVoice
from scriptconv.phonemizers.enums import Alphabet

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _segment(text, phonemes=(), streams=None):
    return types.SimpleNamespace(text=text, phonemes=tuple(phonemes), streams=streams or {})


def _frontend(segments, prepared=""):
    return types.SimpleNamespace(segments=lambda text, stages: iter(segments), prepare=lambda text, stages: prepared)


def _vits_voice(adapter=None, **extra):
    config = {"phoonnx_version": "1.0", "engine": "phoonnx", "phoneme_type": "unicode",
              "alphabet": "unicode", "lang_code": "en-US", "audio": {"sample_rate": 16000},
              "num_symbols": 64,
              "phoneme_id_map": {"_": [0], "^": [1], "$": [2], **{c: [i + 3] for i, c in enumerate(LETTERS)}},
              **extra}
    return TTSVoice(session=types.SimpleNamespace(), config=VoiceConfig.from_dict(config),
                    adapter=adapter or get_adapter("vits"))


def _tokenizer():
    """One id per symbol, and a blank id 0 after each: the stream of a symbol repeats on its id, 0 on the blank."""
    def tokenize_with_streams(symbols, streams):
        ids = [x for s in symbols for x in (ord(s), 0)]
        return ids, {name: [v for value in values for v in (value, 0)] for name, values in streams.items()}

    return types.SimpleNamespace(
        tokenize=lambda symbols: [ord(s) for s in symbols], tokenize_with_streams=tokenize_with_streams)


def test_symbol_encoder_tokenizes_each_chunk_and_keeps_side_streams():
    frontend = _frontend([_segment("ab.", "ab", {"language_ids": (3, 4), "other": (5, 5)}),
                          _segment("", ()),
                          _segment("c.", "c")])
    utterances = list(SymbolEncoder(frontend, _tokenizer()).encode("ab. c.", SynthesisConfig()))
    assert utterances == [
        Utterance(text="ab.", symbols=("a", "b"), ids=(97, 0, 98, 0), streams={"language_ids": [3, 0, 4, 0]}),
        Utterance(text="c.", symbols=("c",), ids=(99,)),
    ]


def test_symbol_encoder_symbols_are_one_utterance():
    utterance = SymbolEncoder(_frontend([]), _tokenizer()).encode_symbols(["x", "y"], SynthesisConfig())
    assert utterance == Utterance(text=None, symbols=("x", "y"), ids=(120, 121))


def test_text_encoder_gives_each_chunk_to_the_tokenizer():
    frontend = _frontend([_segment("one two."), _segment("three.")])
    encoder = TextEncoder(frontend, lambda text, config: [[len(text)]])
    assert [(u.text, u.ids, u.symbols) for u in encoder.encode("x", SynthesisConfig())] == [
        ("one two.", (8,), None), ("three.", (6,), None)]


def test_text_encoder_whole_text_hands_the_prepared_text_over_once():
    frontend = _frontend([_segment("never read")], prepared="One. Two.")
    seen = []
    encoder = TextEncoder(frontend, lambda text, config: seen.append(text) or [[1], [], [2]], whole_text=True)
    utterances = list(encoder.encode("x", SynthesisConfig()))
    assert seen == ["One. Two."]
    assert [(u.text, u.ids) for u in utterances] == [(None, (1,)), (None, (2,))]


def test_text_encoder_reads_symbols_as_the_text_they_spell():
    encoder = TextEncoder(_frontend([]), lambda text, config: [[ord(c) for c in text]])
    assert encoder.encode_symbols(["h", "i"], SynthesisConfig()).ids == (104, 105)


def test_the_base_adapter_makes_a_symbol_encoder():
    voice = _vits_voice()
    assert isinstance(voice.encoder, SymbolEncoder)
    assert voice.encoder.tokenizer is voice.tokenizer


def test_an_adapter_with_encode_text_makes_a_whole_text_encoder_that_calls_it():
    class Adapter(get_adapter("vits").__class__):
        def encode_text(self, text, voice, syn_config):
            return [[len(text), len(voice.config.lang_code)]]

    voice = _vits_voice(adapter=Adapter())
    assert isinstance(voice.encoder, TextEncoder) and voice.encoder.whole_text
    assert [list(u.ids) for u in voice.prepare("hi")] == [[2, 5]]


def test_an_object_that_is_no_base_adapter_gets_a_symbol_encoder():
    assert not isinstance(MagicMock(), BaseOnnxAdapter)
    assert isinstance(_vits_voice(adapter=MagicMock()).encoder, SymbolEncoder)


def test_the_encoder_is_built_once_and_rebuilt_with_the_frontend():
    voice = _vits_voice()
    first = voice.encoder
    assert voice.encoder is first
    voice.phonemizer = MagicMock(keep_punctuation=False, normalizer=None,
                                 owned_stages=frozenset())
    assert voice.encoder is not first
    assert voice.encoder.frontend is voice.frontend


def test_phonemic_input_is_not_phonemized_and_reaches_the_symbol_tokenizer():
    voice = _vits_voice()
    utterances = list(voice.prepare("hi", SynthesisConfig(input_alphabet=Alphabet.UNICODE)))
    # blank, BOS and EOS around h=10 and i=11, by the vocabulary written in _vits_voice
    assert [(u.text, u.symbols, list(u.ids)) for u in utterances] == [
        (None, ("h", "i"), [1, 0, 10, 0, 11, 0, 2])]
