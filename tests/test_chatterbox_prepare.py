"""``TTSVoice.prepare`` reaches the adapter's ``encode_text`` for an engine that reads text.

The voices here are built from a hand-written tokenizer and config with a fake
session, so no model is loaded. The expected ids come from the tokenizer's own
vocabulary, read through the ``tokenizers`` package.
"""
import json
import types

from tokenizers import Tokenizer

from phoonnx.config import VoiceConfig
from phoonnx.engines import get_adapter
from phoonnx.voice import TTSVoice

TOKENS = ["[UNK]", "[SPACE]", "[en]", "[pt]", "h", "i", "o", "l", "a"]
VOCAB = {token: index for index, token in enumerate(TOKENS)}


def _tokenizer_json(path):
    path.write_text(json.dumps({
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "\\[[A-Za-z]+\\]|."},
                          "behavior": "Isolated", "invert": False},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": VOCAB, "unk_token": "[UNK]"},
    }))
    return str(path)


def _voice(tmp_path, config):
    voice_config = VoiceConfig.from_dict(config, bpe_tokenizer_json=_tokenizer_json(tmp_path / "tokenizer.json"))
    return TTSVoice(session=types.SimpleNamespace(), config=voice_config,
                    adapter=get_adapter(voice_config.engine.value))


def test_chatterbox_multilingual_ids_carry_the_language_token(tmp_path):
    voice = _voice(tmp_path, {"engine": "chatterbox", "lang_code": "pt-PT", "alphabet": "unicode"})
    utterances = list(voice.prepare("hi"))
    expected = [VOCAB["[pt]"], VOCAB["h"], VOCAB["i"]]
    assert [list(u.ids) for u in utterances] == [expected]


def test_a_voice_that_reads_symbols_gives_the_same_ids(tmp_path):
    letters = "abcdefghijklmnopqrstuvwxyz"
    config = {"phoonnx_version": "1.0", "engine": "phoonnx", "phoneme_type": "unicode",
              "alphabet": "unicode", "lang_code": "en-US", "audio": {"sample_rate": 16000},
              "num_symbols": 64,
              "phoneme_id_map": {"_": [0], "^": [1], "$": [2], **{c: [i + 3] for i, c in enumerate(letters)}}}
    voice = TTSVoice(session=types.SimpleNamespace(), config=VoiceConfig.from_dict(config),
                     adapter=get_adapter("vits"))
    ids = [list(u.ids) for u in voice.prepare("hi")]
    assert ids == [[1, 0, 10, 0, 11, 0, 2]]


def test_each_sentence_is_one_engine_call_with_its_own_language_token(tmp_path):
    voice = _voice(tmp_path, {"engine": "chatterbox", "lang_code": "pt-PT", "alphabet": "unicode"})
    ids = [list(u.ids) for u in voice.prepare("hi. ola.")]
    letters = {VOCAB[c] for c in "hiola"}
    assert len(ids) == 2
    assert [chunk[0] for chunk in ids] == [VOCAB["[pt]"], VOCAB["[pt]"]]
    assert [[i for i in chunk if i in letters] for chunk in ids] == [
        [VOCAB["h"], VOCAB["i"]], [VOCAB["o"], VOCAB["l"], VOCAB["a"]]]
