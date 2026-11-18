"""phoonnx.compat: a third-party config converts to the native config.json and reads back to the same ids."""
import copy
import json
import warnings

import pytest

from phoonnx import compat
from phoonnx.config import Alphabet, Engine, PhonemeType, VoiceConfig
from phoonnx.tokenizer import TTSTokenizer, Vocabulary


def piper_ids(id_map, phonemes):
    """Piper's own id stream: BOS, PAD, then each phoneme followed by PAD, then EOS."""
    ids = [*id_map["^"], *id_map["_"]]
    for phoneme in phonemes:
        ids += [*id_map[phoneme], *id_map["_"]]
    return ids + id_map["$"]


PIPER = {
    "piper_version": "1.0.0", "phoneme_type": "espeak",
    "phoneme_id_map": {"_": [0], "^": [1], "$": [2], "a": [3], "b": [4], "!": [5], " ": [6]},
    "num_symbols": 7, "num_speakers": 1, "audio": {"sample_rate": 22050},
    "espeak": {"voice": "en-us"},
    "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
}
VOSK = {"espeak": {"voice": "ru"},
        "phoneme_id_map": {"_": [0], "^": [1], "$": [2], "a0": [10], "sch": [20]}}
MIMIC3 = {"text_language": "en", "phonemizer": "gruut", "audio": {"sample_rate": 22050},
          "phonemes": {"pad": "_", "bos": "^", "eos": "$", "blank": "_", "blank_word": "#",
                       "blank_between": "tokens_and_words"}}
COQUI = {"audio": {"sample_rate": 22050}, "add_blank": True, "enable_eos_bos_chars": False,
         "characters": {"characters_class": "TTS.tts.models.vits.VitsCharacters", "pad": "<PAD>",
                        "punctuations": "!,", "characters": "ab", "blank": "<BLNK>"},
         "datasets": [{"language": "en"}]}
CANONICAL = {"engine": "coqui", "phoneme_type": "espeak", "alphabet": "ipa", "lang_code": "ca-ES",
             "num_symbols": 4, "audio": {"sample_rate": 22050},
             "phoneme_id_map": {"_": 0, " ": 1, "a": 2, "b": 3}, "pad": "_", "blank": "_"}
TRANSFORMERS_VOCAB = {"_": 0, "a": 1, "b": 2, " ": 3}
TRANSFORMERS_TOKENIZER = {"add_blank": True, "language": "en", "pad_token": "_"}
TOKENS_TXT = "_ 0\n^ 1\n$ 2\na 3\nb 4\n"


def legacy(config, **companions):
    """The voice the 1.x loader builds from the original files, and the warning it gives."""
    with pytest.warns(DeprecationWarning, match="phoonnx.compat.convert_config"):
        return VoiceConfig.from_dict(copy.deepcopy(config), **companions)


def native_read_back(native, tmp_path):
    """The voice that loads from the written native file, with every warning an error."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(native, ensure_ascii=False), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return VoiceConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))


def assert_same_voice(old, new, sentences):
    for sentence in sentences:
        assert new.tokenizer.tokenize(sentence) == old.tokenizer.tokenize(sentence)
    assert new.frontend == old.frontend
    assert (new.engine, new.sample_rate, new.num_symbols) == (old.engine, old.sample_rate, old.num_symbols)


def test_piper_ids_equal_the_loader_and_piper_itself(tmp_path):
    sentences = [["a", "b"], ["a", " ", "b", "!"]]
    old = legacy(PIPER)
    native = compat.convert_config(PIPER)
    new = native_read_back(native, tmp_path)
    assert_same_voice(old, new, sentences)
    for sentence in sentences:
        assert new.tokenizer.tokenize(sentence) == piper_ids(PIPER["phoneme_id_map"], sentence)
    assert native["keep_punctuation"] is True


def test_vosk_ids(tmp_path):
    old = legacy(VOSK)
    new = native_read_back(compat.convert_config(VOSK), tmp_path)
    assert_same_voice(old, new, [["a0", "sch"]])
    # vosk_tts reads ^ 0 p 0 p 0 $
    assert new.tokenizer.tokenize(["a0", "sch"]) == [1, 0, 10, 0, 20, 0, 2]
    assert new.phoneme_type == PhonemeType.VOSK and new.engine == Engine.VOSK


def test_mimic3_pins_its_language(tmp_path):
    tokens = tmp_path / "phonemes.txt"
    tokens.write_text("0 _\n1 ^\n2 $\n3 a\n", encoding="utf-8")
    old = legacy(MIMIC3, tokens_txt=str(tokens))
    native = compat.convert_config(MIMIC3, tokens=tokens, lang_code="fr")
    new = native_read_back(native, tmp_path)
    assert_same_voice(old, new, [["a"]])
    assert new.lang_code == "en" and new.engine == Engine.MIMIC3


def test_coqui_ids(tmp_path):
    old = legacy(COQUI)
    new = native_read_back(compat.convert_config(COQUI), tmp_path)
    assert_same_voice(old, new, [["a", "b"]])
    # VitsCharacters: pad 0, punctuations, characters, blank last; the blank sits between the symbols
    assert new.tokenizer.tokenize(["a", "b"]) == [5, 3, 5, 4, 5]


def test_transformers_ids(tmp_path):
    kwargs = dict(vocab=TRANSFORMERS_VOCAB, tokenizer_config=TRANSFORMERS_TOKENIZER, phoneme_type="graphemes",
                  alphabet="unicode", lang_code="en")
    old = legacy({}, **kwargs)
    new = native_read_back(compat.convert_config({}, **kwargs), tmp_path)
    assert_same_voice(old, new, [["a", "b"], ["b", " ", "a"]])
    # the MMS tokenizer puts its pad token between the characters and at both ends
    assert new.tokenizer.tokenize(["a", "b"]) == [0, 1, 0, 2, 0]


def test_convert_config_reads_files_for_the_companions(tmp_path):
    (tmp_path / "vocab.json").write_text(json.dumps(TRANSFORMERS_VOCAB), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(TRANSFORMERS_TOKENIZER), encoding="utf-8")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    native = compat.convert_config(tmp_path / "config.json", vocab=tmp_path / "vocab.json",
                                   tokenizer_config=tmp_path / "tokenizer_config.json",
                                   phoneme_type="graphemes", alphabet="unicode", lang_code="en")
    assert native["phoneme_id_map"] == TRANSFORMERS_VOCAB


def test_tokens_txt_ids(tmp_path):
    tokens = tmp_path / "tokens.txt"
    tokens.write_text(TOKENS_TXT, encoding="utf-8")
    config = {"pad": "_"}
    old = legacy(config, tokens_txt=str(tokens))
    new = native_read_back(compat.convert_config(config, tokens=tokens), tmp_path)
    assert_same_voice(old, new, [["a", "b"]])
    assert new.tokenizer.tokenize(["a", "b"]) == [0, 3, 0, 4, 0]


def test_canonical_config_ids(tmp_path):
    old = legacy(CANONICAL)
    new = native_read_back(compat.convert_config(CANONICAL), tmp_path)
    assert_same_voice(old, new, [["a", "b"]])
    assert new.tokenizer.tokenize(["a", "b"]) == [0, 2, 0, 3, 0]


def test_raw_text_engine_converts_to_an_empty_symbol_table(tmp_path):
    config = {"engine": "omnivoice"}
    old = legacy(config)
    native = compat.convert_config(config)
    new = native_read_back(native, tmp_path)
    assert native["phoneme_id_map"] == {} and native["engine"] == "omnivoice"
    assert_same_voice(old, new, [["a"]])
    assert new.alphabet == Alphabet.GRAPHEMES == old.alphabet
    assert new.sample_rate == 24000


def test_engine_override_names_a_raw_text_voice_without_a_config_key(tmp_path):
    native = compat.convert_config({}, engine="supertonic")
    assert native["engine"] == "supertonic" and native["audio"]["sample_rate"] == 44100


def test_native_config_round_trips_byte_equal(tmp_path):
    native = compat.convert_config(PIPER, lang_code="en-us")
    again = native_read_back(native, tmp_path).to_dict()
    assert json.dumps(again, ensure_ascii=False) == json.dumps(native, ensure_ascii=False)


def test_native_config_loads_without_a_warning(tmp_path):
    native_read_back(compat.convert_config(PIPER), tmp_path)


@pytest.mark.parametrize("old, new", [
    (lambda: TTSTokenizer.from_piper_config(PIPER), lambda: compat.from_piper_config(PIPER)),
    (lambda: TTSTokenizer.from_vosk_config(VOSK), lambda: compat.from_vosk_config(VOSK)),
    (lambda: TTSTokenizer.from_coqui_config(COQUI), lambda: compat.from_coqui_config(COQUI)),
    (lambda: TTSTokenizer.from_tokens_txt(TOKENS_TXT), lambda: compat.from_tokens_txt(TOKENS_TXT)),
    (lambda: TTSTokenizer.from_mimic3_config(MIMIC3, "0 _\n1 a\n"),
     lambda: compat.from_mimic3_config(MIMIC3, "0 _\n1 a\n")),
    (lambda: Vocabulary.from_piper_config(PIPER), lambda: compat.vocabulary_from_piper_config(PIPER)),
    (lambda: Vocabulary.from_coqui_config(COQUI), lambda: compat.vocabulary_from_coqui_config(COQUI)),
    (lambda: Vocabulary.from_tokens_txt(TOKENS_TXT), lambda: compat.vocabulary_from_tokens_txt(TOKENS_TXT)),
    (lambda: Vocabulary.from_mimic3_config(MIMIC3, "0 _\n"), lambda: compat.vocabulary_from_mimic3_config(MIMIC3, "0 _\n")),
])
def test_moved_constructors_keep_a_shim(old, new):
    with pytest.warns(DeprecationWarning, match=r"phoonnx 2\.0\.0.*phoonnx\.compat\."):
        shimmed = old()
    assert shimmed == new()


def test_config_loaders_module_keeps_a_shim():
    from phoonnx import config_loaders
    with pytest.warns(DeprecationWarning, match="phoonnx.compat.LOADERS"):
        assert config_loaders.LOADERS is compat.LOADERS
