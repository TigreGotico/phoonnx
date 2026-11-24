"""The standalone card example builds the ids phoonnx builds, and names the graph's real feeds."""
import copy
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import scriptconv.frontend
import text2tashkeel
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from card_examples import UnsupportedExample, graph_io, standalone_example  # noqa: E402
import card_examples  # noqa: E402

from phoonnx.config import SynthesisConfig, VoiceConfig  # noqa: E402
from phoonnx.engines.matcha import MatchaAdapter  # noqa: E402
from phoonnx.engines.styletts2 import StyleTTS2Adapter  # noqa: E402
from phoonnx.engines.vocoders.raw import HiFiGANVocoder  # noqa: E402
from phoonnx.engines.vocoders.vocos import VocosVocoder  # noqa: E402
from phoonnx.model_manager import TTSModelInfo  # noqa: E402
from phoonnx.util import normalize_lang  # noqa: E402
from phoonnx.voice import TTSVoice  # noqa: E402
from scriptconv.phonemizers import get_phonemizer  # noqa: E402
from scriptconv.phonemizers.enums import Phonemizer  # noqa: E402
from scriptconv.phonemizers.registry import PHONEMIZER_REGISTRY  # noqa: E402

# The model input that synthesis gives, before the user's phonetic spellings.
NO_SPELLINGS = SynthesisConfig(phonetic_spellings=False)

PIPER = {"piper_version": "1.0.0", "phoneme_type": "espeak", "audio": {"sample_rate": 22050},
         "espeak": {"voice": "en-us"}, "num_speakers": 1, "lang_code": "en-US",
         "inference": {"noise_scale": 0.667, "length_scale": 1, "noise_w": 0.8},
         "phoneme_id_map": {"_": [0], "^": [1], "$": [2], " ": [3], ",": [4], ".": [5], "a": [6], "b": [7],
                            "ɪ": [8], "t": [9], "ʃ": [10], "ˈ": [11]}}

NATIVE = {"phoonnx_version": "1.0", "engine": "phoonnx", "phoneme_type": "espeak", "alphabet": "ipa",
          "lang_code": "en-US", "audio": {"sample_rate": 22050}, "num_speakers": 1,
          "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
          "phoneme_id_map": {"_": 0, "^": 1, "$": 2, " ": 3, "a": 4, "b": 5, "t": 6, "ʃ": 7, "t͡ʃ": 8, "aɪ": 9,
                             "ɪ": 10, "<bos>": 11, "<eos>": 12}}

PHONEMES = list("ab tʃ aɪ, bat.") + ["t", "͡", "ʃ", "?"]

TOKEN_FLAGS = [
    {},
    {"add_blank_char": False},
    {"add_blank_char": False, "use_eos_bos": False},
    {"blank_at_start": False},
    {"blank_at_end": False},
    {"blank_at_start": False, "blank_at_end": False},
    {"add_blank_char": False, "blank_at_start": True},
    {"add_blank_char": False, "blank_at_start": False},
    {"add_blank_char": False, "blank_at_start": False, "blank_at_end": False},
    {"add_blank_word": True},
    {"fold_compounds": False},
    {"pad": "<pad>", "blank": "<pad>", "bos": "<bos>", "eos": "<eos>", "add_blank_char": False},
    {"bos": "<bos>", "eos": "<eos>"},
]


VITS_FILES = {"inputs": {"input": 2, "input_lengths": 1, "scales": 1}}


def run_lines(code, start, names):
    """Run the example's imports, then its lines from *start* to the session, in *names*."""
    exec("\n".join(line for line in code.splitlines() if line.startswith(("import ", "from "))), names)
    exec(code[code.index(f"\n{start}") + 1:code.index("\nsession = ")], names)
    return names["ids"]


def rendered_ids(config, phonemes):
    """The ids that the rendered example builds from *phonemes*: its lines from ``id_map`` to the session."""
    return run_lines(example(config, VITS_FILES), "id_map = ",
                     {"config": copy.deepcopy(config), "phonemes": list(phonemes)})


def phoonnx_ids(config, phonemes):
    return VoiceConfig.from_dict(copy.deepcopy(config)).tokenizer.tokenize(list(phonemes))


def example(config, files, **kwargs):
    block = standalone_example(config, files, repo_id="Phoonnx/test-voice", **kwargs)
    assert block.startswith("```python\n") and block.endswith("\n```")
    code = block.removeprefix("```python\n").removesuffix("\n```")
    compile(code, "example.py", "exec")
    return code


def test_piper_ids_match_phoonnx():
    assert rendered_ids(PIPER, PHONEMES) == phoonnx_ids(PIPER, PHONEMES)


@pytest.mark.parametrize("flags", TOKEN_FLAGS, ids=lambda flags: ",".join(f"{k}={v}" for k, v in flags.items()) or "defaults")
def test_native_ids_match_phoonnx(flags):
    config = {**NATIVE, **flags}
    assert rendered_ids(config, PHONEMES) == phoonnx_ids(config, PHONEMES)


def test_symbols_of_several_characters_match_phoonnx():
    # Shami gives whole symbols ("aː", "<bos>"), not characters
    config = {**NATIVE, "phoneme_id_map": {**NATIVE["phoneme_id_map"], "aː": 13}, "add_blank_char": False,
              "use_eos_bos": False, "blank_at_start": False, "blank_at_end": False}
    phonemes = ["<bos>", "b", "aː", " ", "t", "͡ʃ", "a", "ɪ", "<eos>"]
    assert rendered_ids(config, phonemes) == phoonnx_ids(config, phonemes) == [11, 5, 13, 3, 6, 9, 12]


def test_piper_example_feeds_the_graph_by_its_names():
    code = example(PIPER, {"inputs": {"input": 2, "input_lengths": 1, "scales": 1}}, text="Hello there.")
    assert 'get_phonemizer("espeak", "ipa", keep_punctuation=True)' in code
    assert 'phonemizer.phonemize_to_list(text, "en-US")' in code
    assert "{s: i[0] for s, i in config[\"phoneme_id_map\"].items()}" in code
    assert '"scales": np.array([noise_scale, length_scale, noise_w], dtype=np.float32)' in code
    assert "noise_scale, length_scale, noise_w = 0.667, 1, 0.8" in code
    assert "wav.setframerate(22050)" in code
    assert '"sid"' not in code
    assert '"scriptconv[espeak]"' in code.splitlines()[0]


def test_multi_speaker_example_names_speaker_zero():
    config = {**NATIVE, "num_speakers": 4, "speaker_id_map": {"ana": 0, "rui": 1, "eva": 2, "bia": 3}}
    code = example(config, {"inputs": {"input": 2, "input_lengths": 1, "scales": 1, "sid": 1}})
    assert '"sid": np.array([0], dtype=np.int64),  # speaker 0 of 4 (ana, rui, eva, ...)' in code


def test_batched_scales_input_gets_a_batched_array():
    code = example(NATIVE, {"inputs": {"input": 2, "input_lengths": 1, "scales": 2}})
    assert "np.array([[noise_scale, length_scale, noise_w]], dtype=np.float32)" in code


def test_mms_example_feeds_each_scale():
    config = {**NATIVE, "phoneme_type": "graphemes", "alphabet": "unicode", "keep_punctuation": False,
              "add_blank_char": True, "use_eos_bos": False}
    code = example(config, {"inputs": {"x": 2, "x_length": 1, "noise_scale": 1, "length_scale": 1,
                                       "noise_scale_w": 1}})
    for feed in ('"x": ', '"x_length": ', '"noise_scale": np.array([noise_scale]', '"noise_scale_w": np.array([noise_w]'):
        assert feed in code
    assert 'get_phonemizer("graphemes", "unicode")' in code
    assert '"scriptconv[phonemizers]"' in code.splitlines()[0]


def test_transformers_example_feeds_an_attention_mask():
    code = example({**NATIVE, "engine": "transformers"}, {"inputs": {"input_ids": 2, "attention_mask": 2}})
    assert '"attention_mask": np.ones((1, len(ids)), dtype=np.int64)' in code
    assert "the noise and speed scales are fixed in this graph" in code


def test_prior_split_example_runs_both_graphs():
    code = example({**NATIVE, "engine": None}, {"model": "duration.onnx", "decode": "decode.onnx"})
    assert 'hf_hub_download(repo, "decode.onnx")' in code
    assert "default_rng(0)" in code


def test_glowtts_example_vocodes_its_mel():
    config = {**NATIVE, "engine": "glowtts", "phoneme_type": "gruut"}
    vocoder = {"repo": "Phoonnx/hifigan", "file": "model.onnx", "type": "hifigan", "inputs": {"mel": 3}}
    code = example(config, {"inputs": {"input": 2, "input_lengths": 1, "scales": 1}, "vocoder": vocoder})
    assert "np.array([noise_scale, length_scale], dtype=np.float32)" in code
    assert "mel.min() >= -0.5" in code
    assert 'hf_hub_download("Phoonnx/hifigan", "model.onnx")' in code
    assert 'vocoder.run(None, {"mel": mel})' in code
    assert '"scriptconv[gruut]"' in code.splitlines()[0]


def test_matcha_example_trims_the_mel_and_runs_vocos():
    config = {**NATIVE, "engine": "matcha"}
    vocoder = {"file": "vocoder.onnx", "type": "vocos", "inputs": {"mels": 3}, "config": {}}
    code = example(config, {"inputs": {"x": 2, "x_lengths": 1, "scales": 1}, "outputs": ["mel", "mel_lengths"],
                            "vocoder": vocoder})
    assert "mel, mel_lengths = session.run(None, {" in code
    assert "mel = mel[..., :mel_lengths[0]]" in code
    assert 'hf_hub_download(repo, "vocoder.onnx")' in code
    assert "np.zeros_like(mel)" in code  # Matcha denoises by default
    assert "n_fft, hop = 1024, 256" in code


def test_matcha_export_with_its_own_vocoder_needs_none():
    code = example({**NATIVE, "engine": "matcha"}, {"inputs": {"x": 2, "x_lengths": 1, "scales": 1},
                                                    "outputs": ["waveform", "mel_lengths"]})
    assert "audio, mel_lengths = session.run(None, {" in code
    assert "vocoder.run" not in code


def test_mixertts_example_feeds_its_controls():
    vocoder = {"repo": "Phoonnx/vocos", "file": "model.onnx", "type": "vocos", "inputs": {"mels": 3},
               "config": {"feature_extractor": {"init_args": {"n_fft": 2048, "hop_length": 512}}}}
    code = example({**NATIVE, "engine": "mixertts"},
                   {"inputs": {"token_ids": 2, "pace": 1, "speaker": 1, "pitch_mul": 1}, "vocoder": vocoder})
    assert '"pace": np.array([1.0], dtype=np.float32)' in code
    assert '"speaker": np.array([0], dtype=np.int32)' in code
    assert '"emotion"' not in code
    assert "n_fft, hop = 2048, 512" in code
    assert "np.zeros_like(mel)" not in code


def test_styletts2_example_picks_a_style_row_and_pads():
    code = example({**NATIVE, "engine": "styletts2", "add_blank_char": False, "use_eos_bos": False},
                   {"inputs": {"input_ids": 2, "style": 2, "speed": 1}, "style": {"file": "voices/af.bin"}})
    assert 'hf_hub_download(repo, "voices/af.bin")' in code
    elsewhere = example({**NATIVE, "engine": "styletts2"}, {"inputs": {"input_ids": 2, "style": 2, "speed": 1},
                                                           "style": {"repo": "org/voices", "file": "af.bin"}})
    assert 'hf_hub_download("org/voices", "af.bin")' in elsewhere
    assert "style[min(len(ids), len(style) - 1)]" in code
    assert 'pad = id_map["_"]' in code
    assert '"input_ids": np.array([padded], dtype=np.int64)' in code


def test_shami_example_feeds_language_ids():
    config = {**NATIVE, "engine": "shami", "phoneme_type": "shami", "add_blank_char": False, "use_eos_bos": False}
    code = example(config, {"inputs": {"phoneme_ids": 2, "phoneme_lengths": 1, "language_ids": 2}}, lang="apc")
    assert 'phonemize_with_language_ids_lazy(text, "apc")' in code
    assert '"language_ids": np.array([language_ids], dtype=np.int64)' in code


def test_optispeech_example_feeds_its_factors():
    code = example({**NATIVE, "engine": "optispeech"}, {"inputs": {"x": 2, "x_lengths": 1, "scales": 1}})
    assert '"scales": np.array([1.0, 1.0, 1.0], dtype=np.float32),  # duration, pitch and energy' in code


def test_text_and_language_overrides():
    code = example(NATIVE, {"inputs": {"input": 2, "input_lengths": 1, "scales": 1}}, text='Say "hi".', lang="en-GB")
    assert 'text = "Say \\"hi\\"."' in code
    assert 'phonemize_to_list(text, "en-GB")' in code
    placeholder = example(NATIVE, {"inputs": {"input": 2, "input_lengths": 1, "scales": 1}})
    assert 'text = "..."  # a sentence in en-US' in placeholder


@pytest.mark.parametrize("config, files, reason", [
    ({**NATIVE, "phoneme_id_map": {}}, {"inputs": {"input": 2}}, "no phoneme_id_map"),
    ({**NATIVE, "phonemes": ["a"]}, {"inputs": {"input": 2}}, "mimic3"),
    ({**NATIVE, "phoneme_type": "graphemes", "alphabet": "pinyin"}, {"inputs": {"input": 2}}, "pinyin"),
    (NATIVE, {"inputs": {"input": 2, "d_vector": 2}}, "d_vector"),
    ({**NATIVE, "engine": "glowtts"}, {"inputs": {"input": 2, "input_lengths": 1, "scales": 1},
                                       "vocoder": {"file": "v.onnx", "type": "griffinlim", "inputs": {"mel": 3}}},
     "griffinlim"),
    ({**NATIVE, "engine": "vosk"}, {"inputs": {"text": 2}}, "vosk"),
    ({**NATIVE, "engine": "styletts2"}, {"inputs": {"input_ids": 2, "style": 2, "speed": 1}}, "style file"),
])
def test_uncovered_shapes_are_refused(config, files, reason):
    with pytest.raises(UnsupportedExample, match=reason):
        standalone_example(config, files, repo_id="Phoonnx/test-voice")


def test_graph_io_reads_names_and_ranks():
    inputs = [helper.make_tensor_value_info("input", TensorProto.INT64, [1, None]),
              helper.make_tensor_value_info("input_lengths", TensorProto.INT64, [1]),
              helper.make_tensor_value_info("scales", TensorProto.FLOAT, [1, 3])]
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, None])
    node = helper.make_node("Cast", ["input"], ["output"], to=TensorProto.FLOAT)
    model = helper.make_model(helper.make_graph([node], "voice", inputs, [output]))
    assert graph_io(model.SerializeToString()) == {"inputs": {"input": 2, "input_lengths": 1, "scales": 2},
                                                   "outputs": {"output": 3}}


# --- the text-to-ids path against phoonnx's ------------------------------------

def example_text_ids(config, text, lang=None, overrides=None):
    """The ids the rendered example builds for *text*: its lines from ``text`` to the session, run."""
    code = example(config, VITS_FILES, text=text, lang=lang, **({"overrides": overrides} if overrides else {}))
    return run_lines(code, "text = ", {"config": copy.deepcopy(config)})


def voice_of(config, lang=None, overrides=None):
    return TTSVoice(session=None, config=VoiceConfig.from_dict(copy.deepcopy(config), lang_code=lang,
                                                               **(overrides or {})))


def phoonnx_text_ids(config, text, lang=None, overrides=None):
    """phoonnx's ids for *text*: ``TTSVoice.phonemize``, then ``phonemes_to_ids``."""
    voice = voice_of(config, lang, overrides)
    return [i for phonemes in voice.phonemize(text) for i in voice.phonemes_to_ids(phonemes)]


PUNCTUATED = ["It's twenty-one.", 'He said "yes" (twice) [really].', "Это по-русски, «да»!",
              "O guarda-chuva é «bom».", "Това е по-добре.", "Rock 'n' roll — «fine», (ok)?"]
SYMBOLS = sorted({c for text in PUNCTUATED for c in text.lower() + text} | set("'-«»\"()[],.:;?!— "))


def raw_text_config(phoneme_type, keep, capitals=False, form="NFC"):
    import unicodedata
    symbols = sorted({unicodedata.normalize(form, c) for c in SYMBOLS} | set("".join(
        unicodedata.normalize("NFD", c) for c in SYMBOLS)))
    symbols = [c for c in symbols if capitals or not c.isupper()]
    id_map = {"_": 0, "^": 1, "$": 2, **{c: n for n, c in enumerate(symbols, 3) if c not in "_^$"}}
    return {**NATIVE, "phoneme_type": phoneme_type, "alphabet": "unicode", "lang_code": "en",
            "keep_punctuation": keep, "phoneme_id_map": id_map}


@pytest.mark.parametrize("text", PUNCTUATED)
@pytest.mark.parametrize("phoneme_type, keep, capitals", [
    ("graphemes", False, False), ("graphemes", True, False), ("unicode", False, False), ("unicode", True, False),
    ("unicode", True, True), ("unicode", False, True)])
def test_raw_text_ids_match_phoonnx(phoneme_type, keep, capitals, text):
    # apostrophes, hyphens, quotes, guillemets and brackets go through scriptconv's own rules in phoonnx
    config = raw_text_config(phoneme_type, keep, capitals)
    assert example_text_ids(config, text) == phoonnx_text_ids(config, text)


@pytest.mark.parametrize("text", PUNCTUATED)
def test_piper_text_voice_ids_match_phoonnx(text):
    id_map = {c: [n] for c, n in raw_text_config("unicode", True)["phoneme_id_map"].items()}
    config = {**PIPER, "phoneme_type": "text", "phoneme_id_map": id_map}
    assert example_text_ids(config, text) == phoonnx_text_ids(config, text)


def phonemizer_lang_of(config):
    """The language phoonnx phonemizes *config* in, normalised: the espeak-ng voice keeps its own spelling."""
    loaded = VoiceConfig.from_dict(copy.deepcopy(config))
    return normalize_lang(loaded.phonemizer_lang or loaded.lang_code, default_region=False)


def normalised_tag_of(config):
    """The language tag the card gives text2spoken for *config*."""
    text = card_examples.without_phoonnx(config)
    return text.split('normalize(text, "', 1)[1].split('"', 1)[0]


def test_a_side_car_with_no_lang_code_takes_the_language_as_phoonnx():
    # an espeak model reads with espeak.voice, as Piper does; a text model reads language.code
    for config, expected in (({**PIPER, "lang_code": None, "language": {"code": "pt"}}, "en-US"),
                             ({**PIPER, "lang_code": None, "language": {"code": "pt"}, "espeak": {"voice": "pt"},
                               "phoneme_type": "text"}, "pt")):
        code = example(config, VITS_FILES)
        lang = code.split("phonemize_to_list(text, ", 1)[1].split(")", 1)[0].strip('"')
        assert normalize_lang(lang, default_region=False) == phonemizer_lang_of(config)
        assert normalize_lang(lang, default_region=False) == expected
    for config in ({**PIPER, "lang_code": None},  # no language block: espeak.voice
                   {**raw_text_config("graphemes", False), "lang_code": None}):  # a voice that reads any text
        code = example(config, VITS_FILES)
        lang = code.split("phonemize_to_list(text, ", 1)[1].split(")", 1)[0].strip('"')
        assert normalize_lang(lang, default_region=False) == phonemizer_lang_of(config)


def test_the_card_normalises_in_the_language_of_the_text_as_phoonnx():
    # Piper's language block is the language of the text; its espeak-ng voice only phonemizes
    config = {**PIPER, "lang_code": None, "language": {"code": "pt_PT"}}
    loaded = VoiceConfig.from_dict(copy.deepcopy(config))
    assert (loaded.lang_code, loaded.phonemizer_lang) == ("pt-PT", "en-us")
    assert normalised_tag_of(config) == "pt-PT"


def test_a_native_config_phonemizes_in_its_phonemizer_lang():
    native = {**NATIVE, "lang_code": "es-MX", "phonemizer_lang": "es-419"}
    code = example(native, VITS_FILES)
    assert code.split("phonemize_to_list(text, ", 1)[1].split(")", 1)[0].strip('"') == "es-419"
    assert phonemizer_lang_of(native) == "es-419"
    assert normalised_tag_of(native) == "es-MX"


@pytest.mark.parametrize("config", [
    # a side-car that is not Piper's: phoonnx reads no espeak block
    {**{k: v for k, v in raw_text_config("graphemes", False).items() if k != "phoonnx_version"}, "engine": None,
     "lang_code": None, "espeak": {"voice": "de"}},
    # a phoonnx config reads only lang_code
    {**raw_text_config("graphemes", False), "lang_code": None, "language": {"code": "pt"}},
    {**raw_text_config("graphemes", False), "lang_code": None, "espeak": {"voice": "de"}},
], ids=["side-car-espeak", "phoonnx-language", "phoonnx-espeak"])
def test_a_config_with_no_lang_code_reads_only_the_blocks_phoonnx_reads(config):
    code = example(config, VITS_FILES)
    lang = code.split("phonemize_to_list(text, ", 1)[1].split(")", 1)[0].strip('"')
    assert lang == VoiceConfig.from_dict(copy.deepcopy(config)).lang_code == "und"


def keeps_punctuation(code):
    return "keep_punctuation=True" in code


@pytest.mark.parametrize("config, overrides", [
    ({key: value for key, value in PIPER.items()}, {}),  # upstream Piper: trained with punctuation
    (PIPER, {"alphabet": "ipa"}),  # the index alphabet is no mark of a phoonnx side-car
    (PIPER, {"alphabet": "ipa", "phoneme_type": "espeak"}),
    (PIPER, {"phoneme_type": "gruut"}),  # phoonnx still reads the config with its Piper loader
    ({**PIPER, "alphabet": "ipa"}, {}),  # a phoonnx side-car names its alphabet
    ({**PIPER, "piper_version": "phoonnx-1.0"}, {}),
    ({**PIPER, "alphabet": "ipa", "keep_punctuation": True}, {}),
    ({key: value for key, value in NATIVE.items()}, {}),
    ({**NATIVE, "keep_punctuation": True}, {}),
    ({**{k: v for k, v in NATIVE.items() if k != "phoonnx_version"}, "engine": None,
      "phoneme_id_map": {k: [v] for k, v in NATIVE["phoneme_id_map"].items()}}, {}),  # Piper by shape
    ({**{k: v for k, v in NATIVE.items() if k != "phoonnx_version"}, "engine": None}, {}),  # a flat map: not Piper
    ({**{k: v for k, v in NATIVE.items() if k not in ("phoonnx_version", "alphabet")}, "engine": None,
      "phoneme_id_map": {k: [v] for k, v in NATIVE["phoneme_id_map"].items()}}, {"phoneme_type": "gruut"}),
])
def test_keep_punctuation_is_phoonnx_s(config, overrides):
    code = example(config, VITS_FILES, **({"overrides": overrides} if overrides else {}))
    assert keeps_punctuation(code) == voice_of(config, overrides=overrides).config.keep_punctuation


def test_a_flat_map_config_with_no_version_uses_its_own_token_flags():
    config = {**{k: v for k, v in NATIVE.items() if k != "phoonnx_version"}, "engine": None, "add_blank_char": False,
              "use_eos_bos": False}
    assert rendered_ids(config, PHONEMES) == phoonnx_ids(config, PHONEMES)


def test_piper_folds_symbols_of_several_characters():
    config = {**PIPER, "phoneme_id_map": {**PIPER["phoneme_id_map"], "aɪ": [12], "tʃ": [13]}}
    phonemes = list("baɪ tʃa.")
    assert rendered_ids(config, phonemes) == phoonnx_ids(config, phonemes)
    assert 13 in rendered_ids(config, phonemes)


@pytest.mark.parametrize("fold", [True, False])
def test_word_blank_folds_no_symbol_across_a_space(fold):
    config = {**NATIVE, "add_blank_word": True, "fold_compounds": fold}
    for phonemes in (list("ba ɪt"), list("a ɪ"), list(" aɪ "), list("t ͡ʃ")):
        assert rendered_ids(config, phonemes) == phoonnx_ids(config, phonemes)


# --- the vowel marks phoonnx adds before the phonemizer ------------------------

def stub_diacritize(text, lang, diacritizer_model=None, **_):
    """Stands in for the tashkeel model: the model's name, then a mark after each letter.

    The name goes first, because a word after the final question mark would be a sentence of its own."""
    return (f"{diacritizer_model} " if diacritizer_model else "") + "".join(c + "َ" if c.isalpha() else c for c in text)


ARABIC = {**NATIVE, "phoneme_type": "graphemes", "alphabet": "unicode", "lang_code": "ar", "keep_punctuation": False,
          "phoneme_id_map": {"_": 0, "^": 1, "$": 2, " ": 3, "َ": 4, "ص": 5, "ب": 6, "ا": 7, "ح": 8, "ل": 9,
                             "خ": 10, "ي": 11, "ر": 12, "ك": 13, "ف": 14, "m": 15, "o": 16, "d": 17, "e": 18}}
ARABIC_TEXT = "صباح الخير، كيف حالك؟"


def recording(calls):
    """:func:`stub_diacritize` that also keeps the text, the language and the model of each call."""
    def diacritize(text, lang, diacritizer_model=None, **_):
        calls.append((text.strip(), lang, diacritizer_model))
        return stub_diacritize(text, lang, diacritizer_model)
    return diacritize


def example_diacritizer(monkeypatch, calls=None):
    """Put :func:`stub_diacritize` in place of text2tashkeel's Diacritizer, which the example calls.

    The default model is written as none, as phoonnx passes it to scriptconv. Each call keeps its text and model.
    """
    class Diacritizer:
        def __init__(self, model=card_examples.RAWI, **_):
            self.model = None if model == card_examples.RAWI else model

        def diacritize(self, text):
            if calls is not None:
                calls.append((text.strip(), self.model))
            return stub_diacritize(text, "ar", self.model)

    monkeypatch.setattr(text2tashkeel, "Diacritizer", Diacritizer)


def without_language(calls):
    """phoonnx's diacritize calls as (text, model): text2tashkeel takes no language."""
    return [(text, model) for text, _, model in calls]


@pytest.mark.parametrize("config", [
    ARABIC,
    {**ARABIC, "inference": {"add_diacritics": True}},
    {**ARABIC, "inference": {"add_diacritics": False}},
    {**ARABIC, "inference": {"add_diacritics": None}},
    {**ARABIC, "inference": {"add_diacritics": True, "diacritizer_model": "mode"}},
    {**ARABIC, "lang_code": "apc"},
    {**{k: v for k, v in ARABIC.items() if k != "phoonnx_version"}, "engine": None},
    {**{k: v for k, v in ARABIC.items() if k != "phoonnx_version"}, "engine": None, "inference": {"add_diacritics": None}},
    {**PIPER, "phoneme_type": "text", "lang_code": "ar", "phoneme_id_map": {k: [v] for k, v in ARABIC["phoneme_id_map"].items()}},
    {**PIPER, "phoneme_type": "text", "lang_code": "ar", "inference": {"add_diacritics": False},
     "phoneme_id_map": {k: [v] for k, v in ARABIC["phoneme_id_map"].items()}},
    # phoonnx's Piper loader does not read a diacritizer model
    {**PIPER, "phoneme_type": "text", "lang_code": "ar", "inference": {"diacritizer_model": "mode"},
     "phoneme_id_map": {k: [v] for k, v in ARABIC["phoneme_id_map"].items()}},
], ids=["phoonnx", "on", "off", "null", "model", "apc", "canonical", "canonical-null", "piper", "piper-off",
        "piper-model"])
def test_vowel_marks_are_added_as_phoonnx_adds_them(config, monkeypatch):
    phoonnx_calls, example_calls = [], []
    monkeypatch.setattr(scriptconv.frontend, "diacritize", recording(phoonnx_calls))
    voice = voice_of(config)
    expected = [i for u in voice.prepare(ARABIC_TEXT, NO_SPELLINGS) for i in u.ids]
    example_diacritizer(monkeypatch, example_calls)
    assert example_text_ids(config, ARABIC_TEXT) == expected
    assert example_calls == without_language(phoonnx_calls)  # the same model, once for the one sentence
    code = example(config, VITS_FILES)
    assert ("from text2tashkeel import Diacritizer" in code) == voice.config.add_diacritics
    assert ("tashkeel" in code.splitlines()[0]) == voice.config.add_diacritics


def test_the_vowel_marks_go_to_each_sentence_on_its_own_as_phoonnx_gives_them(monkeypatch):
    text = "صباح الخير. كيف حالك؟ الخير كبير."
    phoonnx_calls, example_calls = [], []
    monkeypatch.setattr(scriptconv.frontend, "diacritize", recording(phoonnx_calls))
    voice = voice_of(ARABIC)
    list(voice.prepare(text, NO_SPELLINGS))
    example_diacritizer(monkeypatch, example_calls)
    example_text_ids(ARABIC, text)
    assert len(phoonnx_calls) == 3  # the Arabic question mark ends a sentence
    assert example_calls == without_language(phoonnx_calls)


def test_a_voice_diacritized_in_another_language_is_refused():
    config = {**NATIVE, "inference": {"add_diacritics": True}}
    assert voice_of(config).config.add_diacritics
    with pytest.raises(UnsupportedExample, match="diacritizes"):
        standalone_example(config, VITS_FILES, repo_id="Phoonnx/test-voice")


def backend_voice(phoneme_type, lang, texts, **config):
    """A voice of the scriptconv backend *phoneme_type*, whose map holds every symbol the backend gives for *texts*."""
    phonemizer = get_phonemizer(phoneme_type, "ipa")
    symbols = sorted({p for text in texts for p in phonemizer.phonemize_to_list(text, lang)} - set("_^$"))
    return {**NATIVE, "phoneme_type": phoneme_type, "lang_code": lang, "keep_punctuation": False,
            "phoneme_id_map": {"_": 0, "^": 1, "$": 2, **{s: n for n, s in enumerate(symbols, 3)}}, **config}


LISBON_TEXT = "Olá, como estás?"
ARBTOK_TEXT = "ذهب الولد إلى المدرسة."
# tugaphone and arbtok own the diacritize stage, so phoonnx's frontend leaves the text to them
OWNERS = {
    "tugaphone": backend_voice("tugaphone", "pt-PT-x-lisbon", [LISBON_TEXT, stub_diacritize(LISBON_TEXT, "pt")],
                               inference={"add_diacritics": True}),
    "arbtok": backend_voice("arbtok", "ar", [ARBTOK_TEXT, stub_diacritize(ARBTOK_TEXT, "ar")]),
}


@pytest.mark.parametrize("config", [
    OWNERS["tugaphone"],
    {**OWNERS["tugaphone"], "inference": {}},
    OWNERS["arbtok"],
    {**OWNERS["arbtok"], "inference": {"add_diacritics": True}},
    {**OWNERS["arbtok"], "inference": {"add_diacritics": False}},
    {**NATIVE, "lang_code": "ar"},  # espeak owns no stage
    ARABIC,
], ids=["tugaphone-on", "tugaphone", "arbtok", "arbtok-on", "arbtok-off", "espeak-ar", "graphemes-ar"])
def test_the_diacritize_step_follows_the_stage_phoonnx_s_frontend_resolves(config):
    voice = voice_of(config)
    resolved = voice.frontend._resolve(voice._stages(SynthesisConfig()))
    code = example(config, VITS_FILES)
    assert ("from text2tashkeel import Diacritizer" in code) == resolved.diacritize
    assert ("tashkeel" in code.splitlines()[0]) == resolved.diacritize


@pytest.mark.parametrize("owner, text", [("tugaphone", LISBON_TEXT), ("arbtok", ARBTOK_TEXT)])
def test_a_backend_that_owns_the_diacritize_stage_gets_the_text_as_phoonnx_gives_it(owner, text, monkeypatch):
    phoonnx_calls, example_calls = [], []
    monkeypatch.setattr(scriptconv.frontend, "diacritize", recording(phoonnx_calls))
    voice = voice_of(OWNERS[owner])
    expected = [i for u in voice.prepare(text, NO_SPELLINGS) for i in u.ids]
    example_diacritizer(monkeypatch, example_calls)
    assert example_text_ids(OWNERS[owner], text) == expected
    assert example_calls == phoonnx_calls == []


# --- the voice index route: phoonnx loads the voice by its id ------------------

HUB = "https://huggingface.co/Phoonnx/test-voice/resolve/main/"


def index_route_ids(config, entry, text, dictionary=None):
    """phoonnx's ids for *text*, for the voice loaded by its id: ``TTSModelInfo.config`` with *config* as config.json,
    and the local file *dictionary* as the file that the entry's ``dictionary_url`` names."""
    info = TTSModelInfo(**{"voice_id": "test/voice", "model_url": HUB + "model.onnx", "config_url": HUB + "config.json",
                           **entry})
    info.download_config = lambda: copy.deepcopy(config)
    info.download_dictionary = lambda: dictionary
    voice = TTSVoice(session=None, config=info.config)
    return [i for u in voice.prepare(text, NO_SPELLINGS) for i in u.ids]


def card_example_ids(config, entry, text):
    """The ids of the example that ``model_card`` renders for the voice of the index *entry*."""
    import model_card
    section = model_card._example("Phoonnx/test-voice", config, entry, "model.onnx", "config.json", text,
                                  lambda name: {"inputs": VITS_FILES["inputs"], "outputs": {"output": 3}})
    assert section.startswith("### Without phoonnx\n")
    block = section[section.rindex("```python\n"):]
    return run_lines(block.removeprefix("```python\n").removesuffix("\n```"), "text = ",
                     {"repo": "Phoonnx/test-voice", "config": copy.deepcopy(config)})


BASQUE = ["Hamaika lagun etorri ziren atzo.", "Euskal Herriko txakurrak zaunka egiten du."]
AHOTTS = {**NATIVE, "phoneme_type": "ahotts", "lang_code": "eu", "keep_punctuation": False,
          "phoneme_id_map": {"_": 0, "^": 1, "$": 2, **{s: n for n, s in enumerate(sorted(
              {p for model in ("classic", "modern") for text in BASQUE
               for p in get_phonemizer("ahotts", "ipa", model=model).phonemize_to_list(text, "eu")} - set("_^$")), 3)}}}
TDT = {**PIPER, "lang_code": None, "espeak": {"voice": "pt"}, "language": {"code": "pt"},
       "phoneme_id_map": {"_": [0], "^": [1], "$": [2], **{s: [n] for n, s in enumerate(sorted(
           set(get_phonemizer("espeak", "ipa").phonemize_to_list("Bom dia, amigo.", "pt")) - set("_^$")), 3)}}}


@pytest.mark.parametrize("config, entry, text", [
    # the hitz Basque voices: the index names the classic AhoTTS engine, and the config names none
    (AHOTTS, {"lang": "eu-ES", "phoneme_type": "ahotts", "alphabet": "ipa", "phonemizer_model": "classic"}, BASQUE[0]),
    (AHOTTS, {"lang": "eu-ES", "phoneme_type": "ahotts", "alphabet": "ipa", "phonemizer_model": "classic"}, BASQUE[1]),
    # the index wins over a config that names another engine
    ({**AHOTTS, "phonemizer_model": "modern"}, {"lang": "eu-ES", "phonemizer_model": "classic"}, BASQUE[0]),
    ({**AHOTTS, "phonemizer_model": "classic"}, {"lang": "eu-ES", "phonemizer_model": "modern"}, BASQUE[0]),
    # tdt-TL_joao: espeak reads the text in the index phonemizer_lang, not in the voice's own language
    (TDT, {"lang": "tdt-TL", "phonemizer_lang": "pt", "phoneme_type": "espeak", "alphabet": "ipa"}, "Bom dia, amigo."),
], ids=["classic-1", "classic-2", "index-over-config", "index-over-config-modern", "phonemizer-lang"])
def test_the_index_entry_reaches_the_example_as_phoonnx_reads_it(config, entry, text):
    expected = index_route_ids(config, entry, text)
    assert expected
    assert card_example_ids(config, entry, text) == expected


VOSK_TEXT = "Мама мыла раму."
# the stress that the voice's dictionary gives each word; the rules alone stress no vowel of this text
VOSK_DICTIONARY = {"мама": "m a1 m a0", "мыла": "m y0 l a1", "раму": "r a1 m u0"}


def vosk_voice():
    """A vosk config, as the vosk voices publish it: no phoneme_type, and a map of the vosk symbols."""
    symbols = set(get_phonemizer("vosk", "vosk").phonemize_to_list(VOSK_TEXT, "ru-RU"))
    symbols |= {p for phonemes in VOSK_DICTIONARY.values() for p in phonemes.split()}
    return {"audio": {"sample_rate": 22050}, "espeak": {"voice": "ru"}, "num_speakers": 1, "speaker_id_map": {},
            "inference": {"noise_scale": 0.667, "length_scale": 1, "noise_w": 0.8},
            "phoneme_id_map": {"_": [0], "^": [1], "$": [2], **{s: [n] for n, s in enumerate(sorted(symbols - set("_^$")), 3)}}}


def test_a_vosk_voice_reads_its_index_dictionary_as_phoonnx_reads_it(tmp_path, monkeypatch):
    dictionary = tmp_path / "dictionary"
    dictionary.write_text("".join(f"{word} {phonemes}\n" for word, phonemes in VOSK_DICTIONARY.items()), encoding="utf-8")
    entry = {"model_url": "https://huggingface.co/Phoonnx/test-voice/resolve/main/vosk__test/model.onnx",
             "dictionary_url": "https://huggingface.co/Phoonnx/test-voice/resolve/main/vosk__test/dictionary",
             "lang": "ru-RU", "phoneme_type": "vosk", "alphabet": "vosk", "engine": "vosk"}
    downloads = []

    def hub_download(repo, file):
        downloads.append((repo, file))
        return str(dictionary)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", hub_download)
    expected = index_route_ids(vosk_voice(), entry, VOSK_TEXT, dictionary)
    assert expected != index_route_ids(vosk_voice(), entry, VOSK_TEXT)  # the dictionary changes the ids
    assert card_example_ids(vosk_voice(), entry, VOSK_TEXT) == expected
    assert downloads == [("Phoonnx/test-voice", "vosk__test/dictionary")]


def test_the_index_phoneme_type_decides_the_diacritize_step(monkeypatch):
    # the index reads an espeak config with arbtok, which restores the vowel marks itself
    monkeypatch.setattr(scriptconv.frontend, "diacritize", stub_diacritize)
    example_diacritizer(monkeypatch)
    config, entry = {**OWNERS["arbtok"], "phoneme_type": "espeak"}, {"lang": "ar", "phoneme_type": "arbtok"}
    assert card_example_ids(config, entry, ARBTOK_TEXT) == index_route_ids(config, entry, ARBTOK_TEXT)


def test_index_overrides_name_each_key_that_phoonnx_writes_over_the_config():
    # TTSModelInfo.config writes these three keys of the entry over the config's own; the language reaches the
    # example as its lang argument, and the vosk dictionary as a file
    expected = {"phoneme_type": "ahotts", "alphabet": "ipa", "phonemizer_model": "classic"}
    entry = {"lang": "eu-ES", "phonemizer_lang": "eu", **expected}
    info = TTSModelInfo(voice_id="test/voice", model_url=HUB + "model.onnx", config_url=HUB + "config.json", **entry)
    config = {**AHOTTS, "phoneme_type": "espeak", "alphabet": "unicode", "phonemizer_model": "modern"}
    info.download_config = lambda: copy.deepcopy(config)
    loaded = info.config
    assert {key: getattr(loaded, key) for key in expected} == expected
    assert card_examples.index_overrides(entry) == expected


def test_a_backend_that_does_not_import_refuses_the_example(monkeypatch):
    # without the backend, the example cannot tell whether the backend owns the diacritize stage
    monkeypatch.setitem(PHONEMIZER_REGISTRY, Phonemizer.TUGAPHONE,
                        ("scriptconv.phonemizers.no_such_module", "TugaphonePhonemizer", "pt-phonemizers"))
    with pytest.raises(UnsupportedExample, match="does not import"):
        standalone_example(OWNERS["tugaphone"], VITS_FILES, repo_id="Phoonnx/test-voice")


# --- Shami: every sentence, as phoonnx reads it ------------------------------------

SHAMI_TEXT = "مرحبا كيفك؟ أنا منيح. شو أخبارك؟"


def shami_voice():
    symbols = {p for text in (SHAMI_TEXT, stub_diacritize(SHAMI_TEXT, "ar"))
               for chunk, _ in get_phonemizer("shami").phonemize_with_language_ids_lazy(text, "ar") for p in chunk}
    return {**NATIVE, "engine": "shami", "phoneme_type": "shami", "lang_code": "ar", "pad": "<pad>", "blank": "<pad>",
            "bos": "<bos>", "eos": "<eos>", "add_blank_char": False, "use_eos_bos": False,
            "blank_at_start": False, "blank_at_end": False, "inference": {},
            "phoneme_id_map": {"<pad>": 0, **{s: n for n, s in enumerate(sorted(symbols), 1)}}}


def test_shami_example_reads_every_sentence_and_its_language_as_phoonnx(monkeypatch):
    module, name, _ = PHONEMIZER_REGISTRY[Phonemizer("shami")]
    backend = getattr(importlib.import_module(module), name)
    read, langs = backend.phonemize_with_language_ids_lazy, []

    def reading(self, text, lang, *args, **kwargs):
        langs.append(lang)
        yield from read(self, text, lang, *args, **kwargs)

    monkeypatch.setattr(backend, "phonemize_with_language_ids_lazy", reading)
    phoonnx_calls, example_calls = [], []
    monkeypatch.setattr(scriptconv.frontend, "diacritize", recording(phoonnx_calls))
    config = shami_voice()
    voice = voice_of(config)
    sentences = list(voice.prepare(SHAMI_TEXT, NO_SPELLINGS))
    assert len(sentences) == 3
    phoonnx_langs, langs[:] = list(langs), []
    example_diacritizer(monkeypatch, example_calls)
    code = example(config, {"inputs": {"phoneme_ids": 2, "phoneme_lengths": 1, "language_ids": 2}}, text=SHAMI_TEXT)
    names = {"config": copy.deepcopy(config)}
    run_lines(code, "text = ", names)
    assert names["ids"] == [i for u in sentences for i in u.ids]
    assert names["language_ids"] == [i for u in sentences for i in u.streams["language_ids"]]
    assert set(langs) == set(phoonnx_langs) == {"ar"}
    assert example_calls == without_language(phoonnx_calls)  # the vowel marks of each sentence, on its own


# --- the feeds of the graphs against phoonnx's adapters ------------------------

class Graph:
    """Stands in for an onnxruntime session: keeps its feeds and gives fixed outputs."""

    def __init__(self, inputs, outputs):
        self.inputs, self.outputs, self.feeds = inputs, outputs, []

    def get_inputs(self):
        return [types.SimpleNamespace(name=name, shape=[1], type="tensor(float)") for name in self.inputs]

    def get_outputs(self):
        return [types.SimpleNamespace(name=name) for name in self.outputs]

    def run(self, names, feeds):
        self.feeds.append({name: np.asarray(value) for name, value in feeds.items()})
        return [make(feeds) for make in self.outputs.values()]


def mel(feeds):
    return np.linspace(-2.0, 1.0, 80 * 12, dtype=np.float32).reshape(1, 80, 12)


def waveform(feeds):
    return np.linspace(-0.5, 0.5, 3000, dtype=np.float32)[None]


def stft(feeds):
    level = 1.0 + float(np.abs(next(iter(feeds.values()))).mean())
    frames = next(iter(feeds.values())).shape[-1]
    return np.full((1, 513, frames), level, np.float32)


def run_example(config, files, graphs, tmp_path, monkeypatch, style=None):
    """Run the whole rendered example with stand-in graphs; return its names."""
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if style is not None:
        style.astype(np.float32).tofile(tmp_path / "style.bin")
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda repo, name: str(tmp_path / Path(name).name)
    ort = types.ModuleType("onnxruntime")
    ort.InferenceSession = lambda path: graphs[Path(path).name]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.chdir(tmp_path)
    names = {"__name__": "__main__"}
    exec(example(config, files, text="bom dia"), names)
    return names


def assert_same_feeds(example_feeds, phoonnx_feeds):
    assert len(example_feeds) == len(phoonnx_feeds)
    for ours, theirs in zip(example_feeds, phoonnx_feeds):
        assert sorted(ours) == sorted(theirs)
        for name in ours:
            assert ours[name].dtype == theirs[name].dtype, name
            assert np.array_equal(ours[name], theirs[name]), name


GRAPHEME_VOICE = {**NATIVE, "phoneme_type": "graphemes", "alphabet": "unicode", "lang_code": "pt", "keep_punctuation": False,
                  "phoneme_id_map": {"_": 0, "^": 1, "$": 2, " ": 3, "b": 4, "o": 5, "m": 6, "d": 7, "i": 8, "a": 9}}


@pytest.mark.parametrize("inference", [{}, {"noise_scale": 0.3}, {"noise_scale": 0.5, "length_scale": 1.2, "noise_w": 0.1}])
def test_vits_scales_are_phoonnx_s(inference, tmp_path, monkeypatch):
    config = {**{k: v for k, v in GRAPHEME_VOICE.items() if k != "inference"}, **({"inference": inference} if inference else {})}
    outputs = {"output": lambda feeds: np.linspace(-0.5, 0.5, 300, dtype=np.float32).reshape(1, 1, -1)}
    graph = Graph(["input", "input_lengths", "scales"], outputs)
    names = run_example(config, VITS_FILES, {"model.onnx": graph}, tmp_path, monkeypatch)
    reference = Graph(["input", "input_lengths", "scales"], outputs)
    voice = TTSVoice(session=reference, config=VoiceConfig.from_dict(copy.deepcopy(config)))
    voice.phoneme_ids_to_audio(names["ids"])
    assert_same_feeds(graph.feeds, reference.feeds)


@pytest.mark.parametrize("rows", [1, 5])
@pytest.mark.parametrize("engine_params", [{}, {"speed": 1.3}])
def test_styletts2_feeds_are_phoonnx_s(rows, engine_params, tmp_path, monkeypatch):
    config = {**GRAPHEME_VOICE, "engine": "styletts2", "add_blank_char": False, "use_eos_bos": False,
              "inference": {"length_scale": 1.4}, **({"engine_params": engine_params} if engine_params else {})}
    style = np.arange(rows * 256, dtype=np.float32).reshape(rows, 256) / 1000
    outputs = {"waveform": lambda feeds: np.linspace(-0.5, 0.5, 300, dtype=np.float32)[None]}
    inputs = ["input_ids", "style", "speed"]
    graph = Graph(inputs, outputs)
    files = {"inputs": dict.fromkeys(inputs, 2), "style": {"file": "style.bin"}}
    names = run_example(config, files, {"model.onnx": graph}, tmp_path, monkeypatch, style)
    reference = Graph(inputs, outputs)
    voice = TTSVoice(session=reference, config=VoiceConfig.from_dict(copy.deepcopy(config)),
                     adapter=StyleTTS2Adapter(style_pack=style))
    voice.phoneme_ids_to_audio(names["ids"])
    assert_same_feeds(graph.feeds, reference.feeds)


def test_optispeech_duration_factor_is_phoonnx_s(tmp_path, monkeypatch):
    config = {**GRAPHEME_VOICE, "engine": "optispeech", "inference": {"length_scale": 1.25}}
    outputs = {"wav": lambda feeds: np.linspace(-0.5, 0.5, 300, dtype=np.float32)[None]}
    graph = Graph(["x", "x_lengths", "scales"], outputs)
    names = run_example(config, {"inputs": {"x": 2, "x_lengths": 1, "scales": 1}}, {"model.onnx": graph},
                        tmp_path, monkeypatch)
    reference = Graph(["x", "x_lengths", "scales"], outputs)
    TTSVoice(session=reference, config=VoiceConfig.from_dict(copy.deepcopy(config))).phoneme_ids_to_audio(names["ids"])
    assert_same_feeds(graph.feeds, reference.feeds)


MATCHA_PARAMS = [({}, {}), ({"length_scale": 0.9}, {}), ({}, {"temperature": 0.4, "length_scale": 1.1}),
                 ({"temperature": 0.2}, {}), ({}, {"denoise": False}), ({}, {"denoise": True})]


@pytest.mark.parametrize("inference, engine_params", MATCHA_PARAMS)
@pytest.mark.parametrize("kind", ["hifigan", "vocos"])
def test_matcha_feeds_are_phoonnx_s(kind, inference, engine_params, tmp_path, monkeypatch):
    config = {**GRAPHEME_VOICE, "engine": "matcha", "inference": inference,
              **({"engine_params": engine_params} if engine_params else {})}
    outputs = {"mel": mel, "mel_lengths": lambda feeds: np.array([10], dtype=np.int64)}
    vocoder_inputs = ["mel", "denoise"] if kind == "hifigan" else ["mels"]
    vocoder_outputs = {"wav": waveform} if kind == "hifigan" else {"mag": stft, "x": stft, "y": stft}
    graph, vocoder = Graph(["x", "x_lengths", "scales"], outputs), Graph(vocoder_inputs, vocoder_outputs)
    files = {"inputs": {"x": 2, "x_lengths": 1, "scales": 1}, "outputs": list(outputs),
             "vocoder": {"file": "vocoder.onnx", "type": kind, "inputs": dict.fromkeys(vocoder_inputs, 3), "config": {}}}
    names = run_example(config, files, {"model.onnx": graph, "vocoder.onnx": vocoder}, tmp_path, monkeypatch)
    reference, reference_vocoder = Graph(["x", "x_lengths", "scales"], outputs), Graph(vocoder_inputs, vocoder_outputs)
    vocoder_class = HiFiGANVocoder if kind == "hifigan" else VocosVocoder
    voice = TTSVoice(session=reference, config=VoiceConfig.from_dict(copy.deepcopy(config)),
                     adapter=MatchaAdapter(vocoder=vocoder_class(session=reference_vocoder)))
    voice.phoneme_ids_to_audio(names["ids"])
    assert_same_feeds(graph.feeds, reference.feeds)
    assert_same_feeds(vocoder.feeds, reference_vocoder.feeds)


def test_onnx_io_sends_the_hub_token_and_waits_out_the_rate_limit(monkeypatch):
    import requests
    inputs = [helper.make_tensor_value_info("input", TensorProto.INT64, [1, None])]
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, None])
    graph = helper.make_model(helper.make_graph([helper.make_node("Cast", ["input"], ["output"], to=TensorProto.FLOAT)],
                                                "voice", inputs, [output])).SerializeToString()
    calls, waits, limited = [], [], [2]

    def get(url, headers, timeout):
        calls.append(headers)
        response = requests.Response()
        response.status_code, response._content = (429, b"") if len(calls) <= limited[0] else (206, graph)
        if len(calls) == 2:
            response.headers["Retry-After"] = "7"
        return response

    monkeypatch.setenv("HF_TOKEN", "hf_card_test")
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(card_examples.time, "sleep", waits.append)
    assert card_examples.onnx_io("Phoonnx/x", "model.onnx") == {"inputs": {"input": 2}, "outputs": {"output": 2}}
    assert [h["authorization"] for h in calls] == ["Bearer hf_card_test"] * 3
    assert {h["Range"] for h in calls} == {"bytes=-1048576"}
    assert waits == [10, 7]  # the Hub's own wait when it names one
    calls.clear()
    limited[0] = 99
    with pytest.raises(requests.HTTPError):
        card_examples.onnx_io("Phoonnx/x", "model.onnx", tries=3)
    assert len(calls) == 3


def test_onnx_io_doubles_its_wait_when_the_hub_names_none(monkeypatch):
    import requests
    waits = []

    def get(url, headers, timeout):
        response = requests.Response()
        response.status_code, response._content = 429, b""
        return response

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(card_examples.time, "sleep", waits.append)
    with pytest.raises(requests.HTTPError):
        card_examples.onnx_io("Phoonnx/x", "model.onnx", tries=4)
    assert waits == [10, 20, 40]


# --- engines of several graphs, from the voice index entry ------------------

INDEX_DIR = Path(__file__).resolve().parent.parent / "phoonnx" / "voice_index"
INDEX_ENTRIES = [entry for path in sorted(INDEX_DIR.glob("*.json")) for entry in json.loads(path.read_text()).values()
                 if isinstance(entry, dict) and entry.get("engine") in
                 {**card_examples.INDEX_TEMPLATES, **card_examples.REFUSED_ENGINES}]


def test_an_engine_with_no_proven_example_is_refused_with_its_reason(monkeypatch):
    monkeypatch.setitem(card_examples.REFUSED_ENGINES, "llasa", "the reason")
    with pytest.raises(UnsupportedExample) as refused:
        standalone_example({"engine": "llasa", "voice_id": "llasa/x", "lang": "en"}, {}, repo_id="Phoonnx/x")
    assert str(refused.value) == "the reason"


def not_run_against_phoonnx(entry):
    """The voices with no example: a refused engine, or a multilingual Chatterbox voice in a language whose
    text phoonnx changes with another tool (Arabic, Hebrew, Japanese, Russian, Chinese)."""
    language = entry["lang"].split("-")[0]
    return entry["engine"] in card_examples.REFUSED_ENGINES or (
        entry["engine"] == "chatterbox" and entry["voice_id"].split("/")[1] in ("multilingual", "lahgtna")
        and (language in ("ar", "he", "ja", "ru", "zh") or card_examples._is_arabic(entry["lang"])))


@pytest.mark.parametrize("entry", INDEX_ENTRIES, ids=[entry["voice_id"] for entry in INDEX_ENTRIES])
def test_every_index_voice_of_these_engines_compiles_or_is_refused(entry):
    files = card_examples.index_files(entry)
    if not_run_against_phoonnx(entry):
        with pytest.raises(UnsupportedExample):
            standalone_example(entry, files, repo_id=files["model"]["repo"])
        return
    block = standalone_example(entry, files, repo_id=files["model"]["repo"])
    assert block.startswith("```python\n# pip install onnxruntime numpy huggingface_hub") and block.endswith("\n```")
    compile(block.removeprefix("```python\n").removesuffix("\n```"), entry["voice_id"], "exec")


def test_index_files_names_each_file_of_the_entry_in_the_repo_that_holds_it():
    hub = "https://huggingface.co/Phoonnx/{}/resolve/main/{}"
    entry = {"model_url": hub.format("old-name", "lm/model.onnx"), "config_url": None,
             "tokenizer_config_url": hub.format("old-name", "lm/tokenizer.json"),
             "aux_model_urls": {"codec_decoder_path": hub.format("codec", "decoder.onnx")}}
    files = card_examples.index_files(entry, lambda url: url.replace("/old-name/", "/new-name/"))
    assert files == {"model": {"repo": "Phoonnx/new-name", "file": "lm/model.onnx"},
                     "tokenizer_config": {"repo": "Phoonnx/new-name", "file": "lm/tokenizer.json"},
                     "codec_decoder_path": {"repo": "Phoonnx/codec", "file": "decoder.onnx"}}


# --- the text above the example: what prepares the text ------------------------------


def test_a_chatterbox_voice_with_an_arabic_dialect_tag_gets_no_example():
    # phoonnx diacritizes the text of every Arabic tag, a dialect tag such as arz-EG too
    for entry in INDEX_ENTRIES:
        if entry["engine"] == "chatterbox" and entry["voice_id"].startswith("chatterbox/lahgtna/"):
            files = {key[:-4]: {"repo": "R", "url": url, "file": url.rsplit("/", 1)[1]}
                     for key, url in entry.items() if key.endswith("_url") and url}
            with pytest.raises(card_examples.UnsupportedExample, match="vowel marks"):
                card_examples._chatterbox(dict(entry), files, repo_id="R", text="مرحبا")

def test_the_rawi_model_is_the_one_phoonnx_diacritizes_with():
    import inspect
    assert card_examples.RAWI == inspect.signature(text2tashkeel.Diacritizer).parameters["model"].default
    assert card_examples.RAWI in text2tashkeel.available_models(bundled_only=True)  # no [hf] extra needed


def test_the_letters_are_each_accented_letter_that_bifonia_writes_on_a_homograph():
    import bifonia
    # each marked form keeps the length of its word: the letters that differ are the accents bifonia adds
    written = {c for (word, _), form in bifonia._DIACRITIZED.items() for c, b in zip(form, word) if c != b}
    assert written == set(card_examples.BIFONIA_LETTERS)


def test_espeak_reads_the_accents_that_bifonia_writes():
    import unicodedata
    import bifonia

    def bare(word):
        return "".join(c for c in unicodedata.normalize("NFD", word) if not unicodedata.combining(c))
    espeak = get_phonemizer("espeak", "ipa")
    for lang in ("pt-PT", "pt-BR"):
        pairs = {(bare(form), form) for form in bifonia._DIACRITIZED.values()}
        read = {pair for pair in pairs
                if espeak.phonemize_to_list(pair[0], lang) != espeak.phonemize_to_list(pair[1], lang)}
        # each homograph has two marked forms; espeak reads the plain word as one of them, so at least one changes
        by_word = {}
        for word, form in pairs:
            by_word.setdefault(word, []).append((word, form) in read)
        assert all(any(changed) for word, changed in by_word.items() if len(changed) > 1), lang
    assert "espeak" in card_examples.READS_PT_MARKS


def test_a_backend_that_owns_normalisation_gets_no_text2spoken_step():
    module, name, _ = PHONEMIZER_REGISTRY[Phonemizer("dizione")]
    assert "normalize" in getattr(importlib.import_module(module), name).owned_stages
    text = card_examples.without_phoonnx({**NATIVE, "phoneme_type": "dizione"})
    assert "normalize(" not in text
    assert "The dizione phonemizer of this voice normalises the text itself" in text


@pytest.mark.parametrize("lang", ["ptu", "en-US", "es-ES"])
def test_only_a_portuguese_voice_gets_bifonia(lang):
    text = card_examples.without_phoonnx({**NATIVE, "lang_code": lang})
    assert "bifonia" not in text
    assert "from text2spoken import normalize" in text


def test_an_engine_with_its_own_tokenizer_gets_text2spoken_and_no_bifonia():
    entry = {"engine": "supertonic", "lang": "pt-PT", "voice_id": "supertonic/x"}
    text = card_examples.without_phoonnx(entry)
    assert '"text2spoken[pt]"' in text and 'normalize(text, "pt-PT")' in text
    assert "bifonia" not in text and "Diacritizer" not in text


def test_a_brazilian_voice_needs_the_letters_of_its_own_marks():
    # pt-BR marks "lingüiça" with no context; a vocabulary without "ü" does not read it
    from bifonia import context_free_marks
    assert any("ü" in marked for marked in context_free_marks("pt-BR").values())
    graphemes = get_phonemizer("graphemes", "unicode")
    letters = set(graphemes.phonemize_to_list("abcdefghijlmnopqrstuvxzáéêíóôãõç", "pt-BR"))
    base = {**NATIVE, "phoneme_type": "graphemes", "alphabet": "unicode", "lang_code": "pt-BR",
            "keep_punctuation": False}
    without = card_examples.without_phoonnx({**base, "phoneme_id_map": {c: n for n, c in enumerate(letters)}})
    letters |= set(graphemes.phonemize_to_list("ü", "pt-BR"))
    with_it = card_examples.without_phoonnx({**base, "phoneme_id_map": {c: n for n, c in enumerate(letters)}})
    assert "bifonia" not in without
    assert 'add_extra_diacritics(text, lang="pt-BR")' in with_it


def test_a_phonemizer_that_owns_the_diacritize_stage_gets_no_bifonia_even_when_it_reads_the_marks(monkeypatch):
    # tugaphone runs bifonia itself; a second run would mark the text twice. tugaphone also owns normalize,
    # which drops the whole preparation, so the backend here owns only diacritize
    monkeypatch.setattr(card_examples, "READS_PT_MARKS", ("espeak", "tugaphone"))
    monkeypatch.setattr(card_examples, "_owned_stages", lambda phoneme_type: frozenset({"diacritize"}))
    text = card_examples.without_phoonnx({**NATIVE, "phoneme_type": "tugaphone", "lang_code": "pt-PT"})
    assert 'text = normalize(text, "pt-PT")' in text
    assert "add_extra_diacritics" not in text
    assert "The tugaphone phonemizer of this voice puts the accents on homographs itself" in text


# --- the text beside the phoonnx usage of a voice with no standalone example -------------

def phoonnx_diacritizes(config):
    voice = voice_of(config)
    return voice.frontend._resolve(voice._stages(SynthesisConfig())).diacritize


@pytest.mark.parametrize("config", [ARABIC, {**ARABIC, "inference": {"add_diacritics": False}}])
def test_with_phoonnx_names_the_vowel_mark_model_of_phoonnx_s_own_step(config):
    text = card_examples.with_phoonnx(config)
    assert text.startswith("### Prepare the text\n")
    assert 'normalize(text, "ar-001")' in text
    assert "Diacritizer(" not in text  # phoonnx diacritizes; the reader does not do it again
    named = "phoonnx then adds the Arabic vowel marks with [text2tashkeel]" in text
    assert named == phoonnx_diacritizes(config)
    assert ("`rawi-ensemble` model" in text) == named


def test_with_phoonnx_names_the_config_s_own_diacritizer_model():
    config = {**ARABIC, "inference": {"add_diacritics": True, "diacritizer_model": "rawi-v2"}}
    assert "its `rawi-v2` model" in card_examples.with_phoonnx(config)


def test_with_phoonnx_gives_bifonia_to_a_portuguese_voice_that_phoonnx_does_not_diacritize():
    config = {**NATIVE, "lang_code": "pt-PT", "inference": {"add_diacritics": False}}
    assert not phoonnx_diacritizes(config)
    text = card_examples.with_phoonnx(config)
    assert 'text = add_extra_diacritics(text, lang="pt-PT")' in text
    assert "sound different from phoonnx" not in text  # the reader runs bifonia before phoonnx


def test_with_phoonnx_gives_no_bifonia_to_a_portuguese_voice_that_phoonnx_diacritizes():
    config = {**NATIVE, "lang_code": "pt-PT", "inference": {"add_diacritics": True}}
    assert phoonnx_diacritizes(config)
    text = card_examples.with_phoonnx(config)
    assert "add_extra_diacritics" not in text
    assert "phoonnx runs its own vowel-mark step on the text of this voice" in text
    with pytest.raises(card_examples.UnsupportedExample):  # the standalone example refuses this voice
        card_examples.without_phoonnx(config)


@pytest.mark.parametrize("lang, named", [("ar", True), ("en", False)])
def test_with_phoonnx_on_chatterbox_names_the_vowel_marks_only_for_arabic(lang, named):
    text = card_examples.with_phoonnx({"engine": "chatterbox", "voice_id": "chatterbox/lahgtna", "lang": lang})
    assert "from text2spoken import normalize" in text
    assert ("`rawi-ensemble` model" in text) == named


def test_with_phoonnx_asks_for_a_language_file_that_text2spoken_lacks():
    # qaa is a private-use tag, so text2spoken never ships a file for it
    text = card_examples.with_phoonnx({**NATIVE, "lang_code": "qaa"})
    assert 'normalize(text, "qaa")' in text
    assert "text2spoken has no language file for `qaa`, so it leaves text in this language as it is written" in text
    assert card_examples.ADDING_A_LANGUAGE in text


def test_with_phoonnx_gives_no_text2spoken_step_to_a_backend_that_owns_normalisation():
    text = card_examples.with_phoonnx({**NATIVE, "phoneme_type": "dizione"})
    assert "normalize(" not in text
    assert "The dizione phonemizer of this voice normalises the text itself" in text


@pytest.mark.parametrize("inference", [{}, {"noise_scale": None, "length_scale": None, "noise_w": None}])
def test_an_unset_scale_in_the_example_is_the_engine_default(inference):
    # VitsAdapter.default_params(): noise 0.667, length 1.0, noise_w 0.8
    code = example({**NATIVE, "inference": inference}, VITS_FILES)
    assert "noise_scale, length_scale, noise_w = 0.667, 1.0, 0.8" in code
