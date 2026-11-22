"""export_piper and export_sherpa write a voice that Piper and sherpa-onnx read with phoonnx's ids.

The reference id functions below follow the source of each target, not phoonnx.compat:
Piper decomposes the espeak-ng phonemes (NFD), merges the config's vowel_clusters longest
first, skips a symbol that the map lacks and reads ``^ _ (p _)* $``
(src/piper/phonemize_espeak.py and src/piper/phoneme_ids.py of OHF-Voice/piper1-gpl v1.8.0).
sherpa-onnx reads ``tokens.txt`` one code point per line and gives the same sequence
(ReadPiperTokens and PiperPhonemesToIdsVits in sherpa-onnx/csrc/offline-tts-frontend.cc, v2.0.2).
"""
import hashlib
import json
import unicodedata

import onnx
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper

from phoonnx.cli import cli
from phoonnx.compat import export_piper, export_sherpa
from phoonnx.errors import ExportNotSupported
from phoonnx.tokenizer import DEFAULT_IPA_PHONEME_ID_MAP
from phoonnx.voice import TTSVoice

PHONEMES = ["ɪç mˈœçtə", "aɪn baʊm, ja!", "ɔɪ eɪ oʊ.", "hˈaɪsə ʃtˈʁaːsə?"]

INPUT_SPECS = {
    "input": (TensorProto.INT64, [1, "n"]),
    "input_lengths": (TensorProto.INT64, [1]),
    "scales": (TensorProto.FLOAT, [3]),
    "sid": (TensorProto.INT64, [1]),
    "x": (TensorProto.INT64, [1, "n"]),
    "x_lengths": (TensorProto.INT64, [1]),
}


def piper_ids(config, phonemes):
    phones = list(unicodedata.normalize("NFD", phonemes))
    clusters = {tuple(cluster) for cluster in config.get("vowel_clusters", [])}
    if clusters:
        longest, merged, i = max(map(len, clusters)), [], 0
        while i < len(phones):
            for n in range(min(longest, len(phones) - i), 1, -1):
                if tuple(phones[i:i + n]) in clusters:
                    merged.append("".join(phones[i:i + n]))
                    i += n
                    break
            else:
                merged.append(phones[i])
                i += 1
        phones = merged
    id_map = config["phoneme_id_map"]
    ids = id_map["^"] + id_map["_"]
    for phone in phones:
        if phone in id_map:
            ids += id_map[phone] + id_map["_"]
    return ids + id_map["$"]


def sherpa_ids(tokens_txt, phonemes):
    token2id = {}
    for line in tokens_txt.splitlines():
        fields = line.split()
        symbol, idx = (" ", fields[0]) if len(fields) == 1 else fields
        assert len(symbol) == 1 and symbol not in token2id, line  # sherpa-onnx exits on either
        token2id[symbol] = int(idx)
    ids = [token2id["^"], token2id["_"]]
    for code_point in unicodedata.normalize("NFD", phonemes):
        if code_point in token2id:
            ids += [token2id[code_point], token2id["_"]]
    return ids + [token2id["$"]]


def native_config(**overrides):
    id_map = {key: ids[0] for key, ids in DEFAULT_IPA_PHONEME_ID_MAP.items()}
    id_map.update({"ç": [16, 140], "aɪ": 161, "aʊ": 162, "ɔɪ": 163, "eɪ": 164, "oʊ": 165})
    config = {
        "phoonnx_version": "1.0.0", "engine": "piper", "phoneme_type": "espeak", "alphabet": "ipa",
        "lang_code": "de", "audio": {"sample_rate": 22050}, "num_symbols": 256, "num_speakers": 1,
        "keep_punctuation": True, "text_casing": "ignore",
        "inference": {"noise_scale": 0.5, "length_scale": 1.0, "noise_w": 0.333, "add_diacritics": False},
        "phoneme_id_map": id_map, "fold_compounds": True,
    }
    config.update(overrides)
    return config


def load_voice(tmp_path, config, inputs=("input", "input_lengths", "scales")):
    node = helper.make_node("Cast", [inputs[0]], ["output"], to=TensorProto.FLOAT)
    graph = helper.make_graph([node], "tiny",
                              [helper.make_tensor_value_info(name, *INPUT_SPECS[name]) for name in inputs],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, "n"])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 8
    source = tmp_path / "source"
    source.mkdir()
    onnx.save(model, str(source / "model.onnx"))
    (source / "model.onnx.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return TTSVoice.load(str(source / "model.onnx"))


def test_piper_export_gives_the_ids_of_a_voice_that_merges_diphthongs(tmp_path):
    voice = load_voice(tmp_path, native_config())
    model = export_piper(voice, tmp_path / "piper")
    config = json.loads((tmp_path / "piper" / "model.onnx.json").read_text(encoding="utf-8"))
    assert config["vowel_clusters"] == [["a", "ɪ"], ["a", "ʊ"], ["e", "ɪ"], ["o", "ʊ"], ["ɔ", "ɪ"]]
    assert config["espeak"] == {"voice": "de"}
    assert model.read_bytes() == (tmp_path / "source" / "model.onnx").read_bytes()
    for phonemes in PHONEMES:
        assert piper_ids(config, phonemes) == voice.config.tokenizer.tokenize(phonemes), phonemes


def test_piper_export_of_a_voice_that_reads_diphthongs_as_two_symbols(tmp_path):
    voice = load_voice(tmp_path, native_config(fold_compounds=False))
    export_piper(voice, tmp_path / "piper")
    config = json.loads((tmp_path / "piper" / "model.onnx.json").read_text(encoding="utf-8"))
    assert "vowel_clusters" not in config
    assert not [key for key in config["phoneme_id_map"] if len(key) > 1]
    for phonemes in PHONEMES:
        assert piper_ids(config, phonemes) == voice.config.tokenizer.tokenize(phonemes), phonemes


def test_piper_export_merges_a_precomposed_symbol_back_into_its_own_id(tmp_path):
    id_map = {key: ids[0] for key, ids in DEFAULT_IPA_PHONEME_ID_MAP.items()}  # "ç": 40, its own id
    voice = load_voice(tmp_path, native_config(phoneme_id_map=id_map))
    export_piper(voice, tmp_path / "piper")
    config = json.loads((tmp_path / "piper" / "model.onnx.json").read_text(encoding="utf-8"))
    assert config["vowel_clusters"] == [["c", "̧"]]
    assert piper_ids(config, "ɪç")[4] == 40
    for phonemes in PHONEMES:
        assert piper_ids(config, phonemes) == voice.config.tokenizer.tokenize(phonemes), phonemes


def test_sherpa_export_gives_the_ids_of_a_voice(tmp_path):
    voice = load_voice(tmp_path, native_config(fold_compounds=False))
    export_sherpa(voice, tmp_path / "sherpa")
    tokens = (tmp_path / "sherpa" / "tokens.txt").read_text(encoding="utf-8")
    for phonemes in PHONEMES:
        assert sherpa_ids(tokens, phonemes) == voice.config.tokenizer.tokenize(phonemes), phonemes


def test_sherpa_export_writes_metadata_into_a_copy_and_leaves_the_source_byte_equal(tmp_path):
    voice = load_voice(tmp_path, native_config(fold_compounds=False))
    source = tmp_path / "source" / "model.onnx"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    model = export_sherpa(voice, tmp_path / "sherpa")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert not onnx.load(str(source)).metadata_props
    props = {prop.key: prop.value for prop in onnx.load(str(model)).metadata_props}
    assert props == {"model_type": "vits", "comment": "piper", "language": voice.config.lang_code,
                     "voice": "de", "version": "1", "has_espeak": "1", "has_g2pw": "0",
                     "n_speakers": "1", "sample_rate": "22050"}


def test_export_into_the_folder_of_the_source_model_is_refused(tmp_path):
    voice = load_voice(tmp_path, native_config(fold_compounds=False))
    source = tmp_path / "source" / "model.onnx"
    before = source.read_bytes()
    with pytest.raises(ValueError, match="source model"):
        export_sherpa(voice, tmp_path / "source")
    assert source.read_bytes() == before


ARABIC = native_config(lang_code="ar", inference={"add_diacritics": False}, fold_compounds=False)

REFUSALS = {
    "o2i phonemizer": (native_config(phoneme_type="orthography2ipa", lang_code="pt-PT"), None,
                       "orthography2ipa", ("piper", "sherpa")),
    "graph inputs": (native_config(), ("x", "x_lengths", "scales"), "x_lengths", ("piper", "sherpa")),
    "sid input on one speaker": (native_config(), ("input", "input_lengths", "scales", "sid"), "sid",
                                 ("piper", "sherpa")),
    "no punctuation": (native_config(keep_punctuation=False), None, "keep_punctuation", ("piper", "sherpa")),
    "text casing": (native_config(text_casing="lower"), None, "text_casing", ("piper", "sherpa")),
    "diacritics": (native_config(inference={"add_diacritics": True}), None, "diacritics", ("piper", "sherpa")),
    "word blank": (native_config(add_blank_word=True, blank_word=" "), None, "add_blank_word",
                   ("piper", "sherpa")),
    "blank token": (native_config(blank="#"), None, "'#'", ("piper", "sherpa")),
    "symbol with two ids": (native_config(phoneme_id_map={**native_config()["phoneme_id_map"], "x": [36, 5]}),
                            None, "'x'", ("piper", "sherpa")),
    "merged diphthong": (native_config(), None, "'aɪ'", ("sherpa",)),
    "precomposed symbol with its own id": (
        native_config(phoneme_id_map={key: ids[0] for key, ids in DEFAULT_IPA_PHONEME_ID_MAP.items()},
                      fold_compounds=False), None, "'ç'", ("sherpa",)),
    "arabic": (ARABIC, None, "Arabic", ("piper",)),
}


@pytest.mark.parametrize("target", ["piper", "sherpa"])
@pytest.mark.parametrize("reason", list(REFUSALS))
def test_a_voice_the_target_cannot_read_is_refused(tmp_path, reason, target):
    config, inputs, cause, targets = REFUSALS[reason]
    voice = load_voice(tmp_path, config, *([inputs] if inputs else []))
    export = export_piper if target == "piper" else export_sherpa
    if target not in targets:
        export(voice, tmp_path / target)
        return
    with pytest.raises(ExportNotSupported, match=cause):
        export(voice, tmp_path / target)
    assert not (tmp_path / target).exists()


def test_cli_exports_a_voice_and_refuses_one_it_cannot(tmp_path):
    voice = load_voice(tmp_path, native_config())
    runner = CliRunner()
    result = runner.invoke(cli, ["export", voice.model_path, str(tmp_path / "piper"), "--to", "piper"])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [f"Wrote {tmp_path / 'piper' / 'model.onnx'}",
                                          f"Wrote {tmp_path / 'piper' / 'model.onnx.json'}"]
    result = runner.invoke(cli, ["export", voice.model_path, str(tmp_path / "sherpa"), "--to", "sherpa"])
    assert result.exit_code == 1
    assert "sherpa-onnx reads one code point per symbol" in result.output
