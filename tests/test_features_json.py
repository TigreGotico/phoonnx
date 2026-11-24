"""features.json generator (phoonnx_train/features_json.py).

Expected vectors come from the feature sources themselves, turned into
1/-1/0 here, and the map hash is computed here by the recipe of the
``phoonnx-input-features/1`` format.
"""
import hashlib
import json
import re
from importlib.metadata import version

import orthography2ipa.feats as o2i_feats
import phonematcher
import pytest
from click.testing import CliRunner

from phoonnx_train.features_json import cli

TINY_MAP = {"_": [0], "a": [1], "ɐ̃": [2], "ˈ": [3]}
# the seven prosody features of the phoonnx-input-features/1 format, in order
PROSODY_NAMES = ["pros_word_boundary", "pros_pause_minor", "pros_pause_major", "pros_question",
                 "pros_exclamation", "pros_stress_primary", "pros_stress_secondary"]
TONE_NAMES = ["tone_is", "tone_start", "tone_mid", "tone_end"]
NO_PROSODY = [0] * 7
NO_TONE = [0] * 4


def ternary(values):
    out = []
    for v in values:
        if v is True:
            out.append(1)
        elif v is False:
            out.append(-1)
        else:
            assert v is None
            out.append(0)
    return out


def run(tmp_path, phoneme_id_map, source, *args, wrap=True):
    config = tmp_path / "config.json"
    body = {"phoneme_id_map": phoneme_id_map} if wrap else phoneme_id_map
    config.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "features.json"
    result = CliRunner().invoke(cli, [str(config), "--source", source, "--out", str(out), *args])
    table = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    return result, table


def test_phonematcher_tiny_map(tmp_path):
    result, table = run(tmp_path, TINY_MAP, "phonematcher")
    assert result.exit_code == 0, result.output
    assert table["format"] == "phoonnx-input-features/1"
    assert table["source"] == {"name": "phonematcher", "version": phonematcher.__version__,
                               "tone": {"name": "phonematcher", "version": phonematcher.__version__}}
    assert table["feature_names"] == list(phonematcher.FEATURES) + PROSODY_NAMES + TONE_NAMES
    assert set(table["vectors"]) == {"0", "1", "2", "3"}
    assert table["vectors"]["0"] == {"symbol": "_", "vector": None}
    assert table["vectors"]["1"] == {"symbol": "a",
                                     "vector": ternary(phonematcher.vectorize_phones("a")) + NO_PROSODY + NO_TONE}
    assert table["vectors"]["2"]["vector"] == ternary(phonematcher.vectorize_phones("ɐ̃")) + NO_PROSODY + NO_TONE
    assert "ids in: 4" in result.output and "ids out: 4" in result.output
    assert "null ids: 1" in result.output and "0 '_' pad" in result.output


def prosody_of(table, key):
    vector = table["vectors"][key]["vector"]
    names = table["feature_names"]
    assert len(vector) == len(names)
    phone_part, prosody_part, tone_part = vector[:-11], vector[-11:-4], vector[-4:]
    assert phone_part == [0] * (len(names) - 11)
    assert tone_part == NO_TONE
    return {name for name, v in zip(PROSODY_NAMES, prosody_part) if v == 1}, set(prosody_part)


def test_stress_mark_is_stress_primary(tmp_path):
    _, table = run(tmp_path, TINY_MAP, "phonematcher")
    assert prosody_of(table, "3") == ({"pros_stress_primary"}, {0, 1})


def test_question_mark_is_pause_major_and_question(tmp_path):
    _, table = run(tmp_path, {**TINY_MAP, "?": [4], "¿": [5], " ": [6], ",": [7]}, "phonematcher")
    assert prosody_of(table, "4") == ({"pros_pause_major", "pros_question"}, {0, 1})
    assert prosody_of(table, "5") == ({"pros_question"}, {0, 1})
    assert prosody_of(table, "6") == ({"pros_word_boundary"}, {0, 1})
    assert prosody_of(table, "7") == ({"pros_pause_minor"}, {0, 1})


@pytest.mark.parametrize("source", ["phonematcher", "orthography2ipa"])
def test_phone_is_zero_on_prosody(tmp_path, source):
    _, table = run(tmp_path, TINY_MAP, source)
    for key in ("1", "2"):
        assert table["vectors"][key]["vector"][-11:] == NO_PROSODY + NO_TONE


def test_map_sha256_recipe(tmp_path):
    _, table = run(tmp_path, TINY_MAP, "phonematcher")
    serialised = json.dumps(TINY_MAP, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert table["phoneme_id_map_sha256"] == hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def test_map_sha256_sorts_keys(tmp_path):
    unsorted = {"ˈ": [3], "a": [1], "_": [0], "ɐ̃": [2]}
    _, table = run(tmp_path, unsorted, "phonematcher")
    serialised = '{"_":[0],"a":[1],"ɐ̃":[2],"ˈ":[3]}'
    assert table["phoneme_id_map_sha256"] == hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def test_bare_map_with_int_ids(tmp_path):
    flat = {k: v[0] for k, v in TINY_MAP.items()}
    result, table = run(tmp_path, flat, "phonematcher", wrap=False)
    assert result.exit_code == 0, result.output
    assert [table["vectors"][str(i)]["symbol"] for i in range(4)] == ["_", "a", "ɐ̃", "ˈ"]


def test_unreadable_symbol_refused(tmp_path):
    result, table = run(tmp_path, {**TINY_MAP, "̃": [4], "ꭥ": [5]}, "phonematcher")
    assert result.exit_code != 0
    assert table is None
    assert "2 symbols look like phones" in result.output
    assert "id 4 '̃' (U+0303)" in result.output
    assert "id 5 'ꭥ' (U+AB65)" in result.output


def test_non_phone_ruling(tmp_path):
    result, table = run(tmp_path, {**TINY_MAP, "̃": [4]}, "phonematcher", "--non-phone", "̃")
    assert result.exit_code == 0, result.output
    assert table["vectors"]["4"] == {"symbol": "̃", "vector": None}


@pytest.mark.parametrize("symbol", ["7", "|", "·", "-", "(", '"'])
def test_allowlisted_non_phones_are_null(tmp_path, symbol):
    result, table = run(tmp_path, {**TINY_MAP, symbol: [9]}, "phonematcher")
    assert result.exit_code == 0, result.output
    assert table["vectors"]["9"] == {"symbol": symbol, "vector": None}


def test_orthography2ipa_source(tmp_path):
    result, table = run(tmp_path, TINY_MAP, "orthography2ipa")
    assert result.exit_code == 0, result.output
    assert table["source"] == {"name": "orthography2ipa", "version": version("orthography2ipa"),
                               "tone": {"name": "phonematcher", "version": phonematcher.__version__}}
    assert len(table["feature_names"]) == o2i_feats.NUM_FEATURES + 11
    assert table["feature_names"][22] == "nasal_vowel"
    assert table["feature_names"][-11:] == PROSODY_NAMES + TONE_NAMES
    assert table["vectors"]["1"]["vector"] == ternary(o2i_feats.vectorize_phones("a")) + NO_PROSODY + NO_TONE
    assert table["vectors"]["2"]["vector"] == ternary(o2i_feats.vectorize_phones("ɐ̃")) + NO_PROSODY + NO_TONE
    assert table["vectors"]["0"]["vector"] is None
    assert table["vectors"]["3"]["vector"] == [0] * o2i_feats.NUM_FEATURES + [0, 0, 0, 0, 0, 1, 0] + NO_TONE
    assert "ids out: 4" in result.output


WB, MINOR, MAJOR, Q, EXCL = ("pros_word_boundary", "pros_pause_minor", "pros_pause_major",
                             "pros_question", "pros_exclamation")
# the example column of the prosody table, rows read off by hand
SCRIPT_PUNCTUATION = [
    ("　", {WB}),
    (",", {MINOR}), (";", {MINOR}), (":", {MINOR}), ("،", {MINOR}), ("؛", {MINOR}),
    ("、", {MINOR}), ("，", {MINOR}), ("·", {MINOR}), ("፣", {MINOR}),
    ("፤", {MINOR}),
    (".", {MAJOR}), ("…", {MAJOR}), ("。", {MAJOR}), ("।", {MAJOR}),
    ("॥", {MAJOR}), ("։", {MAJOR}), ("።", {MAJOR}),
    ("?", {MAJOR, Q}), ("؟", {MAJOR, Q}), ("？", {MAJOR, Q}), (";", {MAJOR, Q}),
    ("՞", {MAJOR, Q}), ("፧", {MAJOR, Q}), ("¿", {Q}),
    ("!", {MAJOR, EXCL}), ("！", {MAJOR, EXCL}), ("՜", {MAJOR, EXCL}), ("¡", {EXCL}),
    ("⁈", {MAJOR, Q, EXCL}), ("⁉", {MAJOR, Q, EXCL}), ("‽", {MAJOR, Q, EXCL}),
    ("\U00016e9a", {MAJOR, EXCL}), ("\u1361", {WB}), ("‖", {MAJOR}),
]


def test_punctuation_of_every_script(tmp_path):
    extra = {symbol: [100 + n] for n, (symbol, _) in enumerate(SCRIPT_PUNCTUATION)}
    result, table = run(tmp_path, {**TINY_MAP, **extra}, "phonematcher")
    assert result.exit_code == 0, result.output
    for n, (symbol, rows) in enumerate(SCRIPT_PUNCTUATION):
        assert prosody_of(table, str(100 + n)) == (rows, {0, 1}), symbol
    assert "';' GREEK QUESTION MARK: pros_pause_major pros_question" in result.output


def test_punctuation_with_no_rule_is_listed_null(tmp_path):
    result, table = run(tmp_path, {**TINY_MAP, "§": [4]}, "phonematcher")
    assert result.exit_code == 0, result.output
    assert table["vectors"]["4"]["vector"] is None
    assert "4 '§' punctuation SECTION SIGN, no prosody rule" in result.output


def test_middle_dot_is_null_by_default(tmp_path):
    result, table = run(tmp_path, {**TINY_MAP, "·": [4]}, "phonematcher")
    assert result.exit_code == 0, result.output
    assert table["vectors"]["4"] == {"symbol": "·", "vector": None}


def test_pause_flag_rules_a_minor_pause(tmp_path):
    result, table = run(tmp_path, {**TINY_MAP, "·": [4]}, "phonematcher", "--pause", "·")
    assert result.exit_code == 0, result.output
    assert prosody_of(table, "4") == ({MINOR}, {0, 1})


# Chao level 1-5 is (level - 3) / 2; a two-point contour has the mean of its levels in the middle
TONES = [
    ("˥", [1, 1, 1, 1]),            # ˥ level 5
    ("˧˥", [1, 0, 0.5, 1]),    # ˧˥ 3 -> 5
    ("˧˦˧", [1, 0, 0.5, 0]),  # ˧˦˧ 3, 4, 3
    ("⁵¹", [1, 1, 0, -1]),     # ⁵¹ 5 -> 1
]


@pytest.mark.parametrize("source", ["phonematcher", "orthography2ipa"])
@pytest.mark.parametrize("symbol,tone", TONES)
def test_tone_block(tmp_path, source, symbol, tone):
    result, table = run(tmp_path, {**TINY_MAP, symbol: [4]}, source)
    assert result.exit_code == 0, result.output
    assert table["feature_names"][-4:] == TONE_NAMES
    vector = table["vectors"]["4"]["vector"]
    assert vector[-4:] == tone
    assert vector[:-4] == [0] * (len(vector) - 4)


def test_phone_is_zero_on_tone_block(tmp_path):
    _, table = run(tmp_path, {**TINY_MAP, "˥": [4]}, "phonematcher")
    assert table["vectors"]["1"]["vector"][-4:] == NO_TONE
    assert table["vectors"]["3"]["vector"][-4:] == NO_TONE


def o2i_docstring_names():
    """Feature names in index order from orthography2ipa feats.py: its module
    docstring lists 0-20, and the comment above the table names 21 and 22."""
    names = dict(re.findall(r"^(\d+)\s+(\w+)\s+—", o2i_feats.__doc__, re.M))
    with open(o2i_feats.__file__, encoding="utf-8") as f:
        names.update(re.findall(r"# Feature index (\d+): (\w+)", f.read()))
    return [names[str(i)] for i in range(o2i_feats.NUM_FEATURES)]


def test_orthography2ipa_feature_names_in_feats_order(tmp_path):
    _, table = run(tmp_path, TINY_MAP, "orthography2ipa")
    names = o2i_docstring_names()
    assert names[0] == "syllabic" and names[21] == "click" and names[22] == "nasal_vowel"
    assert table["feature_names"][:-11] == names


@pytest.mark.parametrize("source", ["phonematcher", "orthography2ipa"])
def test_tone_points_are_exact_json_numbers(tmp_path, source):
    _, table = run(tmp_path, {**TINY_MAP, "˥": [4], "˦": [5], "˧˦": [6]}, source)
    raw = (tmp_path / "features.json").read_text(encoding="utf-8")
    vectors = json.dumps(json.loads(raw)["vectors"])
    assert "1.0" not in vectors and "0.0" not in vectors
    points = {k: table["vectors"][k]["vector"][-4:] for k in ("4", "5", "6")}
    assert points == {"4": [1, 1, 1, 1], "5": [1, 0.5, 0.5, 0.5], "6": [1, 0, 0.25, 0.5]}
    assert [type(v) for v in points["4"]] == [int] * 4
    assert [type(v) for v in points["5"]] == [int, float, float, float]
    assert [type(v) for v in points["6"]] == [int, int, float, float]


def test_bar_is_a_minor_pause_when_ruled(tmp_path):
    result, table = run(tmp_path, {**TINY_MAP, "|": [4]}, "phonematcher", "--pause", "|")
    assert result.exit_code == 0, result.output
    assert prosody_of(table, "4") == ({MINOR}, {0, 1})


@pytest.mark.parametrize("click", ["ǀ", "ǁ"])
def test_click_letters_are_phones(tmp_path, click):
    result, table = run(tmp_path, {**TINY_MAP, click: [4]}, "phonematcher")
    assert result.exit_code == 0, result.output
    vector = table["vectors"]["4"]["vector"]
    assert vector == ternary(phonematcher.vectorize_phones(click)) + NO_PROSODY + NO_TONE
    # syllabic -, sonorant -, consonantal +
    assert vector[:3] == [-1, -1, 1]


def test_full_stop_is_major_pause_unless_ruled_non_phone(tmp_path):
    _, table = run(tmp_path, {**TINY_MAP, ".": [4]}, "phonematcher")
    assert prosody_of(table, "4") == ({MAJOR}, {0, 1})
    result, table = run(tmp_path, {**TINY_MAP, ".": [4]}, "phonematcher", "--non-phone", ".")
    assert result.exit_code == 0, result.output
    assert table["vectors"]["4"] == {"symbol": ".", "vector": None}


# a phone with IPA tone diacritics: (symbol, base phone, tone block); the marks
# read as the IPA chart's levels, acute 4, grave 2, macron 3, double acute 5,
# double grave 1, circumflex 5 -> 1, caron 1 -> 5
TONE_ACCENTED = [
    ("á", "a", [1, 0.5, 0.5, 0.5]),
    ("à", "a", [1, -0.5, -0.5, -0.5]),
    ("â", "a", [1, 1, 0, -1]),
    ("ǎ", "a", [1, -1, 0, 1]),
    ("ā", "a", [1, 0, 0, 0]),
    ("a\u030b", "a", [1, 1, 1, 1]),
    ("ȁ", "a", [1, -1, -1, -1]),
    ("ɐ̃\u0301", "ɐ̃", [1, 0.5, 0.5, 0.5]),
]


@pytest.mark.parametrize("source", ["phonematcher", "orthography2ipa"])
@pytest.mark.parametrize("symbol,base,tone", TONE_ACCENTED)
def test_tone_accented_phone(tmp_path, source, symbol, base, tone):
    vectorize = phonematcher.vectorize_phones if source == "phonematcher" else o2i_feats.vectorize_phones
    result, table = run(tmp_path, {**TINY_MAP, symbol: [4]}, source)
    assert result.exit_code == 0, result.output
    vector = table["vectors"]["4"]["vector"]
    assert vector[:-11] == ternary(vectorize(base))
    assert vector[-11:-4] == NO_PROSODY
    assert vector[-4:] == tone


@pytest.mark.parametrize("args", [("--pause", ".."), ("--pause", ""), ("--non-phone", "")])
def test_bad_flag_value_writes_nothing(tmp_path, args):
    result, table = run(tmp_path, {**TINY_MAP, "..": [4]}, "phonematcher", *args)
    assert result.exit_code == 2, result.output
    assert table is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.json"]
    assert "ids in" not in result.output


@pytest.mark.parametrize("tied", ["d͡ʒ", "t͡ʃ", "k\u035cp"])
def test_orthography2ipa_refuses_tie_bars_with_advice(tmp_path, tied):
    result, table = run(tmp_path, {**TINY_MAP, tied: [4]}, "orthography2ipa")
    assert result.exit_code != 0
    assert table is None
    assert ("orthography2ipa feats.py does not read tie bars; a tied affricate or "
            "double articulation reads with --source phonematcher") in result.output.replace("\n", " ")
    (tmp_path / "pm").mkdir()
    result, table = run(tmp_path / "pm", {**TINY_MAP, tied: [4]}, "phonematcher")
    assert result.exit_code == 0, result.output
    assert table["vectors"]["4"]["vector"] == ternary(phonematcher.vectorize_phones(tied)) + NO_PROSODY + NO_TONE


def test_merged_segments_glottalised_and_diphthong(tmp_path):
    result, table = run(tmp_path, {**TINY_MAP, "uˀ": [4]}, "phonematcher")
    assert result.exit_code == 0, result.output
    vector = table["vectors"]["4"]["vector"]
    assert vector == ternary(phonematcher.vectorize_phones("uˀ")) + NO_PROSODY + NO_TONE
    # syllabic +, sonorant +, consonantal -
    assert vector[:3] == [1, 1, -1]
    (tmp_path / "diphthong").mkdir()
    result, table = run(tmp_path / "diphthong", {**TINY_MAP, "a͡i": [4]}, "phonematcher")
    assert result.exit_code != 0
    assert table is None
    assert "id 4 'a͡i' (U+0061 U+0361 U+0069)" in result.output
    assert "a diphthong is two segments" in result.output


def test_orthography2ipa_other_refusal_keeps_its_message(tmp_path):
    result, table = run(tmp_path, {**TINY_MAP, "kʰ": [4]}, "orthography2ipa")
    assert result.exit_code != 0
    assert table is None
    assert "Unrecognized phone 'ʰ'" in result.output
    assert "tie bars" not in result.output


@pytest.mark.parametrize("symbol", ["ś", "ć", "ʃ\u0301", "t\u0302", "s\u0300", "ḿ", "ń", "ǹ"])
def test_tone_mark_on_a_non_syllabic_consonant_is_refused(tmp_path, symbol):
    result, table = run(tmp_path, {**TINY_MAP, symbol: [4]}, "phonematcher")
    assert result.exit_code != 0
    assert table is None
    assert f"id 4 {symbol!r}" in result.output


def test_tone_mark_on_a_syllabic_consonant(tmp_path):
    result, table = run(tmp_path, {**TINY_MAP, "m\u0329\u0301": [4]}, "phonematcher")
    assert result.exit_code == 0, result.output
    vector = table["vectors"]["4"]["vector"]
    assert vector[:-11] == ternary(phonematcher.vectorize_phones("m\u0329"))
    # syllabic +, sonorant +, consonantal +
    assert vector[:3] == [1, 1, 1]
    assert vector[-11:] == NO_PROSODY + [1, 0.5, 0.5, 0.5]


def test_bar_reason_says_null_by_default(tmp_path):
    result, table = run(tmp_path, {**TINY_MAP, "|": [4]}, "phonematcher")
    assert result.exit_code == 0, result.output
    assert table["vectors"]["4"]["vector"] is None
    assert "4 '|' null by default; rule it a pause with --pause |" in result.output


def source_with_tone(monkeypatch, tone):
    """A source shaped like phonematcher 3: its features end with the four tone
    features, tone_is a bool and the three pitches floats."""
    from phoonnx_train import features_json
    names = tuple(phonematcher.FEATURES) + tuple(TONE_NAMES)
    monkeypatch.setattr(features_json, "load_source", lambda source: (
        "3.0.0", names, lambda phone: list(phonematcher.vectorize_phones(phone)) + list(tone)))
    return len(phonematcher.FEATURES)


def test_float_features_of_the_source_pass_through(tmp_path, monkeypatch):
    n = source_with_tone(monkeypatch, [True, 1.0, 0.0, -1.0])
    result, table = run(tmp_path, TINY_MAP, "phonematcher")
    assert result.exit_code == 0, result.output
    vector = table["vectors"]["1"]["vector"]
    assert vector[:n] == ternary(phonematcher.vectorize_phones("a"))
    assert vector[n:n + 4] == [1, 1.0, 0.0, -1.0]
    assert vector[n + 4:] == NO_PROSODY


def test_a_source_with_tone_features_gets_no_second_tone_block(tmp_path, monkeypatch):
    n = source_with_tone(monkeypatch, [False, 0.0, 0.0, 0.0])
    _, table = run(tmp_path, {**TINY_MAP, "˧˥": [4], "á": [5]}, "phonematcher")
    names = table["feature_names"]
    assert names == list(phonematcher.FEATURES) + TONE_NAMES + PROSODY_NAMES
    assert all(len(v["vector"]) == len(names) for v in table["vectors"].values() if v["vector"])
    assert table["vectors"]["4"]["vector"] == [0] * n + [1, 0, 0.5, 1] + NO_PROSODY
    assert table["vectors"]["1"]["vector"][n:n + 4] == [-1, 0.0, 0.0, 0.0]
    assert table["vectors"]["5"]["vector"][n:n + 4] == [1, 0.5, 0.5, 0.5]
    assert table["vectors"]["3"]["vector"] == [0] * (n + 4) + [0, 0, 0, 0, 0, 1, 0]
