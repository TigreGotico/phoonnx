import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from split_mirrors import SplitError, card_licence, plan_mirror, target_name  # noqa: E402

SOURCE = "Phoonnx/phoonnx-vits"
README = """---
license: other
---
# phoonnx-vits

| folder | licence |
|---|---|
| `en_US-amy-medium` | cc-by-4.0 |
| `en_US-amy-medium-v2` | mit |
"""


def url(path: str, repo: str = SOURCE) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{path}"


FILES = {"README.md": 900, "en_US-amy-medium/model.onnx": 60, "en_US-amy-medium/model.json": 4,
         "en_US-amy-medium-v2/model.onnx": 61, "en_US-amy-medium-v2/model.json": 5,
         "speakers/amy.wav": 300, "speakers/other.wav": 200}


@pytest.mark.parametrize("folder, expected", [
    ("facebook__mms-tts-agr-Awajún", "Phoonnx/vits-facebook_mms-tts-agr-Awajun"),
    ("SadeghK__persian-tts__epoch=5261-step=2455712", "Phoonnx/vits-SadeghK_persian-tts_epoch_5261-step_2455712"),
    ("a b  c", "Phoonnx/vits-a_b_c"),
])
def test_a_folder_name_becomes_a_valid_repo_name(folder, expected):
    assert target_name(SOURCE, folder) == expected


@pytest.mark.parametrize("folder", ["a--b", "a..b", "x" * 92])
def test_a_name_the_hub_refuses_is_refused(folder):
    with pytest.raises(SplitError, match="not a valid repo name"):
        target_name(SOURCE, folder)


def test_each_model_folder_becomes_one_repo_with_its_files():
    plan = plan_mirror(SOURCE, FILES, README, {})
    assert [(r.target, sorted(r.files), r.size) for r in plan.repos] == [
        ("Phoonnx/vits-en_US-amy-medium", ["en_US-amy-medium/model.json", "en_US-amy-medium/model.onnx"], 64),
        ("Phoonnx/vits-en_US-amy-medium-v2", ["en_US-amy-medium-v2/model.json", "en_US-amy-medium-v2/model.onnx"], 66),
    ]
    assert plan.not_models == ["speakers"]
    assert not plan.stays


def test_a_file_the_index_references_outside_the_folder_is_copied_in():
    index = {"vits/amy": {"model_url": url("en_US-amy-medium/model.onnx"),
                          "aux_model_urls": {"speaker": url("speakers/amy.wav")}}}
    amy = plan_mirror(SOURCE, FILES, README, index).repos[0]
    assert amy.voice_ids == ["vits/amy"]
    assert amy.shared == ["speakers/amy.wav"]
    assert amy.files["speakers/amy.wav"] == 300
    assert amy.size == 364


def test_the_licence_lines_name_the_folder_and_not_a_longer_name():
    amy, amy2 = plan_mirror(SOURCE, FILES, README, {}).repos
    assert amy.licence_lines == ["| `en_US-amy-medium` | cc-by-4.0 |"]
    assert amy2.licence_lines == ["| `en_US-amy-medium-v2` | mit |"]
    assert card_licence(README) == "other"


def test_a_repo_with_one_model_folder_stays():
    files = {"README.md": 1, "voice/model.onnx": 2, "samples/a.wav": 3}
    plan = plan_mirror(SOURCE, files, README, {})
    assert plan.stays and plan.repos == []


def test_a_voice_of_another_repo_must_find_its_file_in_a_new_repo():
    index = {"glow/x": {"model_url": url("x/model.onnx", "Phoonnx/phoonnx-glowtts"),
                        "vocoder_url": url("speakers/other.wav")}}
    assert plan_mirror(SOURCE, FILES, README, index).orphan_references == ["speakers/other.wav"]
    index["glow/x"]["vocoder_url"] = url("en_US-amy-medium/model.onnx")
    assert plan_mirror(SOURCE, FILES, README, index).orphan_references == []


def test_an_index_reference_to_a_missing_file_is_refused():
    index = {"vits/amy": {"model_url": url("en_US-amy-medium/model.onnx"), "config_url": url("gone.json")}}
    with pytest.raises(SplitError, match="does not hold"):
        plan_mirror(SOURCE, FILES, README, index)


def test_two_folders_that_fold_to_one_name_are_refused():
    files = {"Awajún/model.onnx": 1, "Awajun/model.onnx": 1}
    with pytest.raises(SplitError, match="fold to the same repo"):
        plan_mirror(SOURCE, files, README, {})


@pytest.mark.parametrize("folder", ["voice-", "v1.", "x.git", "ب"])
def test_a_name_the_hub_validator_rejects_is_refused(folder):
    with pytest.raises(SplitError, match="not a valid repo name"):
        target_name(SOURCE, folder)


@pytest.mark.parametrize("readme", ["# no front matter\n\n`en_US-amy-medium` cc-by-4.0\n",
                                    "---\npipeline_tag: text-to-speech\n---\n# x\n"])
def test_a_mirror_without_a_licence_gets_no_repos(readme):
    plan = plan_mirror(SOURCE, FILES, readme, {})
    assert plan.licence_gap and plan.repos == []


def test_a_card_with_windows_line_endings_keeps_its_licence():
    assert card_licence(README.replace("\n", "\r\n")) == "other"


def test_a_copied_graph_brings_its_weights_file():
    files = dict(FILES, **{"dac/decoder.onnx": 7, "dac/decoder.onnx_data": 900})
    index = {"vits/amy": {"model_url": url("en_US-amy-medium/model.onnx"),
                          "aux_model_urls": {"codec": url("dac/decoder.onnx")}}}
    amy = next(r for r in plan_mirror(SOURCE, files, README, index).repos if r.folder == "en_US-amy-medium")
    assert amy.shared == ["dac/decoder.onnx", "dac/decoder.onnx_data"]


def test_licence_lines_come_from_the_body_only():
    readme = README.replace("license: other", "license: other\ntags: [en_US-amy-medium]")
    assert plan_mirror(SOURCE, FILES, readme, {}).repos[0].licence_lines == ["| `en_US-amy-medium` | cc-by-4.0 |"]


def test_a_folder_named_inside_a_longer_name_is_not_matched():
    files = {"mini-0.1/model.onnx": 1, "mini-0.1.5/model.onnx": 1}
    readme = README + "\n| `mini-0.1.5` | mit |\n| `mini-0.1` | apache-2.0 |\n"
    plan = {r.folder: r.licence_lines for r in plan_mirror(SOURCE, files, readme, {}).repos}
    assert plan == {"mini-0.1": ["| `mini-0.1` | apache-2.0 |"], "mini-0.1.5": ["| `mini-0.1.5` | mit |"]}


def test_two_folders_that_differ_only_in_case_are_refused():
    with pytest.raises(SplitError, match="fold to the same repo"):
        plan_mirror(SOURCE, {"Amy/model.onnx": 1, "amy/model.onnx": 1}, README, {})


def test_top_level_files_and_unplaced_voices_are_listed():
    files = dict(FILES, **{"notes.txt": 3, "shared.onnx": 4})
    index = {"vits/loose": {"model_url": url("shared.onnx")}}
    plan = plan_mirror(SOURCE, files, README, index)
    assert plan.top_level_files == ["notes.txt"]
    assert plan.top_level_models == ["shared.onnx"]
    assert plan.unplaced_voices == ["vits/loose"]


def test_a_folder_named_at_the_end_of_a_longer_name_is_not_matched():
    files = {"amy/model.onnx": 1, "en_US-amy/model.onnx": 1}
    readme = README + "\n| `en_US-amy` | mit |\n| `amy` | apache-2.0 |\n"
    plan = {r.folder: r.licence_lines for r in plan_mirror(SOURCE, files, readme, {}).repos}
    assert plan == {"amy": ["| `amy` | apache-2.0 |"], "en_US-amy": ["| `en_US-amy` | mit |"]}


def test_a_top_level_licence_file_goes_into_every_new_repo():
    files = dict(FILES, **{"LICENSE": 3, "NOTICE.md": 2, "notes.txt": 1})
    plan = plan_mirror(SOURCE, files, README, {})
    assert [r.shared for r in plan.repos] == [["LICENSE", "NOTICE.md"]] * 2
    assert all(r.files["LICENSE"] == 3 for r in plan.repos)
    assert plan.top_level_files == ["notes.txt"]


def test_an_empty_licence_value_is_a_gap():
    readme = "---\nlicense:\ntags: [tts]\n---\n# x\n"
    assert card_licence(readme) is None
    assert plan_mirror(SOURCE, FILES, readme, {}).licence_gap


def test_a_folder_named_after_a_dot_in_a_longer_name_is_not_matched():
    files = {"1B/model.onnx": 1, "other/model.onnx": 1}
    readme = README + "\nThe 2.1B variant is GPL.\n| `1B` | mit |\n"
    one_b = next(r for r in plan_mirror(SOURCE, files, readme, {}).repos if r.folder == "1B")
    assert one_b.licence_lines == ["| `1B` | mit |"]
