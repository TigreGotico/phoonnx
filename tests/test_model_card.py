"""The model card renderer states each voice's training facts and keeps the live card's text."""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import model_card  # noqa: E402
from model_card import CardError, facts_from, parse_card, render, render_repo  # noqa: E402

OWNERSHIP = """Miro and Dii are the recorded voices of two real people. The voices, the
training data, and this model belong to TigreGotico Lda
(https://tigregotico.pt).

This model is licensed under Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND 4.0)."""

FRONT_MATTER = """language:
- pt
license: cc-by-nc-nd-4.0
pipeline_tag: text-to-speech
datasets:
- TigreGotico/tts-train-synthetic-miro_pt-PT_v2
base_model:
- OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone
tags:
- phoonnx
"""

LIVE_CARD = f"""---
{FRONT_MATTER}---

# phoonnx_pt-PT_miro_unicode

This is a text-to-speech voice model for **European Portuguese**.

## How to use

```python
voice.synthesize_wav("Bom dia.", wav_file)
```

## Training data

Trained on the Miro set.

## Related links

- phoonnx: https://github.com/TigreGotico/phoonnx

## Ownership and licensing

{OWNERSHIP}
"""

REPO = "Phoonnx/phoonnx_pt-PT_miro_unicode"
VOICE_ID = "OpenVoiceOS/phoonnx_pt-PT_miro_unicode"
VERSION_STAMP = ("(the version string of the training code; a development build carries "
                 "the stamp of an earlier release)")


def phoonnx_config(**extra):
    config = {"audio": {"sample_rate": 22050}, "lang_code": "pt-PT", "alphabet": "unicode",
              "phoneme_type": "unicode", "num_speakers": 1, "phoonnx_version": "1.2.0a1"}
    config.update(extra)
    return config


def piper_config(**extra):
    config = {"audio": {"sample_rate": 22050}, "espeak": {"voice": "pt"}, "language": {"code": "pt"},
              "phoneme_type": "espeak", "num_speakers": 1, "piper_version": "1.0.0",
              "lang_code": "pt-PT", "keep_punctuation": True}
    config.update(extra)
    return config


def row(card_text: str, field: str) -> str:
    match = re.search(rf"^\| {field} \| (.*) \|$", card_text, flags=re.M)
    assert match, f"no {field} row"
    return match.group(1)


def rendered(config, provenance=None, card_text=LIVE_CARD, repo=REPO, voice_id=VOICE_ID, files=()) -> str:
    card = parse_card(card_text)
    return render(facts_from(repo, config, card, provenance, voice_id=voice_id, files=files), card)


# --- punctuation -----------------------------------------------------------

def test_a_voice_trained_with_punctuation_lists_the_marks_of_its_map():
    # a unicode map is built from the training text: ";" and "!" never reached this model
    text = rendered(phoonnx_config(keep_punctuation=True,
                                   phoneme_id_map={"a": [5], ",": [6], ".": [7], "?": [8], ":": [9]}))
    assert row(text, "Punctuation") == ("Trained with the clause punctuation , . ? :. "
                                        "phoonnx 1.96.1a1 and later keep it at synthesis.")


def test_a_config_without_a_map_lists_every_clause_mark():
    text = rendered(piper_config())
    assert row(text, "Punctuation") == ("Trained with the clause punctuation , . ! ? : ;. "
                                        "phoonnx 1.96.1a1 and later keep it at synthesis.")


def test_a_voice_trained_without_punctuation_says_so():
    text = rendered(phoonnx_config(keep_punctuation=False))
    assert row(text, "Punctuation") == "Trained without punctuation. phoonnx removes it before synthesis."


def test_a_provenance_note_follows_a_recorded_value():
    text = rendered(phoonnx_config(keep_punctuation=False), {"punctuation_note": "Clause breaks were a token."})
    assert row(text, "Punctuation") == ("Trained without punctuation. phoonnx removes it before synthesis. "
                                        "Clause breaks were a token.")


def test_a_config_without_the_key_is_refused():
    with pytest.raises(CardError, match="keep_punctuation"):
        rendered(phoonnx_config())


def test_a_provenance_note_states_the_unknown_value():
    text = rendered(phoonnx_config(), {"punctuation_note": "The training run is not recorded."})
    assert row(text, "Punctuation") == ("Not recorded in the config. phoonnx removes punctuation before "
                                        "synthesis. The training run is not recorded.")


def test_a_non_boolean_key_is_refused():
    with pytest.raises(CardError, match="not a boolean"):
        rendered(phoonnx_config(keep_punctuation="yes"))


# --- rows read from the config ---------------------------------------------

def test_details_come_from_the_config():
    text = rendered(phoonnx_config(keep_punctuation=True))
    assert row(text, "Languages") == "`pt-PT`"
    assert row(text, "Sample rate") == "22050 Hz"
    assert row(text, "Speakers") == "1"
    assert row(text, "Architecture") == "VITS"
    assert row(text, "Text input") == "Unicode codepoints of the text"
    assert row(text, "Phonemizer") == "None: the model reads Unicode codepoints"
    assert row(text, "Training data") == (
        "[TigreGotico/tts-train-synthetic-miro_pt-PT_v2]"
        "(https://huggingface.co/datasets/TigreGotico/tts-train-synthetic-miro_pt-PT_v2)")
    assert row(text, "Base model") == (
        "[OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone]"
        "(https://huggingface.co/OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone)")
    assert row(text, "Trained by") == "TigreGotico Lda"
    assert row(text, "Licence") == "CC BY-NC-ND 4.0"


def test_the_training_code_is_a_version_stamp():
    assert row(rendered(phoonnx_config(keep_punctuation=True)), "Training code") == \
        f"phoonnx, version stamp 1.2.0a1 {VERSION_STAMP}"
    assert row(rendered(piper_config()), "Training code") == f"Piper, version stamp 1.0.0 {VERSION_STAMP}"
    config = phoonnx_config(keep_punctuation=True)
    del config["phoonnx_version"]
    assert row(rendered(config), "Training code") == "Not recorded in the config"


@pytest.mark.parametrize("extra, expected", [
    ({"phonemizer_identity": {"backend": "espeak", "distribution": "espeak-ng", "version": "1.52.0",
                              "lang": "pt", "revision": None},
      "phonemizer_registry": "scriptconv", "phonemizer_registry_version": "0.0.4a70"},
     "espeak-ng 1.52.0, backend `espeak`, lect `pt`"),
    ({"phonemizer_identity": {"backend": "o2i", "distribution": "orthography2ipa", "version": "7.1.0",
                              "lang": "kab-DZ", "revision": "r42"}},
     "orthography2ipa 7.1.0, backend `o2i`, lect `kab-DZ`, spec revision r42"),
    ({"phonemizer_registry": "scriptconv", "phonemizer_registry_version": "0.0.4a70"},
     "through scriptconv 0.0.4a70; engine version not recorded"),
    ({"phoneme_type": "espeak", "alphabet": "ipa"}, "espeak-ng, version not recorded"),
    ({"phoneme_type": "tugaphone", "alphabet": "ipa"}, "tugaphone, version not recorded"),
])
def test_the_phonemizer_row(extra, expected):
    assert row(rendered(phoonnx_config(keep_punctuation=True, **extra)), "Phonemizer") == expected


ESPEAK_BOUND = {"uploaded": "2025-09-30",
                "phonemizer": {"distribution": "espeak-ng", "version": "1.52.0", "released": "2024-12-12"}}


def test_a_version_bound_follows_an_unrecorded_phonemizer():
    config = phoonnx_config(keep_punctuation=False, phoneme_type="espeak", alphabet="ipa")
    assert row(rendered(config, {"version_at_most": ESPEAK_BOUND}), "Phonemizer") == (
        "espeak-ng, version not recorded. If a released espeak-ng was used, it was at most 1.52.0 "
        "(released 2024-12-12), the newest release before the model was first uploaded on 2025-09-30.")


def test_a_version_bound_follows_unrecorded_training_code():
    config = phoonnx_config(keep_punctuation=True)
    del config["phoonnx_version"]
    bound = {"uploaded": "2026-10-12",
             "training_code": {"distribution": "phoonnx", "version": "1.93.2a1", "released": "2026-10-10"}}
    assert row(rendered(config, {"version_at_most": bound}), "Training code") == (
        "Not recorded in the config. If a released phoonnx was used, it was at most 1.93.2a1 "
        "(released 2026-10-10), the newest release before the model was first uploaded on 2026-10-12.")


@pytest.mark.parametrize("config, bound, message", [
    (phoonnx_config(keep_punctuation=True, phoneme_type="espeak", alphabet="ipa", phonemizer_identity={
        "backend": "espeak", "distribution": "espeak-ng", "version": "1.52.0", "lang": "pt", "revision": None}),
     ESPEAK_BOUND, "phonemizer is recorded"),
    (phoonnx_config(keep_punctuation=True),
     {"uploaded": "2026-10-12", "training_code": {"distribution": "phoonnx", "version": "1.0.0", "released": "2026-01-01"}},
     "version stamp is recorded"),
    (phoonnx_config(keep_punctuation=True, phoneme_type="espeak"),
     {"phonemizer": ESPEAK_BOUND["phonemizer"]}, "without the date"),
])
def test_a_bound_that_contradicts_the_record_is_refused(config, bound, message):
    with pytest.raises(CardError, match=message):
        rendered(config, {"version_at_most": bound})


def test_a_piper_config_names_its_phonemizer_and_text_input():
    text = rendered(piper_config())
    assert row(text, "Phonemizer") == "espeak-ng through piper-phonemize, version not recorded"
    assert row(text, "Text input") == "espeak phonemes, IPA alphabet"


def test_a_multilingual_voice_lists_every_language():
    text = rendered(phoonnx_config(keep_punctuation=True), {"languages": "pt-PT gl-ES mwl"})
    assert row(text, "Languages") == "`pt-PT`, `gl-ES`, `mwl`"


def test_a_config_without_a_sample_rate_is_refused():
    config = phoonnx_config(keep_punctuation=True)
    del config["audio"]
    with pytest.raises(CardError, match="no sample rate"):
        rendered(config)


# --- front matter ----------------------------------------------------------

def test_every_dataset_and_base_model_is_linked():
    card = LIVE_CARD.replace("- TigreGotico/tts-train-synthetic-miro_pt-PT_v2\n",
                             "- TigreGotico/a\n- 'TigreGotico/b'\n").replace(
        "base_model:\n- OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone\n", 'base_model: "OpenVoiceOS/x"\n')
    text = rendered(phoonnx_config(keep_punctuation=True), card_text=card)
    assert row(text, "Training data") == ("[TigreGotico/a](https://huggingface.co/datasets/TigreGotico/a), "
                                          "[TigreGotico/b](https://huggingface.co/datasets/TigreGotico/b)")
    assert row(text, "Base model") == "[OpenVoiceOS/x](https://huggingface.co/OpenVoiceOS/x)"


def test_a_card_without_a_base_model_says_not_recorded():
    card = LIVE_CARD.replace("base_model:\n- OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone\n", "")
    assert row(rendered(phoonnx_config(keep_punctuation=True), card_text=card), "Base model") == "Not recorded"


@pytest.mark.parametrize("field, card_text", [
    ("datasets", LIVE_CARD.replace("datasets:\n- TigreGotico/tts-train-synthetic-miro_pt-PT_v2\n", "")),
    ("licence", LIVE_CARD.replace("license: cc-by-nc-nd-4.0\n", "")),
    ("licence", LIVE_CARD.replace("license: cc-by-nc-nd-4.0\n", "license: other\n")),
])
def test_a_card_without_a_required_fact_is_refused(field, card_text):
    with pytest.raises(CardError, match=f"no {field}"):
        facts_from(REPO, phoonnx_config(keep_punctuation=True), parse_card(card_text))


def test_a_config_without_a_language_is_refused():
    config = phoonnx_config(keep_punctuation=True)
    del config["lang_code"]
    with pytest.raises(CardError, match="no languages"):
        rendered(config)


def test_front_matter_values_pass_through_and_gain_the_library_name():
    text = rendered(phoonnx_config(keep_punctuation=True))
    assert text.split("---\n")[1] == FRONT_MATTER + "library_name: phoonnx\n"


def test_a_piper_card_gains_no_library_name():
    text = rendered(piper_config())
    assert text.split("---\n")[1] == FRONT_MATTER


# --- live card text --------------------------------------------------------

def test_the_licence_and_kept_sections_pass_through_unchanged():
    text = rendered(phoonnx_config(keep_punctuation=True))
    assert text.endswith(f"## Ownership and licensing\n\n{OWNERSHIP}\n")
    assert "## Training data\n\nTrained on the Miro set.\n" in text


def test_a_heading_inside_a_code_block_stays_in_its_section():
    data = "```bash\n## not a heading\nls data\n```"
    card = LIVE_CARD.replace("Trained on the Miro set.", data)
    text = rendered(phoonnx_config(keep_punctuation=True), card_text=card)
    assert f"## Training data\n\n{data}\n\n## Related links" in text


@pytest.mark.parametrize("change, message", [
    (("## Related links", "## Citation\n\nCite us.\n\n## Related links"), "not in the card template"),
    (("## Related links", "## How To Use\n\nx\n\n## Related links"), "not in the card template"),
    (("## Related links", "## Training data\n\nAgain.\n\n## Related links"), "appears twice"),
    (("## Related links", "# Another title\n\n## Related links"), "second title"),
])
def test_a_card_that_would_lose_text_is_refused(change, message):
    with pytest.raises(CardError, match=message):
        parse_card(LIVE_CARD.replace(*change))


def test_a_card_with_another_title_is_refused():
    card = LIVE_CARD.replace("# phoonnx_pt-PT_miro_unicode", "# Miro, European Portuguese")
    with pytest.raises(CardError, match="title"):
        rendered(phoonnx_config(keep_punctuation=True), card_text=card)


def test_sections_come_in_the_standard_order():
    text = rendered(phoonnx_config(keep_punctuation=True))
    assert re.findall(r"^## (.+)$", text, flags=re.M) == list(model_card.SECTION_ORDER)
    assert text.index("This is a text-to-speech voice model") < text.index("## Model details")


def test_a_card_without_the_ownership_section_is_refused():
    card = parse_card(LIVE_CARD.split("## Ownership and licensing")[0])
    with pytest.raises(CardError, match="Ownership and licensing"):
        render(facts_from(REPO, phoonnx_config(keep_punctuation=True), card), card)


def test_rendering_twice_gives_the_same_card():
    first = rendered(phoonnx_config(keep_punctuation=True))
    assert first.count("## Model details") == 1
    assert rendered(phoonnx_config(keep_punctuation=True), card_text=first) == first


# --- one Hub repo ----------------------------------------------------------

def hub_repo(exists=lambda repo, kind: True):
    card = LIVE_CARD.replace("# phoonnx_pt-PT_miro_unicode", "# not_indexed_voice")
    files = {"voice.json": json.dumps(phoonnx_config(keep_punctuation=True)), "README.md": card,
             "voice.onnx": ""}
    return render_repo("Phoonnx/not_indexed_voice", list(files), files.__getitem__, exists, index={})


def test_a_repo_renders_from_its_single_voice_config():
    text = hub_repo()
    assert row(text, "Repository") == "`Phoonnx/not_indexed_voice`"
    assert row(text, "Voice id") == "Not in the phoonnx voice index"


def test_a_card_that_links_a_missing_repo_is_refused():
    def exists(repo, kind):
        return kind == "model"
    with pytest.raises(CardError, match="TigreGotico/tts-train-synthetic-miro_pt-PT_v2"):
        hub_repo(exists)


def test_a_map_without_clause_marks_contradicts_a_true_key():
    with pytest.raises(CardError, match="no clause mark"):
        rendered(phoonnx_config(keep_punctuation=True, phoneme_id_map={"a": [5]}))


def test_a_licence_comment_stays_out_of_the_row():
    card = LIVE_CARD.replace("license: cc-by-nc-nd-4.0\n", "license: cc-by-nc-nd-4.0  # see below\n")
    assert row(rendered(phoonnx_config(keep_punctuation=True), card_text=card), "Licence") == "CC BY-NC-ND 4.0"


def test_one_failing_repo_does_not_stop_the_batch(tmp_path, monkeypatch):
    import huggingface_hub
    from click.testing import CliRunner

    def card(name):
        return LIVE_CARD.replace("# phoonnx_pt-PT_miro_unicode", f"# {name}")

    hub = {
        "Phoonnx/first": {"voice.json": json.dumps(phoonnx_config(keep_punctuation=True)),
                          "voice.onnx": "", "README.md": card("first")},
        "Phoonnx/broken": {"voice.json": json.dumps(phoonnx_config(keep_punctuation=True)), "voice.onnx": ""},
        "Phoonnx/third": {"voice.json": json.dumps(phoonnx_config(keep_punctuation=True)),
                          "voice.onnx": "", "README.md": card("third")},
    }

    class FakeApi:
        def list_repo_files(self, repo):
            return list(hub[repo])

        def repo_exists(self, repo, repo_type=None):
            return True

    def download(repo, name, force_download=False):
        target = tmp_path / "hub" / repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(hub[repo][name])  # KeyError for the README that "broken" lacks
        return str(target)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(model_card, "load_voice_index", dict)
    result = CliRunner().invoke(model_card.main, [*hub, "--out", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "repos 3 rendered 2 refused 1" in result.output
    assert "refused: Phoonnx/broken: KeyError" in result.output
    assert (tmp_path / "out" / "third" / "README.md").is_file()


def test_an_unpublished_repo_is_named_without_a_link():
    card = LIVE_CARD.replace("datasets:\n- TigreGotico/tts-train-synthetic-miro_pt-PT_v2\n", "")
    text = rendered(phoonnx_config(keep_punctuation=True),
                    {"unpublished_datasets": ["TigreGotico/gone"],
                     "unpublished_base_models": ["OpenVoiceOS/also_gone"]}, card_text=card)
    assert row(text, "Training data") == "`TigreGotico/gone` (not published)"
    assert row(text, "Base model") == (
        "[OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone](https://huggingface.co/OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone), "
        "`OpenVoiceOS/also_gone` (not published)")


def test_an_unpublished_repo_is_not_checked_for_existence():
    card = LIVE_CARD.replace("# phoonnx_pt-PT_miro_unicode", "# not_indexed_voice")
    files = {"voice.json": json.dumps(phoonnx_config(keep_punctuation=True)), "README.md": card, "voice.onnx": ""}
    text = render_repo("Phoonnx/not_indexed_voice", list(files), files.__getitem__,
                       lambda repo, kind: repo != "TigreGotico/gone",
                       {"unpublished_datasets": ["TigreGotico/gone"]}, index={})
    assert "`TigreGotico/gone` (not published)" in row(text, "Training data")


# --- how to use ------------------------------------------------------------

def usage(card_text: str) -> str:
    return card_text.split("## How to use\n\n", 1)[1].split("\n\n## Training data", 1)[0]


@pytest.mark.parametrize("config", [phoonnx_config(keep_punctuation=True), piper_config()])
def test_an_indexed_voice_is_used_by_its_voice_id(config):
    text = usage(rendered(config))
    assert 'voice = manager.voices["OpenVoiceOS/phoonnx_pt-PT_miro_unicode"].load()' in text
    assert 'voice.synthesize_wav("Bom dia.", wav_file)' in text
    assert '"module": "ovos-tts-plugin-phoonnx"' in text
    assert '"voice": "OpenVoiceOS/phoonnx_pt-PT_miro_unicode"' in text


@pytest.mark.parametrize("config", [phoonnx_config(keep_punctuation=True), piper_config()])
def test_no_card_names_an_archived_plugin_or_a_missing_call(config):
    text = rendered(config) + hub_repo()
    for gone in ("ovos-tts-plugin-piper", "from_hf", "github.com/OpenVoiceOS/ovos-tts-plugin-phoonnx", "piper --model"):
        assert gone not in text


def test_a_voice_outside_the_index_is_loaded_from_its_files():
    text = usage(hub_repo())
    assert 'model = hf_hub_download("Phoonnx/not_indexed_voice", "voice.onnx")' in text
    assert 'config = hf_hub_download("Phoonnx/not_indexed_voice", "voice.json")' in text
    assert "ovos-tts-plugin-phoonnx" not in text


def test_the_index_entry_is_found_by_the_model_url_not_the_voice_id():
    card = LIVE_CARD.replace("# phoonnx_pt-PT_miro_unicode", "# v")
    files = {"m.onnx.json": json.dumps(phoonnx_config(keep_punctuation=True)), "other.json": "{}",
             "m.onnx": "", "README.md": card}
    index = {"OpenVoiceOS/v": {"model_url": "https://huggingface.co/Phoonnx/v/resolve/main/m.onnx",
                               "config_url": "https://huggingface.co/Phoonnx/v/resolve/main/m.onnx.json"}}
    text = render_repo("Phoonnx/v", list(files), files.__getitem__, lambda repo, kind: True, index=index)
    assert row(text, "Voice id") == "`OpenVoiceOS/v`"
    assert row(text, "Repository") == "`Phoonnx/v`"


def test_two_index_voices_in_one_repo_are_refused():
    entry = {"model_url": "https://huggingface.co/Phoonnx/not_indexed_voice/resolve/main/voice.onnx"}
    card = LIVE_CARD.replace("# phoonnx_pt-PT_miro_unicode", "# not_indexed_voice")
    files = {"voice.json": "{}", "README.md": card, "voice.onnx": ""}
    with pytest.raises(CardError, match="more than one voice"):
        render_repo("Phoonnx/not_indexed_voice", list(files), files.__getitem__, lambda repo, kind: True,
                    index={"a": entry, "b": entry})


def test_a_card_without_a_sample_sentence_is_refused():
    card = LIVE_CARD.replace('voice.synthesize_wav("Bom dia.", wav_file)', "Run it.")
    with pytest.raises(CardError, match="no sample sentence"):
        rendered(phoonnx_config(keep_punctuation=True), card_text=card)


def test_a_placeholder_sample_is_not_a_sample_sentence():
    card = LIVE_CARD.replace('synthesize_wav("Bom dia.", wav_file)', 'synthesize_wav("...", wav_file)')
    with pytest.raises(CardError, match="no sample sentence"):
        rendered(phoonnx_config(keep_punctuation=True), card_text=card)


def test_a_piper_command_line_gives_the_sample_sentence():
    card = LIVE_CARD.replace('voice.synthesize_wav("Bom dia.", wav_file)',
                             'echo "Olá, mundo." | piper --model x --output_file out.wav')
    assert 'voice.synthesize_wav("Olá, mundo.", wav_file)' in usage(rendered(piper_config(), card_text=card))


def test_the_checkpoint_row_names_the_checkpoint_in_the_repo():
    files = ("README.md", "m.onnx", "epoch=8428-step=1308542.ckpt")
    assert row(rendered(piper_config(), files=files), "Training checkpoint") == \
        "`epoch=8428-step=1308542.ckpt`, in this repository for fine-tuning"
    assert row(rendered(piper_config()), "Training checkpoint") == "None in this repository"


def test_an_index_that_names_the_repo_under_its_old_org_is_refused():
    card = LIVE_CARD.replace("# phoonnx_pt-PT_miro_unicode", "# v")
    files = {"m.onnx.json": json.dumps(phoonnx_config(keep_punctuation=True)), "m.onnx": "", "README.md": card}
    index = {"OpenVoiceOS/v": {"model_url": "https://huggingface.co/OpenVoiceOS/v/resolve/main/m.onnx"}}
    with pytest.raises(CardError, match="points at https://huggingface.co/OpenVoiceOS/v/"):
        render_repo("Phoonnx/v", list(files), files.__getitem__, lambda repo, kind: True, index=index)


def test_a_voice_of_another_name_in_the_index_is_no_stale_entry():
    index = {"x": {"model_url": "https://huggingface.co/OpenVoiceOS/other_voice/resolve/main/m.onnx"}}
    card = LIVE_CARD.replace("# phoonnx_pt-PT_miro_unicode", "# not_indexed_voice")
    files = {"voice.json": json.dumps(phoonnx_config(keep_punctuation=True)), "README.md": card, "voice.onnx": ""}
    text = render_repo("Phoonnx/not_indexed_voice", list(files), files.__getitem__, lambda r, k: True, index=index)
    assert row(text, "Voice id") == "Not in the phoonnx voice index"


@pytest.mark.parametrize("phoneme_type, alphabet, install", [
    ("unicode", "unicode", "pip install --pre phoonnx\n"),
    ("espeak", "ipa", "pip install --pre phoonnx\n"),
    ("tugaphone", "ipa", 'pip install --pre phoonnx "scriptconv[pt-phonemizers]"\n'),
    ("orthography2ipa", "ipa", 'pip install --pre phoonnx "scriptconv[o2i]"\n'),
])
def test_the_install_line_brings_the_phonemizer_of_the_voice(phoneme_type, alphabet, install):
    config = phoonnx_config(keep_punctuation=True, phoneme_type=phoneme_type, alphabet=alphabet)
    for text in (usage(rendered(config)), usage(rendered(config, voice_id=None))):
        assert f"```bash\n{install}```" in text
        assert ("espeak-ng program" in text) == (phoneme_type == "espeak")
        assert ('"phoonnx[espeak]"' in text) == (phoneme_type == "espeak")


@pytest.mark.parametrize("sample", ["Bom\\ndia.", "Bom\ndia."])
def test_a_sample_the_template_cannot_quote_is_refused(sample):
    card = LIVE_CARD.replace('synthesize_wav("Bom dia.", wav_file)', f'synthesize_wav("{sample}", wav_file)')
    with pytest.raises(CardError, match="cannot be quoted"):
        rendered(phoonnx_config(keep_punctuation=True), card_text=card)


@pytest.mark.parametrize("config, bound, message", [
    (phoonnx_config(keep_punctuation=False, phoneme_type="espeak", alphabet="ipa"),
     {"uploaded": "2024-01-01", "phonemizer": ESPEAK_BOUND["phonemizer"]}, "released after"),
    (phoonnx_config(keep_punctuation=True, phoonnx_version=None),
     {"uploaded": "2026-10-01",
      "training_code": {"distribution": "phoonnx", "version": "1.93.2a1", "released": "2026-10-10"}},
     "released after"),
    (phoonnx_config(keep_punctuation=True), ESPEAK_BOUND, "reads Unicode codepoints"),
])
def test_an_impossible_bound_is_refused(config, bound, message):
    if config.get("phoonnx_version") is None:
        config.pop("phoonnx_version", None)
    with pytest.raises(CardError, match=message):
        rendered(config, {"version_at_most": bound})


def test_a_provenance_phonemizer_contradicts_a_phonemizer_bound():
    config = phoonnx_config(keep_punctuation=False, phoneme_type="espeak", alphabet="ipa")
    with pytest.raises(CardError, match="phonemizer is recorded"):
        rendered(config, {"version_at_most": ESPEAK_BOUND, "phonemizer": "espeak-ng 1.51"})


def test_a_piper_stamp_contradicts_a_training_bound():
    bound = {"uploaded": "2026-10-12",
             "training_code": {"distribution": "piper", "version": "1.0.0", "released": "2023-01-01"}}
    with pytest.raises(CardError, match="version stamp is recorded"):
        rendered(piper_config(), {"version_at_most": bound})


def test_the_batch_renders_with_the_voice_index_it_loads(tmp_path, monkeypatch):
    import huggingface_hub
    from click.testing import CliRunner

    files = {"voice.json": json.dumps(phoonnx_config(keep_punctuation=True)), "voice.onnx": "",
             "README.md": LIVE_CARD.replace("# phoonnx_pt-PT_miro_unicode", "# v")}

    class FakeApi:
        def list_repo_files(self, repo):
            return list(files)

        def repo_exists(self, repo, repo_type=None):
            return True

    def download(repo, name, force_download=False):
        target = tmp_path / "hub" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(files[name])
        return str(target)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(model_card, "load_voice_index", lambda: {
        "OpenVoiceOS/v": {"model_url": "https://huggingface.co/Phoonnx/v/resolve/main/voice.onnx"}})
    result = CliRunner().invoke(model_card.main, ["Phoonnx/v", "--out", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert row((tmp_path / "out" / "v" / "README.md").read_text(), "Voice id") == "`OpenVoiceOS/v`"


def test_a_bound_released_on_the_upload_day_holds():
    config = phoonnx_config(keep_punctuation=False, phoneme_type="espeak", alphabet="ipa")
    bound = {"uploaded": "2024-12-12", "phonemizer": ESPEAK_BOUND["phonemizer"]}
    assert "at most 1.52.0 (released 2024-12-12)" in row(rendered(config, {"version_at_most": bound}), "Phonemizer")


def test_the_espeak_note_says_espyak_can_differ():
    text = usage(rendered(phoonnx_config(keep_punctuation=True, phoneme_type="espeak", alphabet="ipa")))
    assert "Some espyak phonemes differ from those of\nespeak-ng" in text
