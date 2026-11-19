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


def failed_graph_read(repo, name):
    """onnx_io when the Hub refuses the range request, as it does past the rate limit."""
    import requests
    response = requests.Response()
    response.status_code = 429
    raise requests.HTTPError("429 Client Error: Too Many Requests", response=response)


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
    monkeypatch.setattr(model_card, "onnx_io", failed_graph_read)
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


def example_repo(config):
    card = LIVE_CARD.replace("# phoonnx_pt-PT_miro_unicode", "# not_indexed_voice")
    files = {"voice.json": json.dumps(config), "README.md": card, "voice.onnx": ""}
    io = {"inputs": {"input": 2, "input_lengths": 1, "scales": 1}, "outputs": {"output": 3}}
    return render_repo("Phoonnx/not_indexed_voice", list(files), files.__getitem__, lambda repo, kind: True,
                       index={}, graph_io=lambda name: io)


def example_code(card_text):
    """The standalone example of a card: the last Python block of its "Without phoonnx" section."""
    section = usage(card_text).split("### Without phoonnx", 1)[1]
    return section[section.rindex("```python\n"):].removeprefix("```python\n").split("\n```", 1)[0]


def test_a_card_shows_the_standalone_example_that_compiles():
    config = phoonnx_config(keep_punctuation=False, phoneme_id_map={"_": 0, "^": 1, "$": 2, " ": 3, "b": 4,
                                                                    "o": 5, "m": 6, "d": 7, "i": 8, "a": 9})
    text = usage(example_repo(config))
    assert "### Without phoonnx" in text
    code = example_code(example_repo(config))
    compile(code, "example", "exec")
    assert 'repo = "Phoonnx/not_indexed_voice"' in code
    assert 'hf_hub_download(repo, "voice.onnx")' in code
    assert 'text = "Bom dia."' in code
    assert parse_card(example_repo(config)).sections["How to use"] == text


def without_phoonnx_text(config=None):
    """The prose and the preparation lines of the card's "Without phoonnx" section, above its example."""
    config = config or phoonnx_config(keep_punctuation=False, phoneme_id_map={"_": 0, "^": 1, "$": 2, " ": 3, "b": 4})
    section = usage(example_repo(config)).split("### Without phoonnx", 1)[1]
    return section[:section.rindex("```python")]


@pytest.mark.parametrize("claim", ["English contractions", "typographic apostrophe", 'titles such as "Dr."',
                                   "phoonnx normalises the text before the model reads it",
                                   "runs of spaces to one space"])
def test_the_card_makes_no_claim_about_the_phoonnx_normaliser(claim):
    # the normaliser of phoonnx moves to text2spoken; the card recommends text2spoken and describes no other normaliser
    assert claim not in without_phoonnx_text()


def test_the_card_says_that_phoonnx_reads_each_sentence_on_its_own():
    from phoonnx.config import VoiceConfig
    from phoonnx.voice import TTSVoice
    config = phoonnx_config(keep_punctuation=False, phoneme_type="graphemes", alphabet="unicode", lang_code="pt",
                            phoneme_id_map={"_": 0, "^": 1, "$": 2, " ": 3, "a": 4, "b": 5})
    voice = TTSVoice(session=None, config=VoiceConfig.from_dict(config))
    assert len(voice.phonemize("Ab ba. Ba ab.")) == 2
    text = without_phoonnx_text()
    assert "reads the whole text as one piece" in text
    assert "phoonnx splits the text into sentences" in text


def test_the_card_says_that_phoonnx_cuts_after_an_abbreviation_that_normalisation_keeps():
    from phoonnx.config import VoiceConfig
    from phoonnx.voice import TTSVoice
    from quebra_frases import sentence_tokenize
    from text2spoken import normalize
    sentence = "Mieszkam przy ul. Długiej."
    letters = sorted(set(sentence.lower()) - set(" ."))
    config = phoonnx_config(keep_punctuation=False, phoneme_type="graphemes", alphabet="unicode", lang_code="pl",
                            phoneme_id_map={"_": 0, "^": 1, "$": 2, " ": 3, **{c: n for n, c in enumerate(letters, 4)}})
    assert normalize(sentence, "pl") == sentence
    assert len(TTSVoice(session=None, config=VoiceConfig.from_dict(config)).phonemize(sentence)) == 2
    # the splitter knows other abbreviations, and does not cut after them
    assert sentence_tokenize("I met Dr. Smith today.") == ["I met Dr. Smith today."]
    text = without_phoonnx_text()
    assert "after some abbreviations" in text
    assert 'such as "ul." in Polish' in text


def test_the_card_says_that_the_example_takes_no_inline_phoneme_block():
    from phoonnx.config import VoiceConfig
    from phoonnx.voice import TTSVoice
    config = phoonnx_config(keep_punctuation=False, phoneme_type="graphemes", alphabet="unicode", lang_code="pt",
                            phoneme_id_map={"_": 0, "^": 1, "$": 2, " ": 3, "a": 4, "b": 5, "ʃ": 6})
    phonemes = [p for chunk in TTSVoice(session=None, config=VoiceConfig.from_dict(config)).phonemize("ab [[ʃ]] ba")
                for p in chunk]
    assert "ʃ" in phonemes and "[" not in phonemes
    assert "This code takes plain text." in without_phoonnx_text()
    assert "`[[...]]`" in without_phoonnx_text()


def test_a_voice_no_template_covers_gets_no_standalone_example():
    text = usage(example_repo(phoonnx_config(keep_punctuation=False)))  # no phoneme_id_map
    assert "Without phoonnx" not in text
    assert 'voice.synthesize_wav("Bom dia.", wav_file)' in text


def test_a_voice_no_template_covers_gets_the_text_stack_beside_its_phoonnx_usage():
    text = usage(example_repo(phoonnx_config(keep_punctuation=False)))  # no phoneme_id_map
    prepare = text.index("### Prepare the text")
    assert prepare > text.index('voice.synthesize_wav("Bom dia.", wav_file)')
    assert 'text = normalize(text, "pt-PT")' in text[prepare:]
    assert "before it goes to `voice.synthesize_wav`" in text[prepare:]


@pytest.mark.parametrize("config", [
    phoonnx_config(keep_punctuation=False, phoneme_id_map={"_": 0, "^": 1, "$": 2, " ": 3, "b": 4, "o": 5, "m": 6}),
    phoonnx_config(keep_punctuation=False),  # no template covers it
])
def test_a_graph_that_does_not_read_leaves_out_only_the_example(config, capsys):
    card = LIVE_CARD.replace("# phoonnx_pt-PT_miro_unicode", "# not_indexed_voice")
    files = {"voice.json": json.dumps(config), "README.md": card, "voice.onnx": ""}
    args = ("Phoonnx/not_indexed_voice", list(files), files.__getitem__, lambda repo, kind: True)
    without = render_repo(*args, index={})
    failed = render_repo(*args, index={}, graph_io=lambda name: failed_graph_read("Phoonnx/not_indexed_voice", name))
    # the card keeps the lines that prepare the text, beside the phoonnx usage, and loses only the example
    start = failed.index("\n\n### Prepare the text\n")
    end = failed.index("\n\n## ", start)
    assert failed[:start] + failed[end:] == without
    assert "Without phoonnx" not in failed
    assert "no standalone example" in capsys.readouterr().err


def test_an_upstream_piper_voice_keeps_its_punctuation_under_an_index_alphabet():
    # the voice index sets "alphabet" on espeak Piper voices; phoonnx reads the config without it
    config = piper_config(lang_code="en-US", phoneme_id_map={"_": [0], "^": [1], "$": [2], " ": [3], "a": [4]})
    del config["keep_punctuation"]
    card = LIVE_CARD.replace("# phoonnx_pt-PT_miro_unicode", "# v")
    files = {"voice.onnx.json": json.dumps(config), "voice.onnx": "", "README.md": card}
    index = {"OpenVoiceOS/v": {"model_url": "https://huggingface.co/Phoonnx/v/resolve/main/voice.onnx",
                               "alphabet": "ipa", "lang": "en-US"}}
    io = {"inputs": {"input": 2, "input_lengths": 1, "scales": 1}, "outputs": {"output": 3}}
    text = usage(render_repo("Phoonnx/v", list(files), files.__getitem__, lambda repo, kind: True,
                             {"punctuation_note": "Piper trains with clause punctuation."}, index=index,
                             graph_io=lambda name: io))
    from phoonnx.config import VoiceConfig
    assert VoiceConfig.from_dict(json.loads(files["voice.onnx.json"]), alphabet="ipa").keep_punctuation
    assert 'get_phonemizer("espeak", "ipa", keep_punctuation=True)' in text


def test_the_index_entry_is_found_by_the_model_url_not_the_voice_id():
    card = LIVE_CARD.replace("# phoonnx_pt-PT_miro_unicode", "# v")
    files = {"m.onnx.json": json.dumps(phoonnx_config(keep_punctuation=True)), "other.json": "{}",
             "m.onnx": "", "README.md": card}
    index = {"OpenVoiceOS/v": {"model_url": "https://huggingface.co/Phoonnx/v/resolve/main/m.onnx",
                               "config_url": "https://huggingface.co/Phoonnx/v/resolve/main/m.onnx.json"}}
    text = render_repo("Phoonnx/v", list(files), files.__getitem__, lambda repo, kind: True, index=index)
    assert row(text, "Voice id") == "`OpenVoiceOS/v`"
    assert row(text, "Repository") == "`Phoonnx/v`"


def test_the_index_entry_is_found_at_a_pinned_revision():
    sha = "0123456789abcdef0123456789abcdef01234567"
    card = LIVE_CARD.replace("# phoonnx_pt-PT_miro_unicode", "# v")
    files = {"m.onnx.json": json.dumps(phoonnx_config(keep_punctuation=True)), "other.json": "{}",
             "m.onnx": "", "README.md": card}
    index = {"OpenVoiceOS/v": {"model_url": f"https://huggingface.co/Phoonnx/v/resolve/{sha}/m.onnx",
                               "config_url": f"https://huggingface.co/Phoonnx/v/resolve/{sha}/m.onnx.json"}}
    text = render_repo("Phoonnx/v", list(files), files.__getitem__, lambda repo, kind: True, index=index)
    assert row(text, "Voice id") == "`OpenVoiceOS/v`"


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
    monkeypatch.setattr(model_card, "onnx_io", failed_graph_read)
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


# --- the text that the card prepares: text2spoken, bifonia, text2tashkeel ------------

def preparation(card_text):
    """The Python block of the card's "Prepare the text" part."""
    assert "#### Prepare the text" in usage(card_text)
    section = usage(card_text).split("#### Prepare the text", 1)[1]
    return section.split("```python\n", 1)[1].split("\n```", 1)[0]


def prepared(card_text, text):
    """*text* after the preparation lines of the card, run as the card says: after the line that sets ``text``."""
    names = {"text": text}
    exec(preparation(card_text), names)
    return names["text"]


def text2spoken_extra(tag):
    """The extra that installs the text2spoken language of *tag*, from the distribution's own metadata."""
    from importlib.metadata import metadata
    from text2spoken import supports
    code = supports(tag).code
    return code if code in metadata("text2spoken").get_all("Provides-Extra") else None


def unicode_voice(phoneme_type, letters, lang="pt-PT"):
    """A voice that reads its text, whose map holds each symbol that its phonemizer gives for *letters*."""
    from scriptconv.phonemizers import get_phonemizer
    symbols = sorted(set(get_phonemizer(phoneme_type, "unicode").phonemize_to_list(letters, lang)) - {" "})
    return phoonnx_config(keep_punctuation=False, phoneme_type=phoneme_type, alphabet="unicode", lang_code=lang,
                          phoneme_id_map={"_": 0, "^": 1, "$": 2, " ": 3, **{c: n for n, c in enumerate(symbols, 4)}})


def espeak_voice(lang, **extra):
    return phoonnx_config(keep_punctuation=False, phoneme_type="espeak", alphabet="ipa", lang_code=lang,
                          phoneme_id_map={"_": 0, "^": 1, "$": 2, " ": 3, "a": 4}, **extra)


def test_an_english_voice_gets_text2spoken_with_no_extra():
    from text2spoken import normalize, supports
    card = example_repo(espeak_voice("en-US"))
    assert text2spoken_extra("en-US") is None
    assert "# uv pip install --prerelease=allow text2spoken\n" in preparation(card) + "\n"
    assert f'normalize(text, "{supports("en-US").tag}")' in preparation(card)
    text = "Dr Smith paid $3.50 for 2 tickets."
    assert prepared(card, text) == normalize(text, "en-US") != text
    assert "bifonia" not in usage(card) and "Diacritizer" not in usage(card)
    assert "no language file" not in usage(card)


@pytest.mark.parametrize("config", [
    espeak_voice("pt-PT"),
    # voices that read their text, with each symbol their phonemizer gives for the accented letters
    unicode_voice("unicode", "abcdefghijlmnopqrstuvxzáéêíóôãõç"),
    unicode_voice("graphemes", "abcdefghijlmnopqrstuvxzáéêíóôãõç"),
], ids=["espeak", "unicode", "graphemes"])
def test_a_european_portuguese_voice_gets_the_pt_extra_and_bifonia(config):
    from bifonia import add_extra_diacritics
    from text2spoken import normalize
    card = example_repo(config)
    extra = text2spoken_extra("pt-PT")
    assert extra == "pt"
    assert f'# uv pip install --prerelease=allow "text2spoken[{extra}]" bifonia' in preparation(card)
    text = "A sede da empresa tem 2 andares."
    expected = add_extra_diacritics(normalize(text, "pt-PT"), lang="pt-PT")
    assert expected != normalize(text, "pt-PT")  # bifonia marks "sede" here
    assert prepared(card, text) == expected


@pytest.mark.parametrize("phoneme_type", ["unicode", "graphemes"])
def test_a_portuguese_voice_whose_vocabulary_lacks_the_marks_gets_no_bifonia(phoneme_type):
    from bifonia import add_extra_diacritics
    from scriptconv.phonemizers import get_phonemizer
    config = unicode_voice(phoneme_type, "abcdefghijlmnopqrstuvxzãõç")
    marked = get_phonemizer(phoneme_type, "unicode").phonemize_to_list(
        add_extra_diacritics("a sede da empresa", lang="pt-PT"), "pt-PT")
    assert set(marked) - set(config["phoneme_id_map"])  # the marks would be symbols the voice lacks
    card = example_repo(config)
    assert "bifonia" not in usage(card)
    assert '"text2spoken[pt]"' in preparation(card)


@pytest.mark.parametrize("lang", ["ar", "ar-JO"])
def test_a_diacritized_arabic_voice_gets_text2tashkeel_with_the_rawi_model(lang):
    import inspect
    import scriptconv.diacritics
    import text2tashkeel
    from phoonnx.config import VoiceConfig
    model = inspect.signature(text2tashkeel.Diacritizer).parameters["model"].default
    assert model == scriptconv.diacritics._DEFAULT_DIACRITIZER_MODEL  # the model phoonnx diacritizes with
    assert model.startswith("rawi")
    config = espeak_voice(lang)
    assert VoiceConfig.from_dict(config).add_diacritics
    card = example_repo(config)
    section = usage(card).split("### Without phoonnx", 1)[1]
    assert f'diacritizer = Diacritizer("{model}")' in example_code(card)
    assert section.count("Diacritizer(") == 1  # the marks are added once, in the example
    assert f"[text2tashkeel](https://github.com/TigreGotico/text2tashkeel) and its `{model}` model" in section
    assert f'"text2spoken[{text2spoken_extra(lang)}]"' in preparation(card)
    # the preparation goes in after the sample sentence, so the marks go on the normalised text
    code = example_code(card)
    assert code.index('text = "Bom dia."') < code.index("diacritizer = Diacritizer(")


def espeak_language_without_text2spoken_file(reads_numbers):
    """An espeak language that has no file in the installed text2spoken, and whose numbers it reads or not."""
    from scriptconv.phonemizers import get_phonemizer
    from text2spoken import normalize, supports
    from text2spoken.core.language import language_file
    return next(lang for lang in sorted(get_phonemizer("espeak", "ipa").ESPEAK_LANGS)
                if language_file(supports(lang).code) is None and (normalize("2", lang) != "2") == reads_numbers)


@pytest.mark.parametrize("reads_numbers", [True, False], ids=["reads-numbers", "reads-nothing"])
def test_a_language_with_no_text2spoken_file_gets_a_request_for_one(reads_numbers):
    from text2spoken import supports
    lang = espeak_language_without_text2spoken_file(reads_numbers)
    card = example_repo(espeak_voice(lang))
    assert "#### Prepare the text" in usage(card)
    prose = usage(card).split("#### Prepare the text", 1)[1]
    assert f"text2spoken has no language file for `{supports(lang).tag}`" in prose
    assert "https://github.com/TigreGotico/text2spoken/blob/dev/docs/adding-a-language.md" in prose
    assert "open a pull request" in prose
    assert ("reads only numbers" in prose) == reads_numbers
    assert ("leaves text in this language as it is written" in prose) == (not reads_numbers)


def test_a_language_with_a_text2spoken_file_gets_no_request_for_one():
    # a small file (only numbers, as ko and ja) is still a file
    import card_examples
    from importlib.resources import files
    codes = sorted(f.name[:-3] for f in files("text2spoken").joinpath("lang").iterdir()
                   if f.name.endswith(".py") and not f.name.startswith("_"))
    assert codes
    for code in codes:
        text = card_examples.without_phoonnx({"engine": "f5tts", "lang": code, "voice_id": f"x/{code}"})
        assert "normalize(text" in text, code
        assert "no language file" not in text, code


@pytest.mark.parametrize("phoneme_type, lang", [("tugaphone", "pt-PT"), ("arbtok", "ar")])
def test_a_phonemizer_that_owns_the_diacritize_stage_gets_no_marks_step(phoneme_type, lang):
    import importlib
    from scriptconv.phonemizers.enums import Phonemizer
    from scriptconv.phonemizers.registry import PHONEMIZER_REGISTRY
    module, name, _ = PHONEMIZER_REGISTRY[Phonemizer(phoneme_type)]
    owned = getattr(importlib.import_module(module), name).owned_stages
    assert "diacritize" in owned
    card = example_repo(phoonnx_config(keep_punctuation=False, phoneme_type=phoneme_type, alphabet="ipa",
                                       lang_code=lang, phoneme_id_map={"_": 0, "^": 1, "$": 2, " ": 3, "a": 4}))
    section = usage(card).split("### Without phoonnx", 1)[1]
    assert "add_extra_diacritics" not in section and "Diacritizer" not in section
    assert f"The {phoneme_type} phonemizer of this voice" in section
    # a backend that normalises the text itself gets no second normalisation
    assert ("from text2spoken import normalize" in section) == ("normalize" not in owned)
    assert (f"The {phoneme_type} phonemizer of this voice normalises the text itself" in section) == \
        ("normalize" in owned)


def test_the_text_is_prepared_in_the_language_of_the_voice_not_of_its_phonemizer():
    # a Tetum voice that the voice index reads with the Portuguese phonemizer of espeak
    from text2spoken import supports
    config = espeak_voice("pt")
    card = LIVE_CARD.replace("# phoonnx_pt-PT_miro_unicode", "# v")
    files = {"voice.json": json.dumps(config), "voice.onnx": "", "README.md": card}
    index = {"piper_community/tdt-TL_joao": {"model_url": "https://huggingface.co/Phoonnx/v/resolve/main/voice.onnx",
                                             "config_url": "https://huggingface.co/Phoonnx/v/resolve/main/voice.json",
                                             "lang": "tdt-TL", "phonemizer_lang": "pt", "phoneme_type": "espeak"}}
    io = {"inputs": {"input": 2, "input_lengths": 1, "scales": 1}, "outputs": {"output": 3}}
    text = render_repo("Phoonnx/v", list(files), files.__getitem__, lambda repo, kind: True, index=index,
                       graph_io=lambda name: io)
    assert f'normalize(text, "{supports("tdt-TL").tag}")' in preparation(text)
    assert "bifonia" not in usage(text)
    assert 'phonemizer.phonemize_to_list(text, "pt")' in example_code(text)


def test_a_voice_of_no_single_language_asks_for_the_tag_of_the_text():
    import card_examples
    text = card_examples.without_phoonnx({"engine": "f5tts", "lang": "mul", "voice_id": "f5tts/x"})
    assert "give text2spoken the tag of the language of the text" in text
    assert "no language file" not in text
