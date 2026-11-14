#!/usr/bin/env python3
"""Render the Hugging Face model card of a phoonnx or Piper voice.

Every card has the same sections in the same order. The "Model details"
table is built from the voice's own ``config.json``, so the card states what
phoonnx reads when it loads the voice. The front matter, the lead text and
the "Training data", "Related links" and "Ownership and licensing" sections
are copied from the live card; the licence text is never generated. "How to
use" is generated, so every card shows the same, working phoonnx usage; only
its sample sentence comes from the live card. A live card with any other section, a duplicate section or a
different title is refused. A live "Model details" section is replaced.

A training fact that the config does not record is never guessed. A config
without ``keep_punctuation`` is refused unless ``model_card_provenance.json``
gives a note that says why the value is unknown.

When the config records no phonemizer or training code version, the
provenance file may give ``version_at_most``: the newest release of that
package before the model weights were first uploaded. The card states it as
an upper bound that holds if a released package was used.

Usage::

    python scripts/model_card.py Phoonnx/phoonnx_pt-PT_miro_unicode --out cards/
    python scripts/model_card.py --all-inhouse --out cards/
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import click

PROVENANCE_FILE = Path(__file__).with_name("model_card_provenance.json")

SECTION_ORDER = ("Model details", "How to use", "Training data", "Related links",
                 "Ownership and licensing")

CLAUSE_MARKS = (",", ".", "!", "?", ":", ";")

LICENCE_NAMES = {
    "cc-by-nc-nd-4.0": "CC BY-NC-ND 4.0",
    "cc-by-nc-sa-4.0": "CC BY-NC-SA 4.0",
    "cc-by-nc-4.0": "CC BY-NC 4.0",
    "cc-by-sa-4.0": "CC BY-SA 4.0",
    "cc-by-4.0": "CC BY 4.0",
    "cc0-1.0": "CC0 1.0",
    "mit": "MIT",
    "apache-2.0": "Apache 2.0",
}

# The release from which phoonnx feeds clause punctuation to a model trained with it.
PUNCTUATION_RUNTIME = "phoonnx 1.96.1a1 and later keep it at synthesis."
PUNCTUATION_DROPPED = "Trained without punctuation. phoonnx removes it before synthesis."

DEFAULT_RELATED_LINKS = """- phoonnx: https://github.com/TigreGotico/phoonnx
- TigreGotico: https://tigregotico.pt"""


class CardError(ValueError):
    """The facts needed for a card are missing or contradict each other."""


@dataclass(frozen=True)
class CardFacts:
    """The facts a card states, each read from the voice's config or front matter."""
    repo_id: str
    engine: str
    languages: Tuple[str, ...]
    phoneme_type: str
    alphabet: Optional[str]
    keep_punctuation: Optional[bool]
    clause_marks: Tuple[str, ...]
    punctuation_note: str
    sample_rate: int
    num_speakers: int
    training_code: str
    phonemizer: str
    datasets: Tuple[str, ...]
    base_models: Tuple[str, ...]
    trained_by: str
    licence: str
    upstream: str = ""
    unpublished_datasets: Tuple[str, ...] = ()
    unpublished_base_models: Tuple[str, ...] = ()
    voice_id: Optional[str] = None
    model_file: str = ""
    config_file: str = ""
    checkpoints: Tuple[str, ...] = ()
    sample_text: str = ""


@dataclass(frozen=True)
class LiveCard:
    """A card split into front matter, title, the text above the first section, and its sections."""
    front_matter: str
    title: str
    lead: str
    sections: Dict[str, str]


def parse_card(text: str) -> LiveCard:
    """Split a model card into front matter, title, lead text and ``## `` sections.

    Headings inside fenced code blocks are text. A section the template does
    not hold, or one that appears twice, raises CardError rather than being lost.
    """
    front_matter, body = "", text
    match = re.match(r"---\n(.*?)\n---[ \t]*(?:\n|$)", text, flags=re.S)
    if match:
        front_matter, body = match.group(1), text[match.end():]
    title, current, chunks = "", None, {None: []}
    in_code = False
    for line in body.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
        heading = None if in_code else re.match(r"^(#{1,2}) (.+?)\s*$", line)
        if heading and heading.group(1) == "#" and current is None and not title:
            title = heading.group(2)
            continue
        if heading and heading.group(1) == "##":
            current = heading.group(2)
            if current not in SECTION_ORDER:
                raise CardError(f"section '{current}' is not in the card template")
            if current in chunks:
                raise CardError(f"section '{current}' appears twice")
            chunks[current] = []
            continue
        if heading:
            raise CardError(f"second title '{heading.group(2)}'")
        chunks[current].append(line)
    sections = {name: "\n".join(lines).strip() for name, lines in chunks.items() if name is not None}
    return LiveCard(front_matter, title, "\n".join(chunks[None]).strip(), sections)


def _unquote(value: str) -> str:
    return re.sub(r"\s+#.*$", "", value).strip().strip("'\"")


def front_matter_list(front_matter: str, key: str) -> Tuple[str, ...]:
    """Return the values of ``key`` in the front matter, as a block list or a single scalar."""
    block = re.search(rf"^{key}:[ \t]*\n((?:[ \t]*- .*\n?)+)", front_matter, flags=re.M)
    if block:
        return tuple(_unquote(line.strip()[2:]) for line in block.group(1).splitlines())
    scalar = re.search(rf"^{key}:[ \t]*(\S.*)$", front_matter, flags=re.M)
    return (_unquote(scalar.group(1)),) if scalar else ()


def front_matter_value(front_matter: str, key: str) -> str:
    values = front_matter_list(front_matter, key)
    return values[0] if values else ""


def _training_code(config: Dict[str, Any]) -> str:
    for name, key in (("phoonnx", "phoonnx_version"), ("Piper", "piper_version")):
        if config.get(key):
            return f"{name}, version stamp {config[key]} (the version string of the training code; a development build carries the stamp of an earlier release)"
    return "Not recorded in the config"


def _at_most(text: str, bound: Optional[Dict[str, str]], uploaded: str) -> str:
    if not bound:
        return text
    return (f"{text}. If a released {bound['distribution']} was used, it was at most {bound['version']} "
            f"(released {bound['released']}), the newest release before the model was first uploaded on {uploaded}.")


def _phonemizer(config: Dict[str, Any], engine: str) -> str:
    identity = config.get("phonemizer_identity")
    if identity:
        text = f"{identity['distribution']} {identity['version'] or '(version not recorded)'}"
        text += f", backend `{identity['backend']}`, lect `{identity['lang']}`"
        if identity.get("revision"):
            text += f", spec revision {identity['revision']}"
        return text
    if config.get("phonemizer_registry"):
        version = config.get("phonemizer_registry_version") or "version not recorded"
        return f"through {config['phonemizer_registry']} {version}; engine version not recorded"
    phoneme_type = config.get("phoneme_type", "")
    if phoneme_type == "unicode":
        return "None: the model reads Unicode codepoints"
    if phoneme_type == "espeak" and engine == "piper":
        return "espeak-ng through piper-phonemize, version not recorded"
    if phoneme_type == "espeak":
        return "espeak-ng, version not recorded"
    return f"{phoneme_type}, version not recorded"


def _text_input(phoneme_type: str, alphabet: Optional[str]) -> str:
    if phoneme_type == "unicode":
        return "Unicode codepoints of the text"
    alphabet = alphabet or "ipa"
    return f"{phoneme_type} phonemes, {'IPA' if alphabet == 'ipa' else alphabet} alphabet"


def _sample_rate(repo_id: str, config: Dict[str, Any]) -> int:
    rate = (config.get("audio") or {}).get("sample_rate") or config.get("sample_rate")
    if not rate:
        raise CardError(f"{repo_id}: no sample rate")
    return int(rate)


# The sample sentence of the live "How to use" section: a phoonnx call or a Piper command line.
SAMPLE_PATTERNS = (r'synthesize_wav\(\s*"([^"]+)"', r'echo "([^"]+)" \|')


def _sample_text(card: LiveCard) -> str:
    usage = card.sections.get("How to use", "")
    for pattern in SAMPLE_PATTERNS:
        match = re.search(pattern, usage)
        if match and match.group(1) != "...":
            if "\\" in match.group(1) or "\n" in match.group(1):
                raise CardError(f"the sample sentence {match.group(1)!r} cannot be quoted in the usage example")
            return match.group(1)
    return ""


def facts_from(repo_id: str, config: Dict[str, Any], card: LiveCard,
               provenance: Optional[Dict[str, Any]] = None, voice_id: Optional[str] = None,
               files: Tuple[str, ...] = (), model_file: str = "", config_file: str = "") -> CardFacts:
    """Collect the card facts of one voice; raise CardError when a required fact is missing."""
    provenance = provenance or {}
    engine = "piper" if config.get("piper_version") and not config.get("phoonnx_version") else "phoonnx"
    keep_punctuation = config.get("keep_punctuation")
    note = provenance.get("punctuation_note", "")
    if keep_punctuation is None and not note:
        raise CardError(f"{repo_id}: config has no keep_punctuation and provenance gives no note")
    if keep_punctuation is not None and not isinstance(keep_punctuation, bool):
        raise CardError(f"{repo_id}: keep_punctuation is {keep_punctuation!r}, not a boolean")
    language = config.get("lang_code") or (config.get("language") or {}).get("code")
    languages = tuple(provenance.get("languages", "").split()) or ((language,) if language else ())
    datasets = front_matter_list(card.front_matter, "datasets")
    # Names the card records for repos the Hub does not hold: named, never linked.
    unpublished_datasets = tuple(provenance.get("unpublished_datasets", ()))
    licence = provenance.get("licence") or front_matter_value(card.front_matter, "license")
    if licence == "other":
        licence = ""
    for name, value in (("languages", languages), ("datasets", datasets + unpublished_datasets),
                        ("licence", licence)):
        if not value:
            raise CardError(f"{repo_id}: no {name}")
    at_most = provenance.get("version_at_most", {})
    if at_most and not at_most.get("uploaded"):
        raise CardError(f"{repo_id}: a version bound without the date the model was first uploaded")
    for kind in ("phonemizer", "training_code"):
        if at_most.get(kind) and at_most[kind]["released"] > at_most["uploaded"]:
            raise CardError(f"{repo_id}: the {kind} bound was released after the model was first uploaded")
    if at_most.get("phonemizer") and config.get("phoneme_type") == "unicode":
        raise CardError(f"{repo_id}: a phonemizer version bound for a voice that reads Unicode codepoints")
    if at_most.get("phonemizer") and (config.get("phonemizer_identity") or provenance.get("phonemizer")):
        raise CardError(f"{repo_id}: a phonemizer version bound for a voice whose phonemizer is recorded")
    if at_most.get("training_code") and (config.get("phoonnx_version") or config.get("piper_version")):
        raise CardError(f"{repo_id}: a training code version bound for a voice whose version stamp is recorded")
    # A mark the phoneme map lacks never reached the model, whatever the key says.
    phoneme_map = config.get("phoneme_id_map") or {}
    return CardFacts(
        repo_id=repo_id,
        engine=engine,
        languages=languages,
        phoneme_type=config.get("phoneme_type", ""),
        alphabet=config.get("alphabet"),
        keep_punctuation=keep_punctuation,
        clause_marks=tuple(m for m in CLAUSE_MARKS if m in phoneme_map) if phoneme_map else CLAUSE_MARKS,
        punctuation_note=note,
        sample_rate=_sample_rate(repo_id, config),
        num_speakers=int(config.get("num_speakers", 1)),
        training_code=_at_most(_training_code(config), at_most.get("training_code"), at_most.get("uploaded", "")),
        phonemizer=_at_most(provenance.get("phonemizer") or _phonemizer(config, engine),
                            at_most.get("phonemizer"), at_most.get("uploaded", "")),
        datasets=datasets,
        base_models=front_matter_list(card.front_matter, "base_model"),
        trained_by=provenance.get("trained_by", "TigreGotico Lda"),
        licence=licence,
        upstream=provenance.get("upstream", ""),
        unpublished_datasets=unpublished_datasets,
        unpublished_base_models=tuple(provenance.get("unpublished_base_models", ())),
        voice_id=voice_id,
        model_file=model_file,
        config_file=config_file,
        checkpoints=tuple(sorted(f for f in files if f.endswith(".ckpt"))),
        sample_text=_sample_text(card),
    )


def hf_url(repo: str, kind: str = "") -> str:
    return f"https://huggingface.co/{kind + '/' if kind else ''}{repo}"


def _hf_link(repo: str, kind: str = "") -> str:
    return f"[{repo}]({hf_url(repo, kind)})"


def linked_repos(facts: CardFacts) -> List[Tuple[str, str]]:
    """Every (repo id, repo type) the card links to."""
    return [(d, "dataset") for d in facts.datasets] + [(m, "model") for m in facts.base_models]


def _repo_list(published: Tuple[str, ...], unpublished: Tuple[str, ...], kind: str = "") -> str:
    return ", ".join([_hf_link(r, kind) for r in published] + [f"`{r}` (not published)" for r in unpublished])


def punctuation_row(facts: CardFacts) -> str:
    if facts.keep_punctuation is True and not facts.clause_marks:
        raise CardError(f"{facts.repo_id}: keep_punctuation is true but the phoneme map has no clause mark")
    if facts.keep_punctuation is True:
        text = f"Trained with the clause punctuation {' '.join(facts.clause_marks)}. {PUNCTUATION_RUNTIME}"
    elif facts.keep_punctuation is False:
        text = PUNCTUATION_DROPPED
    else:
        text = "Not recorded in the config. phoonnx removes punctuation before synthesis."
    return f"{text} {facts.punctuation_note}".strip()


def details_table(facts: CardFacts) -> str:
    rows: List[Tuple[str, str]] = [
        ("Repository", f"`{facts.repo_id}`"),
        ("Voice id", f"`{facts.voice_id}`" if facts.voice_id else "Not in the phoonnx voice index"),
        ("Languages", ", ".join(f"`{lang}`" for lang in facts.languages)),
        ("Architecture", "VITS"),
        ("Text input", _text_input(facts.phoneme_type, facts.alphabet)),
        ("Punctuation", punctuation_row(facts)),
        ("Sample rate", f"{facts.sample_rate} Hz"),
        ("Speakers", str(facts.num_speakers)),
        ("Training code", facts.training_code),
        ("Phonemizer", facts.phonemizer),
        ("Training data", _repo_list(facts.datasets, facts.unpublished_datasets, "datasets")),
        ("Base model", _repo_list(facts.base_models, facts.unpublished_base_models) or "Not recorded"),
        ("Training checkpoint", ", ".join(f"`{c}`" for c in facts.checkpoints) + ", in this repository for fine-tuning"
         if facts.checkpoints else "None in this repository"),
    ]
    if facts.upstream:
        rows.append(("Upstream", facts.upstream))
    rows += [
        ("Trained by", facts.trained_by),
        ("Licence", LICENCE_NAMES.get(facts.licence, facts.licence)),
    ]
    lines = ["| Field | Value |", "| --- | --- |"]
    lines += [f"| {field} | {value} |" for field, value in rows]
    return "\n".join(lines)


# What a phoneme type needs beyond phoonnx itself.
EXTRA_PACKAGES = {"tugaphone": ' "scriptconv[pt-phonemizers]"', "orthography2ipa": ' "scriptconv[o2i]"'}
ESPEAK_NOTE = ("\n\nThe phonemes come from the espeak-ng program, so install it too (for example\n"
               "`apt install espeak-ng`). Without it, `pip install --pre \"phoonnx[espeak]\"` gives\n"
               "phoonnx the espyak port of espeak-ng. Some espyak phonemes differ from those of\n"
               "espeak-ng, so the voice can say some words wrong with espyak.")


def _install(facts: CardFacts) -> str:
    line = f"```bash\npip install --pre phoonnx{EXTRA_PACKAGES.get(facts.phoneme_type, '')}\n```"
    return line + (ESPEAK_NOTE if facts.phoneme_type == "espeak" else "")


def how_to_use(facts: CardFacts) -> str:
    """phoonnx usage for the voice: by voice id when the voice index lists it, else from its files."""
    if not facts.sample_text:
        raise CardError(f"{facts.repo_id}: the live 'How to use' section has no sample sentence")
    if not facts.voice_id:
        return f"""The phoonnx voice index does not list this voice. Install
[phoonnx](https://github.com/TigreGotico/phoonnx):

{_install(facts)}

Then load the voice from its files:

```python
import wave
from huggingface_hub import hf_hub_download
from phoonnx.voice import TTSVoice

model = hf_hub_download("{facts.repo_id}", "{facts.model_file}")
config = hf_hub_download("{facts.repo_id}", "{facts.config_file}")
voice = TTSVoice.load(model, config)

with wave.open("output.wav", "wb") as wav_file:
    voice.synthesize_wav("{facts.sample_text}", wav_file)
```"""
    return f"""Install [phoonnx](https://github.com/TigreGotico/phoonnx):

{_install(facts)}

Then run:

```python
import wave
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.load()
manager.merge_default_voices()

voice = manager.voices["{facts.voice_id}"].load()

with wave.open("output.wav", "wb") as wav_file:
    voice.synthesize_wav("{facts.sample_text}", wav_file)
```

In OVOS, phoonnx installs the `ovos-tts-plugin-phoonnx` plugin. Select this voice in
`mycroft.conf`:

```json
{{
  "tts": {{
    "module": "ovos-tts-plugin-phoonnx",
    "ovos-tts-plugin-phoonnx": {{
      "voice": "{facts.voice_id}"
    }}
  }}
}}
```"""


def _front_matter(front_matter: str, facts: CardFacts) -> str:
    if facts.engine == "phoonnx" and not re.search(r"^library_name:", front_matter, flags=re.M):
        front_matter = front_matter.rstrip("\n") + "\nlibrary_name: phoonnx"
    return front_matter.rstrip("\n")


def render(facts: CardFacts, card: LiveCard) -> str:
    """Render the card. Sections the live card has are kept; the rest are generated."""
    name = facts.repo_id.split("/")[-1]
    if card.title and card.title != name:
        raise CardError(f"{facts.repo_id}: the live card title '{card.title}' is not the repo name")
    if "Ownership and licensing" not in card.sections:
        raise CardError(f"{facts.repo_id}: the live card has no 'Ownership and licensing' section")
    sections = {
        "Model details": details_table(facts),
        "How to use": how_to_use(facts),
        "Training data": card.sections.get("Training data") or "Trained on " + _repo_list(
            facts.datasets, facts.unpublished_datasets, "datasets") + ".",
        "Related links": card.sections.get("Related links") or DEFAULT_RELATED_LINKS,
        "Ownership and licensing": card.sections["Ownership and licensing"],
    }
    parts = [f"---\n{_front_matter(card.front_matter, facts)}\n---", f"# {name}"]
    if card.lead:
        parts.append(card.lead)
    parts += [f"## {title}\n\n{sections[title]}" for title in SECTION_ORDER]
    return "\n\n".join(parts) + "\n"


def load_provenance(path: Path = PROVENANCE_FILE) -> Dict[str, Dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def render_repo(repo_id: str, files: List[str], download: Callable[[str], str],
                repo_exists: Callable[[str, str], bool],
                provenance: Optional[Dict[str, Any]] = None,
                index: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """Render one Hub repo's card; raise CardError when it cannot be rendered truthfully."""
    voice_id, entry = _index_entry(repo_id, load_voice_index() if index is None else index)
    if entry:
        model_file = _repo_file(repo_id, entry["model_url"])
        config_file = _repo_file(repo_id, entry["config_url"]) if entry.get("config_url") else _single_config(repo_id, files)
    else:
        models = [f for f in files if f.endswith(".onnx")]
        if len(models) != 1:
            raise CardError(f"{repo_id}: cannot pick the voice model from {models}")
        model_file, config_file = models[0], _single_config(repo_id, files)
    config = json.loads(download(config_file))
    card = parse_card(download("README.md"))
    facts = facts_from(repo_id, config, card, provenance, voice_id=voice_id, files=tuple(files),
                       model_file=model_file, config_file=config_file)
    missing = [repo for repo, kind in linked_repos(facts) if not repo_exists(repo, kind)]
    if missing:
        raise CardError(f"{repo_id}: links to repos that do not exist: {', '.join(missing)}")
    return render(facts, card)


def load_voice_index() -> Dict[str, Dict[str, Any]]:
    """Every voice-index entry phoonnx ships, keyed by voice id."""
    from phoonnx.model_manager import TTSModelManager
    index: Dict[str, Dict[str, Any]] = {}
    for index_file in TTSModelManager.voice_index_files():
        index.update(json.loads(Path(index_file).read_text(encoding="utf-8")))
    return index


def _index_entry(repo_id: str, index: Dict[str, Dict[str, Any]]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """The voice id and entry whose model file lives in ``repo_id``; the id and the repo can differ."""
    prefix = f"https://huggingface.co/{repo_id}/resolve/main/"
    matches = [(voice_id, entry) for voice_id, entry in index.items()
               if (entry.get("model_url") or "").startswith(prefix)]
    if len(matches) > 1:
        raise CardError(f"{repo_id}: the voice index lists more than one voice in this repo")
    org, name = repo_id.split("/", 1)
    elsewhere = [entry["model_url"] for entry in index.values()
                 if re.match(rf"https://huggingface\.co/(?!{re.escape(org)}/)[^/]+/{re.escape(name)}/resolve/",
                             entry.get("model_url") or "")]
    if not matches and elsewhere:
        raise CardError(f"{repo_id}: the voice index points at {elsewhere[0]}; render with an index that "
                        f"names the repo where it lives")
    return matches[0] if matches else (None, None)


def _repo_file(repo_id: str, url: str) -> str:
    return url.split(f"https://huggingface.co/{repo_id}/resolve/main/", 1)[1]


def _single_config(repo_id: str, files: List[str]) -> str:
    configs = [f for f in files if f.endswith(".json") and not f.endswith(".piper.json")]
    if len(configs) != 1:
        raise CardError(f"{repo_id}: cannot pick the voice config from {configs}")
    return configs[0]


@click.command()
@click.argument("repos", nargs=-1)
@click.option("--all-inhouse", is_flag=True,
              help="Render every Phoonnx phoonnx_* and pipertts_* repo.")
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True,
              help="Directory that receives <repo name>/README.md.")
def main(repos: Tuple[str, ...], all_inhouse: bool, out_dir: Path) -> None:
    """Render model cards from the live Hub configs and cards. Uploads nothing."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    repo_ids = list(repos)
    if all_inhouse:
        repo_ids += sorted(m.id for m in api.list_models(author="Phoonnx")
                           if m.id.split("/")[1].startswith(("phoonnx_", "pipertts_")))
    provenance = load_provenance()
    index = load_voice_index()
    failed = []
    for repo_id in repo_ids:
        def download(name: str) -> str:
            return Path(hf_hub_download(repo_id, name, force_download=True)).read_text(encoding="utf-8")
        try:
            text = render_repo(repo_id, api.list_repo_files(repo_id), download,
                               lambda repo, kind: api.repo_exists(repo, repo_type=kind),
                               provenance.get(repo_id), index)
        except Exception as error:  # one repo must not stop the batch; the refusal names the cause
            failed.append(f"{repo_id}: {type(error).__name__}: {error}")
            continue
        target = out_dir / repo_id.split("/")[-1] / "README.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    click.echo(f"repos {len(repo_ids)} rendered {len(repo_ids) - len(failed)} refused {len(failed)}")
    for error in failed:
        click.echo(f"refused: {error}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
