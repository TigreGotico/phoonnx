"""Standalone onnxruntime examples for the model cards of phoonnx voices.

A card's example runs the voice with onnxruntime, numpy, huggingface_hub, the
standard library and scriptconv. scriptconv holds the phonemizer the voice was
trained with, so the phonemes match training. A voice that reads its text
(``graphemes``, ``unicode``) also goes through scriptconv, because phoonnx takes
the punctuation, the hyphens and the case of the text from there. A voice that
phoonnx gives Arabic vowel marks before its phonemizer gets them from scriptconv
too. The example writes out the text-to-id step that phoonnx runs for the repo's
``config.json`` (phoneme_id_map, pad, blanks, BOS/EOS), so the reader sees what
the model reads.

:func:`standalone_example` renders the example. Both card renderers import it:
``model_card.py`` from this directory, and the split-repo renderer by adding this
directory to ``sys.path``. The example names the model's feeds by the graph's real
input names, which :func:`onnx_io` reads from the Hub with one range request.

A model of several graphs reads the voice index entry instead of a voice config:
its template in :data:`INDEX_TEMPLATES` writes out what the phoonnx engine does
(tokenisation, prompt, the generation loop with phoonnx's sampling settings, the
codec decode). Such an example may also need the ``tokenizers`` package, and its
first line names it. :data:`REFUSED_ENGINES` names the engines that have no
example, and why.

A config that no template covers raises :class:`UnsupportedExample`, and the
renderer leaves the example out of that card.
"""
import importlib
import json
import re
import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from scriptconv.phonemizers import get_phonemizer
from scriptconv.phonemizers.enums import Phonemizer
from scriptconv.phonemizers.registry import PHONEMIZER_REGISTRY

PIPER_TOKENS = {"pad": "_", "blank": "_", "bos": "^", "eos": "$"}
RAW_TEXT = ("graphemes", "unicode")
VOCODER_TYPES = ("hifigan", "melgan", "wavenext", "raw", "vocos")
INSTALL = "onnxruntime numpy huggingface_hub"
# the keys of a voice index entry that phoonnx writes over the config's own when it loads the voice by its id
INDEX_OVERRIDES = ("phoneme_type", "alphabet", "phonemizer_model")
TEXT2SPOKEN = "https://github.com/TigreGotico/text2spoken"
ADDING_A_LANGUAGE = TEXT2SPOKEN + "/blob/dev/docs/adding-a-language.md"
BIFONIA = "https://github.com/TigreGotico/bifonia"
TEXT2TASHKEEL = "https://github.com/TigreGotico/text2tashkeel"
# text2tashkeel's default model, a rawi model, and the one scriptconv gives phoonnx when the config names none
RAWI = "rawi-ensemble"
# the accented letters that bifonia writes on a homograph in every lect; context_free_marks adds the lect's own
BIFONIA_LETTERS = "áéêóô"
# the tags of no single language: text2spoken takes the tag of the text's language instead
UNNAMED_LANGUAGES = ("und", "mul", "mis", "zxx")
# the phonemizers that read bifonia's acute and circumflex as the open and closed vowel, measured on each marked form
READS_PT_MARKS = ("espeak",)


class UnsupportedExample(ValueError):
    """The config or graph has a shape that no example template covers."""


def standalone_example(config: Dict[str, Any], files: Dict[str, Any], *, repo_id: str,
                       text: Optional[str] = None, lang: Optional[str] = None,
                       overrides: Optional[Dict[str, str]] = None) -> str:
    """The Python code block of a card: download the voice, build its ids, run it, write a WAV.

    ``config`` is the parsed voice config. ``files`` names the repo's files and the
    graphs' inputs and outputs:

    - ``model``, ``config``: file names in the repo (``model.onnx``, ``config.json``);
    - ``inputs``, ``outputs``: the model graph's names, as :func:`onnx_io` returns them;
    - ``vocoder``: for a spectrogram model, ``{"repo", "file", "type", "inputs",
      "config"}``, where ``config`` is the parsed ``vocoder.json`` and ``repo`` is
      left out when the vocoder is in the voice's own repo;
    - ``style``: for StyleTTS2, the voice's style file as ``{"repo", "file"}``, where
      ``repo`` is left out when the file is in the voice's own repo;
    - ``decode``: for a VITS split at the prior, the decoder graph's file name;
    - ``dictionary``: for a vosk voice, the pronunciation dictionary that the voice
      index names as ``{"repo", "file"}``, where ``repo`` is left out when the file
      is in the voice's own repo. phoonnx loads it as the phonemizer model when the
      index names no ``phonemizer_model``.

    ``text`` is the card's sample sentence, and ``lang`` the phonemizer language when
    the voice index sets one that differs from the config's ``lang_code``.
    ``overrides`` holds :func:`index_overrides` of the voice index entry, which win
    over the config's own keys. phoonnx reads the config without them to tell a
    Piper config from a phoonnx side-car, so the caller passes them here and does
    not merge them into ``config``.

    For an engine in :data:`INDEX_TEMPLATES` (a model of several graphs, a text
    tokenizer or a generation loop), ``config`` is the voice index entry and
    ``files`` is :func:`index_files` of that entry; ``lang`` and ``overrides`` are
    not used, as the entry names the language and the input.
    """
    engine = config.get("engine") or "piper"
    if engine in REFUSED_ENGINES:
        raise UnsupportedExample(REFUSED_ENGINES[engine])
    if engine in INDEX_TEMPLATES:
        return "```python\n" + INDEX_TEMPLATES[engine](config, files, repo_id=repo_id, text=text) + "\n```"
    voice = {**config, **(overrides or {})}
    lang = _lang(config, lang)
    piper = _is_piper(config)
    diacritize, diacritics_extra = _diacritize(config, lang, piper, _phoneme_type(voice))
    model = (f'"{voice["phonemizer_model"]}"' if voice.get("phonemizer_model") else
             _hub({"repo": repo_id, **files["dictionary"]}, repo_id) if files.get("dictionary") else None)
    phonemes, extra = _phonemes(voice, lang, _keeps_punctuation(config, piper), model)
    ids, pad = _ids(voice, piper)
    run = _run(engine, voice, files, pad)
    extras = ",".join(dict.fromkeys(filter(None, (extra, *(diacritics_extra or "").split(",")))))
    install = f'# pip install {INSTALL} "scriptconv[{extras}]"'
    if _phoneme_type(voice) == "espeak":
        install += "  # and the espeak-ng program"
    imports = ["import json", "import wave", "",
               "import numpy as np", "import onnxruntime", "from huggingface_hub import hf_hub_download"]
    if diacritize:
        imports.append("from quebra_frases import sentence_tokenize")
    imports.append("from scriptconv.phonemizers import get_phonemizer")
    if diacritize:
        imports.append("from text2tashkeel import Diacritizer")
    lines = [install, *imports, "",
             f'repo = "{repo_id}"',
             f'config = json.load(open(hf_hub_download(repo, "{files.get("config", "config.json")}"), encoding="utf-8"))',
             "", f"text = {_text(text, config)}", *diacritize, phonemes, "", ids, "", run, "", _wav(config)]
    return "```python\n" + "\n".join(lines) + "\n```"


def without_phoonnx(config: Dict[str, Any], *, lang: Optional[str] = None,
                    overrides: Optional[Dict[str, str]] = None, voice_lang: Optional[str] = None) -> str:
    """The card text above the example of :func:`standalone_example`, which takes the same arguments.

    It names the packages of the example, gives the lines that prepare the text, and says what the
    example leaves out of phoonnx's path. The text is normalised with text2spoken, in *voice_lang*:
    the language of the voice index entry, which can differ from the language the phonemizer reads
    (a Tetum voice on a Portuguese phonemizer). Without it, the phonemizer's language stands for it.
    A language with no text2spoken file gets a request for one. A Portuguese voice whose input reads
    bifonia's marks gets them after the normalisation, unless its phonemizer owns the ``diacritize``
    stage. A voice that phoonnx diacritizes gets the vowel marks in the example itself.
    """
    stack = _text_stack(config, lang, overrides, voice_lang)
    if stack.diacritized and stack.diacritizer is None and not stack.own_tokenizer:
        raise UnsupportedExample(f"phoonnx diacritizes the {stack.phonemizer_lang} text before the phonemizer")
    text = ["### Without phoonnx", "",
            "This code runs the voice without phoonnx. Its first line names the packages it needs.", "",
            "#### Prepare the text", ""]
    if "normalize" in stack.owned:
        text += [f"The {stack.voice['phoneme_type']} phonemizer of this voice normalises the text itself, as in "
                 "phoonnx, so the code needs no step that prepares the text.", ""]
    else:
        text += _preparation(stack, with_phoonnx=False)
    text += ["This code reads the whole text as one piece, in one run of the model. phoonnx splits the text into "
             "sentences or chunks and runs the model once for each piece. Its sentence splitter also cuts the "
             "text after some abbreviations, such as \"ul.\" in Polish, when normalisation keeps them. So give "
             "this code one sentence at a time. For a sentence that phoonnx reads as one piece and does not "
             "rewrite, this code builds the same model input as phoonnx.", "",
             "This code takes plain text. phoonnx reads an inline `[[...]]` block as phonemes, but this code gives "
             "the brackets and their contents to the phonemizer as text.", "", ""]
    return "\n".join(text)


def with_phoonnx(config: Dict[str, Any], *, lang: Optional[str] = None,
                 overrides: Optional[Dict[str, str]] = None, voice_lang: Optional[str] = None) -> str:
    """The card text beside the phoonnx usage of a voice that has no standalone example.

    It takes the arguments of :func:`without_phoonnx` and recommends the same text stack, before the text
    goes to phoonnx: text2spoken, the request for a language file, and bifonia. phoonnx runs its own
    vowel-mark step, so the text names the text2tashkeel model of that step for an Arabic voice, and a
    Portuguese voice that phoonnx diacritizes gets no bifonia step.
    """
    stack = _text_stack(config, lang, overrides, voice_lang)
    text = ["### Prepare the text", ""]
    if "normalize" in stack.owned:
        text += [f"The {stack.voice['phoneme_type']} phonemizer of this voice normalises the text itself, so the "
                 "text needs no step before phoonnx.", ""]
    else:
        text += _preparation(stack, with_phoonnx=True)
    return "\n".join(text).rstrip("\n")


class _TextStack(NamedTuple):
    """What decides the text stack of a voice: the voice, the tags, the stages its backend owns, and phoonnx's
    vowel-mark step (*diacritized*, with its text2tashkeel model for an Arabic voice)."""
    voice: Dict[str, Any]
    tag: Optional[str]
    phonemizer_lang: Optional[str]
    owned: frozenset
    diacritized: bool
    diacritizer: Optional[str]
    own_tokenizer: bool


def _text_stack(config: Dict[str, Any], lang: Optional[str], overrides: Optional[Dict[str, str]],
                voice_lang: Optional[str]) -> _TextStack:
    """The :class:`_TextStack` of a voice, from the arguments of :func:`without_phoonnx`.

    An engine of :data:`INDEX_TEMPLATES` reads the text with its own tokenizer, and its config is the index
    entry. Of those engines, phoonnx's loader turns the vowel-mark step on only for Chatterbox, in Arabic and
    Hebrew.
    """
    engine = config.get("engine") or "piper"
    if engine in INDEX_TEMPLATES:
        tag = voice_lang or config.get("lang")
        diacritized = engine == "chatterbox" and (_is_arabic(tag) or (tag or "").lower().startswith("he"))
        diacritizer = _diacritizer_model(config, False) if diacritized and _is_arabic(tag) else None
        return _TextStack(config, tag, tag, frozenset(), diacritized, diacritizer, True)
    voice = {**config, **(overrides or {})}
    phoneme_type = _phoneme_type(voice)
    phonemizer_lang = _lang(config, lang) or ("und" if phoneme_type in RAW_TEXT else None)
    owned = _owned_stages(phoneme_type)
    piper = _is_piper(config)
    diacritized = _phoonnx_diacritizes(config, phonemizer_lang, piper) and "diacritize" not in owned
    diacritizer = _diacritizer_model(config, piper) if diacritized and _is_arabic(phonemizer_lang) else None
    return _TextStack(voice, voice_lang or phonemizer_lang, phonemizer_lang, owned, diacritized, diacritizer, False)


def _preparation(stack: _TextStack, *, with_phoonnx: bool) -> List[str]:
    """The text and the lines that prepare the text: text2spoken, then bifonia when it applies.

    *with_phoonnx* words the text for the phoonnx usage of the card, and otherwise for the standalone example,
    whose own vowel-mark step uses the text2tashkeel model of *stack*.
    """
    from importlib.metadata import metadata

    from text2spoken import Form, supports

    voice, owned = stack.voice, stack.owned
    support = supports(stack.tag or "und")
    extra = support.code if support.code in (metadata("text2spoken").get_all("Provides-Extra") or ()) else None
    package = f'"text2spoken[{extra}]"' if extra else "text2spoken"
    imports = ["from text2spoken import normalize"]
    steps = [f'text = normalize(text, "{support.tag}")  # numbers, dates, money and abbreviations as words']
    prose = [f"[text2spoken]({TEXT2SPOKEN}) writes numbers, dates, times, money, units and abbreviations as the "
             "words that a speaker says."]
    if support.code in UNNAMED_LANGUAGES:
        steps = ['text = normalize(text, "en")  # the tag of the language of the text']
        prose.append("This voice does not name one language, so give text2spoken the tag of the language of "
                     "the text.")
    elif not support.forms - {Form.CARDINAL, Form.LEXICON}:
        reads = ("so it reads only cardinal numbers in this language" if Form.CARDINAL in support.forms
                 else "so it leaves text in this language as it is written")
        prose.append(f"text2spoken has no language file for `{support.tag}`, {reads}. To add the language, follow "
                     f"[Adding a language]({ADDING_A_LANGUAGE}) and open a pull request.")
    portuguese = support.tag.split("-")[0] == "pt"
    if portuguese and not stack.diacritized and _bifonia(voice, support.tag, owned):
        package += " bifonia"
        imports.insert(0, "from bifonia import add_extra_diacritics")
        steps.append(f'text = add_extra_diacritics(text, lang="{support.tag}")  # open and closed vowels of homographs')
        prose.append(f"[bifonia]({BIFONIA}) then puts an accent on a word whose sound depends on its sense, such as "
                     "\"sede\" (\"sêde\", thirst, or \"séde\", seat). This voice reads the accent as an open or a "
                     "closed vowel.")
        if not with_phoonnx:
            prose.append("phoonnx does not run bifonia for this voice, so with this step some words sound "
                         "different from phoonnx.")
    elif portuguese and "diacritize" in owned:
        prose.append(f"The {voice['phoneme_type']} phonemizer of this voice puts the accents on homographs itself, "
                     "so the code has no bifonia step.")
    elif portuguese and stack.diacritized:
        prose.append("phoonnx runs its own vowel-mark step on the text of this voice, so the code has no bifonia "
                     "step.")
    if stack.diacritizer and (with_phoonnx or not stack.own_tokenizer):
        prose.append(f"phoonnx then adds the Arabic vowel marks with [text2tashkeel]({TEXT2TASHKEEL}) and its "
                     f"`{stack.diacritizer}` model." if with_phoonnx else
                     f"The code then adds the Arabic vowel marks to each sentence with "
                     f"[text2tashkeel]({TEXT2TASHKEEL}) and its `{stack.diacritizer}` model, as phoonnx does.")
    elif _is_arabic(stack.tag) and "diacritize" in owned:
        prose.append(f"The {voice['phoneme_type']} phonemizer of this voice adds the Arabic vowel marks itself.")
    where = ("Run these lines on the text before it goes to `voice.synthesize_wav`:" if with_phoonnx else
             "Put these lines into the code below, after the line that sets `text` to the sample sentence:")
    return [where, "", "```python", f"# uv pip install --prerelease=allow {package}", *imports, "", *steps, "```", "",
            " ".join(prose), ""]


def _bifonia(voice: Dict[str, Any], tag: str, owned: frozenset) -> bool:
    """True when the Portuguese voice reads bifonia's marks and runs no homograph stage of its own.

    A voice that reads its text reads them when its phonemizer turns each accented letter that bifonia writes
    for *tag* into symbols of the voice's map; a ``unicode`` voice reads a letter as its base and a combining
    mark. A phonemizer reads them when it is in :data:`READS_PT_MARKS`. The voice of an engine with its own
    text tokenizer has no vocabulary in the config, so it gets no marks.
    """
    if "diacritize" in owned or voice.get("engine") in INDEX_TEMPLATES:
        return False
    phoneme_type = _phoneme_type(voice)
    if phoneme_type not in RAW_TEXT:
        return phoneme_type in READS_PT_MARKS
    from bifonia import context_free_marks

    letters = set(BIFONIA_LETTERS) | {c for word, marked in context_free_marks(tag).items()
                                      for c in marked if c not in word}
    symbols = get_phonemizer(phoneme_type, "unicode").phonemize_to_list(" ".join(sorted(letters)), tag)
    ids = voice.get("phoneme_id_map") or {}
    return all(symbol in ids or symbol.lower() in ids for symbol in symbols if symbol != " ")


def index_overrides(entry: Dict[str, Any]) -> Dict[str, str]:
    """The keys of :data:`INDEX_OVERRIDES` that the voice index *entry* sets: the ``overrides`` of :func:`standalone_example`."""
    return {key: entry[key] for key in INDEX_OVERRIDES if entry.get(key)}


def index_files(entry: Dict[str, Any], live: Callable[[str], str] = lambda url: url,
                io: bool = False) -> Dict[str, Dict[str, Any]]:
    """The files of a voice index entry, by key, as ``{"repo", "file"}``.

    The keys are ``model``, every other ``*_url`` key of the entry without its
    ``_url`` suffix, and every key of ``aux_model_urls``. ``live`` maps an index
    URL to the URL of the file in the repo that holds it now. With ``io``, each
    ONNX file also gets the ``inputs`` and ``outputs`` that :func:`onnx_io` reads,
    and a template checks the names it feeds against them.
    """
    urls = {"model": entry["model_url"]}
    urls.update({key.removesuffix("_url"): url for key, url in entry.items()
                 if key.endswith("_url") and key != "model_url" and isinstance(url, str)})
    urls.update(entry.get("aux_model_urls") or {})
    files = {}
    for key, url in urls.items():
        match = re.fullmatch(r"https://huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)", live(url))
        if match:
            files[key] = {"repo": match.group(1), "file": match.group(2)}
            if io and match.group(2).endswith(".onnx"):
                files[key].update(onnx_io(match.group(1), match.group(2)))
    return files


def _hub(file: Dict[str, str], repo_id: str) -> str:
    """The hf_hub_download call of one file: ``repo`` names the voice's repo in the example."""
    source = "repo" if file["repo"] == repo_id else f'"{file["repo"]}"'
    return f'hf_hub_download({source}, "{file["file"]}")'


def _write_wav(rate: str) -> str:
    """The lines that write ``audio`` (float samples) to output.wav; ``rate`` is a literal or an expression."""
    return "\n".join([
        'with wave.open("output.wav", "wb") as wav:',
        "    wav.setnchannels(1)",
        "    wav.setsampwidth(2)",
        f"    wav.setframerate({rate})",
        "    wav.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())"])


def _text(text: Optional[str], config: Dict[str, Any]) -> str:
    if text:
        return json.dumps(text, ensure_ascii=False)
    return f'"..."  # a sentence in {config.get("lang_code") or "the voice language"}'


def _phoneme_type(config: Dict[str, Any]) -> str:
    phoneme_type = config.get("phoneme_type") or "espeak"
    return {"text": "unicode", "pygoruut": "goruut"}.get(phoneme_type, phoneme_type)


def _keeps_punctuation(config: Dict[str, Any], piper: bool) -> bool:
    """phoonnx's keep_punctuation for *config*, read before the voice index overrides apply."""
    if "keep_punctuation" in config:
        return bool(config["keep_punctuation"])
    # Piper trains with clause punctuation; phoonnx's piper-shaped side-cars do not
    made_by_phoonnx = str(config.get("piper_version", "")).startswith("phoonnx") or "alphabet" in config
    return piper and not made_by_phoonnx


def _lang(config: Dict[str, Any], lang: Optional[str]) -> Optional[str]:
    """The language phoonnx phonemizes in: the index language, then the config's, as its loaders read them."""
    lang = lang or config.get("lang_code")
    if lang or "phoonnx_version" in config:
        return lang
    language = config.get("language")
    lang = language.get("code") if isinstance(language, dict) else None
    if not lang and _is_piper(config):
        lang = (config.get("espeak") or {}).get("voice")
    return lang


def _is_arabic(lang: Optional[str]) -> bool:
    """phoonnx's test for Arabic: ``ar`` or an individual Arabic language, in Arabic script."""
    from linguatag import lang_distance
    return bool(lang) and lang_distance(lang, "ar") <= 5


def _diacritize(config: Dict[str, Any], lang: Optional[str], piper: bool,
                phoneme_type: str) -> Tuple[List[str], Optional[str]]:
    """The lines that add the vowel marks phoonnx adds before the phonemizer, and their scriptconv extra.

    phoonnx turns diacritization on for an Arabic Piper voice, and for a voice
    whose config sets ``inference.add_diacritics``. A phoonnx config without the
    key, or with it null, turns it on when its language is Arabic. The text
    frontend then skips the step when the backend of *phoneme_type* owns the
    ``diacritize`` stage, because that backend restores the vowel marks itself.
    """
    if not _phoonnx_diacritizes(config, lang, piper) or "diacritize" in _owned_stages(phoneme_type):
        return [], None
    if not _is_arabic(lang):
        raise UnsupportedExample(f"phoonnx diacritizes the {lang} text before the phonemizer")
    return ["# the vowel marks phoonnx adds before the phonemizer, to each sentence on its own",
            f'diacritizer = Diacritizer("{_diacritizer_model(config, piper)}")',
            'text = " ".join(diacritizer.diacritize(sentence) for sentence in sentence_tokenize(text))'], \
        "tashkeel,phonemizers"


def _phoonnx_diacritizes(config: Dict[str, Any], lang: Optional[str], piper: bool) -> bool:
    """True when phoonnx turns its vowel-mark step on for the voice, before a backend that owns it skips it."""
    if piper:
        return _is_arabic(lang)
    enabled = (config.get("inference") or {}).get("add_diacritics", None if "phoonnx_version" in config else False)
    return _is_arabic(lang) if enabled is None else bool(enabled)


def _diacritizer_model(config: Dict[str, Any], piper: bool) -> str:
    """The text2tashkeel model that phoonnx diacritizes with: the config's, or :data:`RAWI`; Piper configs name none."""
    return (None if piper else (config.get("inference") or {}).get("diacritizer_model")) or RAWI


def _backend(phoneme_type: str) -> Tuple[str, str, Optional[str]]:
    """The module, class name and scriptconv extra of the backend of *phoneme_type*."""
    try:
        return PHONEMIZER_REGISTRY[Phonemizer(phoneme_type)]
    except (KeyError, ValueError):
        raise UnsupportedExample(f"no scriptconv phonemizer for {phoneme_type!r}") from None


def _owned_stages(phoneme_type: str) -> frozenset:
    """The text stages that the backend of *phoneme_type* runs itself, so phoonnx's frontend skips them."""
    module, name, _ = _backend(phoneme_type)
    try:
        return getattr(importlib.import_module(module), name).owned_stages
    except ImportError as error:
        raise UnsupportedExample(f"the {phoneme_type} backend does not import, so its own stages are unknown: "
                                 f"{error}") from None


def _phonemes(config: Dict[str, Any], lang: Optional[str], keep: bool, model: Optional[str]) -> Tuple[str, str]:
    """The text-to-phonemes lines, and the scriptconv extra they need. *model* is the code of the phonemizer model."""
    phoneme_type = _phoneme_type(config)
    alphabet = config.get("alphabet") or ("unicode" if phoneme_type in RAW_TEXT else "ipa")
    if phoneme_type in RAW_TEXT and alphabet != "unicode":
        raise UnsupportedExample(f"{phoneme_type} input in the {alphabet} alphabet")
    _, _, extra = _backend(phoneme_type)
    # a backend of scriptconv itself splits the text into sentences with the phonemizers extra
    extra = extra or "phonemizers"
    if phoneme_type in RAW_TEXT:
        lang = lang or "und"
    elif not lang:
        raise UnsupportedExample("the config names no language")
    if phoneme_type == "shami":
        return "\n".join([
            'phonemizer = get_phonemizer("shami")',
            "# each symbol comes with the language it belongs to (Arabic, English, neutral);",
            "# the phonemizer gives one chunk per sentence, and this code reads them as one",
            "phonemes, language_ids = [], []",
            f'for chunk, chunk_language_ids in phonemizer.phonemize_with_language_ids_lazy(text, "{lang}"):',
            "    phonemes += chunk",
            "    language_ids += chunk_language_ids"]), extra
    model = f", model={model}" if model else ""
    punctuation = (", keep_punctuation=True)  # the voice was trained with clause punctuation" if keep
                   else ")")
    lines = [f'phonemizer = get_phonemizer("{phoneme_type}", "{alphabet}"{model}{punctuation}',
             f'phonemes = phonemizer.phonemize_to_list(text, "{lang}")']
    if phoneme_type == "unicode":
        lines.append('phonemes = [p if p in config["phoneme_id_map"] else p.lower() for p in phonemes]'
                     "  # a capital the map lacks reads as its small letter")
    return "\n".join(lines), extra


def _is_piper(config: Dict[str, Any]) -> bool:
    """A config phoonnx reads with its Piper loader: Piper's fixed special tokens apply."""
    if "phoonnx_version" in config:
        return False
    if "piper_version" in config:
        return True
    id_map = config.get("phoneme_id_map") or {}
    return (config.get("engine") in (None, "piper") and config.get("phoneme_type") in ("text", "espeak")
            and bool(id_map) and all(isinstance(v, list) for v in id_map.values()))


def _ids(config: Dict[str, Any], piper: bool) -> Tuple[str, Optional[str]]:
    """The ids lines, as phoonnx's tokenizer builds them for this config, and the pad symbol.

    *piper* tells that phoonnx reads the config with its Piper loader, whose special
    tokens and flags are fixed.
    """
    id_map = config.get("phoneme_id_map")
    if not id_map:
        raise UnsupportedExample("the config has no phoneme_id_map")
    if "phonemes" in config or "characters" in config:
        raise UnsupportedExample("a mimic3 or Coqui config, which phoonnx reads with other files")
    lists = any(isinstance(v, list) for v in id_map.values())
    ids = {k: v[0] if isinstance(v, list) else v for k, v in id_map.items()}
    if piper:
        tokens, add_blank, at_start, at_end, eos_bos, blank_word, fold = PIPER_TOKENS, True, True, True, True, False, True
    else:
        tokens = {key: config.get(key) or PIPER_TOKENS[key] for key in PIPER_TOKENS}
        add_blank, at_start, at_end = (config.get(k, True) for k in ("add_blank_char", "blank_at_start", "blank_at_end"))
        eos_bos, blank_word = config.get("use_eos_bos", True), config.get("add_blank_word", False)
        fold = config.get("fold_compounds", True)
    blank, bos, eos = (tokens[k] if tokens[k] in ids else None for k in ("blank", "bos", "eos"))
    lines = ['id_map = {s: i[0] for s, i in config["phoneme_id_map"].items()}  # each id is a one-item list'
             if lists else 'id_map = config["phoneme_id_map"]']
    folds = fold and any(len(s) > 1 for s in ids)
    if blank_word and not folds:
        lines.append('phonemes = [p for p in phonemes if p != " "]  # the config gives no word blank symbol')
    if folds:
        lines += ["# a symbol of several characters (\"aɪ\", \"t͡ʃ\") has its own id: match the longest first",
                  "symbols = sorted((s for s in id_map if len(s) > 1), key=len, reverse=True)",
                  "ids, i = [], 0",
                  "while i < len(phonemes):"]
        if blank_word:
            lines += ['    if phonemes[i] == " ":  # a word boundary: nothing folds across it, and the config '
                      "gives no word blank symbol",
                      "        i += 1",
                      "        continue"]
        lines += ['    n = next((len(s) for s in symbols if "".join(phonemes[i:i + len(s)]) == s), 1)',
                  '    if "".join(phonemes[i:i + n]) in id_map:',
                  '        ids.append(id_map["".join(phonemes[i:i + n])])',
                  "    i += n"]
    else:
        lines.append("ids = [id_map[p] for p in phonemes if p in id_map]")
    if blank and add_blank:
        interleave = {(True, True): "[x for i in ids for x in (blank, i)] + [blank]",
                      (True, False): "[x for i in ids for x in (blank, i)]",
                      (False, True): "[x for i in ids for x in (i, blank)]",
                      (False, False): "[x for i in ids for x in (i, blank)][:-1]"}[(at_start, at_end)]
        where = {(True, True): " and at both ends", (True, False): " and at the start",
                 (False, True): " and at the end", (False, False): ""}[(at_start, at_end)]
        lines += [f'blank = id_map["{blank}"]', f"ids = {interleave}  # a blank between symbols{where}, as in training"]
    elif blank and at_start:
        lines += [f'ids = [id_map["{blank}"]] + ids  # a blank at the start, as in training']
    if eos_bos and bos and eos:
        lines.append(f'ids = [id_map["{bos}"]] + ids + [id_map["{eos}"]]  # start and end of sentence')
    return "\n".join(lines), tokens["pad"] if tokens["pad"] in ids else None


def _feed(name: str, value: str) -> str:
    return f'    "{name}": {value},'


def _speaker(config: Dict[str, Any]) -> str:
    speakers = int(config.get("num_speakers") or 1)
    names = list(config.get("speaker_id_map") or {})
    named = f" ({', '.join(names[:3])}{', ...' if len(names) > 3 else ''})" if names else ""
    return f"speaker 0 of {speakers}{named}" if speakers > 1 else "the only speaker"


def _inputs(files: Dict[str, Any], key: str = "inputs") -> Dict[str, int]:
    names = files.get(key)
    if not names:
        raise UnsupportedExample(f"files[{key!r}] does not name the graph's inputs")
    return names


def _check(inputs: Dict[str, int], known: Tuple[str, ...], engine: str) -> None:
    unknown = sorted(set(inputs) - set(known))
    if unknown:
        raise UnsupportedExample(f"{engine} graph inputs {unknown} have no template")


def _params(config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """The synthesis parameters phoonnx gives an engine: its *defaults*, the config's scales, then ``engine_params``."""
    inference = config.get("inference") or {}
    scales = {"noise_scale": inference.get("noise_scale", 0.667), "length_scale": inference.get("length_scale", 1.0),
              "noise_w_scale": inference.get("noise_w", 0.8)}
    return {**defaults, **{k: v for k, v in scales.items() if v is not None}, **(config.get("engine_params") or {})}


def _scales(config: Dict[str, Any]) -> Tuple[float, float, float]:
    params = _params(config, {})
    return params["noise_scale"], params["length_scale"], params["noise_w_scale"]


def _session(name: str, file: str, repo: Optional[str] = None) -> str:
    source = f'"{repo}"' if repo else "repo"
    return f'{name} = onnxruntime.InferenceSession(hf_hub_download({source}, "{file}"))'


def _run(engine: str, config: Dict[str, Any], files: Dict[str, Any], pad: Optional[str]) -> str:
    """The lines that run the graphs and leave the waveform in ``audio``."""
    model = files.get("model", "model.onnx")
    if files.get("decode"):
        return _vits_prior_split(config, files)
    inputs = _inputs(files)
    if engine in ("piper", "coqui", "phoonnx", "transformers"):
        return _vits(config, model, inputs, engine)
    if engine == "shami":
        _check(inputs, ("phoneme_ids", "phoneme_lengths", "language_ids", "speaker_id"), engine)
        feeds = [_feed("phoneme_ids", "np.array([ids], dtype=np.int64)"),
                 _feed("phoneme_lengths", "np.array([len(ids)], dtype=np.int64)"),
                 _feed("language_ids", "np.array([language_ids], dtype=np.int64)")]
        if "speaker_id" in inputs:
            feeds.append(_feed("speaker_id", "np.array([0], dtype=np.int64)") + f"  # {_speaker(config)}")
        return "\n".join([_session("session", model), "# the noise and speed scales are fixed in this graph",
                          "audio = session.run(None, {", *feeds, "})[0].squeeze()"])
    if engine == "optispeech":
        _check(inputs, ("x", "x_lengths", "scales", "sids", "lids"), engine)
        params = _params(config, {"d_factor": 1.0, "p_factor": 1.0, "e_factor": 1.0})
        factors = [params[k] for k in ("d_factor", "p_factor", "e_factor")]
        feeds = [_feed("x", "np.array([ids], dtype=np.int64)"),
                 _feed("x_lengths", "np.array([len(ids)], dtype=np.int64)"),
                 _feed("scales", f"np.array({factors}, dtype=np.float32)") + "  # duration, pitch and energy"]
        if "sids" in inputs:
            feeds.append(_feed("sids", "np.array([0], dtype=np.int64)") + f"  # {_speaker(config)}")
        if "lids" in inputs:
            feeds.append(_feed("lids", "np.array([0], dtype=np.int64)"))
        return "\n".join([_session("session", model), "audio = session.run(None, {", *feeds, "})[0].squeeze()"])
    if engine == "styletts2":
        return _styletts2(config, files, model, inputs, pad)
    if engine == "glowtts":
        return _glowtts(config, model, inputs) + "\n\n" + _vocoder(files.get("vocoder"), _denoise(config, False))
    if engine == "matcha":
        return _matcha(config, files, model, inputs)
    if engine in ("mixertts", "fastpitch"):
        return _mixertts(config, model, inputs) + "\n\n" + _vocoder(files.get("vocoder"), _denoise(config, False))
    raise UnsupportedExample(f"no template for the {engine} engine")


def _vits(config: Dict[str, Any], model: str, inputs: Dict[str, int], engine: str) -> str:
    _check(inputs, ("input", "input_lengths", "scales", "sid", "langid", "x", "x_length", "noise_scale",
                    "length_scale", "noise_scale_w", "input_ids", "attention_mask"), engine)
    noise, length, noise_w = _scales(config)
    lines = [_session("session", model)]
    if "scales" in inputs or "noise_scale" in inputs:
        lines.append(f'noise_scale, length_scale, noise_w = {noise}, {length}, {noise_w}  # config.json "inference"')
    else:
        lines.append("# the noise and speed scales are fixed in this graph")
    lines.append("audio = session.run(None, {")
    if "input" in inputs:
        lines += [_feed("input", "np.array([ids], dtype=np.int64)"),
                  _feed("input_lengths", "np.array([len(ids)], dtype=np.int64)")]
    elif "x" in inputs:
        lines += [_feed("x", "np.array([ids], dtype=np.int64)"), _feed("x_length", "np.array([len(ids)], dtype=np.int64)")]
    else:
        lines.append(_feed("input_ids", "np.array([ids], dtype=np.int64)"))
        if "attention_mask" in inputs:
            lines.append(_feed("attention_mask", "np.ones((1, len(ids)), dtype=np.int64)"))
    if "scales" in inputs:
        scales = "[noise_scale, length_scale, noise_w]"
        lines.append(_feed("scales", f"np.array({f'[{scales}]' if inputs['scales'] >= 2 else scales}, dtype=np.float32)"))
    for name, value in (("noise_scale", "noise_scale"), ("length_scale", "length_scale"), ("noise_scale_w", "noise_w")):
        if name in inputs:
            lines.append(_feed(name, f"np.array([{value}], dtype=np.float32)"))
    if "sid" in inputs:
        lines.append(_feed("sid", "np.array([0], dtype=np.int64)") + f"  # {_speaker(config)}")
    if "langid" in inputs:
        lines.append(_feed("langid", "np.array([0], dtype=np.int64)") + "  # language 0")
    lines.append("})[0].squeeze()")
    return "\n".join(lines)


def _vits_prior_split(config: Dict[str, Any], files: Dict[str, Any]) -> str:
    noise, length, _ = _scales(config)
    return "\n".join([
        _session("duration", files.get("model", "duration.onnx")),
        _session("decode", files["decode"]),
        f"noise_scale, length_scale = {noise}, {length}",
        'm_p_exp, logs_p_exp, y_mask = duration.run(["m_p_exp", "logs_p_exp", "y_mask"], {',
        _feed("tokens", "np.array([ids], dtype=np.int64)"),
        _feed("lengths", "np.array([len(ids)], dtype=np.int64)"),
        _feed("length_scale", "np.array(length_scale, dtype=np.float32)"),
        "})",
        "# the prior noise is drawn outside the graph: a fixed seed gives the same audio each run",
        "zp_noise = np.random.default_rng(0).standard_normal(m_p_exp.shape).astype(np.float32)",
        'audio = decode.run(["waveform"], {',
        '    "m_p_exp": m_p_exp, "logs_p_exp": logs_p_exp, "y_mask": y_mask, "zp_noise": zp_noise,',
        _feed("noise_scale", "np.array(noise_scale, dtype=np.float32)"),
        "})[0].reshape(-1)"])


def _styletts2(config: Dict[str, Any], files: Dict[str, Any], model: str, inputs: Dict[str, int],
               pad: Optional[str]) -> str:
    _check(inputs, ("input_ids", "tokens", "attention_mask", "style", "speed"), "styletts2")
    lines = [_session("session", model)]
    if "style" in inputs:
        if not files.get("style"):
            raise UnsupportedExample("a StyleTTS2 graph with a style input, and no style file")
        source = f'"{files["style"]["repo"]}"' if files["style"].get("repo") else "repo"
        lines += [f'style = np.fromfile(hf_hub_download({source}, "{files["style"]["file"]}"), dtype=np.float32)'
                  ".reshape(-1, 256)",
                  "style_row = style[min(len(ids), len(style) - 1)][None]  # the row for this many ids"]
    lines.append(f'pad = id_map["{pad}"]' if pad else "pad = 0")
    if "style" in inputs:
        lines.append("padded = [pad] + ids + [pad] * (len(style) > 1)  # a pack of several style rows pads both ends")
    else:
        lines.append("padded = [pad] + ids")
    name = "input_ids" if "input_ids" in inputs else "tokens"
    lines += ["audio = session.run(None, {", _feed(name, "np.array([padded], dtype=np.int64)")]
    if "attention_mask" in inputs:
        lines.append(_feed("attention_mask", "np.ones((1, len(padded)), dtype=np.int32)"))
    if "style" in inputs:
        lines.append(_feed("style", "style_row"))
    lines += [_feed("speed", f'np.array([{_params(config, {"speed": 1.0})["speed"]}], dtype=np.float32)'),
              "})[0].squeeze()"]
    return "\n".join(lines)


def _glowtts(config: Dict[str, Any], model: str, inputs: Dict[str, int]) -> str:
    _check(inputs, ("input", "input_lengths", "scales", "sid"), "glowtts")
    noise, length, _ = _scales(config)
    lines = [_session("session", model), f"noise_scale, length_scale = {noise}, {length}",
             "mel = session.run(None, {", _feed("input", "np.array([ids], dtype=np.int64)"),
             _feed("input_lengths", "np.array([len(ids)], dtype=np.int64)"),
             _feed("scales", "np.array([noise_scale, length_scale], dtype=np.float32)")]
    if "sid" in inputs:
        lines.append(_feed("sid", "np.array([0], dtype=np.int64)") + f"  # {_speaker(config)}")
    lines += ["})[0]",
              "if mel.min() >= -0.5:",
              "    # a Larynx mel is normalised: undo it as Larynx does before its vocoder",
              "    mel = (np.clip(mel, -1.0, 1.0) + 1.0) * 100.0 / 2.0 - 100.0 + 20.0",
              "    mel = np.log(np.clip(np.power(10.0, mel), 1e-5, None))"]
    return "\n".join(lines)


def _matcha(config: Dict[str, Any], files: Dict[str, Any], model: str, inputs: Dict[str, int]) -> str:
    _check(inputs, ("x", "x_lengths", "scales", "spks"), "matcha")
    outputs = list(files.get("outputs") or [])
    signal = [name for name in outputs if name != "mel_lengths"]
    if len(outputs) != 2 or len(signal) != 1:
        raise UnsupportedExample(f"Matcha outputs {outputs}")
    feeds = [_feed("x", "np.array([ids], dtype=np.int64)"), _feed("x_lengths", "np.array([len(ids)], dtype=np.int64)"),
             _feed("scales", "np.array([temperature, length_scale], dtype=np.float32)")]
    if "spks" in inputs:
        feeds.append(_feed("spks", "np.array([0], dtype=np.int64)") + f"  # {_speaker(config)}")
    names = ", ".join({"mel_lengths": "mel_lengths", "mel": "mel"}.get(name, "audio") for name in outputs)
    params = _params(config, {"temperature": 0.667, "length_scale": 1.0})
    lines = [_session("session", model), f"temperature, length_scale = {params['temperature']}, {params['length_scale']}",
             f"{names} = session.run(None, {{", *feeds, "})"]
    if signal[0] != "mel":
        return "\n".join(lines + ["audio = audio.squeeze()  # this export holds its vocoder"])
    lines.append("mel = mel[..., :mel_lengths[0]]")
    return "\n".join(lines) + "\n\n" + _vocoder(files.get("vocoder"), _denoise(config, True))


def _denoise(config: Dict[str, Any], default: bool) -> bool:
    """Whether phoonnx asks the vocoder to denoise; only a Vocos vocoder does it."""
    return bool(_params(config, {}).get("denoise", default))


def _mixertts(config: Dict[str, Any], model: str, inputs: Dict[str, int]) -> str:
    _check(inputs, ("token_ids", "pace", "speaker", "emotion", "pitch_mul", "pitch_add"), "mixertts")
    values = {"pace": "np.array([1.0], dtype=np.float32)", "speaker": "np.array([0], dtype=np.int32)",
              "emotion": "np.array([0], dtype=np.int32)", "pitch_mul": "np.array([1.0], dtype=np.float32)",
              "pitch_add": "np.array([0.0], dtype=np.float32)"}
    feeds = [_feed("token_ids", "np.array([ids], dtype=np.int64)")]
    feeds += [_feed(name, value) + (f"  # {_speaker(config)}" if name == "speaker" else "")
              for name, value in values.items() if name in inputs]
    return "\n".join([_session("session", model), "mel = session.run(None, {", *feeds, "})[0]"])


def _vocoder(vocoder: Optional[Dict[str, Any]], denoise: bool) -> str:
    if not vocoder:
        raise UnsupportedExample("a spectrogram model with no vocoder")
    kind = vocoder.get("type")
    if kind not in VOCODER_TYPES:
        raise UnsupportedExample(f"no template for the {kind} vocoder")
    names = list(_inputs(vocoder))
    config = vocoder.get("config") or {}
    lines = ["# the vocoder turns the spectrogram into audio", _session("vocoder", vocoder["file"], vocoder.get("repo"))]
    if config.get("stats_norm") and config.get("mel_mean") is not None:
        config_file = vocoder["file"].rsplit("/", 1)[0] + "/vocoder.json" if "/" in vocoder["file"] else "vocoder.json"
        source = f'"{vocoder["repo"]}"' if vocoder.get("repo") else "repo"
        lines += [f'vocoder_config = json.load(open(hf_hub_download({source}, "{config_file}")))',
                  'mean, std = (np.array(vocoder_config[k], dtype=np.float32)[None, :, None] for k in ("mel_mean", "mel_std"))',
                  "mel = (mel - mean) / std  # this vocoder reads a standardised mel"]
    if kind != "vocos":
        # a vocoder that gives the waveform does not denoise: phoonnx feeds 0 to each input after the mel
        feeds = ", ".join([f'"{names[0]}": mel'] + [f'"{n}": np.array([0.0], dtype=np.float32)' for n in names[1:]])
        return "\n".join(lines + [f"audio = vocoder.run(None, {{{feeds}}})[0].squeeze()"])
    init = (config.get("feature_extractor") or {}).get("init_args") or {}
    n_fft, hop = config.get("n_fft", init.get("n_fft", 1024)), config.get("hop_length", init.get("hop_length", 256))
    lines += [f'mag, x, y = vocoder.run(None, {{"{names[0]}": mel}})',
              "spectrum = (mag * (x + 1j * y))[0]  # Vocos gives the STFT, not the waveform"]
    if denoise:
        lines += ["# take away a little of the spectrum the vocoder gives for silence",
                  f'mag, x, y = vocoder.run(None, {{"{names[0]}": np.zeros_like(mel)}})',
                  "bias = np.abs(mag * (x + 1j * y))[0]",
                  "spectrum = np.clip(np.abs(spectrum) - bias * 0.0025, 0.0, None) * np.exp(1j * np.angle(spectrum))"]
    lines += [f"n_fft, hop = {n_fft}, {hop}",
              "window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)",
              "frames = np.fft.irfft(spectrum, n=n_fft, axis=0) * window[:, None]",
              "audio = np.zeros((frames.shape[1] - 1) * hop + n_fft)",
              "envelope = np.zeros_like(audio)",
              "for t in range(frames.shape[1]):  # overlap-add inverse STFT",
              "    audio[t * hop:t * hop + n_fft] += frames[:, t]",
              "    envelope[t * hop:t * hop + n_fft] += window ** 2",
              "audio = (audio / np.maximum(envelope, 1e-11))[n_fft // 2:-(n_fft // 2)]"]
    return "\n".join(lines)


def _wav(config: Dict[str, Any]) -> str:
    rate = (config.get("audio") or {}).get("sample_rate") or config.get("sample_rate")
    if not rate:
        raise UnsupportedExample("the config gives no sample rate")
    return "\n".join([
        'with wave.open("output.wav", "wb") as wav:',
        "    wav.setnchannels(1)",
        "    wav.setsampwidth(2)",
        f"    wav.setframerate({rate})  # the voice's sample rate",
        "    wav.writeframes((audio / np.abs(audio).max() * 32767).astype(np.int16).tobytes())"])


_RAPIDO_IO = {"model": ({"x", "condition", "t", "guidance"}, {"velocity"}),
              "text_encoder_path": ({"ids"}, {"features", "durations"}),
              "aligner_path": ({"features", "durations", "word", "word_start", "frame_word", "frame_pos", "speaker"},
                               {"condition"}),
              "decoder_path": ({"latents"}, {"audio"})}


def _rapido(config: Dict[str, Any], files: Dict[str, Any], *, repo_id: str, text: Optional[str]) -> str:
    for key, (inputs, outputs) in _RAPIDO_IO.items():
        if key not in files:
            raise UnsupportedExample(f"a Rápido voice with no {key}")
        if "inputs" in files[key] and (set(files[key]["inputs"]) != inputs or set(files[key]["outputs"]) != outputs):
            raise UnsupportedExample(f"Rápido {key} names {sorted(files[key]['inputs'])}")
    if "config" not in files:
        raise UnsupportedExample("a Rápido voice with no config.json")
    sentence = json.dumps(text, ensure_ascii=False) if text else f'"..."  # a sentence in {config.get("lang")}'
    return "\n".join([
        "# pip install onnxruntime numpy huggingface_hub",
        "import json",
        "import unicodedata",
        "import wave",
        "",
        "import numpy as np",
        "import onnxruntime",
        "from huggingface_hub import hf_hub_download",
        "",
        f'repo = "{repo_id}"',
        f"config = json.load(open({_hub(files['config'], repo_id)}, encoding=\"utf-8\"))",
        "text_encoder, aligner, model, decoder = (onnxruntime.InferenceSession(hf_hub_download(repo, f)) for f in (",
        f'    "{files["text_encoder_path"]["file"]}", "{files["aligner_path"]["file"]}", '
        f'"{files["model"]["file"]}", "{files["decoder_path"]["file"]}"))',
        "",
        f"text = {sentence}",
        "id_map = config[\"phoneme_id_map\"]  # one id per letter, space and punctuation mark",
        'ids = [id_map[c] for c in unicodedata.normalize("NFC", " ".join(text.lower().split())) if c in id_map]',
        "features, durations = text_encoder.run(None, {\"ids\": np.array([ids], dtype=np.int64)})",
        "",
        "# each letter belongs to a word; a space or a clause mark ends the word",
        'is_boundary = np.isin(ids, [id_map[c] for c in " .?!,:;" if c in id_map])',
        "starts = ~is_boundary & np.concatenate([[True], is_boundary[:-1]])",
        "word = np.maximum(np.cumsum(starts) - 1, 0)",
        "word_start = np.concatenate([[0], np.flatnonzero(starts)[1:]])[word]",
        "totals = np.zeros(word.max() + 1, dtype=np.float32)",
        "np.add.at(totals, word, durations[0])",
        "counts = np.clip(np.round(totals), 1, 250).astype(np.int64)  # frames per word, at 25 frames a second",
        "frame_word = np.repeat(np.arange(len(counts)), counts)",
        "frame_pos = (np.arange(len(frame_word)) - (np.cumsum(counts) - counts)[frame_word]) / counts[frame_word]",
        "condition = aligner.run(None, {",
        '    "features": features, "durations": durations, "word": word[None], "word_start": word_start[None],',
        '    "frame_word": frame_word[None], "frame_pos": frame_pos[None].astype(np.float32),',
        f'    "speaker": np.array([0], dtype=np.int64),  # speaker 0 of {config.get("num_speakers", 42)}'
        ' in config.json "speaker_id_map"',
        "})[0]",
        "",
        "# flow matching: 8 midpoint steps from noise (t=0) to the latents (t=1)",
        "steps, guidance, seed = 8, 4.0, 0",
        "mean, std = (np.array(config[\"engine_params\"][k], dtype=np.float32) for k in (\"latent_mean\", \"latent_std\"))",
        "x = np.random.default_rng(seed).standard_normal((1, len(frame_word), len(mean))).astype(np.float32)",
        "",
        "",
        "def velocity(x, t):",
        "    return model.run(None, {\"x\": x, \"condition\": condition, \"t\": np.array([t], dtype=np.float32),",
        "                            \"guidance\": np.array([guidance], dtype=np.float32)})[0]",
        "",
        "",
        "times = np.linspace(0.0, 1.0, steps + 1).astype(np.float32)",
        "for a, b in zip(times[:-1], times[1:]):",
        "    x = x + (b - a) * velocity(x + (b - a) / 2 * velocity(x, a), (a + b) / 2)",
        "",
        "latents = (x[0] * std + mean).T[None].astype(np.float32)",
        "audio = decoder.run(None, {\"latents\": latents})[0].reshape(-1)",
        "",
        _write_wav('config["audio"]["sample_rate"]')])


_SUPERTONIC_IO = {"duration_predictor_path": ({"text_ids", "style_dp", "text_mask"}, None),
                  "text_encoder_path": ({"text_ids", "style_ttl", "text_mask"}, None),
                  "model": ({"noisy_latent", "text_emb", "style_ttl", "text_mask", "latent_mask", "current_step",
                             "total_step"}, None),
                  "vocoder_path": ({"latent"}, None)}
_SUPERTONIC_LANGS = ("en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu",
                     "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na")


def _supertonic(config: Dict[str, Any], files: Dict[str, Any], *, repo_id: str, text: Optional[str]) -> str:
    for key, (inputs, _) in _SUPERTONIC_IO.items():
        if key not in files:
            raise UnsupportedExample(f"a SuperTonic voice with no {key}")
        if "inputs" in files[key] and set(files[key]["inputs"]) != inputs:
            raise UnsupportedExample(f"SuperTonic {key} names {sorted(files[key]['inputs'])}")
    for key in ("tts_config_path", "unicode_indexer_path", "style_path"):
        if key not in files:
            raise UnsupportedExample(f"a SuperTonic voice with no {key}")
    lang = (config.get("lang") or "en").split("-")[0].lower()
    if lang not in _SUPERTONIC_LANGS:
        raise UnsupportedExample(f"SuperTonic was not trained on {lang}")
    sentence = json.dumps(text, ensure_ascii=False) if text else f'"..."  # a sentence in {lang}'
    graphs = ",\n    ".join(f'"{files[k]["file"]}"' for k in ("duration_predictor_path", "text_encoder_path", "model",
                                                          "vocoder_path"))
    return "\n".join([
        "# pip install onnxruntime numpy huggingface_hub",
        "import json",
        "import re",
        "import unicodedata",
        "import wave",
        "",
        "import numpy as np",
        "import onnxruntime",
        "from huggingface_hub import hf_hub_download",
        "",
        f'repo = "{repo_id}"',
        f"tts = json.load(open({_hub(files['tts_config_path'], repo_id)}))",
        f"indexer = json.load(open({_hub(files['unicode_indexer_path'], repo_id)}))"
        "  # the token id of each codepoint, -1 for none",
        "# the speaker is a fixed style; voice_styles/ holds the others (F1-F5, M1-M5)",
        f"style = json.load(open({_hub(files['style_path'], repo_id)}))",
        "style_ttl, style_dp = (np.array(style[k][\"data\"], dtype=np.float32).reshape(1, *style[k][\"dims\"][1:])",
        "                       for k in (\"style_ttl\", \"style_dp\"))",
        "duration_predictor, text_encoder, vector_estimator, vocoder = (",
        "    onnxruntime.InferenceSession(hf_hub_download(repo, f)) for f in (",
        f"    {graphs}))",
        "",
        f"text = {sentence}",
        f'lang = "{lang}"',
        "# phoonnx also drops emoji and maps dashes and curly quotes; a plain sentence has none",
        'text = re.sub(r"\\s+", " ", unicodedata.normalize("NFKD", text)).strip()',
        "if not re.search(r\"[.!?;:,'\\\")\\]}…。』】〉》›»]$\", text):",
        '    text += "."',
        'ids = [indexer[ord(c)] for c in f"<{lang}>{text}</{lang}>" if ord(c) < len(indexer) and indexer[ord(c)] >= 0]',
        "text_ids, text_mask = np.array([ids], dtype=np.int64), np.ones((1, 1, len(ids)), dtype=np.float32)",
        "",
        "speed, total_step, seed = 1.05, 8, 0",
        'duration = duration_predictor.run(None, {"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask})[0]',
        "seconds = float((duration / speed).reshape(-1)[0])",
        'text_emb = text_encoder.run(None, {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask})[0]',
        "",
        "# flow matching: start from noise as long as the predicted duration, then 8 steps",
        'rate, chunk = tts["ae"]["sample_rate"], tts["ae"]["base_chunk_size"] * tts["ttl"]["chunk_compress_factor"]',
        "frames = max((max(int(seconds * rate), 1) + chunk - 1) // chunk, 1)",
        'channels = tts["ttl"]["latent_dim"] * tts["ttl"]["chunk_compress_factor"]',
        "latent = np.random.default_rng(seed).standard_normal((1, channels, frames)).astype(np.float32)",
        "for step in range(total_step):",
        "    latent = vector_estimator.run(None, {",
        '        "noisy_latent": latent, "text_emb": text_emb, "style_ttl": style_ttl, "text_mask": text_mask,',
        '        "latent_mask": np.ones((1, 1, frames), dtype=np.float32),',
        '        "current_step": np.array([step], dtype=np.float32), "total_step": np.array([total_step], dtype=np.float32),',
        "    })[0]",
        'audio = vocoder.run(None, {"latent": latent})[0].reshape(-1)',
        "",
        _write_wav("rate")])


_F5_IO = {"preprocess_path": ({"audio", "text_ids", "max_duration"},
                              ["noise", "rope_cos_q", "rope_sin_q", "rope_cos_k", "rope_sin_k", "cat_mel_text",
                               "cat_mel_text_drop", "ref_signal_len"]),
          "decode_path": ({"denoised", "ref_signal_len"}, None)}


def _f5tts(config: Dict[str, Any], files: Dict[str, Any], *, repo_id: str, text: Optional[str]) -> str:
    for key in ("model", "config", "preprocess_path", "decode_path"):
        if key not in files:
            raise UnsupportedExample(f"an F5-TTS voice with no {key}")
    for key, (inputs, outputs) in _F5_IO.items():
        if "inputs" in files[key] and (set(files[key]["inputs"]) != inputs
                                       or (outputs and list(files[key]["outputs"]) != outputs)):
            raise UnsupportedExample(f"F5-TTS {key} names {sorted(files[key]['inputs'])}")
    if "inputs" in files["model"] and len(files["model"]["inputs"]) != 8:
        raise UnsupportedExample(f"an F5-TTS transformer with inputs {sorted(files['model']['inputs'])}")
    lang = config.get("lang") or "the voice language"
    sentence = json.dumps(text, ensure_ascii=False) if text else f'"..."  # a sentence in {lang}'
    graphs = ", ".join(f'"{files[k]["file"]}"' for k in ("preprocess_path", "model", "decode_path"))
    return "\n".join([
        "# pip install onnxruntime numpy huggingface_hub",
        "import json",
        "import unicodedata",
        "import wave",
        "",
        "import numpy as np",
        "import onnxruntime",
        "from huggingface_hub import hf_hub_download",
        "",
        f'repo = "{repo_id}"',
        f"config = json.load(open({_hub(files['config'], repo_id)}, encoding=\"utf-8\"))",
        "preprocess, transformer, decode = (onnxruntime.InferenceSession(hf_hub_download(repo, f)) for f in (",
        f"    {graphs}))",
        "",
        "# F5-TTS speaks in the voice of a reference clip: put a few seconds of mono 16-bit 24 kHz",
        "# speech in reference.wav, and the words it says in reference.txt",
        'reference_text = open("reference.txt", encoding="utf-8").read()',
        'with wave.open("reference.wav") as wav:',
        '    assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 24000), "use mono 16-bit 24 kHz"',
        "    reference = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)",
        f"text = {sentence}",
        "",
        'id_map = config["phoneme_id_map"]',
        'compounds = sorted((s for s in id_map if len(s) > 1), key=len, reverse=True) if config.get("fold_compounds", True) else []',
        "",
        "",
        "def to_ids(words):  # lower case, one id per character, as phoonnx reads this voice",
        '    if not config.get("keep_punctuation"):  # the config does not keep punctuation',
        '        words = "".join(c for c in words if not unicodedata.category(c).startswith("P"))',
        '    chars, ids, i = " ".join(words.lower().split()), [], 0',
        "    while i < len(chars):  # a run of characters the map holds as one symbol (\"er\") is read as that symbol",
        "        n = next((len(s) for s in compounds if chars.startswith(s, i)), 1)",
        "        if chars[i:i + n] in id_map:",
        "            ids.append(id_map[chars[i:i + n]])",
        "        i += n",
        "    return ids",
        "",
        "",
        "reference_ids, text_ids = to_ids(reference_text), to_ids(text)",
        'params = config.get("engine_params") or {}',
        'dialect = params.get("dialect")  # Habibi Unified reads a dialect tag',
        "if dialect:",
        '    tag = "⓪①②③④⑤⑥⑦"["UNK MSA SAU UAE ALG IRQ EGY MAR".split().index(dialect)]',
        '    text_ids = [id_map[tag], id_map["〈"]] + text_ids + [id_map["〉"]]',
        "ids = reference_ids + text_ids",
        "",
        "# the length to generate: the reference's frames, and as many again per character as the reference took",
        "frames = len(reference) // 256 + 1  # 256 samples per frame",
        'speed, nfe = params.get("speed", 1.0), params.get("nfe", 32)  # nfe: flow-matching steps',
        "max_duration = frames + int(frames / len(reference_ids) * len(text_ids) / speed)",
        "audio_in = reference if \"int16\" in preprocess.get_inputs()[0].type else reference / np.float32(32768)",
        "noise, *rope, cat_mel_text, cat_mel_text_drop, ref_signal_len = preprocess.run(None, {",
        '    "audio": audio_in.reshape(1, 1, -1), "text_ids": np.array([ids], dtype=np.int32),',
        '    "max_duration": np.array([max_duration], dtype=np.int64)})',
        "names = [i.name for i in transformer.get_inputs()]  # the step input may be named time_step.1",
        "time_step = np.array([0], dtype=np.int32)",
        "for _ in range(nfe - 1):  # each call is one Euler step and returns the next time step",
        "    noise, time_step = transformer.run(None, dict(zip(names, [noise, *rope, cat_mel_text, cat_mel_text_drop, time_step])))",
        'audio = decode.run(None, {"denoised": noise, "ref_signal_len": ref_signal_len})[0].reshape(-1)',
        "if audio.dtype == np.int16:",
        "    audio = audio / np.float32(32768)",
        "",
        _write_wav('config["audio"]["sample_rate"]')])


_POCKET_KEYS = ("model", "bundle_path", "tokenizer_path", "text_conditioner_path", "flow_lm_flow_path",
                "mimi_decoder_path", "voice_state_path")
_POCKET_IO = {"text_conditioner_path": {"token_ids"}, "flow_lm_flow_path": {"c", "s", "t", "x"},
              "mimi_decoder_path": {"latent"}, "model": {"sequence", "text_embeddings"}}
_POCKET_LANGS = ("en", "fr", "de", "it", "pt", "es")


def _pockettts(config: Dict[str, Any], files: Dict[str, Any], *, repo_id: str, text: Optional[str]) -> str:
    for key in _POCKET_KEYS:
        if key not in files:
            raise UnsupportedExample(f"a Pocket TTS voice with no {key}")
    for key, needed in _POCKET_IO.items():
        if "inputs" in files[key] and not needed <= {n for n in files[key]["inputs"] if not n.startswith("state_")}:
            raise UnsupportedExample(f"Pocket TTS {key} names {sorted(files[key]['inputs'])}")
    lang = (config.get("lang") or "en").split("-")[0]
    if lang not in _POCKET_LANGS:
        raise UnsupportedExample(f"Pocket TTS was not trained on {lang}")
    sentence = json.dumps(text, ensure_ascii=False) if text else f'"..."  # a sentence in {lang}'
    graphs = ",\n    ".join(f'"{files[k]["file"]}"' for k in ("text_conditioner_path", "model", "flow_lm_flow_path",
                                                              "mimi_decoder_path"))
    return "\n".join([
        "# pip install onnxruntime numpy huggingface_hub tokenizers",
        "import json",
        "import math",
        "import struct",
        "import wave",
        "",
        "import numpy as np",
        "import onnxruntime",
        "from huggingface_hub import hf_hub_download",
        "from tokenizers import Tokenizer, models",
        "",
        f'repo = "{repo_id}"',
        f"bundle = json.load(open({_hub(files['bundle_path'], repo_id)}))  # sizes and the graphs' state tensors",
        "text_conditioner, flow_lm_main, flow_lm_flow, mimi_decoder = (",
        "    onnxruntime.InferenceSession(hf_hub_download(repo, f)) for f in (",
        f"    {graphs}))",
        "",
        "",
        "def fields(data):  # the (number, value) fields of a protobuf message",
        "    i = 0",
        "    while i < len(data):",
        "        key, i = varint(data, i)",
        "        if key & 7 == 0:",
        "            value, i = varint(data, i)",
        "        elif key & 7 == 5:",
        '            value, i = struct.unpack("<f", data[i:i + 4])[0], i + 4',
        "        else:",
        "            size, i = varint(data, i)",
        "            value, i = data[i:i + size], i + size",
        "        yield key >> 3, value",
        "",
        "",
        "def varint(data, i):",
        "    value = shift = 0",
        "    while data[i] & 128:",
        "        value, shift, i = value | (data[i] & 127) << shift, shift + 7, i + 1",
        "    return value | data[i] << shift, i + 1",
        "",
        "",
        "# tokenizer.model is a SentencePiece unigram model: its pieces and scores give the same ids",
        f"pieces = [dict(fields(p)) for n, p in fields(open({_hub(files['tokenizer_path'], repo_id)}, \"rb\").read()) if n == 1]",
        "unk = next(i for i, p in enumerate(pieces) if p.get(3) == 2)",
        "tokenizer = Tokenizer(models.Unigram([(p[1].decode(), p.get(2, 0.0)) for p in pieces], unk))",
        "tokenizer = tokenizer.compile() if hasattr(tokenizer, \"compile\") else tokenizer  # tokenizers 1.x encodes once compiled",
        "",
        f"text = {sentence}",
        "# the model was trained on sentences that start with a capital and end with a mark",
        'text = text.strip().replace("\\n", " ").replace("  ", " ")',
        'if bundle.get("remove_semicolons"):',
        '    text = text.replace(";", ",")',
        "frames_after_eos = 3 if len(text.split()) <= 4 else 1",
        "text = text[0].upper() + text[1:] + (\".\" if text[-1].isalnum() else \"\")",
        'if bundle.get("pad_with_spaces_for_short_inputs") and len(text.split()) < 5:',
        '    text = " " * 8 + text',
        'ids = tokenizer.encode(("▁" + text).replace(" ", "▁")).ids  # SentencePiece writes a space as "▁", one in front',
        "",
        "# the voice is the main transformer's state after it heard the speaker; voices/ holds the other speakers.",
        "# To clone a clip instead, run it through mimi_encoder.onnx, put bos_before_voice.npy in front, and",
        "# feed that to flow_lm_main as text_embeddings with an empty sequence, from the start state.",
        f"data = open({_hub(files['voice_state_path'], repo_id)}, \"rb\").read()",
        'size = int.from_bytes(data[:8], "little")',
        "saved = {name: np.frombuffer(data[8 + size + a:8 + size + b], dtype={\"F32\": np.float32, \"I64\": np.int64,",
        '                                                                  "BOOL": np.bool_}[t["dtype"]]).reshape(t["shape"])',
        '         for name, t in json.loads(data[8:8 + size]).items() if name != "__metadata__" for a, b in [t["data_offsets"]]}',
        'dtypes = {"float32": np.float32, "float16": np.float16, "int64": np.int64, "bool": np.bool_}',
        "",
        "",
        "def start(entry):  # a state tensor at the start of a stream",
        '    return np.full(entry["shape"], {"nan": np.nan, "ones": 1}.get(entry["fill"], 0), dtype=dtypes[entry["dtype"]])',
        "",
        "",
        "voice_state = {}",
        'for entry in bundle["flow_lm_state_manifest"]:',
        "    tensor = start(entry)",
        "    source = saved.get(f\"{entry['module']}/{entry['key']}\")",
        '    if source is None and entry["key"] == "step":  # a saved voice names its step count "offset"',
        "        source = saved.get(f\"{entry['module']}/offset\")",
        "    if source is not None:  # a saved cache shorter than the graph's goes at the start",
        "        source = source.astype(tensor.dtype).reshape(tensor.shape if source.size == tensor.size else source.shape)",
        "        tensor[tuple(slice(0, n) for n in source.shape)] = source[tuple(slice(0, n) for n in tensor.shape)]",
        '    voice_state[entry["input_name"]] = tensor',
        "",
        "",
        "def step(session, manifest, offset, **feeds):  # run a graph and carry its state to the next call",
        "    outputs = session.run(None, {**feeds, **state[manifest]})",
        "    for entry in bundle[manifest]:",
        '        state[manifest][entry["input_name"]] = outputs[offset + entry["index"]]',
        "    return outputs",
        "",
        "",
        "state = {\"flow_lm_state_manifest\": voice_state,",
        "         \"mimi_state_manifest\": {e[\"input_name\"]: start(e) for e in bundle[\"mimi_state_manifest\"]}}",
        "temperature, steps, eos_threshold, seed = 0.7, 1, -4.0, 0",
        "dt = 1 / steps",
        "rng = np.random.default_rng(seed)",
        'latent_dim = bundle["latent_dim"]',
        'embeddings = text_conditioner.run(None, {"token_ids": np.array([ids], dtype=np.int64)})[0]',
        'step(flow_lm_main, "flow_lm_state_manifest", 2, sequence=np.zeros((1, 0, latent_dim), dtype=np.float32),',
        "     text_embeddings=embeddings.reshape(1, -1, embeddings.shape[-1]))",
        "after_eos = bundle.get(\"model_recommended_frames_after_eos\")",
        "after_eos = frames_after_eos + 2 if after_eos is None else after_eos  # frames to make once the end logit fires",
        "frame, codes, eos = np.full((1, 1, latent_dim), np.nan, dtype=np.float32), [], None",
        'for n in range(math.ceil((len(ids) / 3 + 2) * bundle["frame_rate"])):  # about 3 tokens a second, and 2 s more',
        '    conditioning, eos_logit = step(flow_lm_main, "flow_lm_state_manifest", 2, sequence=frame,',
        '                                   text_embeddings=np.zeros((1, 0, bundle["conditioning_dim"]), dtype=np.float32))[:2]',
        "    if eos is None and float(eos_logit.squeeze()) > eos_threshold:",
        "        eos = n",
        "    if eos is not None and n >= eos + after_eos:",
        "        break",
        "    x = rng.normal(0.0, math.sqrt(temperature), (1, latent_dim)).astype(np.float32)",
        "    for j in range(steps):  # Euler steps of the flow from noise to the next latent frame",
        '        x = x + flow_lm_flow.run(None, {"c": conditioning, "s": np.array([[j / steps]], dtype=np.float32),',
        '                                        "t": np.array([[j / steps + dt]], dtype=np.float32), "x": x})[0] * dt',
        "    frame = x.reshape(1, 1, latent_dim)",
        "    codes.append(frame)",
        "",
        "codes = np.concatenate(codes, axis=1)",
        'audio = np.concatenate([step(mimi_decoder, "mimi_state_manifest", 1, latent=codes[:, i:i + 15])[0].reshape(-1)',
        "                        for i in range(0, codes.shape[1], 15)])",
        "",
        _write_wav('bundle["sample_rate"]')])


OUTETTS_LM_INPUTS = {"input_ids", "attention_mask", "position_ids"}


def _outetts(config: Dict[str, Any], files: Dict[str, Any], *, repo_id: str, text: Optional[str]) -> str:
    for key in ("model", "tokenizer_path", "speakers_path", "codec_decoder_path"):
        if key not in files:
            raise UnsupportedExample(f"an OuteTTS voice with no {key}")
    model, codec = files["model"], files["codec_decoder_path"]
    if "inputs" in model and not OUTETTS_LM_INPUTS <= set(model["inputs"]):
        raise UnsupportedExample(f"OuteTTS LM inputs {sorted(model['inputs'])}")
    if "outputs" in model and "logits" not in model["outputs"]:
        raise UnsupportedExample(f"OuteTTS LM outputs {sorted(model['outputs'])}")
    if "inputs" in codec and list(codec["inputs"]) != ["audio_codes"]:
        raise UnsupportedExample(f"DAC decoder inputs {sorted(codec['inputs'])}")
    sentence = json.dumps(text, ensure_ascii=False) if text else f'"..."  # a sentence in {config.get("lang") or "the voice language"}'
    data = dict(model, file=model["file"] + "_data")
    return f'''# pip install onnxruntime numpy huggingface_hub tokenizers
import json
import unicodedata
import wave

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

repo = "{repo_id}"
tokenizer = Tokenizer.from_file({_hub(files["tokenizer_path"], repo_id)})
# The speaker profile: the words of a short clip with their DAC codes. The model goes on in that voice.
# To use your own voice, build a profile in the same format with the outetts package (Whisper and the DAC encoder).
speaker = json.load(open({_hub(files["speakers_path"], repo_id)}, encoding="utf-8"))

text = {sentence}
text = " ".join(unicodedata.normalize("NFKC", text).replace('"', "").split())  # phoonnx also folds quotes and dashes
# The profile's audio is the start of the utterance, so its transcript comes first.
# (A profile in Chinese or Japanese joins with "。" instead.)
joiner = " " if speaker["text"].strip()[-1] in ".?!" else ". "
words = [dict(w) for w in speaker["words"]]
words[-1]["word"] += joiner.strip()
lines = [f"<|word_start|>{{w['word']}}<|features|><|t_{{w['duration']:.2f}}|>"
         + "".join(f"<|{{k}}_{{(w.get('features') or {{}}).get(k, 0)}}|>" for k in ("energy", "spectral_centroid", "pitch"))
         + "<|code|>" + "".join(f"<|c1_{{a}}|><|c2_{{b}}|>" for a, b in zip(w["c1"], w["c2"])) + "<|word_end|>"
         for w in words]
prompt = (f"<|im_start|>\\n<|text_start|>{{speaker['text'].strip()}}{{joiner}}{{text}}<|text_end|>\\n<|audio_start|>\\n"
          + "\\n".join(lines) + "\\n<|word_start|>")
ids = tokenizer.encode(prompt, add_special_tokens=False).ids

{_hub(data, repo_id)}  # the weights, beside the graph
session = onnxruntime.InferenceSession({_hub(model, repo_id)})
cache = {{i.name: np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32)
         for i in session.get_inputs() if i.name.startswith("past_key_values.")}}
names = [o.name for o in session.get_outputs()]
stop = {{tokenizer.token_to_id("<|audio_end|>"), tokenizer.token_to_id("<|im_end|>")}}


def softmax(x):
    e = np.exp(np.asarray(x, np.float64) - np.max(x))
    return e / e.sum()


temperature, top_k, top_p, min_p, repetition_penalty = 0.4, 40, 0.9, 0.05, 1.1  # as phoonnx and upstream
rng = np.random.default_rng()
history, generated, step_ids = list(ids), [], ids
for _ in range(4096):  # about 44 s of speech
    past = len(history) - len(step_ids)
    out = dict(zip(names, session.run(None, {{
        "input_ids": np.array([step_ids], np.int64),
        "attention_mask": np.ones((1, len(history)), np.int64),
        "position_ids": np.arange(past, len(history), dtype=np.int64)[None],
        **cache,
    }})))
    cache = {{name: out[name.replace("past_key_values", "present")] for name in cache}}
    logits = np.array(out["logits"][0, -1], np.float32)
    seen = np.unique(history[-64:])  # the repetition penalty looks back 64 tokens
    logits[seen] = np.where(logits[seen] <= 0, logits[seen] * repetition_penalty, logits[seen] / repetition_penalty)
    if temperature > 0:  # temperature, then top-k, top-p and min-p, in Hugging Face's order
        logits = logits / temperature
        logits[logits < np.partition(logits, -top_k)[-top_k]] = -np.inf
        order = np.argsort(logits)[::-1]
        drop = np.ones(len(order), bool)
        drop[1:] = np.cumsum(softmax(logits[order]))[:-1] >= top_p
        drop[0] = False
        logits[order[drop]] = -np.inf
        probs = softmax(logits)
        logits[probs < min_p * probs.max()] = -np.inf
        token = int(rng.choice(len(logits), p=softmax(logits)))
    else:
        token = int(np.argmax(logits))
    if token in stop:
        break
    generated.append(token)
    history.append(token)
    step_ids = [token]

# each frame is a pair of tokens, one per DAC codebook
c1 = [tokenizer.id_to_token(t) for t in generated if tokenizer.id_to_token(t).startswith("<|c1_")]
c2 = [tokenizer.id_to_token(t) for t in generated if tokenizer.id_to_token(t).startswith("<|c2_")]
n = min(len(c1), len(c2))
codes = np.array([[[int(t[5:-2]) for t in c1[:n]], [int(t[5:-2]) for t in c2[:n]]]], np.int64)
codec = onnxruntime.InferenceSession({_hub(codec, repo_id)})
audio = codec.run(None, {{"audio_codes": codes}})[0].reshape(-1)
fade = min(360, len(audio) // 2)  # 15 ms in and out
audio[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
audio[len(audio) - fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
audio = audio * 0.9 / np.abs(audio).max()  # phoonnx sets the level to -18 LUFS instead

{_write_wav("24000")}'''


NEUTTS_LM_INPUTS = {"input_ids", "attention_mask", "position_ids"}


def _neutts(config: Dict[str, Any], files: Dict[str, Any], *, repo_id: str, text: Optional[str]) -> str:
    for key in ("model", "lm_weights_path", "tokenizer_path", "voices_path", "codec_decoder_path"):
        if key not in files:
            raise UnsupportedExample(f"a NeuTTS voice with no {key}")
    if not (config.get("lang") or "").startswith("tw"):
        # phoonnx reads every NeuTTS voice with espeak-ng's lfn voice, which only Akiti-TTS (Twi) was trained on
        raise UnsupportedExample(f"a NeuTTS voice in {config.get('lang')!r}: the lfn espeak voice fits Twi only")
    model, codec = files["model"], files["codec_decoder_path"]
    if "inputs" in model and not NEUTTS_LM_INPUTS <= set(model["inputs"]):
        raise UnsupportedExample(f"NeuTTS LM inputs {sorted(model['inputs'])}")
    if "outputs" in model and "logits" not in model["outputs"]:
        raise UnsupportedExample(f"NeuTTS LM outputs {sorted(model['outputs'])}")
    if "inputs" in codec and list(codec["inputs"]) != ["codes"]:
        raise UnsupportedExample(f"NeuCodec decoder inputs {sorted(codec['inputs'])}")
    voice = (config.get("engine_options") or {}).get("voice")
    preset = f'voices["presets"]["{voice}"]' if voice else 'voices["presets"][voices["default_voice"]]'
    sentence = json.dumps(text, ensure_ascii=False) if text else f'"..."  # a sentence in {config.get("lang")}'
    return f'''# pip install onnxruntime numpy huggingface_hub tokenizers "scriptconv[espeak]"  # and the espeak-ng program
import json
import re
import wave

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from scriptconv.phonemizers import get_phonemizer
from tokenizers import Tokenizer

repo = "{repo_id}"
tokenizer = Tokenizer.from_file({_hub(files["tokenizer_path"], repo_id)})
# A preset is the transcript of a short clip and the NeuCodec codes of that clip. The model goes on in its voice.
# For your own voice, add a preset with your clip's transcript and its NeuCodec codes (neucodec package).
voices = json.load(open({_hub(files["voices_path"], repo_id)}, encoding="utf-8"))
preset = {preset}

phonemizer = get_phonemizer("espeak")
marks = '[;:,.!?¡¿—…"«»“”]+'


def phonemize(text):
    """espeak-ng with its lfn voice, as in training; espeak drops punctuation, so it goes back in."""
    phones = ""
    for part in re.split(f"({{marks}})", text):
        if re.fullmatch(marks, part):
            phones += part
        elif part.strip() and (said := phonemizer.phonemize_string(part.strip(), "lfn").strip()):
            phones += (" " if phones else "") + said
    return phones


text = {sentence}
reference = "".join(f"<|speech_{{c}}|>" for c in preset["codes"][:200])
prompt = (f"<|TEXT_PROMPT_START|>{{phonemize(preset['text'])}} {{phonemize(text)}}<|TEXT_PROMPT_END|>"
          f"<|SPEECH_GENERATION_START|>{{reference}}")
ids = tokenizer.encode(prompt, add_special_tokens=False).ids

{_hub(files["lm_weights_path"], repo_id)}  # the weights, beside the graph
session = onnxruntime.InferenceSession({_hub(model, repo_id)})
cache = {{i.name: np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32)
         for i in session.get_inputs() if i.name.startswith("past_")}}
names = [o.name for o in session.get_outputs()]
stop = {{tokenizer.token_to_id("<|SPEECH_GENERATION_END|>"), tokenizer.token_to_id("<|endoftext|>")}}


def softmax(x):
    e = np.exp(np.asarray(x, np.float64) - np.max(x))
    return e / e.sum()


temperature, top_k, top_p, repetition_penalty = 0.4, 50, 0.8, 1.3  # as phoonnx and upstream
rng = np.random.default_rng()
history, generated, step_ids = list(ids), [], ids
for _ in range(600):  # 50 codes a second: 12 s
    past = len(history) - len(step_ids)
    out = dict(zip(names, session.run(None, {{
        "input_ids": np.array([step_ids], np.int64),
        "attention_mask": np.ones((1, len(history)), np.int64),
        "position_ids": np.arange(past, len(history), dtype=np.int64)[None],
        **cache,
    }})))
    cache = {{name: out[name.replace("past_", "present_")] for name in cache}}
    logits = np.array(out["logits"], np.float32).reshape(-1)
    seen = np.unique(history[-64:])  # the repetition penalty looks back 64 tokens, as llama.cpp does
    logits[seen] = np.where(logits[seen] < 0, logits[seen] * repetition_penalty, logits[seen] / repetition_penalty)
    if temperature > 0:  # top-k and top-p on the raw scores, then temperature, in llama.cpp's order
        logits[logits < np.partition(logits, -top_k)[-top_k]] = -np.inf
        order = np.argsort(logits)[::-1]
        drop = np.ones(len(order), bool)
        drop[1:] = np.cumsum(softmax(logits[order]))[:-1] >= top_p
        drop[0] = False
        logits[order[drop]] = -np.inf
        token = int(rng.choice(len(logits), p=softmax(logits / temperature)))
    else:
        token = int(np.argmax(logits))
    if token in stop:
        break
    generated.append(token)
    history.append(token)
    step_ids = [token]

codes = [int(m.group(1)) for t in generated if (m := re.fullmatch(r"<\\|speech_(\\d+)\\|>", tokenizer.id_to_token(t)))]
codec = onnxruntime.InferenceSession({_hub(codec, repo_id)})
audio = codec.run(None, {{"codes": np.array(codes, np.int32).reshape(1, 1, -1)}})[0].reshape(-1)

{_write_wav("24000")}'''


_SPARKTTS_FEEDS = {"model": ({"input_ids", "attention_mask", "position_ids"}, {"logits"}),
                   "vocoder_path": ({"semantic_tokens", "global_tokens"}, set())}


def _sparktts(config: Dict[str, Any], files: Dict[str, Any], *, repo_id: str, text: Optional[str]) -> str:
    for key, (inputs, outputs) in _SPARKTTS_FEEDS.items():
        if key not in files:
            raise UnsupportedExample(f"a Spark-TTS voice with no {key}")
        if "inputs" in files[key] and not (inputs <= set(files[key]["inputs"]) and outputs <= set(files[key]["outputs"])):
            raise UnsupportedExample(f"Spark-TTS {files[key]['file']} has inputs {sorted(files[key]['inputs'])}")
    if "speaker_tokens_path" not in files:
        raise UnsupportedExample("a Spark-TTS voice with no preset speaker tokens")
    model = files["model"]
    weights = {"repo": model["repo"], "file": model["file"] + "_data"}
    sentence = json.dumps(text, ensure_ascii=False) if text else f'"..."  # a sentence in {config.get("lang")}'
    return "\n".join([
        "# pip install onnxruntime numpy huggingface_hub tokenizers",
        "import json",
        "import wave",
        "",
        "import numpy as np",
        "import onnxruntime",
        "from huggingface_hub import hf_hub_download",
        "from tokenizers import Tokenizer",
        "",
        f'repo = "{repo_id}"',
        f"tokenizer = Tokenizer.from_file({_hub(files['bpe_tokenizer_path'], repo_id)})",
        "token = tokenizer.token_to_id",
        f"{_hub(weights, repo_id)}  # the weights, beside the graph",
        f"lm = onnxruntime.InferenceSession({_hub(model, repo_id)})",
        f"vocoder = onnxruntime.InferenceSession({_hub(files['vocoder_path'], repo_id)})",
        "# the speaker is 32 BiCodec global tokens; for your own voice, phoonnx turns 6 s of audio into them",
        "# with the repo's speaker_encoder_tokenizer.onnx (phoonnx.engines.sparktts.tokenize_reference)",
        f"global_tokens = json.load(open({_hub(files['speaker_tokens_path'], repo_id)}))[\"global_tokens\"]",
        "",
        f"text = {sentence}",
        'ids = [token("<|task_tts|>"), token("<|start_content|>")] + tokenizer.encode(text).ids',
        'ids += [token("<|end_content|>"), token("<|start_global_token|>")]',
        'ids += [token("<|bicodec_global_0|>") + t for t in global_tokens] + [token("<|end_global_token|>")]',
        "",
        "temperature, top_k, top_p = 0.8, 50, 0.95  # as Spark-TTS samples; temperature 0 is greedy",
        "rng = np.random.default_rng()",
        'past = {i.name: np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32) for i in lm.get_inputs() if i.name.startswith("past")}',
        "names = [o.name for o in lm.get_outputs()]",
        'feed = {"input_ids": np.array([ids]), "attention_mask": np.ones((1, len(ids)), np.int64),',
        '        "position_ids": np.arange(len(ids))[None]}',
        "codes = []",
        "for step in range(3000):  # about 60 s of audio at 50 tokens a second",
        "    out = dict(zip(names, lm.run(None, {**feed, **past})))",
        '    logits = out["logits"][0, -1].astype(np.float64)',
        "    if temperature == 0:",
        "        next_token = int(logits.argmax())",
        "    else:  # temperature, then top-k, then top-p, as Hugging Face generate",
        "        logits /= temperature",
        "        logits[logits < np.sort(logits)[-top_k]] = -np.inf",
        "        probs = np.exp(logits - logits.max())",
        "        probs /= probs.sum()",
        "        order = np.argsort(probs)[::-1]",
        "        keep = order[:np.searchsorted(np.cumsum(probs[order]), top_p) + 1]",
        "        next_token = int(rng.choice(keep, p=probs[keep] / probs[keep].sum()))",
        '    if next_token == token("<|im_end|>"):',
        "        break",
        "    codes.append(next_token)",
        '    past = {name: out[name.replace("past_key_values", "present")] for name in past}',
        '    feed = {"input_ids": np.array([[next_token]]), "attention_mask": np.ones((1, len(ids) + step + 1), np.int64),',
        '            "position_ids": np.array([[len(ids) + step]])}',
        "",
        'semantic = [c - token("<|bicodec_semantic_0|>") for c in codes if 0 <= c - token("<|bicodec_semantic_0|>") < 8192]',
        "audio = vocoder.run(None, {",
        '    "semantic_tokens": np.array([semantic], dtype=np.int64),',
        '    "global_tokens": np.array(global_tokens, dtype=np.int64).reshape(1, 1, -1),',
        "})[0].reshape(-1)",
        "",
        _write_wav("16000"),
    ])


_ARKTTS_FEEDS = {"model": ({"codes", "input_pos"}, {"logits", "slow_hidden"}),
                 "fast_ar_path": ({"slow_hidden", "token_id", "use_slow_hidden", "input_pos"}, {"logits"}),
                 "codec_decoder_path": ({"codes"}, set())}


def _arktts(config: Dict[str, Any], files: Dict[str, Any], *, repo_id: str, text: Optional[str]) -> str:
    for key, (inputs, outputs) in _ARKTTS_FEEDS.items():
        if key not in files:
            raise UnsupportedExample(f"an ArkTTS voice with no {key}")
        if "inputs" in files[key] and not (inputs <= set(files[key]["inputs"]) and outputs <= set(files[key]["outputs"])):
            raise UnsupportedExample(f"ArkTTS {files[key]['file']} has inputs {sorted(files[key]['inputs'])}")
    for key in ("bpe_tokenizer_path", "voice_codes_path"):
        if key not in files:
            raise UnsupportedExample(f"an ArkTTS voice with no {key}")
    sentence = json.dumps(text, ensure_ascii=False) if text else f'"..."  # a sentence in {config.get("lang")}'
    return "\n".join([
        "# pip install onnxruntime numpy huggingface_hub tokenizers",
        "import json",
        "import wave",
        "",
        "import numpy as np",
        "import onnxruntime",
        "from huggingface_hub import hf_hub_download",
        "from tokenizers import Tokenizer",
        "",
        f'repo = "{repo_id}"',
        f"tokenizer = Tokenizer.from_file({_hub(files['bpe_tokenizer_path'], repo_id)})",
        f"slow = onnxruntime.InferenceSession({_hub(files['model'], repo_id)})  # one token per frame: codebook 0",
        f"fast = onnxruntime.InferenceSession({_hub(files['fast_ar_path'], repo_id)})  # codebooks 1 to 9 of the frame",
        f"decoder = onnxruntime.InferenceSession({_hub(files['codec_decoder_path'], repo_id)})",
        "# the voice is the codec codes (10 codebooks) of a short clip and what the clip says; to clone",
        "# your own voice, encode a clip with the upstream ArkTTS codec encoder into the same JSON",
        f"voice = json.load(open({_hub(files['voice_codes_path'], repo_id)}, encoding=\"utf-8\"))",
        'reference, speaker = np.array(voice["codes"]), voice["reference_text"].strip()',
        'if "<|speaker:" not in speaker:',
        '    speaker = "<|speaker:0|>" + speaker',
        "",
        f"text = {sentence}",
        "# each part is encoded on its own, as upstream builds the prompt",
        'prefix = ["<|im_start|>system\\n", "convert the provided text to speech reference to the following:\\n\\nText:\\n",',
        '          speaker, "\\n\\nSpeech:\\n"]',
        'suffix = ["<|im_end|>\\n", "<|im_start|>user\\n", " ".join(text.split()), "<|im_end|>\\n", "<|im_start|>assistant\\n<|voice|>"]',
        "prefix, suffix = ([i for part in parts for i in tokenizer.encode(part).ids] for parts in (prefix, suffix))",
        "semantic = 151678  # the text vocabulary id of codebook-0 value 0",
        "ids = prefix + [semantic + c for c in reference[0]] + suffix",
        "step = np.zeros((1, 11, len(ids)), np.int64)  # row 0: the tokens; rows 1-10: the reference codes under them",
        "step[0, 0] = ids",
        "step[0, 1:, len(prefix):len(prefix) + reference.shape[1]] = reference",
        "",
        "temperature, top_k, top_p = 0.8, 50, 0.95  # the model cards' settings: greedy decoding loops",
        "rng = np.random.default_rng()",
        "",
        "",
        "def nucleus(logits, top_p):  # upstream's order: top-k and top-p on the raw logits, before temperature",
        '    order = np.argsort(-logits, kind="stable")',
        "    p = np.exp(logits[order] - logits[order].max())",
        "    drop = (np.cumsum(p / p.sum()) > top_p) | (np.arange(logits.size) >= top_k)",
        "    drop[0] = False",
        "    kept = logits.copy()",
        "    kept[order[drop]] = -np.inf",
        "    return kept",
        "",
        "",
        "def draw(scores):",
        "    p = np.exp(scores - scores.max())",
        "    return int(rng.choice(p.size, p=p / p.sum()))",
        "",
        "",
        "def window_cache(session):  # a fixed window: each step writes its keys and values at its positions",
        '    return {i.name: np.zeros(i.shape, np.float16 if "16" in i.type else np.float32)',
        '            for i in session.get_inputs() if i.name.startswith("cache_")}',
        "",
        "",
        "def write(cache, out, positions):",
        "    for name in cache:",
        '        kind, layer = name.split("_")[1:]',
        '        cache[name][:, :, positions] = out[f"{kind}_delta_{layer}"]',
        "",
        "",
        "slow_names, fast_names = [o.name for o in slow.get_outputs()], [o.name for o in fast.get_outputs()]",
        "cache, positions, window, frames = window_cache(slow), np.arange(len(ids)), [], []",
        "for _ in range(min(512, 2048 - len(ids))):  # 512 frames: about 24 s",
        '    out = dict(zip(slow_names, slow.run(None, {"codes": step, "input_pos": positions, **cache})))',
        "    write(cache, out, positions)",
        '    logits = out["logits"][0, -1].astype(np.float64)  # 4096 codebook-0 values, then the end of speech',
        "    first = draw(nucleus(logits, top_p) / temperature)",
        "    if first != 4096 and semantic + first in window:  # a repeat: draw again, tighter, as upstream",
        "        first = draw(nucleus(logits, 0.9))",
        "    if first == 4096:",
        "        break",
        '    hidden, fast_cache, frame = out["slow_hidden"][:, -1:], window_cache(fast), [first]',
        "    for position in range(10):  # position 0 reads the slow hidden state; its logits are not used",
        "        out = dict(zip(fast_names, fast.run(None, {",
        "            \"slow_hidden\": hidden if position == 0 else np.zeros_like(hidden),",
        '            "token_id": np.array([[frame[-1] if position else 0]]),',
        '            "use_slow_hidden": np.array([position == 0]),',
        '            "input_pos": np.array([position]), **fast_cache})))',
        "        write(fast_cache, out, np.array([position]))",
        "        if position:",
        '            frame.append(draw(nucleus(out["logits"][0, -1].astype(np.float64), top_p) / temperature))',
        "    frames.append(frame)",
        "    window = (window + [semantic + first])[-10:] if window else [0] * 10",
        "    positions = positions[-1:] + 1",
        "    step = np.array([semantic + first] + frame).reshape(1, 11, 1)",
        "",
        "codes = np.array(frames).T  # (10, frames)",
        'audio = decoder.run(None, {"codes": codes[None]})[0].reshape(-1)',
        "",
        _write_wav("44100"),
    ])


_MAGPIE_GRAPHS = ("model", "text_encoder_path", "cross_kv_path", "local_step_path", "audio_embed_path", "lt_embed_path",
                  "codec_decoder_path")
_MAGPIE_IO = {"model": ({"x", "pos", "self_k", "self_v", "cross_k", "cross_v", "cond_mask", "attn_prior"},
                        {"logits", "dec_out", "cross_attn_probs"}),
              "text_encoder_path": ({"text", "text_mask"}, {"encoder_out"}),
              "cross_kv_path": ({"cond"}, {"cross_k", "cross_v"}),
              "local_step_path": ({"h", "pos", "cache_k", "cache_v", "cb"}, {"logits"}),
              "audio_embed_path": ({"codes"}, {"emb"}),
              "lt_embed_path": ({"tok", "cb"}, {"emb"}),
              "codec_decoder_path": ({"codes"}, {"audio"})}
# phoonnx.engines.magpie: the languages whose sub-tokenizer is a byte or a character table
_MAGPIE_BYTES, _MAGPIE_CHARS = ("fr", "it", "vi", "ko"), ("ar",)


def _magpie(config: Dict[str, Any], files: Dict[str, Any], *, repo_id: str, text: Optional[str]) -> str:
    options = config.get("engine_options") or {}
    lang = str(options.get("lang") or config.get("lang") or "")
    base = lang.split("-")[0].split("_")[0].lower()
    if base not in _MAGPIE_BYTES + _MAGPIE_CHARS:
        raise UnsupportedExample(f"Magpie reads {lang!r} through NeMo's IPA G2P, which phoonnx does not run")
    if not options.get("exact_decode", True):
        raise UnsupportedExample("Magpie with a KV cache (exact_decode false) has no template")
    for key in _MAGPIE_GRAPHS + ("tokenizer_path", "context_embeddings_path", "magpie_config_path"):
        if key not in files or files[key]["repo"] != files["model"]["repo"]:
            raise UnsupportedExample(f"the Magpie entry has no {key} in the voice's repo")
    for key, (inputs, outputs) in _MAGPIE_IO.items():
        if "inputs" in files[key] and not (inputs <= set(files[key]["inputs"]) and outputs <= set(files[key]["outputs"])):
            raise UnsupportedExample(f"Magpie graph {files[key]['file']} has other inputs or outputs")
    speaker = int(options.get("speaker_id", 0))
    sentence = json.dumps(text, ensure_ascii=False) if text else f'"..."  # a sentence in {lang}'
    graphs = ", ".join(f'"{files[k]["file"]}"' for k in _MAGPIE_GRAPHS)
    if base in _MAGPIE_BYTES:
        tokens = ("offset = tokens[\"tokenizer_offsets\"][tokenizer]\n"
                  "ids = [offset + 3 + b for b in text.encode()] + [tokens[\"eos_id\"]]"
                  "  # byt5: pad, end, unknown, then the 256 bytes")
    else:
        tokens = ("offset, count = tokens[\"tokenizer_offsets\"][tokenizer], tokens[\"num_tokens_per_tokenizer\"][tokenizer]\n"
                  "symbols = {s: offset + i for i, s in enumerate(tokens[\"tokens\"][offset:offset + count])}\n"
                  "ids = [symbols[c] for c in text if c in symbols] + [tokens[\"eos_id\"]]  # one id a character")
    return f'''# pip install onnxruntime numpy huggingface_hub
import json
import wave

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download

repo = "{repo_id}"
text = {sentence}
speaker = {speaker}  # {options.get("speaker") or "the voice"}; assets/speakers.json names the five voices of the model

config = json.load(open(hf_hub_download(repo, "{files["magpie_config_path"]["file"]}")))
inference = config.get("inference", {{}})
graphs = ({graphs})
for name in graphs:
    hf_hub_download(repo, name + "_data")  # the weights, read from beside each graph
decoder, encoder, cross_kv, local, audio_embed, lt_embed, codec = (
    onnxruntime.InferenceSession(hf_hub_download(repo, name)) for name in graphs)

# the {lang} sub-tokenizer of the model's aggregated tokenizer
tokens = json.load(open(hf_hub_download(repo, "{files["tokenizer_path"]["file"]}")))
tokenizer = next(names[0] for key, names in config["language_to_tokenizer"].items() if key.split("-")[0].lower() == "{base}")
{tokens}

n_cb, stack = config.get("num_audio_codebooks", 8), config.get("frame_stacking_factor", 2)  # 8 codebooks, 2 frames a step
vocab, size = config.get("num_all_tokens_per_codebook", 2024), config.get("codebook_size", 2016)
bos, eos = size, size + 1  # the 8 special tokens follow the codec tokens
heads, head_dim = config.get("sa_n_heads", 12), config.get("sa_d_head", 64)
epsilon, window = inference.get("attention_prior_epsilon", 0.1), inference.get("attention_prior_lookahead_window", 6)
sink = inference.get("attention_sink_threshold", 4)

# classifier-free guidance: row 0 reads the text, row 1 a zeroed text that keeps only its first place
cond = encoder.run(None, {{"text": np.array([ids], np.int64), "text_mask": np.ones((1, len(ids)), np.float32)}})[0]
cond = np.concatenate([cond, np.zeros_like(cond)])
cond_mask = np.zeros((2, len(ids)), np.float32)
cond_mask[0, :], cond_mask[1, 0] = 1.0, 1.0
cross_k, cross_v = cross_kv.run(None, {{"cond": cond}})
context = np.load(hf_hub_download(repo, "{files["context_embeddings_path"]["file"]}"))[speaker][None].astype(np.float32)
start = audio_embed.run(None, {{"codes": np.full((2, n_cb * stack), bos, np.int64)}})[0]
x = np.concatenate([np.concatenate([context, np.zeros_like(context)]), start], axis=1)  # the voice, then audio start

temperature, top_k, cfg_scale = 0.6, 80, 2.5  # as phoonnx
rng = np.random.default_rng(0)  # a fixed seed gives the same audio each run


def sample(logits, forbid_eos, temperature, top_k):
    logits = logits.astype(np.float64).copy()
    logits[[size + i for i in range(8) if forbid_eos or i != 1]] = -np.inf  # special tokens; the end only when allowed
    logits[logits < np.partition(logits, -top_k)[-top_k]] = -np.inf
    if temperature <= 0:
        return int(np.argmax(logits))
    p = np.exp(logits / temperature - np.max(logits / temperature))
    return int(rng.choice(p.size, p=p / p.sum()))


empty = np.zeros((config.get("decoder_n_layers", 12), 2, 0, heads, head_dim), np.float32)
prior = np.ones((2, 1, len(ids)), np.float32)
frames, counter, last, end_frame, finished = [], {{}}, 1, None, 0
for step in range(inference.get("max_decoder_steps", 500) // stack):
    # no KV cache: the attention prior changes every step, and phoonnx re-applies it over the whole history
    logits, dec_out, _, _, attention = decoder.run(None, {{
        "x": x, "pos": np.arange(x.shape[1]), "self_k": empty, "self_v": empty, "cross_k": cross_k,
        "cross_v": cross_v, "cond_mask": cond_mask, "attn_prior": prior}})
    guided = cfg_scale * logits[0, -1] + (1.0 - cfg_scale) * logits[1, -1]

    # the attention prior: find the text place being spoken, and keep the next step near it
    layers = [layer for layer in range(len(attention))  # the layers whose attention tracks the text
              if layer in (config.get("estimate_alignment_from_layers") or [layer])
              and layer in (config.get("transcript_decoder_layers") or [layer])]
    alignment = np.stack([attention[layer].mean(axis=1) for layer in layers], axis=1).mean(axis=1)[0, -1]
    if counter.get(last, 0) >= sink:  # a place attended too often is a sink: step past it
        last += 1
    scores = alignment[last:min(last + window, len(ids) - 3)]
    last = int(np.argmax(scores)) + last if scores.size else len(ids) - 1
    counter[last] = counter.get(last, 0) + 1
    prior = np.full((2, 1, len(ids)), epsilon, np.float32)
    if len(ids) <= 5:
        prior[:] = 1.0
    else:
        prior[0, 0, [max(1, last - 1), last] + [min(last + k, len(ids) - 1) for k in range(1, window + 1)]] = 1.0
        for place, count in counter.items():
            if count >= sink:
                prior[0, 0, :place + 1] = epsilon
        prior[1] = prior[0]

    # the end of speech is forbidden until the text runs out
    if last >= len(ids) - 2 or end_frame is not None:
        finished += 1
    forbid_eos = step * stack < inference.get("min_generated_frames", 0) or (
        last < len(ids) - 3 and end_frame is None and finished <= 5)

    # the local transformer writes the 16 tokens of the step one by one
    hidden = dec_out[:, -1:, :].astype(np.float32)
    cache_k = cache_v = np.zeros((config.get("local_transformer_n_layers", 2), 2, 0, heads, head_dim), np.float32)
    picks = []
    for cb in range(n_cb * stack):
        lt_logits, cache_k, cache_v = local.run(None, {{
            "h": hidden, "pos": np.array([cb]), "cache_k": cache_k, "cache_v": cache_v, "cb": np.array(cb)}})
        picks.append(sample(cfg_scale * lt_logits[0] + (1.0 - cfg_scale) * lt_logits[1], forbid_eos, temperature, top_k))
        hidden = lt_embed.run(None, {{"tok": np.full(2, picks[-1]), "cb": np.array(cb)}})[0]
    frame = np.array(picks).reshape(stack, n_cb).T
    greedy = np.array([[sample(guided[(cb + n_cb * fs) * vocab:(cb + n_cb * fs + 1) * vocab], forbid_eos, 0, 1)
                        for fs in range(stack)] for cb in range(n_cb)])
    if end_frame is None:  # the first frame where the greedy read or the sampled tokens end
        ends = [np.flatnonzero((codes == eos).any(axis=0)) for codes in (greedy, frame)]
        if any(e.size for e in ends):
            end_frame = step * stack + min(int(e[0]) for e in ends if e.size)
    frames.append(frame)
    if end_frame is not None and len(frames) >= 4:
        break
    x = np.concatenate([x, audio_embed.run(None, {{"codes": np.repeat(frame.T.reshape(1, -1), 2, axis=0)}})[0]], axis=1)

codes = np.concatenate(frames, axis=-1)
codes = codes[:, :max(4, end_frame if end_frame is not None else codes.shape[-1])]
audio = codec.run(None, {{"codes": codes[None]}})[0].reshape(-1)

{_write_wav("22050")}'''


_MOSSTTS_FILES = ("model", "decode_step_path", "local_fixed_frame_path", "sp_model_path", "codec_encode_path",
                  "codec_decode_path", "global_weights_path", "local_weights_path", "codec_encode_weights_path",
                  "codec_decode_weights_path")
_MOSSTTS_IO = {"model": ({"input_ids", "attention_mask"}, {"global_hidden"}),
               "decode_step_path": ({"input_ids", "past_valid_lengths"}, {"global_hidden"}),
               "local_fixed_frame_path": ({"global_hidden", "repetition_seen_mask", "assistant_random_u",
                                           "audio_random_u"}, {"should_continue", "frame_token_ids"}),
               "codec_encode_path": ({"waveform", "input_lengths"}, {"audio_codes", "audio_code_lengths"}),
               "codec_decode_path": ({"audio_codes", "audio_code_lengths"}, {"audio", "audio_lengths"})}
# the chat prompt MOSS-TTS-Nano was trained with, as SentencePiece ids (phoonnx.engines.mosstts)
_MOSSTTS_USER_PREFIX = [600, 289, 10356, 13, 10356, 10425, 1860, 4546, 12907, 10363, 13325, 11492]
_MOSSTTS_AFTER_REFERENCE = [10356, 10425, 3965, 7738, 11492, 505, 587, 10356, 10425, 352, 500, 856, 11492, 505, 587,
                            10356, 10425, 2128, 1247, 594, 11492, 505, 587, 10356, 10425, 348, 909, 561, 3648, 11492,
                            505, 587, 10356, 10425, 2818, 1305, 355, 348, 909, 11492, 505, 587, 10356, 10425, 484,
                            339, 10367, 783, 11492, 505, 587, 10356, 10425, 2427, 980, 11492]
_MOSSTTS_ASSISTANT_PREFIX = [10356, 14, 5, 4, 8165, 430]


def _mosstts(config: Dict[str, Any], files: Dict[str, Any], *, repo_id: str, text: Optional[str]) -> str:
    missing = [key for key in _MOSSTTS_FILES if key not in files]
    if missing:
        raise UnsupportedExample(f"the MOSS-TTS-Nano entry names no {', '.join(missing)}")
    for key, (inputs, outputs) in _MOSSTTS_IO.items():
        if "inputs" in files[key] and not (inputs <= set(files[key]["inputs"]) and outputs <= set(files[key]["outputs"])):
            raise UnsupportedExample(f"MOSS-TTS-Nano graph {files[key]['file']} has other inputs or outputs")
    sentence = json.dumps(text, ensure_ascii=False) if text else f'"..."  # a sentence in {config.get("lang") or "the voice language"}'
    weights = ",\n             ".join(f'"{files[k]["file"]}"' for k in _MOSSTTS_FILES[6:])
    return f'''# pip install onnxruntime numpy huggingface_hub
import struct
import wave

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download

repo = "{repo_id}"
text = {sentence}
# MOSS-TTS-Nano has no built-in voices: it speaks in the voice of a reference clip.
# Give a WAV file of one speaker, about 5 to 15 seconds long.
reference = "reference.wav"

for name in ({weights}):
    hf_hub_download(repo, name)  # the graphs read their weights from these files beside them
prefill, decode_step, local_frame, codec_encode, codec_decode = (onnxruntime.InferenceSession(path) for path in (
    {_hub(files["model"], repo_id)}, {_hub(files["decode_step_path"], repo_id)},
    {_hub(files["local_fixed_frame_path"], repo_id)}, {_hub(files["codec_encode_path"], repo_id)},
    {_hub(files["codec_decode_path"], repo_id)}))


def varint(data, i):
    value = shift = 0
    while True:
        value, shift, i = value | (data[i] & 127) << shift, shift + 7, i + 1
        if data[i - 1] < 128:
            return value, i


def fields(data):  # the (field number, value) pairs of a protobuf message
    i = 0
    while i < len(data):
        key, i = varint(data, i)
        if key & 7 == 0:
            value, i = varint(data, i)
        elif key & 7 == 5:
            value, i = data[i:i + 4], i + 4
        else:
            size, i = varint(data, i)
            value, i = data[i:i + size], i + size
        yield key >> 3, value


# the SentencePiece tokenizer (BPE with byte fallback), read from tokenizer.model
pieces, charsmap = {{}}, b""
for number, value in fields(open({_hub(files["sp_model_path"], repo_id)}, "rb").read()):
    if number == 1:  # a piece: its text and its score
        piece = dict(fields(value))
        pieces[piece[1].decode()] = (len(pieces), struct.unpack("<f", piece.get(2, bytes(4)))[0])
    elif number == 3:  # the nmt_nfkc normaliser, as a double-array trie and its replacement strings
        charsmap = dict(fields(value))[2]
size = struct.unpack("<I", charsmap[:4])[0]
trie, replacements = struct.unpack(f"<{{size // 4}}I", charsmap[4:4 + size]), charsmap[4 + size:]
data, normalized, i = text.encode(), b"", 0
while i < len(data):  # replace the longest match of the trie at each place
    node, match = trie[0] >> 10 << (trie[0] >> 6 & 8), None
    for j in range(i, len(data)):
        node ^= data[j]
        unit = trie[node]
        if unit & 0x800000FF != data[j]:
            break
        node ^= unit >> 10 << (unit >> 6 & 8)
        if unit >> 8 & 1:
            match = j + 1, trie[node] & 0x7FFFFFFF
    if match:
        normalized, i = normalized + replacements[match[1]:replacements.index(0, match[1])], match[0]
    else:
        n = 1 + (data[i] >= 0xC0) + (data[i] >= 0xE0) + (data[i] >= 0xF0)
        normalized, i = normalized + data[i:i + n], i + n
symbols = list("▁" + "▁".join(normalized.decode().split()))
while True:  # merge the adjacent pair whose piece has the best score, leftmost first
    pairs = [(pieces[a + b][1], -k) for k, (a, b) in enumerate(zip(symbols, symbols[1:])) if a + b in pieces]
    if not pairs:
        break
    k = -max(pairs)[1]
    symbols[k:k + 2] = [symbols[k] + symbols[k + 1]]
ids = [x for s in symbols for x in ([pieces[s][0]] if s in pieces else [pieces[f"<0x{{b:02X}}>"][0] for b in s.encode()])]

with wave.open(reference) as wav:
    rate, channels = wav.getframerate(), wav.getnchannels()
    clip = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").reshape(-1, channels).mean(axis=1) / 32768
if rate != 48000:  # phoonnx resamples with scipy's polyphase filter when scipy is installed
    clip = np.interp(np.linspace(0, len(clip) - 1, round(len(clip) * 48000 / rate)), np.arange(len(clip)), clip)
stereo = np.repeat(clip[None, None, :], 2, axis=1).astype(np.float32)  # the codec reads 48 kHz stereo
reference_codes, n = codec_encode.run(None, {{"waveform": stereo, "input_lengths": np.array([stereo.shape[-1]], np.int32)}})

# each row is 17 wide: a text or slot token, then the 16 codebook tokens (1024 pads a text row)
pad = [1024] * 16
rows = ([[t] + pad for t in {_MOSSTTS_USER_PREFIX} + [6]]  # 6 starts audio
        + [[8] + [int(c) for c in frame] for frame in reference_codes[0, :int(n[0])]]  # 8: the user's audio
        + [[t] + pad for t in [7] + {_MOSSTTS_AFTER_REFERENCE}  # 7 ends audio
           + ids + {_MOSSTTS_ASSISTANT_PREFIX} + [6]])

out = prefill.run(None, {{"input_ids": np.array([rows], np.int32), "attention_mask": np.ones((1, len(rows)), np.int32)}})
names = [o.name for o in prefill.get_outputs()]
# the sampling (temperature 0.8, top-p 0.95, top-k 25, repetition penalty 1.2) is inside the local graph;
# the graph takes its random draws as inputs, so a fixed seed gives the same audio each run
rng = np.random.default_rng(0)
seen = np.zeros((1, 16, 1024), np.int32)  # the tokens each codebook used so far, for the repetition penalty
codes = []
for step in range(375):  # at most 30 seconds: 12.5 frames a second
    state = dict(zip(names, out))
    hidden = state["global_hidden"][:, -1]
    past = {{k.replace("present_", "past_"): v for k, v in state.items() if k.startswith("present_")}}
    draws = {{"assistant_random_u": np.array([min(0.99999994, rng.random())], np.float32),
             "audio_random_u": np.array([[min(0.99999994, rng.random()) for _ in range(16)]], np.float32)}}
    should_continue, frame = local_frame.run(["should_continue", "frame_token_ids"], {{
        "global_hidden": hidden, "repetition_seen_mask": seen, **draws}})
    if not should_continue.reshape(-1)[0]:
        break
    frame = [int(t) for t in frame.reshape(-1)]
    seen[0, np.arange(16), frame] = 1
    codes.append(frame)
    out = decode_step.run(None, {{"input_ids": np.array([[[9] + frame]], np.int32),  # 9: the assistant's audio
                                 "past_valid_lengths": np.array([len(rows) + step], np.int32), **past}})
    names = [o.name for o in decode_step.get_outputs()]

audio, lengths = codec_decode.run(None, {{"audio_codes": np.array([codes], np.int32),
                                         "audio_code_lengths": np.array([len(codes)], np.int32)}})
audio = audio[0, :, :lengths[0]].mean(axis=0)  # stereo to mono

{_write_wav("48000")}'''


_CHATTERBOX_GRAPHS = ("speech_encoder", "embed_tokens", "model", "conditional_decoder")
# Languages whose text phoonnx changes before the multilingual tokenizer with a tool the example
# cannot use: Arabic and Hebrew get their vowel marks restored, and the script transform of
# Japanese (pykakasi), Russian (stress marks) and Chinese (Cangjie) needs a package of its own.
_CHATTERBOX_REFUSED = {"ar": "phoonnx restores the Arabic vowel marks with a diacritizer model",
                       "he": "phoonnx restores the Hebrew vowel marks with a diacritizer model",
                       "ja": "phoonnx writes Japanese in hiragana with pykakasi",
                       "ru": "phoonnx adds Russian stress marks with a stress model",
                       "zh": "phoonnx writes Chinese in Cangjie codes with a table package"}


def _chatterbox(config: Dict[str, Any], files: Dict[str, Any], *, repo_id: str, text: Optional[str]) -> str:
    variant = config["voice_id"].split("/")[1]
    lang = config.get("lang") or ""
    language = lang.replace("_", "-").split("-")[0].lower()
    multilingual = variant in ("multilingual", "lahgtna")
    if multilingual and language in _CHATTERBOX_REFUSED:
        raise UnsupportedExample(_CHATTERBOX_REFUSED[language])
    for key in (*_CHATTERBOX_GRAPHS, "tokenizer_config"):
        if key not in files or files[key]["repo"] != repo_id:
            raise UnsupportedExample(f"a Chatterbox voice with no {key} in its repo")
    embed_inputs = set(files["embed_tokens"].get("inputs") or
                       (["input_ids"] if variant == "turbo" else ["input_ids", "position_ids", "exaggeration"]))
    lm_inputs = set(files["model"].get("inputs") or
                    (["inputs_embeds", "attention_mask", "position_ids"] if variant == "turbo"
                     else ["inputs_embeds", "attention_mask"]))
    if not {"input_ids"} <= embed_inputs <= {"input_ids", "position_ids", "exaggeration"}:
        raise UnsupportedExample(f"Chatterbox embed_tokens inputs {sorted(embed_inputs)}")
    if not {"inputs_embeds", "attention_mask"} <= lm_inputs or \
            {n for n in lm_inputs if not n.startswith("past_key_values")} - {"inputs_embeds", "attention_mask", "position_ids"}:
        raise UnsupportedExample(f"Chatterbox language model inputs {sorted(lm_inputs)}")
    positions = "position_ids" in lm_inputs
    sentence = json.dumps(text, ensure_ascii=False) if text else f'"..."  # a sentence in {lang}'
    graphs = ", ".join(f'"{files[k]["file"]}"' for k in _CHATTERBOX_GRAPHS)
    if multilingual:
        tokens = [f"# the multilingual model reads lowercase NFKD text after a [{language}] language token, "
                  "with [SPACE] for each space",
                  f'ids = tokenizer.encode("[{language}]" + unicodedata.normalize("NFKD", text.lower())'
                  '.replace(" ", "[SPACE]")).ids']
    else:
        tokens = ["ids = tokenizer.encode(text).ids"]

    def embed_feed(ids, position):
        feed = [f'"input_ids": {ids}']
        if "position_ids" in embed_inputs:
            feed.append(f'"position_ids": {position}')
        if "exaggeration" in embed_inputs:
            feed.append('"exaggeration": np.array([exaggeration], np.float32)')
        return ", ".join(feed)

    lm_feed = '"inputs_embeds": embeds, "attention_mask": np.ones((1, length), np.int64)'
    if positions:
        lm_feed += ', "position_ids": np.arange(length - embeds.shape[1], length)[None]'
    return "\n".join([
        "# pip install onnxruntime numpy huggingface_hub tokenizers",
        *(["import unicodedata"] if multilingual else []),
        "import wave",
        "",
        "import numpy as np",
        "import onnxruntime",
        "from huggingface_hub import hf_hub_download",
        "from tokenizers import Tokenizer",
        "",
        f'repo = "{repo_id}"',
        f'tokenizer = Tokenizer.from_file(hf_hub_download(repo, "{files["tokenizer_config"]["file"]}"))',
        "graphs = []",
        f"for name in ({graphs}):",
        '    hf_hub_download(repo, name + "_data")  # the weights, beside the graph',
        "    graphs.append(onnxruntime.InferenceSession(hf_hub_download(repo, name)))",
        "encoder, embed, lm, decoder = graphs",
        "",
        "# Chatterbox clones the voice of a reference clip, and no clip ships with the model:",
        "# give a WAV of a few seconds of one speaker (mono, 16 bit)",
        'with wave.open("reference.wav", "rb") as clip:',
        "    rate = clip.getframerate()",
        '    reference = np.frombuffer(clip.readframes(clip.getnframes()), dtype="<i2").astype(np.float32) / 32768',
        "if rate != 24000:  # the speech encoder reads 24 kHz",
        "    reference = np.interp(np.linspace(0, len(reference) - 1, round(len(reference) * 24000 / rate)),",
        "                          np.arange(len(reference)), reference).astype(np.float32)",
        'cond_emb, prompt_token, x_vector, prompt_feat = encoder.run(None, {"audio_values": reference[None]})',
        "",
        f"text = {sentence}",
        *tokens,
        "",
        "exaggeration, temperature, top_p, repetition_penalty = 0.5, 0.8, 0.95, 1.2  # as phoonnx; temperature 0 is greedy",
        "rng = np.random.default_rng()",
        "start, stop = 6561, 6562  # the speech tokens that open and close the speech",
        "text_ids = np.array([ids])",
        f"embeds = embed.run(None, {{{embed_feed('text_ids', 'np.where(text_ids >= start, 0, np.arange(len(ids)) - 1)')}}})[0]",
        "embeds = np.concatenate((cond_emb, embeds), axis=1)  # the speaker conditioning comes first",
        'past = {i.name: np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32) for i in lm.get_inputs() if i.name.startswith("past")}',
        "generated, length = [start], embeds.shape[1]",
        "for step in range(1, 1001):",
        f"    logits, *present = lm.run(None, {{{lm_feed}, **past}})",
        "    scores = logits[0, -1]",
        "    seen = scores[generated]  # every token said so far is penalised, not only the last",
        "    scores[generated] = np.where(seen < 0, seen * repetition_penalty, seen / repetition_penalty)",
        "    if temperature > 0:",
        "        x = scores.astype(np.float64) / temperature",
        "        probs = np.exp(x - x.max())",
        "        probs /= probs.sum()",
        "        order = np.argsort(probs)[::-1]",
        "        keep = order[:np.searchsorted(np.cumsum(probs[order]), top_p) + 1]",
        "        token = int(rng.choice(keep, p=probs[keep] / probs[keep].sum()))",
        "    else:",
        "        token = int(scores.argmax())",
        "    generated.append(token)",
        "    if token == stop:",
        "        break",
        "    past = dict(zip(past, present))",
        f"    embeds = embed.run(None, {{{embed_feed('np.array([[token]])', 'np.array([[step]])')}}})[0]",
        "    length += 1",
        "",
        "codes = generated[1:-1]  # the speech tokens, without the start and the end",
        "# the reference's own tokens go first, and three silence tokens last",
        "tokens = np.concatenate([prompt_token, np.array([codes + [4299] * 3])], axis=1)",
        "audio = decoder.run(None, {",
        '    "speech_tokens": tokens, "speaker_embeddings": x_vector, "speaker_features": prompt_feat,',
        "})[0].reshape(-1)",
        "",
        _write_wav("24000"),
    ])


_LLASA_LM_IO = ({"input_ids", "attention_mask", "position_ids"}, {"logits"})


def _llasa(config: Dict[str, Any], files: Dict[str, Any], *, repo_id: str, text: Optional[str]) -> str:
    for key in ("model", "lm_weights_path", "tokenizer_path", "voices_path", "codec_decoder_path"):
        if key not in files:
            raise UnsupportedExample(f"a Llasa voice with no {key}")
    model = files["model"]
    if "inputs" in model and not (_LLASA_LM_IO[0] <= set(model["inputs"]) and _LLASA_LM_IO[1] <= set(model["outputs"])):
        raise UnsupportedExample(f"Llasa LM inputs {sorted(model['inputs'])}")
    if "inputs" in files["codec_decoder_path"] and list(files["codec_decoder_path"]["inputs"]) != ["codes"]:
        raise UnsupportedExample(f"XCodec2 decoder inputs {sorted(files['codec_decoder_path']['inputs'])}")
    voice = (config.get("engine_options") or {}).get("voice")
    preset = f'voices["presets"]["{voice}"]' if voice else 'voices["presets"][voices["default_voice"]]'
    sentence = json.dumps(text, ensure_ascii=False) if text else f'"..."  # a sentence in {config.get("lang")}'
    return f'''# pip install onnxruntime numpy huggingface_hub tokenizers
import json
import wave

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

repo = "{repo_id}"
tokenizer = Tokenizer.from_file({_hub(files["tokenizer_path"], repo_id)})
# A preset is the transcript of a short clip and the XCodec2 codes of that clip. The model goes on in its voice.
# For your own voice, add a preset with your clip's transcript and its XCodec2 codes (xcodec2 package).
voices = json.load(open({_hub(files["voices_path"], repo_id)}, encoding="utf-8"))
preset = {preset}

text = {sentence}
# the Llama 3.2 chat template the model was trained on, with a fixed date
prompt = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n"
          "Cutting Knowledge Date: December 2023\\nToday Date: 26 Jul 2024\\n\\n"
          "<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\nConvert the text to speech:"
          f"<|TEXT_UNDERSTANDING_START|>{{preset['text'].strip()}} {{text.strip()}}<|TEXT_UNDERSTANDING_END|>"
          "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n<|SPEECH_GENERATION_START|>")
speech = tokenizer.token_to_id("<|s_0|>")  # <|s_0|> .. <|s_65535|> are the XCodec2 codes
ids = tokenizer.encode(prompt, add_special_tokens=False).ids + [speech + c for c in preset["codes"][:500]]

{_hub(files["lm_weights_path"], repo_id)}  # the weights, beside the graph
session = onnxruntime.InferenceSession({_hub(files["model"], repo_id)})
cache = {{i.name: np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32)
         for i in session.get_inputs() if i.name.startswith("past_key_values.")}}
names = [o.name for o in session.get_outputs()]
end = tokenizer.token_to_id("<|SPEECH_GENERATION_END|>")

temperature, top_p = 0.9, 0.95  # as phoonnx and the upstream model card
rng = np.random.default_rng()
history, generated, step_ids = list(ids), [], ids
for _ in range(1000):  # 50 codes a second: 20 s
    past = len(history) - len(step_ids)
    out = dict(zip(names, session.run(None, {{
        "input_ids": np.array([step_ids], np.int64),
        "attention_mask": np.ones((1, len(history)), np.int64),
        "position_ids": np.arange(past, len(history), dtype=np.int64)[None],
        **cache,
    }})))
    cache = {{name: out[name.replace("past_key_values", "present")] for name in cache}}
    logits = np.full(out["logits"].size, -np.inf)
    logits[speech:speech + 65536] = out["logits"].reshape(-1)[speech:speech + 65536]  # speech tokens only,
    if generated:  # and the end marker once the speech has started
        logits[end] = out["logits"].reshape(-1)[end]
    if temperature > 0:  # temperature, then top-p, in Hugging Face's order
        scores = logits / temperature
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        order = np.argsort(probs)[::-1]
        keep = order[np.concatenate([[True], np.cumsum(probs[order])[:-1] < top_p])]
        token = int(rng.choice(keep, p=probs[keep] / probs[keep].sum()))
    else:
        token = int(np.argmax(logits))
    if token == end:
        break
    generated.append(token)
    history.append(token)
    step_ids = [token]

codes = [t - speech for t in generated]
codec = onnxruntime.InferenceSession({_hub(files["codec_decoder_path"], repo_id)})
audio = codec.run(None, {{"codes": np.array(codes, np.int64).reshape(1, 1, -1)}})[0].reshape(-1)

{_write_wav("16000")}'''


_QWEN3TTS_IO = {"model": ({"inputs_embeds", "position_ids", "past_key_values.0.key"}, {"logits", "last_hidden", "present.0.key"}),
                "text_embed_path": ({"input_ids"}, {"hidden"}),
                "codec_embed_path": ({"input_ids"}, {"hidden"}),
                "code_predictor_prefill_path": ({"inputs_embeds"}, {"logits", "present.0.key"}),
                "code_predictor_step_path": ({"inputs_embeds", "step", "position_ids", "past_key_values.0.key"},
                                             {"logits", "present.0.key"}),
                "sub_codec_embed_path": ({"input_ids", "tables"}, {"hidden"}),
                "codec_decoder_path": ({"codes"}, {"wav"})}
# phoonnx.engines.qwen3tts: the talker tokens of the speakers and the languages
_QWEN3TTS_SPEAKERS = {"vivian": 3065, "serena": 3066, "uncle_fu": 3010, "dylan": 2878, "eric": 2875, "ryan": 3061,
                      "aiden": 2861, "ono_anna": 2873, "sohee": 2864}
_QWEN3TTS_LANGUAGES = {"chinese": 2055, "english": 2050, "german": 2053, "italian": 2070, "portuguese": 2071,
                       "spanish": 2054, "japanese": 2058, "korean": 2064, "french": 2061, "russian": 2069,
                       "beijing_dialect": 2074, "sichuan_dialect": 2062}
_QWEN3TTS_CODES = {"zh": "chinese", "en": "english", "de": "german", "it": "italian", "pt": "portuguese",
                   "es": "spanish", "ja": "japanese", "ko": "korean", "fr": "french", "ru": "russian"}
_QWEN3TTS_DIALECTS = {"eric": "sichuan_dialect", "dylan": "beijing_dialect"}


def _qwen3tts(config: Dict[str, Any], files: Dict[str, Any], *, repo_id: str, text: Optional[str]) -> str:
    for key, (inputs, outputs) in _QWEN3TTS_IO.items():
        if key not in files:
            raise UnsupportedExample(f"the Qwen3-TTS entry names no {key}")
        if "inputs" in files[key] and not (inputs <= set(files[key]["inputs"]) and outputs <= set(files[key]["outputs"])):
            raise UnsupportedExample(f"Qwen3-TTS graph {files[key]['file']} has other inputs or outputs")
    options = config.get("engine_options") or {}
    speaker = str(options.get("speaker", "")).lower()
    if speaker not in _QWEN3TTS_SPEAKERS:
        raise UnsupportedExample(f"Qwen3-TTS has no speaker {speaker!r}")
    tag = str(options.get("language") or config.get("lang") or "auto").lower().replace("-", "_")
    language = tag if tag in _QWEN3TTS_LANGUAGES or tag == "auto" else _QWEN3TTS_CODES.get(tag.split("_")[0], "auto")
    if language in ("chinese", "auto") and speaker in _QWEN3TTS_DIALECTS:
        language = _QWEN3TTS_DIALECTS[speaker]  # this voice always speaks its dialect
    if language == "auto":
        think = "[2155, 2156, 2157]  # no language: the model finds it in the text"
    else:
        think = f"[2154, 2156, {_QWEN3TTS_LANGUAGES[language]}, 2157]  # think, then the language: {language}"
    sentence = json.dumps(text, ensure_ascii=False) if text else f'"..."  # a sentence in {config.get("lang") or "the voice language"}'
    voices = ", ".join(_QWEN3TTS_SPEAKERS)
    return f'''# pip install onnxruntime numpy huggingface_hub tokenizers
import wave

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

repo = "{repo_id}"
text = {sentence}
speaker = {_QWEN3TTS_SPEAKERS[speaker]}  # {speaker}; the nine voices: {voices}

talker, text_embed, codec_embed, predictor_prefill, predictor_step, sub_codec_embed, decoder = (
    onnxruntime.InferenceSession(hf_hub_download(repo, name)) for name in (
        "{files["model"]["file"]}", "{files["text_embed_path"]["file"]}", "{files["codec_embed_path"]["file"]}",
        "{files["code_predictor_prefill_path"]["file"]}", "{files["code_predictor_step_path"]["file"]}",
        "{files["sub_codec_embed_path"]["file"]}", "{files["codec_decoder_path"]["file"]}"))
tokenizer = Tokenizer.from_file({_hub(files["bpe_tokenizer_path"], repo_id)})
ids = tokenizer.encode(f"<|im_start|>assistant\\n{{text}}<|im_end|>\\n<|im_start|>assistant\\n").ids


def embed(session, ids):  # token ids to hidden states (1, n, 1024)
    return session.run(None, {{"input_ids": np.array([ids], np.int64)}})[0]


def sub_embed(tokens, tables):  # code group tokens to hidden states, each through its group's table
    return sub_codec_embed.run(None, {{"input_ids": np.array(tokens, np.int64).reshape(-1),
                                      "tables": np.array(tables, np.int64).reshape(-1)}})[0]


# The talker reads embeddings: at each place a text hidden state plus a codec hidden state.
hidden = embed(text_embed, ids)
specials = embed(text_embed, [151672, 151673, 151671])
tts_bos, tts_eos, tts_pad = specials[:, 0:1], specials[:, 1:2], specials[:, 2:3]
think = {think}
codec_prefix = np.concatenate([embed(codec_embed, think),
                               embed(codec_embed, [speaker]),
                               embed(codec_embed, [2148, 2149])], axis=1)  # codec pad, codec start
head = np.concatenate([np.repeat(tts_pad, codec_prefix.shape[1] - 2, axis=1), tts_bos], axis=1) + codec_prefix[:, :-1]
body = hidden[:, 3:-5]  # the text, without "<|im_start|>assistant\\n" and the closing tokens
body = np.concatenate([body, tts_eos], axis=1) + embed(codec_embed, [2148] * (body.shape[1] + 1))
tail = tts_pad + embed(codec_embed, [2149])  # the audio starts here
step_embeds = np.concatenate([hidden[:, :3], head, body, tail], axis=1).astype(np.float32)

temperature, top_k, repetition_penalty = 0.9, 50, 1.05  # as phoonnx, for the talker and the code predictor
rng = np.random.default_rng()


def sample(logits):  # temperature, then top-k, as Hugging Face generate does
    x = np.asarray(logits, np.float64).reshape(-1) / temperature
    x = np.where(x < np.partition(x, -top_k)[-top_k], -np.inf, x)
    p = np.exp(x - x.max())
    return int(rng.choice(p.size, p=p / p.sum()))


def next_past(out, layers):
    return {{f"past_key_values.{{i}}.{{kind}}": out[f"present.{{i}}.{{kind}}"] for i in range(layers) for kind in ("key", "value")}}


past = {{f"past_key_values.{{i}}.{{kind}}": np.zeros((1, 8, 0, 128), np.float32) for i in range(28) for kind in ("key", "value")}}
talker_names = [o.name for o in talker.get_outputs()]
prefill_names = [o.name for o in predictor_prefill.get_outputs()]
step_names = [o.name for o in predictor_step.get_outputs()]
position, codes, generated = 0, [], []
for step in range(4096):  # one frame (80 ms) a step, until the end-of-speech token 2150
    width = step_embeds.shape[1]
    out = dict(zip(talker_names, talker.run(None, {{
        "inputs_embeds": step_embeds, "position_ids": np.arange(position, position + width)[None], **past}})))
    position, past = position + width, next_past(out, 28)
    logits = out["logits"][0, -1].astype(np.float64)
    if generated:  # the repetition penalty over the tokens so far
        seen = np.unique(generated)
        logits[seen] = np.where(logits[seen] < 0, logits[seen] * repetition_penalty, logits[seen] / repetition_penalty)
    end = logits[2150]
    logits[2048:] = -np.inf  # control tokens are never sampled, except the end of speech after 2 frames
    logits[2150] = end if step >= 2 else -np.inf
    token = sample(logits)
    generated.append(token)
    if token == 2150:
        break
    # the code predictor writes code groups 1 to 15 of this frame, one group a step
    group0 = embed(codec_embed, [token])
    sub = dict(zip(prefill_names, predictor_prefill.run(None, {{
        "inputs_embeds": np.concatenate([out["last_hidden"], group0], axis=1).astype(np.float32)}})))
    groups = [sample(sub["logits"][0])]
    for group in range(1, 15):
        sub = dict(zip(step_names, predictor_step.run(None, {{
            "inputs_embeds": sub_embed(groups[-1], group - 1).astype(np.float32), "step": np.array(group, np.int64),
            "position_ids": np.array([[group + 1]], np.int64), **next_past(sub, 5)}})))
        groups.append(sample(sub["logits"][0]))
    codes.append([token] + groups)
    step_embeds = (group0 + sub_embed(groups, range(15)).sum(axis=1, keepdims=True) + tts_pad).astype(np.float32)

# 12.5 frames a second; phoonnx decodes 300 frames (24 s) at a time, and one sentence fits in one call
audio = decoder.run(None, {{"codes": np.array(codes, np.int64).T[None]}})[0].reshape(-1)

{_write_wav("24000")}'''


_PARLER_FEEDS = {"text_encoder_path": {"input_ids", "attention_mask"},
                 "model": {"codec_input_ids", "prompt_input_ids", "encoder_hidden_states", "encoder_attention_mask"},
                 "decoder_decode_path": {"codec_input_ids", "encoder_attention_mask"},
                 "dac_decoder_path": {"audio_codes"}}


def _indic_parler(config: Dict[str, Any], files: Dict[str, Any], *, repo_id: str, text: Optional[str]) -> str:
    for key, names in _PARLER_FEEDS.items():
        if key not in files:
            raise UnsupportedExample(f"an Indic Parler voice with no {key}")
        if "inputs" in files[key] and not names <= set(files[key]["inputs"]):
            raise UnsupportedExample(f"Indic Parler {files[key]['file']} has inputs {sorted(files[key]['inputs'])}")
    if "inputs" in files["decoder_decode_path"]:
        past = {n for n in files["decoder_decode_path"]["inputs"] if n.startswith("past_key_values.")}
        if set(files["decoder_decode_path"]["inputs"]) - past - _PARLER_FEEDS["decoder_decode_path"]:
            raise UnsupportedExample("the Indic Parler decode graph takes inputs the example does not feed")
    for key in ("prompt_tokenizer_path", "description_tokenizer_path"):
        if key not in files:
            raise UnsupportedExample(f"an Indic Parler voice with no {key}")
    options = config.get("engine_options") or {}
    if not options.get("description"):
        raise UnsupportedExample("an Indic Parler voice with no description")
    sentence = json.dumps(text, ensure_ascii=False) if text else f'"..."  # a sentence in {config.get("lang")}'
    return "\n".join([
        "# pip install onnxruntime numpy huggingface_hub tokenizers",
        "import wave",
        "",
        "import numpy as np",
        "import onnxruntime",
        "from huggingface_hub import hf_hub_download",
        "from tokenizers import Tokenizer",
        "",
        f'repo = "{repo_id}"',
        f"prompt_tokenizer = Tokenizer.from_file({_hub(files['prompt_tokenizer_path'], repo_id)})",
        f"description_tokenizer = Tokenizer.from_file({_hub(files['description_tokenizer_path'], repo_id)})",
        f"text_encoder = onnxruntime.InferenceSession({_hub(files['text_encoder_path'], repo_id)})",
        f"prefill = onnxruntime.InferenceSession({_hub(files['model'], repo_id)})",
        f"decode = onnxruntime.InferenceSession({_hub(files['decoder_decode_path'], repo_id)})",
        f"dac = onnxruntime.InferenceSession({_hub(files['dac_decoder_path'], repo_id)})",
        "",
        "# the voice is a description in words; write your own to get another voice",
        f"description = {json.dumps(options['description'], ensure_ascii=False)}",
        f"text = {sentence}",
        "ids = prompt_tokenizer.encode(text).ids",
        "description_ids = np.array([description_tokenizer.encode(description).ids])",
        "mask = np.ones_like(description_ids)",
        'hidden = text_encoder.run(None, {"input_ids": description_ids, "attention_mask": mask})[0]',
        "",
        "# 9 DAC codebooks under a delay pattern: codebook k runs k steps behind codebook 0",
        "max_length, column, book = 1501, np.arange(1501), np.arange(9)[:, None]",
        "pattern = np.where(column <= book, 1025, np.where(column >= max_length - 8 + book, 1024, -1))  # BOS, PAD, predict",
        "raw = np.full((9, 1), 1025)",
        "out = dict(zip([o.name for o in prefill.get_outputs()], prefill.run(None, {",
        '    "codec_input_ids": raw[None], "prompt_input_ids": np.array([ids]),',
        '    "encoder_hidden_states": hidden, "encoder_attention_mask": mask})))',
        "",
        "temperature = 1.0  # as upstream: plain sampling, as greedy decoding talks on past the sentence",
        "rng = np.random.default_rng()",
        "decode_names = [o.name for o in decode.get_outputs()]",
        "first, unfinished, cache = 0, np.ones(9, bool), {}",
        "for _ in range(max_length - 1):",
        "    # the prefill gives the self- and cross-attention cache; each step renews the self-attention part",
        '    cache.update({k.replace("present", "past_key_values"): v for k, v in out.items() if k != "logits"})',
        '    scores = out["logits"].reshape(9, -1).astype(np.float32)',
        "    # end of speech (1024) moves one codebook at a time, in order",
        "    if (raw[first] == 1024).any() and first < 8:",
        "        first += 1",
        "    scores[np.arange(9) > first, 1024] = -np.inf",
        "    x = scores.astype(np.float64) / temperature",
        "    probs = np.exp(x - x.max(axis=1, keepdims=True))",
        "    probs /= probs.sum(axis=1, keepdims=True)",
        "    step = np.array([rng.choice(probs.shape[1], p=row) for row in probs])",
        "    step = np.where(unfinished, step, 1024)",
        "    raw = np.concatenate([raw, step[:, None]], axis=1)",
        "    unfinished &= step != 1024",
        "    if not unfinished.any() or raw.shape[1] >= max_length:",
        "        break",
        "    forced = np.where(pattern[:, raw.shape[1] - 1] == -1, raw[:, -1], pattern[:, raw.shape[1] - 1])",
        "    out = dict(zip(decode_names, decode.run(None, {",
        '        "codec_input_ids": forced[None, :, None], "encoder_attention_mask": mask, **cache})))',
        "",
        "# undo the delay: codebook k keeps columns k + 1 to n - 9 + k; stop at the first end of speech",
        "codes = np.stack([raw[k, k + 1:raw.shape[1] - 8 + k] for k in range(9)])",
        "end = np.nonzero((codes >= 1024).any(axis=0))[0]",
        "codes = codes[:, :end[0] if len(end) else None]",
        'audio = dac.run(None, {"audio_codes": codes[None]})[0].reshape(-1)',
        "",
        _write_wav("44100"),
    ])


_OMNIVOICE_IO = {"model": ({"input_ids", "audio_mask"}, {"logits"}), "decoder_path": ({"codes"}, {"waveform_24k"})}


def _omnivoice(config: Dict[str, Any], files: Dict[str, Any], *, repo_id: str, text: Optional[str]) -> str:
    for key, (inputs, outputs) in _OMNIVOICE_IO.items():
        if key not in files:
            raise UnsupportedExample(f"an OmniVoice voice with no {key}")
        if "inputs" in files[key] and not (set(files[key]["inputs"]) == inputs and outputs <= set(files[key]["outputs"])):
            raise UnsupportedExample(f"OmniVoice {files[key]['file']} has inputs {sorted(files[key]['inputs'])}")
    if "bpe_tokenizer_path" not in files:
        raise UnsupportedExample("an OmniVoice voice with no tokenizer")
    lang = (config.get("engine_options") or {}).get("lang") or config.get("lang") or ""
    if not re.fullmatch(r"[a-z]{2,3}", lang):
        raise UnsupportedExample(f"OmniVoice language {lang!r} is not a language id of the model")
    sentence = json.dumps(text, ensure_ascii=False) if text else f'"..."  # a sentence in {lang}'
    return f'''# pip install onnxruntime numpy huggingface_hub tokenizers pydub "audioop-lts; python_version >= '3.13'"
import bisect
import math
import re
import unicodedata
import wave

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, split_on_silence
from tokenizers import Tokenizer

repo = "{repo_id}"
tokenizer = Tokenizer.from_file({_hub(files["bpe_tokenizer_path"], repo_id)})
backbone = onnxruntime.InferenceSession({_hub(files["model"], repo_id)})
decoder = onnxruntime.InferenceSession({_hub(files["decoder_path"], repo_id)})

lang, text = "{lang}", {sentence}

# OmniVoice writes a fixed number of codec frames (25 a second), estimated from the text:
# each character weighs its speaking time against a Latin letter, by Unicode category or by script
ENDS = [0x58F, 0x8FF, 0xDFF, 0xEFF, 0x109F, 0x10FF, 0x11FF, 0x139F, 0x177F, 0x17FF, 0x18FF, 0x1C7F, 0x1CBF, 0x1CFF,
        0x1EFF, 0x30FF, 0x312F, 0x318F, 0xA4CF, 0xA7FF, 0xA82F, 0xA87F, 0xA95F, 0xA97F, 0xAAFF, 0xAB2F, 0xABBF, 0xABFF,
        0xD7AF, 0xFAFF, 0xFDFF, 0xFE6F, 0xFEFF, 0xFFEF]  # the last code point of each run of scripts
WEIGHTS = [1, 1.5, 1.8, 1.5, 1.8, 1, 2.5, 3, 1, 1.8, 1, 1.8, 1, 1.8, 1, 2.2, 3, 2.5, 3, 1, 1.8, 1, 1.8, 2.5, 1.8, 3, 1,
           1.8, 2.5, 3, 1.5, 1, 1.5, 1]


def weight(char):
    category = unicodedata.category(char)[0]
    if "A" <= char <= "Z" or "a" <= char <= "z":
        return 1.0
    if char == " " or category == "Z":
        return 0.2
    if char == "\\u0640" or category == "M":  # marks and the Arabic tatweel are silent
        return 0.0
    if category in "PS":
        return 0.5
    if category == "N":
        return 3.5
    i = bisect.bisect_left(ENDS, ord(char))
    return WEIGHTS[i] if i < len(ENDS) else 3.0 if ord(char) > 0x20000 else 1.0


frames = sum(map(weight, text)) / (sum(map(weight, "Nice to meet you.")) / 25)  # "Nice to meet you." takes 25 frames
if frames < 50:
    frames = 50 * (frames / 50) ** (1 / 3)  # a short text gets more time
length = max(1, int(frames))

text = re.sub(r"[ \\t]+", " ", re.sub(r"[\\r\\n]+", "", text.strip()).replace("（", "(").replace("）", ")"))
text = re.sub(r"(?<=[一-鿿])\\s+|\\s+(?=[一-鿿])", "", text)  # no spaces beside Chinese characters
tags = r"(\\[(?:laughter|sigh|confirmation-en|question-(?:en|ah|oh|ei|yi)|surprise-(?:ah|oh|wa|yo)|dissatisfaction-hnn)\\])"
ids = tokenizer.encode(f"<|lang_start|>{{lang}}<|lang_end|><|instruct_start|>None<|instruct_end|>").ids
ids += [i for part in re.split(tags, f"<|text_start|>{{text}}<|text_end|>") if part
        for i in tokenizer.encode(part).ids]  # a non-verbal tag such as [laughter] is encoded on its own

# masked diffusion: all 8 codebooks of every frame start as the mask token 1024, and 32 steps fill them in
cond = np.array([[ids + [1024] * length] * 8], np.int64)  # the text, copied to the 8 codebook rows, then the frames
cond_mask = np.arange(cond.shape[2])[None] >= len(ids)
uncond, uncond_mask = cond[:, :, len(ids):].copy(), cond_mask[:, len(ids):]  # the frames alone, for guidance
tokens = np.full((8, length), 1024)
t = np.linspace(0, 1, 33)
t = 0.1 * t / (1 - 0.9 * t)  # most slots stay masked until the late steps
remaining = 8 * length
rng = np.random.default_rng()


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


for step in range(32):
    take = remaining if step == 31 else min(math.ceil(8 * length * (t[step + 1] - t[step])), remaining)
    remaining -= take
    if take <= 0:
        continue
    c = log_softmax(backbone.run(None, {{"input_ids": cond, "audio_mask": cond_mask}})[0][0, :, len(ids):].astype(np.float32))
    u = log_softmax(backbone.run(None, {{"input_ids": uncond, "audio_mask": uncond_mask}})[0][0].astype(np.float32))
    log_probs = log_softmax(c + 2.0 * (c - u))  # classifier-free guidance, scale 2
    log_probs[..., 1024] = -np.inf
    predicted = log_probs.argmax(-1)
    # fill the most confident slots, the low codebooks first, with Gumbel noise at temperature 5
    scores = log_probs.max(-1) - np.arange(8, dtype=np.float32)[:, None] * 5.0
    scores = scores / 5.0 - np.log(-np.log(rng.random(scores.shape) + 1e-10) + 1e-10)
    scores[tokens != 1024] = -np.inf
    chosen = np.argpartition(-scores.ravel(), take - 1)[:take]
    tokens.flat[chosen] = predicted.flat[chosen]
    cond[0, :, len(ids):] = tokens
    uncond[0] = tokens

audio = decoder.run(None, {{"codes": tokens[:, None, :]}})[0].reshape(-1)

# as phoonnx: cut silences longer than 0.5 s, keep 0.1 s at each edge, scale the peak to 0.5, fade and pad 0.1 s
segment = AudioSegment(data=(audio * 32768).clip(-32768, 32767).astype(np.int16).tobytes(), sample_width=2,
                       frame_rate=24000, channels=1)
segment = sum(split_on_silence(segment, min_silence_len=500, silence_thresh=-50, keep_silence=500, seek_step=10),
              AudioSegment.silent(duration=0, frame_rate=24000))
for _ in range(2):  # the start, then the end
    segment = segment[max(0, detect_leading_silence(segment, silence_threshold=-50) - 100):].reverse()
trimmed = np.array(segment.get_array_of_samples()).astype(np.float32) / 32768
audio = trimmed if trimmed.size else audio
audio = audio / np.abs(audio).max() * 0.5
fade = min(2400, audio.size // 2)
audio[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
audio[audio.size - fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
audio = np.concatenate([np.zeros(2400, np.float32), audio, np.zeros(2400, np.float32)])

{_write_wav("24000")}'''


_ORPHEUS_IO = {"model": ({"input_ids", "attention_mask"}, {"logits"}), "snac_decoder_path": (set(), {"audio_values"})}


def _orpheus(config: Dict[str, Any], files: Dict[str, Any], *, repo_id: str, text: Optional[str]) -> str:
    for key, (inputs, outputs) in _ORPHEUS_IO.items():
        if key not in files:
            raise UnsupportedExample(f"an Orpheus voice with no {key}")
        if "inputs" in files[key] and not (inputs <= set(files[key]["inputs"]) and outputs <= set(files[key]["outputs"])):
            raise UnsupportedExample(f"Orpheus {files[key]['file']} has inputs {sorted(files[key]['inputs'])}")
    if {n for n in files["model"].get("inputs", ()) if not n.startswith("past_key_values.")} - _ORPHEUS_IO["model"][0]:
        raise UnsupportedExample(f"Orpheus {files['model']['file']} takes inputs the example does not feed")
    if "inputs" in files["snac_decoder_path"] and len(files["snac_decoder_path"]["inputs"]) != 3:
        raise UnsupportedExample("an Orpheus SNAC decoder that does not read three code streams")
    if "tokenizer_path" not in files:
        raise UnsupportedExample("an Orpheus voice with no tokenizer")
    weights = [f for key, f in files.items() if key.startswith("lm_weights_path")]
    if not weights or any(f["repo"] != repo_id for f in weights):
        raise UnsupportedExample("an Orpheus voice whose weights are not in its repo")
    names = sorted((f["file"] for f in weights), key=lambda name: (len(name), name))
    base = files["model"]["file"] + "_data"
    if names == [base] + [f"{base}_{i}" for i in range(1, len(names))]:  # model.onnx_data, model.onnx_data_1, ...
        weight_names = f'["{base}"] + [f"{base}_{{i}}" for i in range(1, {len(names)})]'
    else:
        weight_names = "[" + ",\n             ".join(f'"{name}"' for name in names) + "]"
    voice = (config.get("engine_options") or {}).get("voice")
    if not voice:
        raise UnsupportedExample("an Orpheus voice with no voice name")
    sentence = json.dumps(text, ensure_ascii=False) if text else f'"..."  # a sentence in {config.get("lang")}'
    return f'''# pip install onnxruntime numpy huggingface_hub tokenizers
import wave

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

repo = "{repo_id}"
tokenizer = Tokenizer.from_file({_hub(files["tokenizer_path"], repo_id)})
for name in {weight_names}:
    hf_hub_download(repo, name)  # the weights, beside the graph: a 3B model in fp32 needs about 14 GB of memory
lm = onnxruntime.InferenceSession({_hub(files["model"], repo_id)})
snac = onnxruntime.InferenceSession({_hub(files["snac_decoder_path"], repo_id)})

voice, text = "{voice}", {sentence}
# the prompt as upstream's server reads it: <|begin_of_text|> (128000) twice, the start of the human turn between,
# then the end of the turn, the start of the AI turn and the start of speech
ids = [128000, 128259] + tokenizer.encode(f"{{voice}}: {{text}}").ids + [128009, 128260, 128261, 128257]

temperature, top_p, repetition_penalty = 0.6, 0.8, 1.3  # upstream's serving settings, as phoonnx
rng = np.random.default_rng()
past = {{i.name: np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32) for i in lm.get_inputs() if i.name.startswith("past")}}
history, tokens, step_ids = list(ids), [], ids
for _ in range(1200):  # about 82 tokens a second: 14.6 s
    logits, *present = lm.run(None, {{"input_ids": np.array([step_ids], np.int64),
                                     "attention_mask": np.ones((1, len(history)), np.int64), **past}})
    past = dict(zip(past, present))
    scores = logits[0, -1].astype(np.float32)
    seen = np.unique(history)  # the prompt and every token so far, as vLLM penalises
    scores[seen] = np.where(scores[seen] < 0, scores[seen] * repetition_penalty, scores[seen] / repetition_penalty)
    if temperature > 0:  # temperature, then top-p, in vLLM's order
        x = (scores / temperature).astype(np.float64)
        probs = np.exp(x - x.max())
        probs /= probs.sum()
        order = np.argsort(probs)[::-1]
        keep = order[:np.searchsorted(np.cumsum(probs[order]), top_p) + 1]
        p = np.zeros_like(probs)
        p[keep] = probs[keep]
        token = int(rng.choice(p.size, p=p / p.sum()))
    else:
        token = int(scores.argmax())
    if not 128266 <= token <= 156937:  # end of speech, end of turn, or a text token
        break
    tokens.append(token)
    history.append(token)
    step_ids = [token]

# 7 tokens a frame, each from its own 4096-code table, into the 3 SNAC streams at rates 1, 2 and 4
codes = [t - 128266 - i % 7 * 4096 for i, t in enumerate(tokens)]
frames = [codes[i:i + 7] for i in range(0, len(codes) - 6, 7)]
frames = [f for f in frames if all(0 <= c < 4096 for c in f)]  # a code outside its table makes the frame corrupt
streams = [[f[0] for f in frames], [c for f in frames for c in (f[1], f[4])],
           [c for f in frames for c in (f[2], f[3], f[5], f[6])]]
audio = snac.run(None, {{i.name: np.array([s], np.int64) for i, s in zip(snac.get_inputs(), streams)}})[0].reshape(-1)

{_write_wav("24000")}'''


# engines whose example a template writes out whole from the voice index entry: see standalone_example
INDEX_TEMPLATES: Dict[str, Callable[..., str]] = {
    "arktts": _arktts,
    "chatterbox": _chatterbox,
    "f5tts": _f5tts,
    "indic_parler": _indic_parler,
    "llasa": _llasa,
    "magpie": _magpie,
    "mosstts": _mosstts,
    "neutts": _neutts,
    "omnivoice": _omnivoice,
    "orpheus": _orpheus,
    "outetts": _outetts,
    "pockettts": _pockettts,
    "qwen3tts": _qwen3tts,
    "rapido": _rapido,
    "sparktts": _sparktts,
    "supertonic": _supertonic,
}
# engines of several graphs with no example, and why: a template that was not run against phoonnx is not shipped
REFUSED_ENGINES: Dict[str, str] = {}


def _varint(data: bytes, i: int) -> Tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[i]
        value |= (byte & 0x7F) << shift
        i += 1
        if not byte & 0x80:
            return value, i
        shift += 7


def _fields(data: bytes) -> List[Tuple[int, Any]]:
    """The (field number, value) pairs of a serialized protobuf message."""
    out, i = [], 0
    while i < len(data):
        key, i = _varint(data, i)
        wire = key & 7
        if wire == 0:
            value, i = _varint(data, i)
        elif wire == 2:
            size, i = _varint(data, i)
            value, i = data[i:i + size], i + size
        elif wire in (1, 5):
            value, i = data[i:i + (8 if wire == 1 else 4)], i + (8 if wire == 1 else 4)
        else:
            raise ValueError(f"protobuf wire type {wire}")
        out.append((key >> 3, value))
    return out


def _value_info(data: bytes) -> Optional[Tuple[str, int]]:
    """The name and rank of an ONNX ValueInfoProto, or None if *data* is not one."""
    try:
        fields = dict(_fields(data))
        name = fields[1].decode()
        tensor = dict(_fields(dict(_fields(fields[2]))[1]))
        rank = sum(1 for number, _ in _fields(tensor.get(2, b"")) if number == 1)
    except (KeyError, IndexError, ValueError, TypeError, AttributeError):
        return None
    return (name, rank) if re.fullmatch(r"[A-Za-z_][\w.:/-]*", name) else None


def graph_io(data: bytes) -> Dict[str, Dict[str, int]]:
    """The graph inputs (field 11) and outputs (field 12) found in the bytes of an ONNX file, with their rank."""
    found: Dict[str, Dict[str, int]] = {"inputs": {}, "outputs": {}}
    for tag, kind in ((0x5A, "inputs"), (0x62, "outputs")):
        for match in re.finditer(bytes([tag]), data):
            try:
                size, start = _varint(data, match.start() + 1)
            except IndexError:
                continue
            info = _value_info(data[start:start + size]) if start + size <= len(data) else None
            if info:
                found[kind].setdefault(*info)
    return found


def onnx_io(repo_id: str, filename: str, tail: int = 1 << 20, tries: int = 5) -> Dict[str, Dict[str, int]]:
    """The graph inputs and outputs of an ONNX file on the Hub, by name with their rank.

    An ONNX file serialises the graph's inputs and outputs after its weights, so
    the last megabyte holds them, and one range request reads it. The request
    sends the user's Hub token, as ``hf_hub_download`` does, so a gated repo
    reads and the rate limit is the account's. A rate-limited request is tried
    again after the wait the Hub names, or after a wait that doubles each time.
    """
    import requests
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    headers = {**build_hf_headers(), "Range": f"bytes=-{tail}"}
    for attempt in range(tries):
        response = requests.get(hf_hub_url(repo_id, filename), headers=headers, timeout=60)
        if response.status_code != 429 or attempt == tries - 1:
            break
        wait = response.headers.get("Retry-After", "")
        time.sleep(int(wait) if wait.isdigit() else 10 * 2 ** attempt)
    response.raise_for_status()
    return graph_io(response.content)
