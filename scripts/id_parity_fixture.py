"""Build and check the token-id fixture that phoonnx and phoonnx.js share.

Each row of ``tests/fixtures/id_parity.json`` names a voice of the voice index,
a text and the token ids phoonnx feeds the model for that text, one list per
sentence. The fixture also stores the small text files of each voice (config,
vocabulary), so a runtime computes the ids with no download and no model.

    python scripts/id_parity_fixture.py           # rewrite the fixture
    python scripts/id_parity_fixture.py --check   # exit 1 when it is out of date

The voice files come from the pinned URLs of the voice index. A new row needs
the voice in ``ROWS`` and a run of this script.
"""
import argparse
import contextlib
import json
import re
import sys
import tempfile
import urllib.request
from pathlib import Path
from unittest import mock

from phoonnx import model_manager, voice as voice_module
from phoonnx.engines import get_adapter
from scriptconv_copyleft.phonemizers.mul import EspeakPhonemizer

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "id_parity.json"
FILE_FIELDS = ("config_url", "vocab_url", "tokenizer_config_url", "tokens_url")

ROWS = {
    "OpenVoiceOS/phoonnx_oc_dii_unicode": [
        "bonjorn lo monde",
        "Bonjorn, lo monde",
        "Ço qu'a dit: « Òc ! »",
        "Bonjorn, lo monde. Aquesta votz parla dins ton navegador.",
    ],
    "OpenVoiceOS/phoonnx_an_dii_unicode": [
        "ola esta voz charra en o tuyo navegador",
        "Ola! Esta voz charra en o tuyo navegador.",
    ],
    "OpenVoiceOS/phoonnx_eu-ES_dii_espeak": [
        "Kaixo mundua",
        "Kaixo mundua. Hau proba bat da.",
    ],
    "OpenVoiceOS/phoonnx_es-ES_dii_espeak": [
        "¡Hola! Esta voz habla en tu navegador.",
    ],
    "OpenVoiceOS/pipertts_en-GB_dii": [
        "Hello this voice runs entirely in your browser",
        "Hello! This voice runs entirely in your browser.",
    ],
    "facebook/mms-tts-eng-English": [
        "Hello world. This voice speaks in your browser.",
        "How are you today? Fine, thanks.",
    ],
    "coqui/bg-cv-vits": [
        "Здравей, свят. Това е тест.",
    ],
}


def voice_files(entry: dict) -> dict:
    """The text files of a voice index entry, keyed by the entry field that names them."""
    files = {}
    for field in FILE_FIELDS:
        if entry.get(field):
            with urllib.request.urlopen(entry[field], timeout=60) as response:
                files[field] = json.loads(response.read()) if field != "tokens_url" \
                    else response.read().decode("utf-8")
    return files


@contextlib.contextmanager
def loaded_voice(entry: dict, files: dict):
    """The voice of an index entry, loaded from ``files`` with no model.

    The ONNX session is absent and no file is downloaded.
    """
    with tempfile.TemporaryDirectory() as folder:
        paths = {}
        for field, content in files.items():
            path = Path(folder) / field
            path.write_text(content if isinstance(content, str) else json.dumps(content),
                            encoding="utf-8")
            paths[entry[field]] = path
        with mock.patch.object(model_manager, "_resolve", lambda url, *a, **k: paths[url]), \
                mock.patch.object(model_manager.TTSModelInfo, "_fetch_onnx",
                                  lambda self, url: Path(folder) / "model.onnx"), \
                mock.patch.object(model_manager, "make_session", lambda *a, **k: None), \
                mock.patch.object(voice_module, "make_session", lambda *a, **k: None), \
                mock.patch.object(voice_module, "detect_engine", lambda *a, **k: get_adapter("vits")):
            yield model_manager.TTSModelInfo(**entry).load()


def compute_ids(entry: dict, files: dict, text: str) -> list:
    """The token ids of each sentence of ``text``, as ``TTSVoice.prepare`` gives them."""
    with loaded_voice(entry, files) as voice:
        return [[int(i) for i in utterance.ids] for utterance in voice.prepare(text)]


def espeak_voice(entry: dict, files: dict) -> str:
    """The espeak-ng voice name phoonnx reads the voice's text with."""
    with loaded_voice(entry, files) as voice:
        return EspeakPhonemizer.get_lang(voice.config.lang_code)


def build() -> dict:
    manager = model_manager.TTSModelManager()
    manager.merge_default_voices()
    rows = []
    voices = {}
    for voice_id, texts in ROWS.items():
        entry = manager.voices[voice_id].to_dict()
        files = voice_files(entry)
        voices[voice_id] = {"phoneme_type": entry["phoneme_type"], "files": files}
        if entry["phoneme_type"] == "espeak":
            voices[voice_id]["espeak_voice"] = espeak_voice(entry, files)
        rows += [{"voice_id": voice_id, "text": text, "ids": compute_ids(entry, files, text)}
                 for text in texts]
    return {"voices": voices, "rows": rows}


def render(fixture: dict) -> str:
    """The fixture as JSON text, with each list of numbers on one line."""
    text = json.dumps(fixture, ensure_ascii=False, indent=1)
    return re.sub(r"\[\s*(-?\d+(?:,\s*-?\d+)*)\s*\]",
                  lambda m: "[" + re.sub(r"\s+", "", m.group(1)) + "]", text) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare, write nothing")
    args = parser.parse_args()
    text = render(build())
    if args.check:
        if FIXTURE.read_text(encoding="utf-8") != text:
            print(f"{FIXTURE} is out of date", file=sys.stderr)
            return 1
        return 0
    FIXTURE.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
