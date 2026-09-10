"""
A corpus of voice configs covering every loader in :mod:`phoonnx.config_loaders`.

Each case is a config dict plus the companion arguments that identify its
format, and every case is run against :data:`KWARG_SETS` — so the corpus
exercises the detection registry and the override contract together.  Cases
that *raise* are part of the corpus: which error a malformed config produces is
behaviour too.

The corpus is data, not assertions.  :mod:`tests.differential.runner` turns it
into a snapshot; ``tests/test_config_differential.py`` compares that snapshot
against the committed golden file, and the same runner compares two git refs
when a refactor needs to prove it changed nothing.
"""
import json
import os
from typing import Any, Dict, List, Tuple


def _write_tokens(tmpdir: str) -> Tuple[str, str]:
    """Write the two shapes of external token file, returning their paths."""
    txt = os.path.join(tmpdir, "phonemes.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(f"{i} {c}" for i, c in enumerate("abcdefgh_ ")))
    js = os.path.join(tmpdir, "tokens.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump({c: i for i, c in enumerate("abcdef_")}, f)
    return txt, js


def _piper(phoneme_type: str) -> Dict[str, Any]:
    return {"piper_version": "1.0", "phoneme_type": phoneme_type,
            "phoneme_id_map": {"a": [1], "b": [2], "_": [0]},
            "language": {"code": "pt"}, "audio": {"sample_rate": 22050},
            "num_symbols": 3}


def _mimic3(phonemizer: str) -> Dict[str, Any]:
    return {"phonemizer": phonemizer,
            "phonemes": {"blank_between": "tokens", "phoneme_separator": " "},
            "text_language": "de", "audio": {"sample_rate": 22050}}


COQUI = {"characters": {"characters_class": "TTS.tts.models.vits.VitsCharacters",
                        "characters": "abcdef", "punctuations": "!?.", "pad": "_",
                        "eos": "&", "bos": "*", "blank": "<BLNK>"},
         "datasets": [{"language": "gl"}], "audio": {"sample_rate": 22050}}
NATIVE = {"phoonnx_version": "1.0", "engine": "vits", "phoneme_type": "espeak",
          "alphabet": "ipa", "lang_code": "pt", "audio": {"sample_rate": 22050},
          "phoneme_id_map": {"a": 1, "b": 2, "_": 0},
          "inference": {"add_diacritics": True, "diacritizer_model": "rawi"}}
# a config whose keys are present but explicitly null: every fallback must be
# `get(k) or default`, never `get(k, default)`
NATIVE_NULLS = {"phoonnx_version": "1.0", "engine": None, "phoneme_type": None,
                "alphabet": None, "lang_code": None, "pad": None, "blank": None,
                "phoneme_id_map": {"a": 1, "_": 0}, "inference": {"add_diacritics": None}}
# a native config whose phoneme_id_map has piper's list-valued shape: only the
# registry order keeps the piper loader from claiming it
NATIVE_PIPER_SHAPED = {"phoonnx_version": "1.0", "phoneme_type": "espeak",
                       "phoneme_id_map": {"a": [1], "b": [2], "_": [0]},
                       "lang_code": "pt", "audio": {"sample_rate": 22050}}
NATIVE_BAD_ALPHABET = {"phoonnx_version": "1.0", "alphabet": "not-an-alphabet",
                       "phoneme_id_map": {"a": 1, "_": 0}}
CANONICAL = {"engine": "matcha", "phoneme_type": "gruut", "alphabet": "ipa",
             "phoneme_id_map": {"a": 1, "_": 0}, "audio": {"sample_rate": 22050},
             "inference": {"add_diacritics": True}}
CANONICAL_BARE = {"phoneme_id_map": {"a": 1, "_": 0}}
CANONICAL_BAD_ALPHABET = {"engine": "matcha", "alphabet": "not-an-alphabet",
                          "phoneme_id_map": {"a": 1, "_": 0}}
# alphacep vosk-tts: piper-shaped, but - unlike a real piper config - carries
# no phoneme_type of its own; that absence (which PiperLoader's own detect()
# already requires) plus the romanised-Russian phoneme inventory ("a0",
# "sch") is what routes it to VoskLoader
VOSK = {"espeak": {"voice": "ru"},
        "phoneme_id_map": {"a0": [1], "sch": [2], "_": [0]},
        "audio": {"sample_rate": 22050}}
# a real piper config that happens to define the same phonemes: the inventory
# alone is not the signature, a declared phoneme_type keeps it as piper
VOSK_LOOKALIKE_PIPER = {"phoneme_type": "espeak", "espeak": {"voice": "ru"},
                        "phoneme_id_map": {"a0": [1], "sch": [2], "_": [0]},
                        "audio": {"sample_rate": 22050}}
# an explicitly named vosk engine without the tell-tale inventory: the
# declared name must still win over the shape-sniffer
VOSK_EXPLICIT_ENGINE = {"engine": "vosk", "phoneme_id_map": {"p": [1], "_": [0]},
                        "audio": {"sample_rate": 22050}}

# (engine name, its codec's sample rate); the loaders differ in nothing else
CODEC_ENGINES = [("llasa", 16000), ("neutts", 24000), ("orpheus", 24000),
                 ("mosstts", 48000), ("supertonic", 44100), ("pockettts", 24000),
                 ("sparktts", 16000), ("indic_parler", 44100), ("omnivoice", 24000),
                 ("magpie", 22050), ("arktts", 44100), ("qwen3tts", 24000),
                 ("outetts", 24000)]

KWARG_SETS: List[Dict[str, Any]] = [
    {},
    {"lang_code": "xx"},
    {"phoneme_type": "gruut"},
    {"alphabet": "arpa"},
    {"engine": "coqui"},
    {"lang_code": "xx", "phoneme_type": "espeak", "alphabet": "ipa", "engine": "matcha"},
    {"lang_tokens": {"pt": "<pt>"}, "engine_params": {"x": 1}},
]


def build(tmpdir: str) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Return ``(name, config, companion kwargs)`` triples, writing token files to *tmpdir*."""
    tokens_txt, tokens_json = _write_tokens(tmpdir)
    cases: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for phoneme_type in ("espeak", "text", "pygoruut", "bogus"):
        cases.append((f"piper:{phoneme_type}", _piper(phoneme_type), {}))
    for phonemizer in ("gruut", "espeak", "epitran", "symbols"):
        cases.append((f"mimic3:{phonemizer}", _mimic3(phonemizer), {"tokens_txt": tokens_txt}))
        cases.append((f"mimic3-notokens:{phonemizer}", _mimic3(phonemizer), {}))
    cases += [
        ("coqui", COQUI, {}),
        ("native", NATIVE, {}),
        ("native-nulls", NATIVE_NULLS, {}),
        ("native-piper-shaped", NATIVE_PIPER_SHAPED, {}),
        ("native-bad-alphabet", NATIVE_BAD_ALPHABET, {}),
        ("canonical", CANONICAL, {}),
        ("canonical-bare", CANONICAL_BARE, {}),
        ("canonical-bad-alphabet", CANONICAL_BAD_ALPHABET, {}),
        ("vosk", VOSK, {}),
        ("vosk-lookalike-piper", VOSK_LOOKALIKE_PIPER, {}),
        ("vosk-explicit-engine", VOSK_EXPLICIT_ENGINE, {}),
        ("transformers", {"num_symbols": 5}, {"vocab": {"a": 1, "b": 2, "_": 0}}),
        ("transformers-tokcfg", {"num_symbols": 5},
         {"vocab": {"a": 1, "_": 0},
          "tokenizer_config": {"add_blank": False, "language": "ro", "pad_token": "<pad>"}}),
        ("transformers-tokcfg-nolang", {"num_symbols": 5},
         {"vocab": {"a": 1, "_": 0}, "tokenizer_config": {"add_blank": True}}),
        ("sherpa-txt", {}, {"tokens_txt": tokens_txt}),
        ("sherpa-json", {"pad": "_"}, {"tokens_txt": tokens_json}),
    ]
    for engine, _ in CODEC_ENGINES:
        cases.append((f"codec-cfg:{engine}", {"engine": engine}, {}))
        cases.append((f"codec-kwarg:{engine}", {}, {"engine": engine}))
    cases.append(("chatterbox-nobpe", {"engine": "chatterbox"}, {}))
    # an engine kwarg disagreeing with a config that names a codec engine: the
    # config wins, because the codec loader was selected by that name
    cases.append(("codec-conflict", {"engine": "llasa"}, {"engine": "piper"}))
    return cases
