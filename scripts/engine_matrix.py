"""Write the architecture matrix of docs/engines.md from the engine registries.

Run ``python scripts/engine_matrix.py`` after a change to an engine registry,
to ``phoonnx/voice_index`` or to ``tests/fixtures/id_parity.json``.
``tests/test_engine_matrix.py`` fails while docs/engines.md differs from what
this script writes.
"""
import json
from collections import Counter
from pathlib import Path

from phoonnx.config import ENGINE_ALIASES
from phoonnx.engines import get_adapter, list_engines
from phoonnx_train.engines import list_engines as list_training_engines

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "engines.md"
INDEX = ROOT / "phoonnx" / "voice_index"
FIXTURE = ROOT / "tests" / "fixtures" / "id_parity.json"
CONVERSION = ROOT / "scripts" / "conversion"
BEGIN = "<!-- engine-matrix:begin -->"
END = "<!-- engine-matrix:end -->"

# Config formats that TTSVoice loads with a VITS adapter.
VITS_FORMATS = {"phoonnx", "piper", "mimic3", "coqui", "transformers"}

# Training engines registered under a name that is not their architecture.
TRAINING_ARCH = {
    "mixer": "mixertts",
    "speedyspeech": "fastpitch",
    "styletts2-aligner": "styletts2",
    "styletts2-plbert": "styletts2",
    "styletts2-pitch": "styletts2",
}

# Export paths besides the export_onnx of a training engine.
CONVERTERS = {
    "vits": "`export_piper`, `export_sherpa`",
    "vits_streaming": "`ensure_split_vits` on a `vits` export",
}

# The folders of scripts/conversion by the architecture whose upstream checkpoint they
# export. None: the folder holds tools around an export, not a model exporter.
CONVERSION_DIRS = {
    "arktts": "arktts",
    "coqui_fastpitch_export": "fastpitch",
    "coqui_glowtts_export": "glowtts",
    "coqui_vits_export": "vits",
    "indicparler": "indic_parler",
    "inflect": "vits",  # a single-graph VITS
    "llasa": "llasa",
    "magpie": "magpie",
    "neutts": "neutts",
    "orpheus": None,  # parity checks of the upstream graphs
    "outetts": "outetts",
    "qwen3tts": "qwen3tts",
    "shami_tts": "shami",
    "styletts2": "styletts2",
    "yourtts": "yourtts",
    "zipvoice": None,  # the mel front end only
}

# The roadmap task that closes a gap: an architecture with no training engine,
# or an inference engine with no index voice.
TRAINING_TASKS = {name: "R4.8" for name in (
    "arktts", "chatterbox", "f5tts", "indic_parler", "llasa", "magpie", "mosstts", "neutts",
    "omnivoice", "orpheus", "outetts", "pockettts", "qwen3tts", "sparktts", "tinyvoca", "xvits",
    "shami", "vits_prior_split", "vosk", "vits_streaming")}
VOICE_TASKS = {
    "tinyvoca": "R3.4",
    "vits_streaming": "R2.7",
    "xvits": "R2.7",
    "yourtts": "R2.7",
    "zipvoice": "R2.7",
}


def architecture(entry: dict) -> str:
    """The inference engine that loads an index entry, as TTSVoice picks it."""
    engine = ENGINE_ALIASES.get(entry["engine"], entry["engine"])
    if engine in list_engines():
        return engine
    if engine in VITS_FORMATS:
        return "vits_prior_split" if "decode_path" in (entry.get("aux_model_urls") or {}) else "vits"
    raise ValueError(f"{entry['voice_id']}: no inference engine loads the format {engine!r}")


def index_entries() -> dict:
    entries = {}
    for path in sorted(INDEX.glob("*.json")):
        entries.update(json.loads(path.read_text(encoding="utf-8")))
    return entries


def voice_counts() -> Counter:
    return Counter(architecture(entry) for entry in index_entries().values())


def fixture_architectures() -> set:
    entries = index_entries()
    voices = json.loads(FIXTURE.read_text(encoding="utf-8"))["voices"]
    return {architecture(entries[voice_id]) for voice_id in voices}


def training_engines() -> dict:
    engines = {}
    for name in list_training_engines():
        engines.setdefault(TRAINING_ARCH.get(name, name), []).append(name)
    return engines


def render_table() -> str:
    counts = voice_counts()
    fixtures = fixture_architectures()
    training = training_engines()
    inference = {name: type(get_adapter(name)).__name__ for name in list_engines()}
    lines = [
        "| Architecture | Inference | Training | Export | Index voices | Id fixture | Open task |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arch in sorted(set(inference) | set(training)):
        names = training.get(arch, [])
        export = ["`export_onnx`"] if names else []
        if arch in CONVERTERS:
            export.append(CONVERTERS[arch])
        export += [f"`scripts/conversion/{folder}`" for folder, target in CONVERSION_DIRS.items() if target == arch]
        lines.append("| " + " | ".join((
            f"`{arch}`",
            f"`{inference[arch]}`" if arch in inference else "none",
            ", ".join(f"`{name}`" for name in names) or TRAINING_TASKS.get(arch, "none"),
            ", ".join(export) or "none",
            str(counts[arch]),
            "yes" if arch in fixtures else "no",
            VOICE_TASKS.get(arch, ""),
        )) + " |")
    return "\n".join(lines)


def render() -> str:
    """docs/engines.md with the table between the markers replaced."""
    text = DOCS.read_text(encoding="utf-8")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{BEGIN}\n{render_table()}\n{END}{tail}"


if __name__ == "__main__":
    DOCS.write_text(render(), encoding="utf-8")
