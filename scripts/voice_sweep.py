#!/usr/bin/env python3
"""Load and synthesise every voice in the phoonnx index, and compare two such runs.

`run` takes each index entry through the index path (TTSModelInfo.load), synthesises
a few sentences in the voice's language with TTSVoice.synthesize, and writes one JSON
line per voice: load and synthesis status, duration, peak, RMS, NaN, silence, real-time
factor, and, where a verified ASR model exists for the language, the transcript with
its WER and CER. A failure or a timeout is a result, recorded like any other.

Voices that share model files form a group. A group runs in one worker process, so
its files download once, and with --evict they leave the cache when the group ends.
A worker that crashes or stops answering is replaced; the voice it was on is recorded
as "crash" or "timeout", and the rest of the group continues in a new worker.

`floor` checks each ASR model on real speech (the FLEURS test split) and writes the
languages whose WER is low enough for a voice's WER to mean something. `run` reads
that file with --asr-floor; a language that is not in it gets no WER.

`diff` compares two runs and lists the regressions: a voice that loaded and spoke in
the first run and fails in the second, or whose WER rose by more than --wer-margin.

    voice_sweep.py floor --out floor.json [--clips 15] [--max-wer 0.30]
    voice_sweep.py run --sentences sentences.tsv --out run.jsonl [--asr-floor floor.json]
                       [--jobs 3 --cpus 0-11] [--timeout 600] [--evict] [--engine NAME ...]
    voice_sweep.py diff A.jsonl B.jsonl [--wer-margin 0.15]

The sentence file is the voice snapshot's TSV (lang, id, category, text, source). Each
voice gets the rows of its language whose id is in --sentence-ids. A sentence written
for a script and not for the language (source "script:...") is synthesised but gets no
WER. The FLEURS sentences that `floor` stores are added for languages it verified.
Every count line states N in and N out, and the command fails when they differ.
"""
import argparse
import csv
import io
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import unicodedata
import warnings
from collections import Counter, defaultdict
from pathlib import Path

#: A file with one of these endings holds a graph or its weights.
GRAPH_FILE = re.compile(r"(\.onnx|\.ort)([._]data(_\d+)?)?$|\.data$|\.bin$|\.safetensors$")
HF_URL = re.compile(r"https?://huggingface\.co/(?P<repo>[^/]+/[^/]+)/(?:resolve|raw)/(?P<rev>[^/]+)/(?P<path>.+)")

#: onnx-asr model per primary language subtag, tried in order by `floor`.
ASR_CANDIDATES = {
    **{lang: ["nemo-parakeet-tdt-0.6b-v3"] for lang in
       "bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk".split()},
    "ru": ["gigaam-v2-rnnt"],
    "pt": ["OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-pt-onnx", "nemo-parakeet-tdt-0.6b-v3"],
    "pl": ["OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-pl-onnx", "nemo-parakeet-tdt-0.6b-v3"],
    "nl": ["OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-nl-onnx", "nemo-parakeet-tdt-0.6b-v3"],
    "et": ["OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-et-onnx", "nemo-parakeet-tdt-0.6b-v3"],
    "sl": ["OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-sl-onnx", "nemo-parakeet-tdt-0.6b-v3"],
    "ca": ["OpenVoiceOS/nvidia-ca-conformer-transducer-large-onnx"],
    "gl": ["OpenVoiceOS/proxectonos-gl-conformer-ctc-large-onnx"],
    "fa": ["OpenVoiceOS/nvidia-fa-fastconformer-hybrid-large-onnx"],
    "ja": ["OpenVoiceOS/nvidia-parakeet-tdt_ctc-0.6b-ja-onnx"],
    "vi": ["OpenVoiceOS/nvidia-parakeet-ctc-0.6b-vietnamese-onnx"],
    "be": ["OpenVoiceOS/nvidia-be-conformer-transducer-large-onnx"],
    "uz": ["OpenVoiceOS/asr-uz-fastconformer-large-onnx"],
    "fil": ["OpenVoiceOS/stt-tl-fastconformer-hybrid-large-onnx"],
}
#: Whisper large-v3-turbo is the fallback for every FLEURS language without a model above.
WHISPER = "onnx-community/whisper-large-v3-turbo"
QUANTIZATION = "int8"
#: Whisper's own codes where they differ from the ISO 639-1 code.
WHISPER_CODES = {"nb": "no", "fil": "tl", "jv": "jw"}
#: Scripts without spaces between words: WER is not defined there, so CER decides.
UNSEGMENTED = {"zh", "yue", "ja", "th", "lo", "my", "km", "bo"}
#: Members of a macrolanguage that an ASR model for the macrolanguage serves.
MACRO = {
    **{code: "ar" for code in "arb arq ary arz aeb apc apd ars avl shu acm ajp afb ayl acq abv "
                               "adf aec auz ayh ayn ayp pga ssh".split()},
    "cmn": "zh", "pes": "fa", "prs": "fa", "zsm": "ms", "swh": "sw", "ekk": "et", "lvs": "lv",
    "uzn": "uz", "khk": "mn", "npi": "ne", "ory": "or", "azj": "az", "als": "sq", "nob": "nb",
    "plt": "mg", "gaz": "om", "ydd": "yi", "tl": "fil", "fil": "fil", "no": "nb",
}
BLOCK = re.compile(r"\[\[.*?\]\]")
ARABIC_LETTERS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ـ": None})
SILENCE_RMS = 1e-3


# ---------------------------------------------------------------- language and text

def asr_lang(tag: str) -> str:
    """The language an ASR model must serve to judge a voice tagged `tag`."""
    import langcodes
    primary = tag.split("-")[0].lower()
    if primary in MACRO:
        return MACRO[primary]
    try:
        primary = langcodes.standardize_tag(primary).split("-")[0]
    except ValueError:
        pass
    return MACRO.get(primary, primary)


def spoken(text: str, lang: str) -> str:
    import text2spoken
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return text2spoken.normalize(text, lang)
        except Exception:
            return text


def normalise(text: str, lang: str) -> str:
    """text2spoken's reading, NFC, lower case, no combining marks, no punctuation."""
    text = unicodedata.normalize("NFC", spoken(BLOCK.sub(" ", text), lang)).lower()
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = "".join(" " if unicodedata.category(c)[0] in "PSZC" else c for c in text)
    if lang == "ar":
        text = text.translate(ARABIC_LETTERS)
    return " ".join(text.split())


def edits(a: list, b: list) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def wer(ref: str, hyp: str) -> float:
    words = ref.split()
    return edits(words, hyp.split()) / max(len(words), 1)


def cer(ref: str, hyp: str) -> float:
    chars = ref.replace(" ", "")
    return edits(list(chars), list(hyp.replace(" ", ""))) / max(len(chars), 1)


class Recognisers:
    """onnx-asr models, loaded once each, with a fixed thread count."""

    def __init__(self, threads: int):
        self.threads = threads
        self.models: dict = {}

    def get(self, name: str):
        if name not in self.models:
            import onnx_asr
            import onnx_asr.utils
            import onnxruntime
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
            if name == WHISPER:
                from huggingface_hub import snapshot_download
                path = snapshot_download(name, allow_patterns=[
                    "*.json", "*.txt", f"onnx/encoder_model_{QUANTIZATION}.onnx",
                    f"onnx/decoder_model_merged_{QUANTIZATION}.onnx"])
                self.models[name] = onnx_asr.load_model("whisper", path, quantization=QUANTIZATION,
                                                        sess_options=options)
            else:
                try:
                    self.models[name] = onnx_asr.load_model(name, quantization=QUANTIZATION,
                                                            sess_options=options)
                except onnx_asr.utils.ModelFileNotFoundError:
                    # Some exports ship only the full-precision graphs.
                    self.models[name] = onnx_asr.load_model(name, sess_options=options)
        return self.models[name]

    def transcribe(self, name: str, lang: str, audio, rate: int) -> str:
        import numpy as np
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate not in (8000, 16000, 22050, 24000, 32000, 44100, 48000):
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(rate, 16000)
            audio, rate = resample_poly(audio, 16000 // g, rate // g).astype(np.float32), 16000
        options = {"language": WHISPER_CODES.get(lang, lang)} if name == WHISPER else {}
        return self.get(name).recognize(audio, sample_rate=rate, **options)


def score(lang: str, reference: str, transcript: str) -> dict:
    ref, hyp = normalise(reference, lang), normalise(transcript, lang)
    return {"reference": ref, "transcript": transcript, "transcript_normalised": hyp,
            "wer": wer(ref, hyp), "cer": cer(ref, hyp)}


# ---------------------------------------------------------------- floor

def fleurs_codes() -> dict:
    """FLEURS locale per ASR language, from the dataset's own file list."""
    from huggingface_hub import HfApi
    codes = {}
    for name in HfApi().list_repo_files("google/fleurs", repo_type="dataset"):
        match = re.match(r"parquet-data/([a-z_]+)/test-", name)
        if match and match.group(1) != "all":
            code = match.group(1)
            lang = asr_lang(code.split("_")[0])
            codes.setdefault(lang, code)
    return codes


def floor(args) -> int:
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    asr = Recognisers(args.threads)
    codes = fleurs_codes()
    wanted = args.lang or sorted(codes)
    result = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    print(f"languages in: {len(wanted)}", flush=True)
    for lang in wanted:
        if lang in result or lang not in codes:
            result.setdefault(lang, {"verified": False, "reason": "no FLEURS test split"})
            continue
        try:
            path = hf_hub_download("google/fleurs", f"parquet-data/{codes[lang]}/test-00000-of-00001.parquet",
                                   repo_type="dataset")
        except Exception as error:
            result[lang] = {"fleurs": codes[lang], "verified": False, "reason": describe(error)}
            continue
        rows = pq.ParquetFile(path).read_row_group(0).slice(0, args.clips + 3).to_pylist()
        tried = []
        for model in ASR_CANDIDATES.get(lang, []) + [WHISPER]:
            started, scored = time.time(), []
            try:
                for row in rows[:args.clips]:
                    audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                    scored.append(score(lang, row["transcription"] or row["raw_transcription"],
                                        asr.transcribe(model, lang, audio, rate)))
            except Exception as error:
                tried.append({"model": model, "error": f"{type(error).__name__}: {error}"[:300]})
                continue
            words = sum(len(s["reference"].split()) for s in scored)
            chars = sum(len(s["reference"].replace(" ", "")) for s in scored)
            tried.append({"model": model, "clips": len(scored),
                          "wer": sum(s["wer"] * len(s["reference"].split()) for s in scored) / max(words, 1),
                          "cer": sum(s["cer"] * len(s["reference"].replace(" ", "")) for s in scored) / max(chars, 1),
                          "seconds": round(time.time() - started, 1)})
            metric = "cer" if lang in UNSEGMENTED else "wer"
            if tried[-1][metric] <= (args.max_cer if metric == "cer" else args.max_wer):
                break
        usable = [t for t in tried if "error" not in t]
        metric = "cer" if lang in UNSEGMENTED else "wer"
        best = min(usable, key=lambda t: t[metric]) if usable else None
        limit = args.max_cer if metric == "cer" else args.max_wer
        result[lang] = {"fleurs": codes[lang], "tried": tried, "metric": metric, "limit": limit,
                        "model": best and best["model"], "wer": best and best["wer"],
                        "cer": best and best["cer"],
                        "verified": bool(best and best[metric] <= limit),
                        # Two clips after the scored ones: real sentences for the voice sweep.
                        "sentences": [r["raw_transcription"] for r in rows[args.clips:args.clips + 2]]}
        if args.reference_dir and len(rows) > args.clips + 2:
            # The clip after those: a speaker reference for voices that clone one.
            os.makedirs(args.reference_dir, exist_ok=True)
            row = rows[args.clips + 2]
            audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            sf.write(Path(args.reference_dir) / f"{lang}.wav", audio, rate)
            (Path(args.reference_dir) / f"{lang}.txt").write_text(row["raw_transcription"], encoding="utf-8")
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(lang, result[lang]["model"], round(best[metric], 3) if best else None,
              "verified" if result[lang]["verified"] else "not verified", flush=True)
        os.remove(os.path.realpath(path))
        os.remove(path)
    print(f"languages in: {len(wanted)}  languages out: {sum(l in result for l in wanted)}  "
          f"verified: {sum(result[l]['verified'] for l in wanted if l in result)}")
    return 0 if all(l in result for l in wanted) else 1


# ---------------------------------------------------------------- index and groups

def read_index() -> dict:
    from phoonnx.model_manager import TTSModelManager
    voices = {}
    for path in TTSModelManager.voice_index_files():
        for voice_id, entry in json.loads(path.read_text(encoding="utf-8")).items():
            voices[voice_id] = (path.name, entry)
    return voices


def urls(entry: dict) -> list[str]:
    found = [v for k, v in entry.items() if k.endswith("_url") and isinstance(v, str)]
    found += [v for v in (entry.get("aux_model_urls") or {}).values() if isinstance(v, str)]
    return found


def model_key(entry: dict) -> tuple:
    """The graph and weight files a voice loads. Voices with one key share a download."""
    return tuple(sorted(u for u in urls(entry) if GRAPH_FILE.search(u.split("?")[0]))) or (entry["model_url"],)


def read_sentences(path: str, ids: list[str]) -> dict:
    rows = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t", quoting=csv.QUOTE_NONE):
            if row["id"] in ids:
                rows[row["lang"]].append({"id": row["id"], "text": row["text"],
                                          "native": not row["source"].startswith("script:")})
    return rows


def repos(entry: dict) -> set[str]:
    """The Hugging Face repos a voice downloads from."""
    return {match["repo"] for u in urls(entry) if (match := HF_URL.match(u.split("?")[0]))}


def evict(names: set[str], cache: str) -> None:
    """Delete these repos from the cache through huggingface_hub, which keeps its shared
    blob store consistent (a blob there can back files of other repos)."""
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir(cache)
    revisions = [rev.commit_hash for repo in info.repos if repo.repo_id in names for rev in repo.revisions]
    if revisions:
        info.delete_revisions(*revisions).execute()


def cache_bytes(path: str) -> int:
    """Bytes held by the files under `path`, symlinks not counted."""
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            if not os.path.islink(full):
                total += os.path.getsize(full)
    return total


def free_bytes(path: str) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


# ---------------------------------------------------------------- worker

class VoiceTimeout(BaseException):
    """Raised by the alarm. A BaseException, so a voice's own error handling cannot swallow it."""


def on_alarm(*_):
    raise VoiceTimeout("no result in the time limit")


def cap_threads(threads: int) -> None:
    """Give every onnxruntime session `threads` intra-op threads.

    phoonnx builds sessions with default options, and onnxruntime then starts one
    pinned thread per core, which overrides taskset. The cap changes how many threads
    run, not what the graph computes.
    """
    import onnxruntime

    class CappedSession(onnxruntime.InferenceSession):
        def __init__(self, path_or_bytes, sess_options=None, *args, **kwargs):
            sess_options = sess_options or onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = threads
            sess_options.inter_op_num_threads = 1
            super().__init__(path_or_bytes, sess_options, *args, **kwargs)

    onnxruntime.InferenceSession = CappedSession


def describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {' '.join(str(error).split())}"[:400]


def reference_clip(directory: str | None, lang: str):
    """A speech clip and its text for voices that clone a speaker, if one is given."""
    if not directory:
        return None
    for name in (asr_lang(lang), "default"):
        wav = Path(directory) / f"{name}.wav"
        if wav.exists():
            text = wav.with_suffix(".txt")
            return str(wav), text.read_text(encoding="utf-8").strip() if text.exists() else None
    return None


def synthesise_voice(record: dict, entry: dict, sentences: list, args) -> None:
    """Fill `record` as the voice loads and speaks, so a timeout keeps what was done."""
    import numpy as np
    import soundfile as sf
    from phoonnx.config import SynthesisConfig
    from phoonnx.model_manager import TTSModelInfo

    record.update(phoneme_type=entry["phoneme_type"], requires_reference=bool(entry.get("requires_reference")),
                  sentences=[])
    voice_id = record["voice"]
    started = time.time()
    try:
        voice = TTSModelInfo(**entry).load()
    except (Exception, SystemExit) as error:
        record.update(status="load_failed", error=describe(error), load_seconds=round(time.time() - started, 2))
        return
    record["load_seconds"] = round(time.time() - started, 2)
    config = voice.config
    record["resolved"] = {"adapter": type(voice.adapter).__name__, "lang_code": config.lang_code,
                          "phoneme_type": getattr(config.phoneme_type, "value", config.phoneme_type),
                          "alphabet": getattr(config.alphabet, "value", config.alphabet),
                          "sample_rate": config.sample_rate}
    model = HF_URL.match(entry["model_url"].split("?")[0])
    if model:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(model["repo"], model["path"], revision=model["rev"])
        if isinstance(path, str) and "/snapshots/" in path:
            record["snapshot"] = path.split("/snapshots/")[1].split("/")[0]
    syn = SynthesisConfig()
    if entry.get("requires_reference"):
        clip = reference_clip(args.reference_dir, entry["lang"])
        if clip is None:
            record.update(status="needs_reference", error="requires_reference and no --reference-dir clip")
            return
        syn.speaker_reference, syn.speaker_reference_text = clip
    failures = 0
    for number, sentence in enumerate(sentences):
        result = {"id": sentence["id"], "text": sentence["text"], "native": sentence["native"]}
        random.seed(number)
        np.random.seed(number)
        began = time.time()
        try:
            chunks = list(voice.synthesize(sentence["text"], syn))
            audio = (np.concatenate([c.audio_float_array.reshape(-1) for c in chunks])
                     if chunks else np.zeros(0, dtype=np.float32))
            rate = chunks[0].sample_rate if chunks else config.sample_rate
            seconds = time.time() - began
            nan = bool(np.isnan(audio).any())
            clean = np.nan_to_num(audio)
            duration = len(audio) / rate if rate else 0.0
            rms = float(np.sqrt(np.mean(clean ** 2))) if audio.size else 0.0
            result.update(duration=round(duration, 3), peak=round(float(np.abs(clean).max()) if audio.size else 0.0, 5),
                          rms=round(rms, 6), nan=nan, silent=bool(rms < SILENCE_RMS),
                          rtf=round(seconds / duration, 4) if duration else None, sample_rate=rate)
            if args.audio_dir and audio.size:
                wav = Path(args.audio_dir) / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', voice_id)}__{sentence['id']}.wav"
                sf.write(wav, clean, rate, subtype="PCM_16")
                result["wav"] = str(wav)
        except (Exception, SystemExit) as error:
            failures += 1
            result["error"] = describe(error)
        record["sentences"].append(result)
    spoken_rows = [s for s in record["sentences"] if "error" not in s]
    if failures == len(sentences):
        record.update(status="synth_failed", error=record["sentences"][0]["error"] if sentences else "no sentences")
    elif not sentences:
        record.update(status="no_sentences")
    elif any(s["nan"] for s in spoken_rows):
        record["status"] = "nan"
    elif all(s["silent"] for s in spoken_rows):
        record["status"] = "silent"
    else:
        record["status"] = "ok" if not failures else "partial"
    if spoken_rows:
        record.update(duration=round(sum(s["duration"] for s in spoken_rows), 3),
                      peak=max(s["peak"] for s in spoken_rows),
                      rms=round(sum(s["rms"] for s in spoken_rows) / len(spoken_rows), 6),
                      rtf=round(sum(s["rtf"] or 0 for s in spoken_rows) / len(spoken_rows), 4))


def worker(args) -> int:
    """Synthesise the voices named on stdin, printing one JSON event per line."""
    cap_threads(args.threads)
    signal.signal(signal.SIGALRM, on_alarm)
    index = read_index()
    sentences = read_sentences(args.sentences, args.sentence_ids)
    floor_file = json.loads(Path(args.asr_floor).read_text()) if args.asr_floor else {}
    for line in sys.stdin:
        voice_id = line.strip()
        if not voice_id:
            continue
        index_file, entry = index[voice_id]
        rows = list(sentences.get(entry["lang"], []))
        verified = floor_file.get(asr_lang(entry["lang"]), {})
        if verified.get("verified"):
            rows += [{"id": f"fleurs_{n}", "text": text, "native": True}
                     for n, text in enumerate(verified.get("sentences", []))]
        print(json.dumps({"event": "start", "voice": voice_id}), flush=True)
        record = {"voice": voice_id, "index_file": index_file, "engine": entry.get("engine"), "lang": entry["lang"]}
        signal.alarm(args.timeout)
        try:
            synthesise_voice(record, entry, rows, args)
        except VoiceTimeout as error:
            spoke = sum("duration" in s for s in record.get("sentences", []))
            record.update(status="timeout", error=f"{error} ({args.timeout} s); {spoke} sentences spoke before it")
        finally:
            signal.alarm(0)
        print(json.dumps({"event": "result", "record": record}, ensure_ascii=False), flush=True)
    return 0


# ---------------------------------------------------------------- run

def cpu_sets(spec: str, jobs: int) -> list[str | None]:
    if not spec:
        return [None] * jobs
    cpus = []
    for part in spec.split(","):
        low, _, high = part.partition("-")
        cpus += list(range(int(low), int(high or low) + 1))
    size = max(1, len(cpus) // jobs)
    return [",".join(map(str, cpus[i * size:(i + 1) * size])) for i in range(jobs)]


def run(args) -> int:
    index = read_index()
    chosen = {v: e for v, e in index.items() if not args.engine or e[1].get("engine") in args.engine}
    if args.voices:
        wanted = set(Path(args.voices).read_text().split())
        chosen = {v: e for v, e in chosen.items() if v in wanted}
    synth_path = Path(args.out + ".synth")
    done = {}
    if synth_path.exists():
        for line in synth_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done[record["voice"]] = record
    groups = defaultdict(list)
    for voice_id, (_, entry) in sorted(chosen.items()):
        if voice_id not in done:
            groups[model_key(entry)].append(voice_id)
    # A large group is split so that its voices run in several slots; files stay cached
    # until no pending or running voice names them.
    queue = sorted((g[i:i + args.group_size] for g in groups.values() for i in range(0, len(g), args.group_size)),
                   key=lambda g: (chosen[g[0]][1].get("engine") or "", g[0]))
    print(f"voices in: {len(chosen)}  groups to run: {len(queue)}  already done: {len(done)}", flush=True)
    if args.asr_floor and not args.audio_dir:
        args.audio_dir = args.out + ".audio"
    if args.audio_dir:
        os.makedirs(args.audio_dir, exist_ok=True)

    command = [sys.executable, os.path.abspath(__file__), "worker", "--sentences", args.sentences,
               "--timeout", str(args.timeout), "--threads", str(args.threads),
               "--sentence-ids", *args.sentence_ids]
    for flag in ("asr_floor", "reference_dir", "audio_dir"):
        if getattr(args, flag):
            command += [f"--{flag.replace('_', '-')}", getattr(args, flag)]
    slots = cpu_sets(args.cpus, args.jobs)
    running = {}
    sink = open(synth_path, "a", encoding="utf-8")
    cache = os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub")
    os.makedirs(cache, exist_ok=True)

    def launch(number, voices):
        slot = slots[number]
        prefix = ["nice", "-n", str(args.nice)] + (["taskset", "-c", slot] if slot else [])
        voice_list = Path(f"{args.out}.slot{number}")
        voice_list.write_text("\n".join(voices) + "\n", encoding="utf-8")
        with open(voice_list, encoding="utf-8") as stdin, open(args.out + ".worker.log", "a") as log:
            process = subprocess.Popen(prefix + command, stdin=stdin, stdout=subprocess.PIPE, stderr=log,
                                       env={**os.environ, "OMP_NUM_THREADS": str(args.threads)})
        os.set_blocking(process.stdout.fileno(), False)
        return {"process": process, "voices": list(voices), "current": None, "seen": time.time(), "buffer": b""}

    def evict_finished(group):
        """Evict a finished group's repos that no pending or running voice downloads from."""
        pending = [v for g in queue for v in g] + [v for job in running.values() for v in job["voices"]]
        in_use = {r for v in pending for r in repos(index[v][1])}
        evict({r for v in group for r in repos(index[v][1])} - in_use, cache)

    def record(entry):
        sink.write(json.dumps(entry, ensure_ascii=False) + "\n")
        sink.flush()
        done[entry["voice"]] = entry

    def stub(voice_id, status, error):
        index_file, entry = index[voice_id]
        return {"voice": voice_id, "index_file": index_file, "engine": entry.get("engine"),
                "lang": entry["lang"], "status": status, "error": error}

    while queue or running:
        for slot in range(len(slots)):
            if slot in running or not queue:
                continue
            if args.min_free_gb and free_bytes(cache) < args.min_free_gb * 1e9 or \
                    args.max_cache_gb and running and cache_bytes(cache) > args.max_cache_gb * 1e9:
                print("cache disk too full, waiting for a running group to end", flush=True)
                time.sleep(60)
                break
            group = queue.pop(0)
            running[slot] = launch(slot, group)
            running[slot]["group"] = group
        time.sleep(0.5)
        for slot, job in list(running.items()):
            process = job["process"]
            try:
                job["buffer"] += os.read(process.stdout.fileno(), 1 << 20)
            except BlockingIOError:
                pass
            *lines, job["buffer"] = job["buffer"].split(b"\n")
            for line in lines:
                if not line.startswith(b"{"):
                    continue
                event = json.loads(line.decode("utf-8"))
                job["seen"] = time.time()
                if event["event"] == "start":
                    job["current"] = event["voice"]
                else:
                    record(event["record"])
                    job["voices"].remove(event["record"]["voice"])
                    job["current"] = None
                    print(f"{len(done)}/{len(chosen)} {event['record']['status']:<12} {event['record']['voice']}",
                          flush=True)
            stalled = time.time() - job["seen"] > args.timeout + 120
            if process.poll() is None and not stalled:
                continue
            if stalled:
                process.kill()
            process.wait()
            process.stdout.close()
            remaining = job["voices"]
            if job["current"] in remaining:
                record(stub(job["current"], "timeout" if stalled else "crash",
                            f"worker {'stopped answering' if stalled else f'exited {process.returncode}'} "
                            f"on this voice"))
                remaining.remove(job["current"])
            del running[slot]
            if remaining:
                running[slot] = launch(slot, remaining)
                running[slot]["group"] = job["group"]
            elif args.evict:
                evict_finished(job["group"])
    sink.close()
    return score_run(args, index, chosen, done)


def score_run(args, index, chosen, done) -> int:
    """Add ASR results to every synthesised sentence of a verified language, then write --out."""
    floor_file = json.loads(Path(args.asr_floor).read_text()) if args.asr_floor else {}
    asr = Recognisers(args.threads * max(args.jobs, 1))
    out_done = set()
    if Path(args.out).exists():
        out_done = {json.loads(l)["voice"] for l in Path(args.out).read_text(encoding="utf-8").splitlines()}
    import soundfile as sf
    pending = sorted((v for v in chosen if v not in out_done),
                     key=lambda v: asr_lang(index[v][1]["lang"]))
    with open(args.out, "a", encoding="utf-8") as sink:
        for voice_id in pending:
            record = done[voice_id]
            lang = asr_lang(record["lang"])
            verified = floor_file.get(lang, {})
            record["asr"] = {"lang": lang, "model": verified.get("model"),
                             "verified": bool(verified.get("verified"))}
            if verified.get("verified"):
                scored = []
                for sentence in record.get("sentences", []):
                    if "wav" not in sentence or not sentence["native"]:
                        continue
                    audio, rate = sf.read(sentence["wav"], dtype="float32")
                    sentence.update(score(lang, sentence["text"],
                                          asr.transcribe(verified["model"], lang, audio, rate)))
                    scored.append(sentence)
                if scored:
                    record["wer"] = round(sum(s["wer"] for s in scored) / len(scored), 4)
                    record["cer"] = round(sum(s["cer"] for s in scored) / len(scored), 4)
            if not args.keep_audio:
                for sentence in record.get("sentences", []):
                    if "wav" in sentence and os.path.exists(sentence["wav"]):
                        os.remove(sentence["wav"])
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
    results = [json.loads(l) for l in Path(args.out).read_text(encoding="utf-8").splitlines()]
    ids = {r["voice"] for r in results}
    statuses = Counter(r["status"] for r in results if r["voice"] in chosen)
    print(f"voices in: {len(chosen)}  results out: {len(ids & set(chosen))}  {dict(statuses)}")
    return 0 if ids >= set(chosen) and len(results) == len(ids) else 1


# ---------------------------------------------------------------- diff

SPEAKING = {"ok", "partial"}


def diff(args) -> int:
    def load(path):
        return {r["voice"]: r for r in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines())}
    old, new = load(args.a), load(args.b)
    lines, regressions = [], 0
    for voice_id in sorted(set(old) | set(new)):
        a, b = old.get(voice_id), new.get(voice_id)
        if a is None or b is None:
            lines.append(f"{'added' if a is None else 'removed'}  {voice_id}")
            continue
        if a["status"] in SPEAKING and b["status"] not in SPEAKING:
            regressions += 1
            lines.append(f"REGRESSION status {voice_id}: {a['status']} -> {b['status']}  {b.get('error', '')}")
        elif a["status"] not in SPEAKING and b["status"] in SPEAKING:
            lines.append(f"fixed  {voice_id}: {a['status']} -> {b['status']}")
        metric = "cer" if b.get("asr", {}).get("lang") in UNSEGMENTED else "wer"
        if metric in a and metric in b and b[metric] - a[metric] > args.wer_margin:
            regressions += 1
            lines.append(f"REGRESSION {metric} {voice_id}: {a[metric]:.3f} -> {b[metric]:.3f}")
    print("\n".join(lines))
    print(f"voices in A: {len(old)}  voices in B: {len(new)}  compared: {len(set(old) & set(new))}  "
          f"regressions: {regressions}")
    return 1 if regressions else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def synth_options(command):
        command.add_argument("--sentences", required=True)
        command.add_argument("--sentence-ids", nargs="+", default=["plain", "digits"])
        command.add_argument("--asr-floor")
        command.add_argument("--reference-dir", help="<lang>.wav and <lang>.txt, or default.wav, for cloning voices")
        command.add_argument("--audio-dir")
        command.add_argument("--timeout", type=int, default=600, help="seconds per voice")
        command.add_argument("--threads", type=int, default=4, help="onnxruntime threads per worker")

    run_args = commands.add_parser("run", help="synthesise every voice and write one JSON line per voice")
    synth_options(run_args)
    run_args.add_argument("--out", required=True)
    run_args.add_argument("--jobs", type=int, default=1)
    run_args.add_argument("--cpus", help="CPU list split between the jobs, e.g. 0-11")
    run_args.add_argument("--nice", type=int, default=19)
    run_args.add_argument("--engine", action="append")
    run_args.add_argument("--voices", help="file with one voice id per line")
    run_args.add_argument("--evict", action="store_true", help="remove a group's files from the cache after it")
    run_args.add_argument("--min-free-gb", type=float, default=0, help="wait while the cache disk has less free")
    run_args.add_argument("--max-cache-gb", type=float, default=0,
                          help="start no new group while the Hugging Face cache holds more than this")
    run_args.add_argument("--keep-audio", action="store_true")
    run_args.add_argument("--group-size", type=int, default=40, help="most voices one worker runs in a row")
    run_args.set_defaults(run=run)

    worker_args = commands.add_parser("worker", help=argparse.SUPPRESS)
    synth_options(worker_args)
    worker_args.set_defaults(run=worker)

    floor_args = commands.add_parser("floor", help="score the ASR models on FLEURS speech")
    floor_args.add_argument("--out", required=True)
    floor_args.add_argument("--lang", action="append", help="ASR language codes; default every FLEURS language")
    floor_args.add_argument("--clips", type=int, default=15)
    floor_args.add_argument("--reference-dir", help="also save one FLEURS clip per language here, for cloning voices")
    floor_args.add_argument("--max-wer", type=float, default=0.30)
    floor_args.add_argument("--max-cer", type=float, default=0.20)
    floor_args.add_argument("--threads", type=int, default=4)
    floor_args.set_defaults(run=floor)

    diff_args = commands.add_parser("diff", help="list the regressions between two runs")
    diff_args.add_argument("a")
    diff_args.add_argument("b")
    diff_args.add_argument("--wer-margin", type=float, default=0.15)
    diff_args.set_defaults(run=diff)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
