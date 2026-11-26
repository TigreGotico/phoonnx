#!/usr/bin/env python3
"""Run the phoonnx-qa gate on the default voice of each leader language.

For each language the plugin picks its default voice from ``lang`` alone. The voice
speaks the fixed sentences through three load paths: manual (``TTSVoice.load`` on the
downloaded files), index (``TTSModelInfo.load``) and plugin (``PhoonnxTTSPlugin.get_tts``).
The onnx-asr model that the ovos-stt-plugin-onnx-asr registry names for the language
transcribes each WAV. The same model transcribes two reference clips of real speech
first; that WER is the ASR floor. A floor above 0.25 makes every path
ASR-UNVERIFIED, with no WER, because then the WER of the voice tells nothing.

WER: text2spoken reads the digits of the reference, then both sides are lower-cased,
stripped of punctuation and combining marks (``voice_sweep.normalise``).

    tools/voice_qa.py [--lang pt-PT ...] [--out docs/data/voice_qa.json]

Each run replaces the records of the languages it ran and keeps the others.
CI does not run this script: it downloads voices and ASR models.
"""
import argparse
import hashlib
import io
import json
import sys
import tempfile
import wave
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from voice_sweep import normalise, wer  # noqa: E402

LANGS = ["pt-PT", "pt-BR", "es-CO", "fr-FR", "it-IT", "de-DE", "nl-NL", "da-DK",
         "kab", "hi-IN", "bn-IN", "mwr"]
SENTENCES = {
    "pt-PT": ["O comboio parte às 8 horas, não te atrases!", "Comprei 3 livros na feira."],
    "pt-BR": ["O ônibus chega em 5 minutos, fique tranquilo.", "Tenho 2 irmãos e uma irmã."],
    "es-CO": ["El bus sale a las 7, ¿vienes conmigo?", "Compré 4 mangos en la plaza."],
    "fr-FR": ["Le train part à 9 heures, ne sois pas en retard !", "J'ai acheté 3 baguettes ce matin."],
    "it-IT": ["Il treno parte alle 6, non fare tardi!", "Ho comprato 2 chili di mele."],
    "de-DE": ["Der Zug fährt um 8 Uhr ab, sei pünktlich!", "Ich habe 3 Bücher gekauft."],
    "nl-NL": ["De trein vertrekt om 7 uur, kom niet te laat!", "Ik heb 4 appels gekocht."],
    "da-DK": ["Toget kører klokken 8, kom ikke for sent!", "Jeg har købt 3 bøger."],
    "kab": ["Azul, ɣriɣ 2 n yedlisen ass-a.", "Ad d-uɣaleɣ ɣef 5, ur ttɛeṭṭil ara!"],
    "hi-IN": ["मेरे पास 3 किताबें हैं।", "ट्रेन 5 बजे आएगी, देर मत करना!"],
    "bn-IN": ["আমার 2 ভাই আছে।", "ট্রেন 5 টায় আসবে, দেরি কোরো না!"],
    "mwr": [],
}
#: FLEURS test config per language, or an LJSpeech-style dataset (metadata.csv of path|text).
REFERENCE = {"pt-PT": "pt_br", "pt-BR": "pt_br", "es-CO": "es_419", "fr-FR": "fr_fr",
             "it-IT": "it_it", "de-DE": "de_de", "nl-NL": "nl_nl", "da-DK": "da_dk",
             "hi-IN": "hi_in", "bn-IN": "bn_in", "kab": "boffire/kabyle-ljspeech-22khz"}
LOW_RESOURCE = {"kab", "hi-IN", "bn-IN"}
NOTES = {"kab": "sentences not checked by a native speaker"}
ISSUES = {"bn-IN": "https://github.com/TigreGotico/phoonnx/issues/1182",
          "da-DK": "https://github.com/TigreGotico/phoonnx/issues/1186"}
PATHS = ("manual", "index", "plugin")


def threshold(lang: str) -> float:
    return 0.5 if lang in LOW_RESOURCE else 0.35


def reference_clips(lang: str, n: int = 2) -> list:
    """(audio, rate, text) of n clips of real speech in the language."""
    source = REFERENCE[lang]
    if "/" in source:
        from huggingface_hub import hf_hub_download
        meta = hf_hub_download(source, "metadata.csv", repo_type="dataset")
        rows = [line.split("|") for line in Path(meta).read_text(encoding="utf-8").splitlines()[:n]]
        return [(*sf.read(hf_hub_download(source, name, repo_type="dataset"), dtype="float32"), text)
                for name, text in rows]
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    path = f"datasets/google/fleurs/parquet-data/{source}/test-00000-of-00001.parquet"
    batch = next(pq.ParquetFile(HfFileSystem().open(path)).iter_batches(
        batch_size=n, columns=["raw_transcription", "audio"]))
    return [(*sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32"), row["raw_transcription"])
            for row in batch.to_pylist()[:n]]


class Asr:
    """The registry model of the language, loaded in fp32."""

    def __init__(self, lang: str, scratch: Path):
        from ovos_stt_plugin_onnxasr import OnnxASR
        from ovos_stt_plugin_onnxasr.defaults import resolve_model
        self.lang = lang
        self.model = resolve_model(lang, {})
        self.stt = None
        if self.model:
            import onnx_asr.utils
            try:
                self.stt = OnnxASR(config={"lang": lang, "model": self.model})
            except onnx_asr.utils.ModelFileNotFoundError:
                # The plugin data folder holds other files of the model (an int8 copy):
                # download the fp32 files to the scratch folder.
                self.stt = OnnxASR(config={"lang": lang, "model": self.model,
                                           "model_dir": str(scratch / "asr")})

    def transcribe(self, audio, rate: int) -> str:
        from ovos_plugin_manager.utils.audio import AudioData
        from scipy.signal import resample_poly
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != 16000:
            from math import gcd
            g = gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // g, rate // g)
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        return self.stt.execute(AudioData(pcm, 16000, 2), language=self.lang)


def score(asr: Asr, lang: str, audio, rate: int, text: str) -> float:
    return round(wer(normalise(text, lang), normalise(asr.transcribe(audio, rate), lang)), 3)


#: An ASR floor above this makes the WER of a voice meaningless.
MAX_FLOOR = 0.25


def classify(wers: list, lang: str, floor) -> str:
    if floor is None or floor > MAX_FLOOR:
        return "ASR-UNVERIFIED"
    return "PASS" if sum(wers) / len(wers) <= threshold(lang) else "FAIL"


def run_path(name: str, info, lang: str, sentences: list, asr: Asr, floor, scratch: Path) -> dict:
    from phoonnx.opm import PhoonnxTTSPlugin
    from phoonnx.voice import TTSVoice
    wavs = []
    try:
        if name == "plugin":
            plugin = PhoonnxTTSPlugin({"lang": lang})
            for i, text in enumerate(sentences):
                wav = scratch / f"{lang}-{name}-{i}.wav"
                plugin.get_tts(text, str(wav), lang=lang)
                wavs.append(wav)
        else:
            if name == "index":
                voice = info.load()
            else:
                voice = TTSVoice.load(
                    model_path=info.download_model(),
                    config_path=info.hub_path(info.config_url),
                    vocab_path=info.hub_path(info.vocab_url),
                    tokenizer_config_path=info.hub_path(info.tokenizer_config_url))
            for i, text in enumerate(sentences):
                wav = scratch / f"{lang}-{name}-{i}.wav"
                with wave.open(str(wav), "wb") as out:
                    voice.synthesize_wav(text, out)
                wavs.append(wav)
    except (ImportError, ModuleNotFoundError) as error:
        return {"class": "ENV-MISSING", "error": f"{type(error).__name__}: {error}"[:200]}
    except Exception as error:
        return {"class": "FAIL", "error": f"{type(error).__name__}: {error}"[:200]}
    sha = [hashlib.sha256(wav.read_bytes()).hexdigest()[:16] for wav in wavs]
    if floor is None or floor > MAX_FLOOR:
        return {"class": "ASR-UNVERIFIED", "audio_sha256": sha}
    wers = [score(asr, lang, *sf.read(str(wav), dtype="float32"), text)
            for wav, text in zip(wavs, sentences)]
    mean = round(sum(wers) / len(wers), 3)
    return {"wer": wers, "mean": mean, "class": classify(wers, lang, floor), "audio_sha256": sha}


def qa(lang: str, scratch: Path) -> dict:
    from phoonnx.opm import PhoonnxTTSPlugin
    record = {"lang": lang, "sentences": SENTENCES[lang], "note": NOTES.get(lang),
              "phoonnx": version("phoonnx"), "text2spoken": version("text2spoken")}
    try:
        info = PhoonnxTTSPlugin({"lang": lang}).get_default_voice(lang)
    except ValueError:
        return {**record, "voice_id": None, "class": "NO-VOICE"}
    cfg = info.config
    record.update(voice_id=info.voice_id, engine=info.engine.value, lang_code=cfg.lang_code,
                  phoneme_type=cfg.phoneme_type and cfg.phoneme_type.value,
                  alphabet=cfg.alphabet and cfg.alphabet.value)
    print(f"{lang}: {info.voice_id} {record['engine']} {cfg.lang_code} "
          f"{record['phoneme_type']} {record['alphabet']}", flush=True)
    asr = Asr(lang, scratch)
    record["asr_model"] = asr.model
    floor = None
    if asr.model and lang in REFERENCE:
        clips = reference_clips(lang)
        floor = round(sum(score(asr, lang, *clip) for clip in clips) / len(clips), 3)
    record["asr_floor"] = floor
    for name in PATHS:
        result = run_path(name, info, lang, SENTENCES[lang], asr, floor, scratch)
        if result["class"] == "FAIL" and lang in ISSUES:
            result["issue"] = ISSUES[lang]
        record[name] = result
        print(f"  {name}: {result}", flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", nargs="+", choices=LANGS, default=LANGS)
    parser.add_argument("--out", default=str(ROOT / "docs/data/voice_qa.json"))
    args = parser.parse_args()
    out = Path(args.out)
    records = {r["lang"]: r for r in json.loads(out.read_text(encoding="utf-8"))} if out.exists() else {}
    print(f"languages in: {len(args.lang)}", flush=True)
    done = 0
    with tempfile.TemporaryDirectory() as scratch:
        for lang in args.lang:
            records[lang] = qa(lang, Path(scratch))
            done += 1
    ordered = [records[lang] for lang in LANGS if lang in records]
    out.write_text(json.dumps(ordered, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"records out: {done} (file holds {len(ordered)})", flush=True)
    return 0 if done == len(args.lang) else 1


if __name__ == "__main__":
    sys.exit(main())
