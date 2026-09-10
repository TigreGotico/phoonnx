"""Synthesize the WER-gate sentences with the Magpie-TTS engine.

    python synth.py [lang ...]     # defaults to all five languages

Writes one WAV per sentence into ./wav/<lang>_<i>.wav. Feeds run_asr.py.
"""
import json
import sys
import time
import wave
from pathlib import Path


def repo_root() -> Path:
    """Walk up from this file until a directory with pyproject.toml is found."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("could not locate phoonnx repo root above " + str(here))


sys.path.insert(0, str(repo_root()))
from phoonnx.model_manager import TTSModelInfo  # noqa: E402

VOICES = {
    "fr": "magpie/Leo/fr",
    "it": "magpie/Aria/it",
    "vi": "magpie/Sofia/vi",
    "ar": "magpie/John/ar",
    "ko": "magpie/Jason/ko",
}


def main(langs):
    idx = json.load(open(Path(__file__).parent / "sentences.json"))
    voice_idx = json.load(open(repo_root() / "phoonnx/voice_index/magpie.json"))
    out_dir = Path(__file__).parent / "wav"
    out_dir.mkdir(exist_ok=True)
    for lang in langs:
        voice_id = VOICES[lang]
        entry = voice_idx[voice_id]
        info = TTSModelInfo(**entry)
        t0 = time.time()
        voice = info.load()
        print(f"[{lang}] loaded in {time.time()-t0:.1f}s", flush=True)
        for i, sentence in enumerate(idx[lang]):
            wav_path = out_dir / f"{lang}_{i}.wav"
            t0 = time.time()
            with wave.open(str(wav_path), "wb") as wf:
                voice.synthesize_wav(sentence, wf)
            dt = time.time() - t0
            print(f"[{lang}] sentence {i} -> {wav_path.name} in {dt:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(VOICES.keys()))
