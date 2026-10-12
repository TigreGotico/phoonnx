#!/usr/bin/env python3
"""Intelligibility and speed gate for a Llasa bundle.

Synthesises a fixed sentence set, transcribes the result, and reports WER (English,
word level) or CER (Chinese, character level) together with the real-time factor.

The floor is the **upstream torch pipeline** on the same sentences: Llasa samples a
different speaker and prosody every call, so an absolute WER means little on its own,
but torch and ONNX driven the same way must land in the same place. A gap between the
two is export loss; a high number in both is the model.

Recognisers (both are the phoonnx org's own ONNX packagings):

* English  — ``nemo-parakeet-tdt-0.6b-v3``
* Chinese  — ``OpenVoiceOS/omnilingual-asr-ctc-1b-onnx``

Usage::

    python bench_wer.py --bundle out/bundle/llasa-1b-onnx --output wer.json \
        [--torch-floor] [--voice en_female_a]
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np

SENTENCES = {
    "en": [
        "The morning train was late again, so she walked the rest of the way.",
        "He kept the letter in a drawer for almost twenty years.",
        "Please close the window before the rain starts.",
        "They agreed to meet at the corner of the old market square.",
        "The doctor said the results would arrive on Thursday.",
        "Nobody expected the river to rise that quickly.",
        "She counted the coins twice and put them back in the jar.",
        "We should leave early if we want to avoid the traffic.",
    ],
    "zh": [
        "他昨天晚上很晚才回到家里。",
        "这本书我已经看了三遍了。",
        "请你把窗户关上，外面开始下雨了。",
        "我们约好在老市场的路口见面。",
        "医生说结果星期四就会出来。",
        "没有人想到河水会涨得这么快。",
        "她把硬币数了两遍又放回罐子里。",
        "如果想避开堵车，我们应该早点出发。",
    ],
}

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str, lang: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def tokens(text: str, lang: str):
    return list(text.replace(" ", "")) if lang == "zh" else text.split()


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_rate(ref: str, hyp: str, lang: str) -> float:
    r, h = tokens(normalize(ref, lang), lang), tokens(normalize(hyp, lang), lang)
    return edit_distance(r, h) / max(1, len(r))


_ASR_CACHE = {}


def load_asr(lang: str):
    """English through parakeet, Chinese through the org's Omnilingual CTC 1B.

    ``omnilingual-ctc`` is not in upstream onnx-asr; it needs the TigreGotico fork's
    ``integration`` branch. The 300M sibling is materially worse on Mandarin (it
    misreads tones on clean synthetic speech), so the gate uses the 1B.
    """
    if lang in _ASR_CACHE:
        return _ASR_CACHE[lang]
    import onnx_asr
    if lang == "en":
        entry = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3"), {}
    else:
        from huggingface_hub import snapshot_download
        path = snapshot_download("OpenVoiceOS/omnilingual-asr-ctc-1b-onnx",
                                 allow_patterns=["*.json", "model.onnx", "model.onnx*", "*.txt"])
        entry = onnx_asr.load_model("omnilingual-ctc", path), {}
    _ASR_CACHE[lang] = entry
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--wav-dir", default=None)
    ap.add_argument("--langs", default="en,zh")
    ap.add_argument("--torch-floor", action="store_true",
                    help="also run the upstream torch pipeline on the same sentences")
    ap.add_argument("--model", default="HKUSTAudio/Llasa-1B")
    args = ap.parse_args()

    import soundfile as sf
    from phoonnx.config import SynthesisConfig
    from phoonnx.voice import TTSVoice

    bundle = Path(args.bundle)
    voice = TTSVoice.load(
        str(bundle / "model.onnx"),
        config_path=str(bundle / "config.json"),
        engine_params={
            "codec_decoder_path": str(bundle / "xcodec2_decoder.onnx"),
            "tokenizer_path": str(bundle / "tokenizer.json"),
            "voices_path": str(bundle / "voices.json"),
        },
    )

    results = {}
    for lang in args.langs.split(","):
        asr, asr_kwargs = load_asr(lang)
        preset = "en_female_a" if lang == "en" else "zh_male_a"
        rows = []
        for idx, sentence in enumerate(SENTENCES[lang]):
            syn = SynthesisConfig(extra_params={"voice": preset, "seed": 1000 + idx})
            start = time.time()
            audio = np.concatenate([c.audio_float_array
                                    for c in voice.synthesize(sentence, syn_config=syn)])
            elapsed = time.time() - start
            path = Path(args.wav_dir or ".") / f"onnx_{lang}_{idx}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), audio, 16000)
            hyp = asr.recognize(str(path), **asr_kwargs)
            duration = len(audio) / 16000
            rows.append({
                "ref": sentence, "hyp": hyp,
                "err": error_rate(sentence, hyp, lang),
                "seconds": duration, "wall": elapsed,
                "rtf": elapsed / max(duration, 1e-6),
            })
            print(f"[onnx/{lang}] {rows[-1]['err']:.3f} rtf={rows[-1]['rtf']:.2f} {hyp!r}",
                  flush=True)
        results[f"onnx_{lang}"] = {
            "metric": "CER" if lang == "zh" else "WER",
            "voice": preset,
            "mean": float(np.mean([r["err"] for r in rows])),
            "rtf_mean": float(np.mean([r["rtf"] for r in rows])),
            "rows": rows,
        }

    if args.torch_floor:
        results.update(torch_floor(args, SENTENCES, args.langs.split(",")))

    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    for key, value in results.items():
        print(f"{key}: {value['metric']} {value['mean']:.4f}")


def torch_floor(args, sentences, langs) -> dict:
    """Run upstream's own transformers + XCodec2 path over the same sentences."""
    import sys
    import soundfile as sf
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(__file__).parent))
    from export_xcodec2 import load_xcodec2

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    except TypeError:  # transformers < 5 spells it torch_dtype
        lm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    codec = load_xcodec2()
    end_id = tok.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    base = tok.convert_tokens_to_ids("<|s_0|>")

    out = {}
    for lang in langs:
        asr, asr_kwargs = load_asr(lang)
        rows = []
        for idx, sentence in enumerate(sentences[lang]):
            chat = [
                {"role": "user",
                 "content": "Convert the text to speech:<|TEXT_UNDERSTANDING_START|>"
                            + sentence + "<|TEXT_UNDERSTANDING_END|>"},
                {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"},
            ]
            ids = tok.apply_chat_template(chat, tokenize=True, return_tensors="pt",
                                          continue_final_message=True,
                                          return_dict=True)["input_ids"]
            torch.manual_seed(1000 + idx)
            start = time.time()
            with torch.no_grad():
                gen = lm.generate(ids, max_length=ids.shape[1] + 1000, eos_token_id=end_id,
                                  do_sample=True, temperature=0.9, top_p=0.95)
            codes = [int(t) - base for t in gen[0][ids.shape[1]:].tolist()
                     if base <= int(t) < base + 65536]
            with torch.no_grad():
                audio = codec.decode_code(torch.tensor(codes).view(1, 1, -1))[0, 0].numpy()
            elapsed = time.time() - start
            path = Path(args.wav_dir or ".") / f"torch_{lang}_{idx}.wav"
            sf.write(str(path), audio, 16000)
            hyp = asr.recognize(str(path), **asr_kwargs)
            duration = len(audio) / 16000
            rows.append({"ref": sentence, "hyp": hyp,
                         "err": error_rate(sentence, hyp, lang),
                         "seconds": duration, "wall": elapsed,
                         "rtf": elapsed / max(duration, 1e-6)})
            print(f"[torch/{lang}] {rows[-1]['err']:.3f} rtf={rows[-1]['rtf']:.2f} {hyp!r}",
                  flush=True)
        out[f"torch_{lang}"] = {
            "metric": "CER" if lang == "zh" else "WER",
            "voice": "unprompted",
            "mean": float(np.mean([r["err"] for r in rows])),
            "rtf_mean": float(np.mean([r["rtf"] for r in rows])),
            "rows": rows,
        }
    return out


if __name__ == "__main__":
    main()
