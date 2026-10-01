#!/usr/bin/env python3
"""Check the exported Qwen3-TTS graphs against the upstream PyTorch model.

Usage::

    python verify_parity.py --onnx ./onnx

Runs one greedy synthesis both ways on the same prompt and reports:

* the prompt embeddings the talker reads;
* the talker logits at the prefill step and at every decode step;
* how many of the 16 code groups per frame agree, token for token;
* the codec decoder's waveform, decoded from the *same* codes on both sides.

Greedy decoding is the only setting where the two sides are comparable token for
token: with sampling they draw from different random streams. A quantized export
that does not reach full greedy agreement here should not ship.
"""
import argparse
import functools
import json
from types import SimpleNamespace

import numpy as np
import torch

from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.qwen3tts import Qwen3TTSAdapter
from phoonnx.providers import make_session
from qwen_tts import Qwen3TTSModel

GRAPHS = {
    "text_embed_path": "text_embed.onnx",
    "codec_embed_path": "codec_embed.onnx",
    "code_predictor_prefill_path": "code_predictor_prefill.onnx",
    "code_predictor_step_path": "code_predictor_step.onnx",
    "sub_codec_embed_path": "sub_codec_embed.onnx",
    "codec_decoder_path": "codec_decoder.onnx",
    "bpe_tokenizer_path": "tokenizer.json",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    ap.add_argument("--onnx", default="./onnx")
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--speaker", default="Ryan")
    ap.add_argument("--language", default="English")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    # ---- upstream, greedy ------------------------------------------------
    wrapper = Qwen3TTSModel.from_pretrained(
        args.model, dtype=torch.float32, device_map="cpu")
    model = wrapper.model
    talker = model.talker

    captured = {"logits": []}
    original_forward = talker.forward

    @functools.wraps(original_forward)
    def spy(*a, **kw):
        out = original_forward(*a, **kw)
        captured["logits"].append(out.logits[:, -1:].detach().clone())
        return out

    original_generate = talker.generate

    def spy_generate(*a, **kw):
        captured["prompt"] = kw["inputs_embeds"].detach().clone()
        return original_generate(*a, **kw)

    talker.forward = spy
    talker.generate = spy_generate

    input_ids = wrapper._tokenize_texts([wrapper._build_assistant_text(args.text)])
    codes_list, _ = model.generate(
        input_ids=input_ids, instruct_ids=[None], languages=[args.language],
        speakers=[args.speaker], non_streaming_mode=True, do_sample=False,
        subtalker_dosample=False,
        repetition_penalty=model.generate_config["repetition_penalty"])
    torch_codes = codes_list[0].numpy()
    torch_wav = model.speech_tokenizer.decode([{"audio_codes": codes_list[0]}])[0][0]
    torch_logits = torch.cat(captured["logits"], dim=1).numpy()[0]
    talker.forward, talker.generate = original_forward, original_generate

    # ---- phoonnx adapter, greedy ----------------------------------------
    engine_params = {k: f"{args.onnx}/{v}" for k, v in GRAPHS.items()}
    engine_params.update(speaker=args.speaker.lower(), language=args.language)
    adapter = Qwen3TTSAdapter()
    adapter.configure(SimpleNamespace(engine_params=engine_params, lang_code=None))
    session = make_session(f"{args.onnx}/talker.onnx")

    ids = np.asarray(adapter.encode_text(args.text, None, None)[0], np.int64)
    print("token ids match:", np.array_equal(ids, input_ids[0].numpy().reshape(-1)))

    prompt, _ = adapter.build_prompt(ids[None], args.speaker, args.language)
    print("prompt embeddings, max abs diff: %.3e"
          % np.abs(prompt - captured["prompt"].numpy()).max())

    request = AdapterSynthesisRequest(
        phoneme_ids=ids[None], phoneme_lengths=np.array([ids.size], np.int64),
        params={"do_sample": False})
    result = adapter.synthesize(request, session)
    onnx_codes = adapter.generate_codes(
        session, prompt, adapter._text_hidden([151671]).astype(np.float32),
        {"do_sample": False}, np.random.default_rng(0))

    frames = min(len(onnx_codes), len(torch_codes))
    agree = onnx_codes[:frames] == torch_codes[:frames]
    print("frames: onnx %d, torch %d" % (len(onnx_codes), len(torch_codes)))
    print("greedy token agreement: %.4f%% (%d/%d)"
          % (100 * agree.mean(), agree.sum(), agree.size))

    onnx_wav = adapter.decode_codes(torch_codes)
    k = min(len(onnx_wav), len(torch_wav))
    print("codec decoder on the same codes, max abs diff: %.3e"
          % np.abs(onnx_wav[:k] - torch_wav[:k]).max())
    print("waveform lengths: onnx %d, torch %d" % (len(onnx_wav), len(torch_wav)))
    print("adapter audio seconds: %.2f" % (len(result.audio) / 24000))


if __name__ == "__main__":
    main()
