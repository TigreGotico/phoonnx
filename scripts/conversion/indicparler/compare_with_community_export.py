#!/usr/bin/env python3
"""Check ``naklitechie/indic-parler-tts-ONNX`` against upstream PyTorch, step by step.

Usage::

    python compare_with_community_export.py

Requires network access (downloads the community ``decoder_with_past_model.onnx`` plus
``ai4bharat/indic-parler-tts`` in float32 torch) and the ``parler_tts`` package.

Why this script exists
-----------------------
``naklitechie/indic-parler-tts-ONNX`` is the only community export of Indic Parler-TTS
on the Hub. Its text encoder and its prefill (``decoder_model.onnx``) graph are correct.
Its decode-with-cache graph (``decoder_with_past_model.onnx``) is not: upstream gates
cross-attention on ``encoder_hidden_states is not None``

    https://github.com/huggingface/parler-tts/blob/main/parler_tts/modeling_parler_tts.py

and exporting the decode step *without* passing that argument makes the traced graph
silently drop cross-attention in all 24 layers. It still produces audio -- just audio
that ignores the voice description. This script reproduces the numbers cited in
phoonnx PR #363: step-1 logits differ from torch by ~32.6 (mean ~15.9 over 40 steps),
greedy argmax agreement 0/40. It is the methodology check, not a one-off narrative claim.

For contrast, run the same forced-decode loop against phoonnx's own
``decoder_decode.onnx`` (see ``export_onnx.py`` in this directory) with ``--onnx`` to
confirm it does *not* reproduce this divergence.
"""
import argparse

import numpy as np
import torch
from huggingface_hub import hf_hub_download

NUM_LAYERS = 24
NUM_CODEBOOKS = 9


def _forced_prompt(model, tokenizer, desc_tokenizer, text, description):
    prompt = tokenizer(text, return_tensors="pt")
    desc = desc_tokenizer(description, return_tensors="pt")
    return prompt, desc


def run_torch_reference(model, prompt, desc, n_steps):
    """Greedy-forced decode: at every step, feed the model's *own* previous choice back
    in (true greedy), and capture the logits before sampling. Returns (logits, codes)
    with logits shape (n_steps, NUM_CODEBOOKS, vocab)."""
    with torch.no_grad():
        enc_out = model.text_encoder(input_ids=desc.input_ids, attention_mask=desc.attention_mask)
        enc_hidden = enc_out.last_hidden_state * desc.attention_mask[..., None].to(enc_out.last_hidden_state.dtype)
        prompt_hidden = model.embed_prompts(prompt.input_ids)

        bos = model.generation_config.decoder_start_token_id
        codec_ids = torch.full((1, NUM_CODEBOOKS, 1), bos, dtype=torch.long)
        past = None
        all_logits = []
        codes = []
        for step in range(n_steps):
            out = model.decoder(
                input_ids=codec_ids.reshape(-1, codec_ids.shape[-1]) if step == 0 else codec_ids[:, :, -1:].reshape(-1, 1),
                encoder_hidden_states=enc_hidden,
                encoder_attention_mask=desc.attention_mask,
                prompt_hidden_states=prompt_hidden if step == 0 else None,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits = out.logits[:, -1].reshape(NUM_CODEBOOKS, -1)
            all_logits.append(logits.numpy())
            next_tok = logits.argmax(-1)
            codes.append(next_tok.numpy())
            codec_ids = torch.cat([codec_ids, next_tok.reshape(1, NUM_CODEBOOKS, 1)], dim=-1)
            past = out.past_key_values
        return np.stack(all_logits), np.stack(codes), enc_hidden.numpy(), desc.attention_mask.numpy()


def run_community_onnx(onnx_path, torch_codes, enc_hidden, enc_mask, n_steps):
    """Feed the *same* torch-chosen codes back into the community decode-with-past graph
    (so a mismatch is attributable to the graph, not to divergent sampling) and record
    its logits at each step."""
    import onnxruntime

    sess = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}
    print("community decoder_with_past inputs:", sorted(input_names))
    has_cross_input = "encoder_hidden_states" in input_names
    print("has encoder_hidden_states input:", has_cross_input)

    past = {n: np.zeros(_zero_shape_for(n), np.float32) for n in input_names
            if n.startswith(("past_key_values", "past."))}
    logits_out = []
    codec_ids = np.full((1, NUM_CODEBOOKS, 1), 1025, np.int64)
    for step in range(n_steps):
        feed = {"input_ids" if "input_ids" in input_names else "codec_input_ids":
                codec_ids[:, :, -1:].reshape(1, NUM_CODEBOOKS, 1)}
        if has_cross_input:
            feed["encoder_hidden_states"] = enc_hidden
        if "encoder_attention_mask" in input_names:
            feed["encoder_attention_mask"] = enc_mask
        feed.update({k: v for k, v in past.items() if k in input_names})
        outs = sess.run(None, feed)
        out_names = [o.name for o in sess.get_outputs()]
        result = dict(zip(out_names, outs))
        logits = result["logits"].reshape(NUM_CODEBOOKS, -1)
        logits_out.append(logits)
        next_tok = torch_codes[step].reshape(1, NUM_CODEBOOKS, 1)
        codec_ids = np.concatenate([codec_ids, next_tok], axis=-1)
        for name in list(past):
            present_name = name.replace("past_key_values", "present").replace("past.", "present.")
            if present_name in result:
                past[name] = result[present_name]
    return np.stack(logits_out)


def _zero_shape_for(name):
    # cross-attention caches carry the source length; self-attention caches start empty.
    return (1, 16, 0, 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ai4bharat/indic-parler-tts")
    ap.add_argument("--community-repo", default="naklitechie/indic-parler-tts-ONNX")
    ap.add_argument("--community-file", default="decoder_with_past_model.onnx")
    ap.add_argument("--text", default="Hello world, this is a test of Indic Parler.")
    ap.add_argument("--description",
                    default="Rohit's voice is clear and expressive, recorded in a "
                             "very close-sounding environment with no background noise.")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    print("loading torch reference model ...", flush=True)
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        args.model, attn_implementation="eager", torch_dtype=torch.float32).eval()
    for cfg in (model.config, model.config.decoder, model.config.text_encoder, model.config.audio_encoder):
        cfg._attn_implementation = "eager"
    prompt_tok = AutoTokenizer.from_pretrained(args.model)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

    prompt, desc = _forced_prompt(model, prompt_tok, desc_tok, args.text, args.description)

    print("running torch greedy decode, %d steps ..." % args.steps, flush=True)
    torch_logits, torch_codes, enc_hidden, enc_mask = run_torch_reference(model, prompt, desc, args.steps)

    print("downloading community decode-with-past graph ...", flush=True)
    onnx_path = hf_hub_download(args.community_repo, args.community_file)
    hf_hub_download(args.community_repo, args.community_file + ".data")

    print("running community decoder_with_past_model.onnx on the same forced codes ...", flush=True)
    community_logits = run_community_onnx(onnx_path, torch_codes, enc_hidden, enc_mask, args.steps)

    diff = np.abs(community_logits - torch_logits)
    step1_diff = diff[0].max()
    mean_diff = diff.max(axis=(1, 2)).mean()
    agree = (community_logits.argmax(-1) == torch_logits.argmax(-1))
    print()
    print("step-1 logits max abs diff: %.3f" % step1_diff)
    print("mean per-step max abs diff over %d steps: %.3f" % (args.steps, mean_diff))
    print("greedy argmax agreement: %d / %d" % (int(agree.sum()), agree.size))
    print()
    print("Compare against phoonnx's own decoder_decode.onnx with the same forced-code")
    print("loop (swap run_community_onnx's feed-building for phoonnx's engine adapter,")
    print("see scripts/conversion/qwen3tts/verify_parity.py for the pattern) to confirm")
    print("the OpenVoiceOS export does not reproduce this divergence.")


if __name__ == "__main__":
    main()
