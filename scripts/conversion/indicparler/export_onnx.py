#!/usr/bin/env python3
"""Export ai4bharat/indic-parler-tts to ONNX for phoonnx.

Parler-TTS is an encoder-decoder codec LM:

    Flan-T5 encoder (description)  ->  encoder_hidden_states
    embed_prompts(prompt_input_ids) ->  prompt_hidden_states  (prepended to the
                                        decoder input embeddings, NOT cross-attended,
                                        because `prompt_cross_attention=False`)
    24-layer AR decoder over 9 delayed DAC codebooks, cross-attending to the encoder
    DAC decoder (44.1 kHz) turns the 9 codebooks back into a waveform

Four graphs are written:

    text_encoder.onnx     input_ids, attention_mask -> encoder_hidden_states
    decoder_prefill.onnx  codec_input_ids, prompt_input_ids, encoder_hidden_states,
                          encoder_attention_mask
                          -> logits, present self-KV (24x2), present cross-KV (24x2)
    decoder_decode.onnx   codec_input_ids, encoder_attention_mask,
                          past self-KV (24x2), cross-KV (24x2)
                          -> logits, present self-KV (24x2)
    dac_decoder.onnx      audio_codes -> audio_values

The cross-attention KV is computed once in the prefill graph and then held constant:
the decode graph consumes it but does not re-emit it. This is the first
encoder-decoder engine in phoonnx -- every previous AR engine is decoder-only, so
there is no cross-KV to carry.
"""
import argparse
import json
import os

import torch
from torch import nn

NUM_CODEBOOKS = 9


class TextEncoderWrapper(nn.Module):
    """Flan-T5 encoder + the mask-zeroing that ParlerTTS applies to its output."""

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids, attention_mask):
        hidden = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return hidden * attention_mask[..., None].to(hidden.dtype)


class PrefillWrapper(nn.Module):
    """First AR step: embed the prompt, prepend it, run the full decoder, keep the cache."""

    def __init__(self, model):
        super().__init__()
        self.embed_prompts = model.embed_prompts
        self.decoder = model.decoder

    def forward(self, codec_input_ids, prompt_input_ids, encoder_hidden_states, encoder_attention_mask):
        prompt_hidden_states = self.embed_prompts(prompt_input_ids)
        out = self.decoder(
            input_ids=codec_input_ids.reshape(-1, codec_input_ids.shape[-1]),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            prompt_hidden_states=prompt_hidden_states,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        if hasattr(past, "to_legacy_cache"):
            past = past.to_legacy_cache()
        flat = []
        for layer in past:
            flat.extend(layer[:4])
        return (out.logits[:, -1],) + tuple(flat)


class DecodeWrapper(nn.Module):
    """Subsequent AR steps: one codec frame in, cache in, cache out."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
        self.num_layers = len(model.decoder.model.decoder.layers)

    def forward(self, codec_input_ids, encoder_hidden_states, encoder_attention_mask, *past):
        legacy = tuple(tuple(past[4 * i: 4 * i + 4]) for i in range(self.num_layers))
        out = self.decoder(
            input_ids=codec_input_ids.reshape(-1, codec_input_ids.shape[-1]),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=legacy,
            use_cache=True,
            return_dict=True,
        )
        new = out.past_key_values
        if hasattr(new, "to_legacy_cache"):
            new = new.to_legacy_cache()
        flat = []
        for layer in new:
            flat.extend(layer[:2])          # self-attention KV only; cross-KV is invariant
        return (out.logits[:, -1],) + tuple(flat)


class DacDecoderWrapper(nn.Module):
    def __init__(self, audio_encoder):
        super().__init__()
        self.audio_encoder = audio_encoder

    def forward(self, audio_codes):
        quantized = self.audio_encoder.quantizer.from_codes(audio_codes)[0]
        return self.audio_encoder.decoder(quantized)


def _kv_names(num_layers, prefix, cross=True):
    names = []
    for i in range(num_layers):
        names += [f"{prefix}.{i}.decoder.key", f"{prefix}.{i}.decoder.value"]
        if cross:
            names += [f"{prefix}.{i}.encoder.key", f"{prefix}.{i}.encoder.value"]
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ai4bharat/indic-parler-tts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--only", default=None,
                    help="comma-separated subset: text_encoder,decoder_prefill,decoder_decode,dac_decoder")
    args = ap.parse_args()

    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    os.makedirs(args.out, exist_ok=True)
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        args.model, attn_implementation="eager", torch_dtype=torch.float32
    ).eval()
    for cfg in (model.config, model.config.decoder, model.config.text_encoder, model.config.audio_encoder):
        cfg._attn_implementation = "eager"

    num_layers = len(model.decoder.model.decoder.layers)
    d_model = model.decoder.config.hidden_size

    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    prompt_tok = AutoTokenizer.from_pretrained(args.model)
    want = set(args.only.split(",")) if args.only else None
    def _do(name):
        return want is None or name in want

    desc = desc_tok("A clear voice, very close recording, no background noise.", return_tensors="pt")
    prompt = prompt_tok("Hello world, this is a test.", return_tensors="pt")

    # ---------------- text encoder ----------------
    if _do("text_encoder"):
     print("exporting text_encoder ...", flush=True)
     torch.onnx.export(
         TextEncoderWrapper(model.text_encoder).eval(),
         (desc.input_ids, desc.attention_mask),
         os.path.join(args.out, "text_encoder.onnx"),
         input_names=["input_ids", "attention_mask"],
         output_names=["encoder_hidden_states"],
         dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                       "attention_mask": {0: "batch", 1: "seq"},
                       "encoder_hidden_states": {0: "batch", 1: "seq"}},
         opset_version=args.opset,
     )

    with torch.no_grad():
        enc_hidden = TextEncoderWrapper(model.text_encoder).eval()(desc.input_ids, desc.attention_mask)

    # ---------------- decoder prefill ----------------
    print("exporting decoder_prefill ...", flush=True)
    codec = torch.full((1, NUM_CODEBOOKS, 1), model.generation_config.decoder_start_token_id, dtype=torch.long)
    prefill = PrefillWrapper(model).eval()
    out_names = ["logits"] + _kv_names(num_layers, "present", cross=True)
    dyn = {"codec_input_ids": {0: "batch", 2: "tgt"},
           "prompt_input_ids": {0: "batch", 1: "prompt"},
           "encoder_hidden_states": {0: "batch", 1: "src"},
           "encoder_attention_mask": {0: "batch", 1: "src"},
           "logits": {0: "batch_codebooks"}}
    for n in out_names[1:]:
        dyn[n] = {0: "batch", 2: "src" if n.endswith((".encoder.key", ".encoder.value")) else "past"}
    torch.onnx.export(
        prefill,
        (codec, prompt.input_ids, enc_hidden, desc.attention_mask),
        os.path.join(args.out, "decoder_prefill.onnx"),
        input_names=["codec_input_ids", "prompt_input_ids", "encoder_hidden_states", "encoder_attention_mask"],
        output_names=out_names,
        dynamic_axes=dyn,
        opset_version=args.opset,
    )

    with torch.no_grad():
        pre_out = prefill(codec, prompt.input_ids, enc_hidden, desc.attention_mask)
    past = list(pre_out[1:])

    # ---------------- decoder decode ----------------
    print("exporting decoder_decode ...", flush=True)
    decode = DecodeWrapper(model).eval()
    in_names = ["codec_input_ids", "encoder_hidden_states", "encoder_attention_mask"] + _kv_names(
        num_layers, "past_key_values", cross=True)
    out_names = ["logits"] + _kv_names(num_layers, "present", cross=False)
    dyn = {"codec_input_ids": {0: "batch", 2: "tgt"},
           "encoder_hidden_states": {0: "batch", 1: "src"},
           "encoder_attention_mask": {0: "batch", 1: "src"},
           "logits": {0: "batch_codebooks"}}
    for n in in_names[3:]:
        dyn[n] = {0: "batch", 2: "src" if n.endswith((".encoder.key", ".encoder.value")) else "past"}
    for n in out_names[1:]:
        dyn[n] = {0: "batch", 2: "past_plus"}
    torch.onnx.export(
        decode,
        (codec, enc_hidden, desc.attention_mask, *past),
        os.path.join(args.out, "decoder_decode.onnx"),
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dyn,
        opset_version=args.opset,
    )

    # ---------------- DAC decoder ----------------
    print("exporting dac_decoder ...", flush=True)
    codes = torch.zeros((1, NUM_CODEBOOKS, 16), dtype=torch.long)
    torch.onnx.export(
        DacDecoderWrapper(model.audio_encoder).eval(),
        (codes,),
        os.path.join(args.out, "dac_decoder.onnx"),
        input_names=["audio_codes"],
        output_names=["audio_values"],
        dynamic_axes={"audio_codes": {0: "batch", 2: "frames"},
                      "audio_values": {0: "batch", 2: "samples"}},
        opset_version=args.opset,
    )

    meta = {
        "engine": "indic_parler",
        "source_model": args.model,
        "sample_rate": model.config.sampling_rate,
        "num_codebooks": NUM_CODEBOOKS,
        "num_layers": num_layers,
        "hidden_size": d_model,
        "num_attention_heads": model.decoder.config.num_attention_heads,
        "audio_vocab_size": model.decoder.config.vocab_size,
        "bos_token_id": model.generation_config.decoder_start_token_id,
        "eos_token_id": model.generation_config.eos_token_id,
        "pad_token_id": model.generation_config.pad_token_id,
        "max_length": model.generation_config.max_length,
        "prompt_vocab_size": model.config.vocab_size,
        "description_tokenizer": model.config.text_encoder._name_or_path,
        "prompt_tokenizer": args.model,
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
