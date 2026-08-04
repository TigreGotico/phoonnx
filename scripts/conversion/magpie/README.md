# Magpie-TTS ONNX export

Converts [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m)
(MagpieTTS Multilingual, v2607) into the seven ONNX graphs that the phoonnx
`magpie` engine loads.

Unlike the other converters here, this one does **not** vendor the model code. The
checkpoint is a NeMo `.nemo` archive whose config names NeMo classes directly, so the
export loads the real `MagpieTTSModel` and re-expresses its forward passes.

## Requirements

NeMo 2.8.0rc0 or newer. The published `nemo_toolkit` 2.7.3 lacks the `pt-BR` locale the
checkpoint needs, so use the source tree:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu 'nemo_toolkit[tts]==2.7.3'
uv pip install --python .venv/bin/python onnx onnxscript onnxruntime
git clone --depth 1 https://github.com/NVIDIA-NeMo/Speech.git nemo-speech
export PYTHONPATH=$PWD/nemo-speech
```

CPU is enough. The whole export takes a few minutes.

## Running

```bash
export MAGPIE_ONNX_DIR=/somewhere/with/2GB
python export_encoder.py     # text_encoder.onnx
python export_decoder.py     # cross_kv.onnx, decoder_step.onnx
python export_local.py       # audio_embed.onnx, local_step.onnx, lt_embed.onnx
python export_codec.py       # codec_decoder.onnx
python dump_assets.py        # context embeddings, tokenizer, config
python parity.py "Some sentence." en        # exact mode parity
python parity.py "Some sentence." en kv     # KV-cache mode parity
```

Each export script checks itself against the torch module it came from and prints the
maximum absolute difference.

## Architecture

Text goes through a 6-layer causal encoder. A 12-layer causal decoder cross-attends to
those states and autoregressively predicts audio codec tokens. The decoder emits 8
codebooks for 2 stacked frames at a time (16 tokens per step). A 2-layer local
transformer then refines those 16 tokens one by one. NanoCodec turns the finished codes
into a 22.05 kHz waveform.

Speaker identity is a baked context embedding of shape (5, 217, 768), prepended to the
decoder input. This checkpoint has no voice cloning, so the embedding is a static asset
rather than a graph.

Classifier-free guidance runs the conditional and unconditional branches as one batch of
2. The unconditional branch gets a zeroed encoder output, a conditioning mask that keeps
only the first text position, and a zeroed context prefix.

## Graphs

| Graph | Inputs | Outputs |
|---|---|---|
| `text_encoder` | `text`, `text_mask` | `encoder_out` |
| `cross_kv` | `cond` | `cross_k`, `cross_v` |
| `decoder_step` | `x`, `pos`, `self_k`, `self_v`, `cross_k`, `cross_v`, `cond_mask`, `attn_prior` | `logits`, `dec_out`, `new_self_k`, `new_self_v`, `cross_attn_probs` |
| `local_step` | `h`, `pos`, `cache_k`, `cache_v`, `cb` | `logits`, `new_k`, `new_v` |
| `audio_embed` | `codes` | `emb` |
| `lt_embed` | `tok`, `cb` | `emb` |
| `codec_decoder` | `codes` | `audio` |

`decoder_step` returns the cross-attention probabilities of all 12 layers because the
inference loop needs them. They drive the attention prior that keeps the model reading
the text in order, and the end-of-speech decision.

## The KV cache is not free here

NeMo ships `use_kv_cache_for_inference: false`. In that mode the decoder recomputes the
whole sequence every step, and the newest attention prior is applied to *every* query
position, not just the newest one. Past positions therefore change from step to step, so
a KV cache gives different tokens — not a rounding difference, a different sample path.

`decoder_step` supports both, because NeMo supports both:

- **exact** — pass the full sequence with an empty cache. Matches the NeMo default.
- **cached** — pass one frame plus the cache. Matches `use_kv_cache_for_inference: true`.

`parity.py` checks each mode against the matching NeMo setting. Both reach 100% greedy
token agreement over all 8 codebooks.

## Attribution

The weights and the architecture are NVIDIA's, under the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license).
The scripts here only change the file format.
