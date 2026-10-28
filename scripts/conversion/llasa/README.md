# Llasa -> ONNX

Converts [Llasa](https://huggingface.co/HKUSTAudio/Llasa-1B) (arXiv 2502.04128) and its
codec [XCodec2](https://huggingface.co/HKUSTAudio/xcodec2) into the two graphs
`phoonnx/engines/llasa.py` drives.

Llasa is a LLaMA-3.2 backbone whose vocabulary was extended with the 65,536 `<|s_N|>`
speech tokens of XCodec2, a single-codebook 16 kHz codec at 50 tokens per second.
Generation is a plain autoregressive loop, so one KV-cached graph serves both prefill
and decode.

Both upstream repositories are **CC BY-NC 4.0**. The conversion inherits that.

## Pipeline

```bash
# 1. the language model  ->  model_fp32.onnx (+ .onnx_data)
python export_llm.py --model HKUSTAudio/Llasa-1B --output out/llasa-1b

# 2. drop the duplicated tied embedding  ->  model.onnx  (7.07 GB -> 5.48 GB)
python dedupe_embedding.py --input out/llasa-1b/model_fp32.onnx \
                           --output out/llasa-1b/model.onnx

# 3. the codec decoder  ->  decoder.onnx
python export_xcodec2.py --output out/xcodec2

# 4. gates
python parity_llm.py   --onnx out/llasa-1b/model.onnx --steps 48
python parity_codec.py --onnx out/xcodec2/decoder.onnx --tokens ref_codes.npy

# 5. bundle voices, then measure intelligibility and speed
python make_presets.py --bundle out/bundle/llasa-1b-onnx --output voices.json
python bench_wer.py    --bundle out/bundle/llasa-1b-onnx --output wer.json --torch-floor
```

## Environments

Two virtualenvs are needed, because the two halves disagree about `transformers`:

* **the LM** — any recent `transformers`. `export_llm.py` and `parity_llm.py` were run
  on 5.14.
* **the codec** — `transformers==4.47.1`, plus `vector_quantize_pytorch`, `einops`,
  `torchtune` and `torchao==0.8.0`. Upstream's `XCodec2Model.__init__` calls
  `Wav2Vec2BertModel.from_pretrained` *inside* itself, which `transformers` 5.x refuses
  (it builds models under a meta-device context).

`hf-audio/xcodec2`, the transformers-native port, is **not** usable: its checkpoint still
carries the pre-rename tensor names (`semantic_model`, `decoder`, `fc_prior`), while
`transformers` 5.x expects `semantic_encoder`, `acoustic_decoder`, `fc_encoder`. There is
no conversion mapping between them, so `from_pretrained` loads zero weights and reports
success. The scripts here use upstream's own `modeling_xcodec2.py` instead.

## What the exports do

**`export_llm.py`** traces the model with a 2-token input over a 3-token past, so one
graph covers prefill (long input, empty past) and decode (one token, long past). Only the
last position's logits leave the graph: over a 193,800-wide vocabulary a whole prefill
would cost about 78 MB per 100 prompt tokens for a value the sampler never reads.

**`dedupe_embedding.py`** rewrites the tied `lm_head`. Llasa shares `lm_head` with
`embed_tokens`, but the exporter writes the 193,800 x 2,048 matrix twice — once for the
`Gather`, once transposed for the head's `MatMul`. The head becomes
`Reshape -> Gemm(transB=1) -> Unsqueeze` over the embedding initialiser, and the copy is
deleted. The script refuses to run unless the two tensors really are transposes of each
other.

**`export_xcodec2.py`** replaces two pieces that ONNX cannot trace:

* the **ISTFT head** builds `mag * exp(i*phase)` and calls `torch.fft.irfft`; ONNX has no
  complex tensors. `RealISTFTHead` re-expresses the inverse real FFT as two constant
  cosine/sine matmuls and the overlap-add `fold` as a `conv_transpose1d` with an identity
  kernel.
* `ResidualFSQ.get_output_from_indices` goes through `einops`, which bakes the traced
  frame count into the graph. Since this is a single FSQ level with an implicit 65,536 x 8
  codebook, the lookup becomes an embedding gather followed by the quantiser's own
  `project_out`.

It also restores the encoder's `weight_norm` tensors: upstream saved them under the
pre-2.1 names (`weight_g`/`weight_v`) while modern torch registers
`parametrizations.weight.original0/1`, so `from_pretrained` silently leaves `CodecEnc`
at its random init.

## Quantization

Every quantized variant was built and **every one was rejected**:

| Variant | prefill mean abs logit diff | decode mean abs logit diff | greedy agreement (en / zh) |
|---|---|---|---|
| fp32 | 7.4e-06 | 5.4e-06 | 48/48, 48/48 |
| dynamic int8, per-tensor | 1.83 | 1.35 | 0/48, 0/48 |
| dynamic int8, per-channel | 3.46 | 3.21 | 0/48, 25/48 |
| 4-bit `MatMulNBits`, block 32 | 0.89 | 1.44 | 1/48, 9/48 |

Only fp32 is published. The published Llasa ONNX repositories that do ship int8/q4/uint8
files have not been shown to clear this gate.
