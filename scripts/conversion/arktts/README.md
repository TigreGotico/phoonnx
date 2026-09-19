# ArkTTS conversion toolchain

ArkTTS is the DualAR text-to-speech architecture behind two Apache-2.0 checkpoints:

| Checkpoint | Languages | Voices |
|---|---|---|
| [`Audio8/Audio8-TTS-Preview-0.6b`](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b) | yue, zh, nl, en, fr, de, it, ja, ko, pl, es | none bundled — zero-shot cloning |
| [`itzune/zortzi-tts`](https://huggingface.co/itzune/zortzi-tts) | eu (Basque) | Maider, Antton |

The second is a fine-tune of the first, and the two ship **byte-identical**
`modeling_arktts.py`, `modeling_arktts_codec.py`, `processing_arktts.py`, `tokenizer.json`
and `codec.pth`. One ONNX contract therefore covers both, one engine
(`phoonnx/engines/arktts.py`) drives both, and the codec decoder graph is interchangeable
between them.

## The contract

Three graphs, named exactly as the official export at
[`itzune/zortzi-tts-onnx`](https://huggingface.co/itzune/zortzi-tts-onnx) names them, so
graphs from either source are drop-in replacements for each other.

```
slow_ar_<precision>.onnx     24-layer backbone; one step over T positions
  in   codes[1, 11, T] int64, input_pos[T] int64,
       cache_key_{0..23} / cache_value_{0..23}  [1, 2, 2048, 64]
  out  logits[1, T, 4097], slow_hidden[1, T, 896],
       key_delta_{i} / value_delta_{i}          [1, 2, T, 64]

fast_ar_<precision>.onnx     4-layer depth transformer over the ten codebooks
  in   slow_hidden[1, 1, 896], token_id[1, 1] int64, use_slow_hidden[1] bool,
       input_pos[1] int64, cache_key_{0..3} / cache_value_{0..3}  [1, 2, 10, 64]
  out  logits[1, 1, 4096], key_delta_{i} / value_delta_{i}        [1, 2, 1, 64]

codec_decoder_fp16.onnx      codes[1, 10, T] int64 -> audio[1, 1, samples] @ 44.1 kHz
```

Two properties are load-bearing:

* **The KV cache is a fixed 2048-wide window.** A graph writes its new keys and values at
  `input_pos` and attends over the whole window, masking by `key <= input_pos[query]`.
  Slots the loop never wrote hold zeros and stay out of the softmax. The graph returns only
  the delta, so the caller keeps the window. Scattering a delta to the wrong offset gives
  wrong attention and no error.
* **The slow logits are sliced to 4097** — the 4096 semantic logits followed by the EOS
  logit. Upstream masks everything else before sampling, so the full 155776-entry vocabulary
  is never read. An index below 4096 is already codebook 0's value.

## Scripts

### `export_arktts_onnx.py`

```bash
python export_arktts_onnx.py --repo Audio8/Audio8-TTS-Preview-0.6b \
    --out-dir ./onnx --fp16 --drop-fp32
```

Nothing is vendored: the checkpoint's own modeling code loads through `transformers`, and
`arktts_wrappers.py` only re-plumbs the KV cache, which upstream hides in `nn.Module`
buffers the exporter cannot carry.

Four upstream constructs need handling before the tracer accepts the model, and each is a
named function with the reasoning in its docstring:

| Construct | Why it blocks | What the script does |
|---|---|---|
| `mask &= ...` in the codec's window transformer | `aten::__iand_` has no opset-17 lowering | registers a symbolic mapping it to `And` |
| `torch.polar` in the codec's rotary table | opset 17 has no complex tensors | rebuilds the table from `cos`/`sin` |
| `_extra_padding`'s `math.ceil` over a traced shape | lowers to an unsupported op | drops it, after asserting every causal convolution *on the decode path* is stride 1, where the padding is provably zero |
| `x.float()` / `.to(x.dtype)` in the RMS norms and rotary application | become `Cast` nodes with hard-coded types that the fp16 converter cannot reconcile | removes them; they are no-ops in a float32 export |

**Use `transformers==4.57.x`.** On transformers 5.x the checkpoint's non-persistent RoPE
buffer is never initialised, and the reference model silently produces `NaN` logits rather
than failing — every parity number would come out as `NaN`.

**The codec decoder is written in single precision only.** Its window transformer builds a
rotary table inside the graph, and the half-precision converter leaves that `Einsum` with
one single- and one half-precision operand, which ONNX Runtime rejects. Since both
checkpoints carry the same `codec.pth`, the fp16 codec decoder published at
`itzune/zortzi-tts-onnx` decodes either model's codes; `verify_parity.py` confirms it
against whichever checkpoint you point it at.

### `verify_parity.py`

```bash
python verify_parity.py --repo itzune/zortzi-tts --onnx-dir ./onnx --precision fp16 \
    --voice-codes voices/maider.json --text "Kaixo mundua." --steps 24
```

Compares four things against the PyTorch checkpoint, and all four must hold before a mirror
is published:

1. the `[1, 11, T]` prompt the engine builds against the one upstream's own processor and
   `_prepare_prompt` build — exact equality, no tolerance;
2. slow-AR logits and hidden states at prefill and at every decode step;
3. fast-AR logits for all nine predicted codebooks of every frame;
4. the codec decoder's waveform for a fixed code matrix.

Both stacks are driven in lockstep by greedy decoding so they follow the same path. Greedy
is used *only here* — both model cards state it loops forever in real synthesis.

The gate is greedy agreement, not the raw difference. For each disagreement the script also
reports the reference's own top-1 minus top-2 margin: a miss whose margin is smaller than
the logit difference is the two stacks splitting a tie that precision decided, while a miss
with a wide margin is a defect.

### `mint_voice.py`

An ArkTTS voice is not a speaker id — the model has no speaker table and always emits
`<|speaker:0|>`. A voice is the codec codes of a short clip plus that clip's transcription,
and the prompt embeds both.

```bash
python mint_voice.py --repo Audio8/Audio8-TTS-Preview-0.6b \
    --audio antton.wav --text "Inguru hura gerrillarien esku izan da denbora luzean." \
    --name antton --out voices/antton.json
```

The codec *encoder* runs here and never ships: phoonnx only needs the decoder at synthesis
time, because voices arrive pre-encoded. Minting is an offline step.

Keep the clip short and prosodically flat — upstream selected its own references by
pitch-range ratio precisely because an expressive reference bleeds its intonation into
every sentence the voice later speaks.

### `synthesize_samples.py` and `wer_gate.py`

Drive the adapter over a sentence list, write WAVs and per-voice CPU RTF, then transcribe
the result with a CPU `onnx-asr` model and score it. Sentences may carry a `lang|` prefix so
one run can cover several languages; `zh`, `yue` and `ja` are scored by character.

The WER gate is an intelligibility check, not an ASR benchmark. It catches what logit parity
cannot see: a wrong prompt layout, a mis-scattered cache, or a sampler drifting into a
repetition loop.

## Known limitations

* **INT4 is not shipped.** The official `*_int4.onnx` graphs fail parity badly — see the
  table in the pull request. They are left out rather than published.
* **No codec encoder graph.** Zero-shot cloning from a user's own clip therefore needs
  `mint_voice.py` offline. The encoder's strided causal convolutions do need the dynamic
  right-padding this script removes for the decode path, so exporting it is real work rather
  than a flag.
* **The slow AR carries the tied embedding twice**, about 270 MB of half-precision weights,
  because the tracer materialises the output projection separately from the embedding table.
  `deduplicate_initializers` collapses byte-identical initializers and does not catch this
  one.
