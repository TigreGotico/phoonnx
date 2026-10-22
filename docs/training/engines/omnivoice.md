# OmniVoice

[OmniVoice](https://github.com/k2-fsa/OmniVoice) (k2-fsa) is a zero-shot text-to-speech
model for 600+ languages. Set `engine: omnivoice`.

Every other engine in phoonnx writes speech either one token at a time (Chatterbox,
Spark-TTS, OuteTTS) or by integrating a smooth path (ZipVoice, F5-TTS, Pocket TTS).
OmniVoice does neither. It is a **masked diffusion language model**: the audio it is about
to write starts as a grid of MASK tokens, and a fixed number of steps replace the most
confident MASK slots with real codec codes until none are left.

Three consequences follow, and they drive the whole adapter:

1. **Attention is bidirectional.** A slot in the middle must see the slots after it, so
   the backbone reads the whole sequence at once. There is no causal mask and no KV cache.
2. **Every step is a full forward pass.** Nothing is cached between steps, so cost scales
   with steps times sequence length rather than with tokens produced.
3. **Length is decided in advance.** The model never emits an end-of-speech token, so the
   number of frames is estimated from the text before decoding starts.

## Graphs

Published at
[`OpenVoiceOS/phoonnx-omnivoice`](https://huggingface.co/OpenVoiceOS/phoonnx-omnivoice).

| `engine_params` key | Graph | Contract |
|---|---|---|
| *(the voice's own model)* | `omnivoice_backbone.onnx` | `(input_ids[B,8,S] int64, audio_mask[B,S] bool)` → `logits[B,8,S,1025]` |
| `acoustic_encoder_path` | `acoustic_encoder.onnx` | reference wav @24 kHz `(1,1,T)` → `(1,256,T')` |
| `semantic_encoder_path` | `semantic_encoder.onnx` | reference wav @16 kHz `(1,T)` → `(1,768,T')` |
| `quantizer_encoder_path` | `quantizer_encoder.onnx` | acoustic + semantic → codes `(8,1,T')` |
| `decoder_path` | `higgs_decoder.onnx` | codes `(8,1,T')` → waveform @24 kHz |
| `bpe_tokenizer_path` | `tokenizer.json` | the model's own Qwen3 subword BPE |

The backbone is one graph: audio embeddings, the 28-layer Qwen3 core and the per-codebook
heads are fused, which removes the dtype negotiation a three-way split needs. The three
encoder graphs only open when a call actually clones.

The codec is the **Higgs Audio V2** tokenizer — a DAC acoustic branch at 24 kHz and a
HuBERT semantic branch at 16 kHz, quantized into 8 residual codebooks at 25 frames per
second.

## Prompt layout

```
<|denoise|>                                   (cloning only)
<|lang_start|>{language or None}<|lang_end|>
<|instruct_start|>{instruct or None}<|instruct_end|>
<|text_start|>{reference transcription + " " + text}<|text_end|>
{reference codec codes}                       (cloning only)
{MASK × target_len}                           ← what gets decoded
```

All eight codebook rows carry the same text ids. `audio_mask` marks where the rows stop
being copies of the text and start being codec streams; that is what tells the embedding
layer which table to look each position up in.

The reference transcription is **joined to the target text as one string**, not
concatenated as a second token sequence. Cloning is noticeably worse without it.

## Cloning

OmniVoice is an **in-context** cloner, like ZipVoice: it needs the reference clip *and*
its transcription. See [cloning.md](../../cloning.md).

```python
from phoonnx.model_manager import TTSModelManager

voice = TTSModelManager().load_voice("omnivoice/en")
audio = voice.synthesize(
    "Machine learning models can now speak in hundreds of languages.",
    speaker_reference="reference.wav",
    speaker_reference_text="Transcription of the reference clip.",
)
```

A reference clip of 3–10 seconds works best. Longer clips make every step slower, because
their codes sit in the sequence the backbone re-reads 32 times.

Without a reference, the model picks a voice itself. `instruct` describes one in words
("a calm older man") — that is OmniVoice's voice-design mode.

## Parameters

| Parameter | Default | What it does |
|---|---|---|
| `num_step` | 32 | Unmasking steps. Fewer is faster and rougher. |
| `guidance_scale` | 2.0 | Classifier-free guidance. 0 disables it and halves the cost. |
| `t_shift` | 0.1 | Warps the time grid. Smaller leaves more slots for the late steps. |
| `layer_penalty_factor` | 5.0 | Biases unmasking towards the low codebooks, which carry more of the signal. |
| `position_temperature` | 5.0 | Randomness in *which* slot is filled next. 0 is greedy. |
| `class_temperature` | 0.0 | Randomness in *which code* is chosen. 0 is greedy. |
| `length_scale` | 1.0 | Scales the estimated duration. Above 1 is slower speech. |
| `lang` | voice's language | Fills the `<|lang_start|>` slot. Unknown tags fall back to language-agnostic. |
| `instruct` | none | Voice-design description. |
| `seed` | none | Makes the stochastic path reproducible. |

Setting both temperatures to 0 makes generation fully deterministic, which is what the
parity checks below use.

## Classifier-free guidance

Each step runs the backbone twice: once over the full prompt, and once over the target
span alone with no text, language or reference. Upstream batches both rows behind a
`[2B,1,S,S]` block mask. For a single item that block mask says exactly "each row attends
within its own real length", which is what two separate forwards of different lengths
compute — so the adapter runs two forwards and gets the same numbers.

## A warning about the community export

`onnx-community/OmniVoice-Onnx` routes the backbone through
`com.microsoft::GroupQueryAttention`, which is **unconditionally causal**. Measured
against upstream PyTorch on a fixed input, its `llm_decoder` matches a *causal* run
(cos 0.99945) and not the bidirectional one the model needs (cos 0.954), leaving **18 %**
greedy-token agreement end to end. Do not point this adapter at it.

The re-export in `OpenVoiceOS/phoonnx-omnivoice` keeps the bidirectional mask and reaches
cos 1.0000000 and 100 % agreement, and 100 % codec-token agreement through the full
32-step sampler. The community **codec** graphs are exact and are mirrored unchanged.

## Cost

OmniVoice is not cheap on CPU. Measured on 12 CPU cores with the fp32 backbone, the
real-time factor is about **6–9** — that is, roughly seven seconds of compute per second
of audio. The paper's 0.025 figure is a GPU number. The cost is `num_step × 2` full
forwards over the whole prompt, so shorter reference clips and shorter text help
directly, and `guidance_scale: 0` halves it at some quality cost.
