# ArkTTS Engine (Zortzi + Audio8)

This page is for integrators who want ArkTTS voices in phoonnx. After reading it you can
pick a checkpoint and a voice, you know what each of the three ONNX graphs does, and you
know what one second of audio costs.

> Related: [adapter architecture](../../engines.md) ·
> [configuration](../../configuration.md) ·
> [Qwen3-TTS — the sibling two-stage codec-LM engine](qwen3tts.md)

## What it is

**ArkTTS** predicts the tokens of a 21.5 Hz neural codec and turns them into a waveform at
44.1 kHz — the highest sample rate of any engine in phoonnx. Like Qwen3-TTS it has two
stacked autoregressive models, not one:

* the **slow AR** — 24 decoder layers that emit one token per audio frame. That token is
  codebook 0, and the slow AR's hidden state conditions the second model;
* the **fast AR** — 4 decoder layers that read the slow hidden state and write codebooks
  1 to 9 of the *same* frame, one codebook per step.

One 46 ms frame therefore costs one slow step and nine fast steps. The ten codebooks of a
frame are summed into a single embedding and fed back to the slow AR at the next position.

The prompt is a `[1, 11, T]` matrix, not a flat token sequence. Row 0 carries the token
stream — a system block, then the reference clip's codes shifted into the model's semantic
range, then the text. Rows 1 to 10 carry the reference clip's ten codebooks, aligned under
those semantic positions and zero everywhere else.

Two checkpoints share this architecture, and phoonnx drives both through one adapter
because they ship byte-identical model code, tokenizer and codec weights:

| Checkpoint | Mirror | Languages |
| --- | --- | --- |
| [`Audio8/Audio8-TTS-Preview-0.6b`](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b) | `OpenVoiceOS/phoonnx-audio8-tts` | Cantonese, Chinese, Dutch, English, French, German, Italian, Japanese, Korean, Polish, Spanish |
| [`itzune/zortzi-tts`](https://huggingface.co/itzune/zortzi-tts) | `OpenVoiceOS/phoonnx-zortzi-tts` | Basque |

Both are Apache-2.0. Zortzi is a Basque fine-tune of Audio8 and speaks only Basque.

## When to pick it

Pick ArkTTS when you want **Basque**, where it is currently the only codec-LM voice
phoonnx offers, or when you want 44.1 kHz output and can afford the cost.

Do not pick it when you need real-time synthesis. Two autoregressive stages at ten steps
per frame are expensive: measured on twelve cores, one second of audio takes about eight
seconds of CPU time. Both upstream model cards say the same thing and point at a GPU for
real-time use. For a cheaper multilingual voice use [Qwen3-TTS](qwen3tts.md); for cloning
on CPU use [Spark-TTS](sparktts.md).

## Voices

A voice here is **not** a speaker id. ArkTTS has no speaker table and always emits the
`<|speaker:0|>` token; the speaker is carried by the codec codes of a short reference clip,
which the prompt embeds directly. The mirrors therefore ship the codes rather than the
audio, in one small JSON per voice that also holds the clip's transcription.

| Voice | Timbre | Reference clip |
| --- | --- | --- |
| `maider` | female | HiTZ-Aholab Basque TTS, 2.2 s |
| `antton` | male | HiTZ-Aholab Basque TTS, 3.3 s |

Audio8 bundles no reference voices of its own — upstream ships it as a cloning-only model —
so its mirror carries the same two clips. They condition timbre, not language: both voices
are listed for every language Audio8 was trained on.

Voice ids follow `arktts/<checkpoint>-<voice>/<language>`, for example
`arktts/zortzi-maider/eu` or `arktts/audio8-antton/ja`.

To add a voice, encode a clip offline with `scripts/conversion/arktts/mint_voice.py`. Keep
it short and prosodically flat: upstream selected its own references by pitch-range ratio
because an expressive reference bleeds its intonation into every later sentence.

## Sampling

**ArkTTS must sample.** Both model cards state that greedy decoding runs into repetition
loops that never reach the end-of-speech token, so this engine does not offer a greedy
mode. The defaults are the ones both cards pin: temperature `0.8`, top-p `0.95`, top-k `50`.

The sampler reproduces upstream's own order, which is not the HuggingFace one: the semantic
mask runs first, then top-k and top-p together against the softmax of the **unscaled**
logits, and temperature divides last. Applying temperature first, as HuggingFace does,
changes which tokens survive the nucleus cut.

On top of that sits repetition-aware sampling. If a draw lands on a semantic token that is
already among the last ten emitted, it is discarded and redrawn under tighter settings
(top-p `0.9` at temperature `1.0`). That fallback is what stops the model looping, and it
is why greedy decoding cannot terminate — with sampling off, both draws are the same argmax.

Pass a `seed` in `engine_params` to make a call reproducible.

## Graphs

| File | Role |
| --- | --- |
| `slow_ar_fp16.onnx` | 24-layer backbone, KV-cached; this is the voice's `model_url` |
| `fast_ar_fp16.onnx` | 4-layer depth transformer over the ten codebooks |
| `codec_decoder_fp16.onnx` | `(1, 10, T)` codes to a 44.1 kHz waveform |
| `tokenizer.json` | the model's own Qwen2 subword BPE |
| `voices/<name>.json` | one voice's reference codes and transcription |

The KV cache is a fixed 2048-position window rather than a growing tensor: each graph writes
its new keys and values at `input_pos` and attends over the whole window. The slow AR's
logits are sliced to 4097 entries — the 4096 semantic logits followed by end-of-speech — so
an index below 4096 is already codebook 0's value.

Only half precision is published. The `int4` graphs upstream also publishes fail numeric
parity against the PyTorch checkpoint and are deliberately not mirrored.

Text longer than about 200 characters is split on sentence boundaries and synthesized one
chunk per autoregressive pass, because the model's own frame budget runs out before a long
chunk does.

## Limitations

These are upstream's, and they are not worked around in phoonnx:

* **Numbers are not spoken correctly.** Both cards report that digits come out in a mix of
  languages even when expanded to words. Spell numbers out in the text.
* **No text normalization.** Text is tokenized verbatim, so expand acronyms yourself —
  upstream's own example is "TTS" written as "te te ese".
* **Zortzi speaks only Basque**, with these two voices and no others.
* **Cloning is offline.** phoonnx ships the codec decoder but not the encoder, so a new
  reference clip becomes a voice through `mint_voice.py` rather than at synthesis time.

## Licence and attribution

Both checkpoints are Apache-2.0. The reference clips that carry the voices come from the
[HiTZ-Aholab Basque TTS dataset](https://doi.org/10.5281/zenodo.17952596), which is
CC BY 4.0, so redistributing the voices or audio generated from them carries that
attribution. The mirrors state it, and it is repeated here because it travels with the
audio, not only with the files.
