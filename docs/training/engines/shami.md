# Shami Engine

This page is for integrators running Levantine-Arabic / English code-switched
voices in phoonnx. After reading it you can export a ShamiVITS / HamsVITS
checkpoint to ONNX, load it, and understand how per-phoneme language IDs flow
through the pipeline.

> Related: [adapter architecture](../../engines.md) ·
> [configuration](../../configuration.md) ·
> [training reference](../training.md)

## What it is

The **Shami** engine supports
[ShamiVITS](https://huggingface.co/Tushe/shami-tts) /
[HamsVITS](https://github.com/Al-aminI/hams-levantine-tts), a VITS variant
trained for **Levantine Arabic** with code-switched **English**. It is
registered at high detect priority (checked early, ahead of the generic VITS
adapters) because its ONNX contract differs from standard VITS.

## When to pick it

Choose Shami for Levantine Arabic text that mixes in English words — the engine
tags each phoneme with its source language so the model pronounces the
Arabic and English segments in their own phonology.

## Extras needed

Inference needs no engine-specific extra. The text front end optionally diacritizes Arabic
through a pluggable backend and uses an espeak-based English G2P; both come with the vendored
`scriptconv.phonemizers._thirdparty.shami` package.

## What makes it different from VITS

Standard VITS adapters in phoonnx consume:

- `phoneme_ids` / `phoneme_lengths` / `scales`
- optional `speaker_id` / `language_id`

ShamiVITS ONNX exports **do not have a `scales` input** and **do not have a
single utterance-level `language_id`**. Instead they expect a parallel
**per-phoneme language stream**:

| Input | Shape | Meaning |
|---|---|---|
| `phoneme_ids` | `[1, T]` | tokenized IPA phonemes |
| `phoneme_lengths` | `[1]` | `T` |
| `language_ids` | `[1, T]` | `0=PAD`, `1=AR`, `2=EN`, `3=NEUTRAL` |
| `speaker_id` (optional) | `[1]` | speaker identifier |

The output is a single float32 `waveform` of shape `[1, samples]` at 24 kHz.

## Text front-end and the language-ID dispatch

`ShamiPhonemizer` (`phoonnx/phonemizers/shami.py`) wraps the vendored
`scriptconv.phonemizers._thirdparty.shami` package (ported from `hams_tts.text`). It performs:

1. Arabic text normalization and optional diacritization (pluggable `diacritizer_backend`: `auto` tries `camel` then `catt`; `passthrough` skips it).
2. English G2P using the same espeak-based approach as the upstream model.
3. Code-switch detection, tagging Arabic and English segments with `AR` / `EN`
   language IDs.

Unlike phoneme engines that emit only a token stream, `ShamiPhonemizer` exposes
a **`phonemize_with_language_ids`** method. `TTSVoice._phonemize`
(`phoonnx/voice.py`, ~lines 395–404) checks for this hook: when the phonemizer
has it, phoonnx builds the phoneme stream and the per-phoneme `language_ids`
stream **together** (from the same call), so the two can never fall out of
alignment. `ShamiAdapter` then forwards both to the ONNX graph. Engines without
the hook take the ordinary `encode_text` path.

Because the front-end already emits BOS/EOS symbols, the phoonnx tokenizer is
configured with `use_eos_bos=False` and `add_blank_char=False`.

## Obtaining / exporting

Use the conversion script to export the upstream PyTorch checkpoint:

```bash
python scripts/conversion/shami_tts/export.py \
    --checkpoint-dir ~/.cache/huggingface/hub/models--Tushe--shami-tts/snapshots/<sha> \
    --output-dir ./shami_phoonnx
```

This writes `model.onnx` and a phoonnx-compatible `config.json` using the
vendored symbol vocabulary.

## Synthesis example

```python
from phoonnx.voice import TTSVoice

voice = TTSVoice.load("model.onnx", "config.json")
audio = voice.synthesize("مَرحَبا، this is a test.")
```

`TTSVoice` routes the model to `ShamiAdapter` and forwards the per-phoneme
`language_ids` produced by `ShamiPhonemizer`.

## Gotchas / aliases

- **Detect aliases:** the config `engine` field may be `shami` or `hams`.
- **No runtime scales:** noise scale and speaking rate are baked into the
  exported graph. The adapter exposes them as read-only labels for UIs; they
  cannot be changed at inference time.
- **Language IDs are per-phoneme**, not per-utterance — this is what enables
  in-sentence AR/EN code-switching.

## References

- [Tushe/shami-tts](https://huggingface.co/Tushe/shami-tts)
- [Al-aminI/hams-levantine-tts](https://github.com/Al-aminI/hams-levantine-tts)
