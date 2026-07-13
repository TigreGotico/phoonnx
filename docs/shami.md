# Shami Engine

The **Shami** engine supports [ShamiVITS](https://huggingface.co/Tushe/shami-tts) / [HamsVITS](https://github.com/Al-aminI/hams-levantine-tts), a VITS variant trained for **Levantine Arabic** with code-switched **English**.

## What makes it different from VITS

Standard VITS adapters in phoonnx consume:

- `phoneme_ids` / `phoneme_lengths` / `scales`
- optional `speaker_id` / `language_id`

ShamiVITS ONNX exports **do not have a `scales` input** and **do not have a single utterance-level `language_id`**. Instead they expect a parallel **per-phoneme language stream**:

| Input | Shape | Meaning |
|---|---|---|
| `phoneme_ids` | `[1, T]` | tokenized IPA phonemes |
| `phoneme_lengths` | `[1]` | `T` |
| `language_ids` | `[1, T]` | `0=PAD`, `1=AR`, `2=EN`, `3=NEUTRAL` |
| `speaker_id` (optional) | `[1]` | speaker identifier |

The output is a single float32 `waveform` of shape `[1, samples]` at 24 kHz.

## Text front-end

`ShamiPhonemizer` (`phoonnx/phonemizers/shami.py`) wraps the vendored `phoonnx/thirdparty.shami` package ported from `hams_tts.text`. It performs:

1. Arabic text normalization and optional diacritization (via `libtashkeel`).
2. English G2P using the same espeak-based approach as the upstream model.
3. Code-switch detection, so Arabic and English segments are tagged with `AR` / `EN` language IDs.

Because the front-end already emits BOS/EOS symbols, the phoonnx tokenizer is configured with `use_eos_bos=False` and `add_blank_char=False`.

## ONNX export

Use the conversion script to export the upstream PyTorch checkpoint:

```bash
python scripts/conversion/shami_tts/export.py \
    --checkpoint-dir ~/.cache/huggingface/hub/models--Tushe--shami-tts/snapshots/<sha> \
    --output-dir ./shami_phoonnx
```

This writes `model.onnx` and a phoonnx-compatible `config.json` using the vendored symbol vocabulary.

## Loading a Shami voice

```python
from phoonnx.voice import TTSVoice

voice = TTSVoice.load("model.onnx", "config.json")
audio = voice.synthesize("مَرحَبا، this is a test.")
```

`TTSVoice` automatically routes the model to `ShamiAdapter` and forwards the per-phoneme `language_ids` produced by `ShamiPhonemizer`.

## Runtime parameters

Noise scale and speaking rate are baked into the exported graph; the adapter exposes them as read-only labels for UIs but they cannot be changed at inference time.

## References

- [Tushe/shami-tts](https://huggingface.co/Tushe/shami-tts)
- [Al-aminI/hams-levantine-tts](https://github.com/Al-aminI/hams-levantine-tts)
- [docs/engines.md](./engines.md) — the engine adapter framework
