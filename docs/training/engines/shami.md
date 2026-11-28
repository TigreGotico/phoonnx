# Shami Engine

<!-- agent-hint
covers: how to export, load and run Levantine-Arabic and English code-switched ShamiVITS voices
source: phoonnx/engines/shami.py, phoonnx/voice_index/shami.json, scripts/conversion/shami_tts/export.py, scriptconv.phonemizers.shami:ShamiPhonemizer
generated: no
verify: ! comm -23 <(grep -oP -- '(?<![\w-])--[a-z][a-z0-9-]*' docs/training/engines/shami.md | sort -u) <(.venv/bin/python scripts/conversion/shami_tts/export.py --help | grep -oP -- '(?<![\w-])--[a-z][a-z0-9-]*' | sort -u) | grep . && .venv/bin/python -W ignore -c "import json, re, sys; from phoonnx.engines.shami import ShamiAdapter; from scriptconv.phonemizers._thirdparty.shami.phoneme_inventory import Lang; p = open('docs/training/engines/shami.md').read(); ids = re.search(r'\x60(0=PAD[^\x60]*)\x60', p).group(1); want = ', '.join(f'{int(x)}={x.name}' for x in Lang); voices = re.findall(r'^voice = manager.voices\[.(.+?).\]', p, re.M); bad = ([] if ids == want else [ids, want]) + [v for v in voices if v not in json.load(open('phoonnx/voice_index/shami.json'))] + [a for a in ('shami', 'hams') if not ShamiAdapter.detect(config={'engine': a})]; print(bad); sys.exit(bool(bad) or not voices)"
see: ../training.md, ../../engines.md, ../../configuration.md
-->

This page shows how to load, run and export ShamiVITS voices, which speak Levantine Arabic mixed with English.

See also the [adapter architecture](../../engines.md), the [configuration reference](../../configuration.md) and the [training reference](../training.md).

## What it is

The Shami engine runs [ShamiVITS](https://huggingface.co/Tushe/shami-tts) and [HamsVITS](https://github.com/Al-aminI/hams-levantine-tts) models. These are VITS models trained on Levantine Arabic with English words mixed in. Use the engine for Levantine Arabic text that contains English words. The phonemizer tags each phoneme with its language, so the model says the Arabic and English parts each in their own phonology.

Inference needs no engine-specific extra. The text frontend comes with scriptconv, in the package `scriptconv.phonemizers._thirdparty.shami`.

## Run the index voice

The bundled index holds one Shami voice, `Tushe/shami-tts`, with the language tag `apc`:

```python
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.load()
manager.merge_default_voices()
voice = manager.voices["Tushe/shami-tts"].load()
for chunk in voice.synthesize("مَرحَبا، this is a test."):
    ...
```

`TTSVoice` gives the model to `ShamiAdapter`, together with the language id of each phoneme from `ShamiPhonemizer`.

## Export a checkpoint

The conversion script exports the upstream PyTorch checkpoint. The checkpoint folder holds `hams_vits.pt` and `hams_vits_config.json`:

```bash
python scripts/conversion/shami_tts/export.py \
    --checkpoint-dir ~/.cache/huggingface/hub/models--Tushe--shami-tts/snapshots/<sha> \
    --output-dir ./shami_phoonnx
```

This writes `model.onnx` and a phoonnx `config.json` with the symbol vocabulary of the frontend. Load the result as any voice:

```python
from phoonnx.voice import TTSVoice

voice = TTSVoice.load("shami_phoonnx/model.onnx", "shami_phoonnx/config.json")
```

## How it differs from VITS

The standard VITS adapters of phoonnx feed `phoneme_ids`, `phoneme_lengths` and `scales`, and an optional `speaker_id` or `language_id`. A ShamiVITS graph has no `scales` input and no single `language_id` for the utterance. It takes a language id for each phoneme:

| Input | Shape | Meaning |
|---|---|---|
| `phoneme_ids` | `[1, T]` | The tokenized IPA phonemes |
| `phoneme_lengths` | `[1]` | `T` |
| `language_ids` | `[1, T]` | `0=PAD, 1=AR, 2=EN, 3=NEUTRAL` |
| `speaker_id` (optional) | `[1]` | The speaker id |

The output is one float32 `waveform` of shape `[1, samples]` at 24 kHz.

The engine has detect priority 45. phoonnx checks the engines in order of priority, lowest first, so it checks Shami before the generic VITS adapters (48 to 51). A graph with the inputs `phoneme_ids` and `language_ids` is a Shami graph.

## Text frontend and language ids

`ShamiPhonemizer` (`scriptconv.phonemizers.shami`) wraps the frontend in `scriptconv.phonemizers._thirdparty.shami`, which is a port of `hams_tts.text`. It does three things:

1. It normalizes the Arabic text and can add diacritics. The `diacritizer_backend` option selects the diacritizer: `auto` tries `camel` and then `catt`, and `passthrough` adds none.
2. It converts the English words with an espeak-based G2P, as the upstream model does.
3. It finds the Arabic and English parts and tags them with the `AR` and `EN` language ids.

Each segment that the phonemizer gives carries a `language_ids` stream, with one value per phoneme. `TTSVoice` passes the phonemes and the stream together to `ShamiAdapter`, so the two always align. When a request has no stream, the adapter feeds a zero for every phoneme, and the model speaks one language.

The frontend writes its own BOS and EOS symbols, so the config of an export sets `use_eos_bos` and `add_blank_char` to `false`.

## Notes

- The `engine` field of the config can be `shami` or `hams`.
- The graph holds the noise scale and the speaking rate. They cannot change at inference. The adapter gives fixed default values for them, so the generic `TTSVoice` code finds the keys it expects.
- The language ids are per phoneme, not per utterance. This lets one sentence switch between Arabic and English.

## References

- [Tushe/shami-tts](https://huggingface.co/Tushe/shami-tts)
- [Al-aminI/hams-levantine-tts](https://github.com/Al-aminI/hams-levantine-tts)

---

[← MOSS-TTS-Nano](mosstts.md) · [Home](../../index.md) · [OuteTTS →](outetts.md)
