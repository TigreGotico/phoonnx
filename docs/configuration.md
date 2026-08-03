# Configuration Reference

This page is the exhaustive reference for advanced users: every `VoiceConfig` and
`SynthesisConfig` field, the `model.json` schema, engine detection, and execution providers.
For a task-oriented walkthrough see [Usage](usage.md) instead.

## VoiceConfig

`VoiceConfig` holds all model-level configuration parsed from the voice's `model.json`.

```python
from phoonnx.config import VoiceConfig
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_symbols` | `int` | — | Number of phoneme/token IDs in the vocabulary |
| `num_speakers` | `int` | — | Number of speakers (1 for single-speaker) |
| `num_langs` | `int` | — | Number of languages (1 for monolingual) |
| `sample_rate` | `int` | `16000` | Output audio sample rate in Hz |
| `lang_code` | `str` \| `None` | `"und"` | BCP-47 language code (e.g. `"en-US"`) |
| `phoneme_type` | `PhonemeType` | — | Phonemizer backend |
| `alphabet` | `Alphabet` | — | Phoneme representation |
| `engine` | `Engine` | `PHOONNX` | Training framework the model was built with |
| `noise_scale` | `float` | `0.667` | Default generator noise scale |
| `length_scale` | `float` | `1.0` | Default phoneme length scale |
| `noise_w_scale` | `float` | `0.8` | Default phoneme width noise scale |
| `hop_length` | `int` | `256` | Model frames → audio samples factor for [phoneme alignment](alignment.md) |
| `add_diacritics` | `bool` | `False` | Auto-add diacritics (Arabic/Hebrew) |
| `diacritizer_model` | `str` | `rawi-ensemble` | Diacritizer model name (Arabic `text2tashkeel`); round-tripped through the config's `inference` block |
| `phonemizer_model` | `str` \| `None` | `None` | Model path/id or variant selector for phonemizers that take one (ByT5, AhoTTS, Cotovia, arbtok) |
| `speaker_id_map` | `dict` | `{}` | Maps speaker names to integer IDs |
| `lang_id_map` | `dict` | `{}` | Maps language codes to integer IDs (multi-lang voices) |
| `lang_tokens` | `dict` | `{}` | Optional BCP-47 → model language-token override (dialect models) |
| `engine_params` | `dict` | `{}` | Adapter-specific parameters (vocoder/speaker-encoder/style paths, etc.) |
| `tokenizer` | `TTSTokenizer` | — | Tokenizer instance |
| `blank_at_start` | `bool` | `True` | Insert a blank token before the sequence |
| `blank_at_end` | `bool` | `True` | Insert a blank token after the sequence |
| `pad_token` | `str` \| `None` | `<pad>` default | Padding token |
| `blank_token` | `str` \| `None` | pad default | Blank token inserted between tokens |
| `bos_token` | `str` \| `None` | `<bos>` default | Beginning-of-sequence token |
| `eos_token` | `str` \| `None` | `<eos>` default | End-of-sequence token |
| `word_sep_token` | `str` \| `None` | word-blank default | Token inserted between words |
| `blank_between` | `BlankBetween` | `TOKENS_AND_WORDS` | Where blank tokens are inserted |

### Loading from a Config File

`VoiceConfig` is not usually constructed directly. It is loaded automatically from a `model.json` file via `TTSVoice.load()`. To load manually:

```python
config = VoiceConfig.from_dict(my_config_dict)
```

### Engine Auto-Detection

`VoiceConfig.from_dict()` automatically detects the training engine from the config structure:

- **Chatterbox** — an explicit `engine: "chatterbox"` (needs a BPE `tokenizer.json`)
- **NeuTTS** — an explicit `engine: "neutts"` (the adapter loads its own BPE `tokenizer.json` and `voices.json` from `engine_params`)
- **Pocket TTS** — an explicit `engine: "pockettts"` (multi-graph flow-matching codec LM with explicit stream state; raw-text, no phonemizer)
- **phoonnx** — presence of `phoonnx_version`
- **Piper** — presence of `piper_version`, or a list-valued `phoneme_id_map` + `phoneme_type: "espeak"|"text"` (an explicitly declared non-piper engine wins over shape-sniffing)
- **Mimic3** — presence of `phonemizer` + `phonemes` dict (requires an external `phonemes.txt`)
- **Coqui VITS** — presence of a `characters` dict with a recognized `characters_class`
- **Transformers** — driven by an external `vocab` argument passed to `from_dict()` (the voice manager supplies it from `vocab.json`), not by the ONNX shape
- **Canonical / fallback** — a config that ships its own `phoneme_id_map` and declares its `engine`/`phoneme_type`/`alphabet` is honored as-is

A `config.json` that explicitly declares its `engine` is trusted over shape-sniffing.

## SynthesisConfig

`SynthesisConfig` controls synthesis behavior per request. All fields are optional and fall back to model defaults when `None`.

```python
from phoonnx.config import SynthesisConfig

syn_config = SynthesisConfig(
    speaker_id=None,
    lang_id=None,
    length_scale=None,
    noise_scale=None,
    noise_w_scale=None,
    normalize_audio=True,
    volume=1.0,
    enable_phonetic_spellings=True,
    add_diacritics=True,
)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `speaker_id` | `int` \| `None` | `None` | Speaker index for multi-speaker models |
| `lang_id` | `int` \| `None` | `None` | Language index for multi-language models |
| `length_scale` | `float` \| `None` | `None` | Speed control: < 1 faster, > 1 slower |
| `noise_scale` | `float` \| `None` | `None` | Generator noise: affects naturalness |
| `noise_w_scale` | `float` \| `None` | `None` | Phoneme duration noise |
| `normalize_audio` | `bool` | `True` | Scale audio to full amplitude range |
| `volume` | `float` | `1.0` | Volume multiplier |
| `enable_phonetic_spellings` | `bool` | `True` | Apply word-level pronunciation overrides |
| `add_diacritics` | `bool` \| `None` | `None` | Add vowel diacritics before phonemization; `None` defers to the voice config |
| `diacritizer_model` | `str` \| `None` | `None` | Override the diacritizer model for this call; `None` inherits the voice config's `diacritizer_model` |
| `speaker_reference` | `str` \| `(audio, sr)` \| `None` | `None` | Reference clip for zero-shot [voice cloning](cloning.md) |
| `speaker_reference_text` | `str` \| `None` | `None` | Reference transcription, required by in-context cloning engines (ZipVoice) |
| `speaker_reference_lang` | `str` \| `None` | `None` | Language of the transcription, for cross-lingual cloning (defaults to the voice's `lang_code`) |
| `exaggeration` | `float` \| `None` | `None` | Expressiveness/intensity for engines that support it (Chatterbox, ~0.5) |
| `temperature` | `float` \| `None` | `None` | Sampling temperature for autoregressive engines (Chatterbox, ~0.8; `0` = greedy) |
| `top_p` | `float` \| `None` | `None` | Nucleus sampling cutoff for autoregressive engines (Chatterbox, ~0.95) |
| `extra_params` | `dict` | `{}` | Engine-specific per-call parameters (e.g. `d_factor`, `p_factor`, `e_factor`) |

## model.json Format

phoonnx supports multiple JSON config schemas. The native phoonnx format looks like:

```json
{
  "phoneme_type": "espeak",
  "alphabet": "ipa",
  "num_symbols": 256,
  "num_speakers": 1,
  "num_langs": 1,
  "phoneme_id_map": { "a": [1], "b": [2], ... },
  "audio": { "sample_rate": 22050 },
  "inference": {
    "noise_scale": 0.667,
    "length_scale": 1.0,
    "noise_w": 0.8,
    "add_diacritics": false,
    "diacritizer_model": "rawi-ensemble"
  },
  "blank": "_",
  "pad": "<pad>",
  "bos": "<bos>",
  "eos": "<eos>",
  "blank_at_start": true,
  "blank_at_end": true,
  "add_blank": true,
  "speaker_id_map": {},
  "lang_code": "en-US"
}
```

Piper, Mimic3, and Coqui config formats are also parsed automatically.

## Engine Enum

`Engine` (in `phoonnx.config`) records which framework a voice was built with. It has 17
members. `piper`, `mimic3` and `coqui` all run through the single VITS adapter; the rest map
to a dedicated adapter. The adapter registry itself is described in [Engines](engines.md).

| `Engine` value | Kind | Per-engine guide |
|---|---|---|
| `phoonnx` | Native VITS | [Engines](engines.md) |
| `piper` | VITS (shared adapter) | [Engines](engines.md) |
| `mimic3` | VITS (shared adapter) | [Engines](engines.md) |
| `coqui` | VITS (shared adapter) | [Engines](engines.md) |
| `transformers` | HuggingFace VITS/MMS | [Engines](engines.md) |
| `matcha` | Two-stage flow-matching mel | [Matcha](training/engines/matcha.md) |
| `optispeech` | FastSpeech2-style + GAN vocoder | [OptiSpeech](training/engines/optispeech.md) |
| `glowtts` | Flow-based mel + vocoder | [GlowTTS](training/engines/glowtts.md) |
| `mixertts` | MLP-Mixer mel + vocoder | [MixerTTS](training/engines/mixertts.md) |
| `fastpitch` | FastSpeech2-style mel + vocoder | [FastPitch](training/engines/fastpitch.md) |
| `styletts2` | StyleTTS2 / Kokoro end-to-end | [Training](training/training.md) |
| `yourtts` | Multilingual VITS + d-vector cloning | [Cloning](cloning.md) |
| `zipvoice` | Flow-matching in-context cloning | [ZipVoice](training/engines/zipvoice.md) |
| `shami` | Levantine Arabic / English (HamsVITS) | [Shami](training/engines/shami.md) |
| `f5tts` | DiT flow-matching (F5-TTS / Habibi) | [F5-TTS](training/engines/f5tts.md) |
| `chatterbox` | Autoregressive codec-LM cloning | [Chatterbox](training/engines/chatterbox.md) |
| `supertonic` | Multi-graph flow-matching, raw-text (no phonemizer) | [Engines](engines.md) |
| `neutts` | Autoregressive NeuCodec LM, preset cloning @24 kHz | [Cloning](cloning.md) |
| `pockettts` | Flow-matching codec LM with explicit stream state, raw-text (no phonemizer) | [Pocket TTS](pockettts.md) |
| `sparktts` | Autoregressive codec-LM (Qwen2 + BiCodec), en/zh | [Spark-TTS](training/engines/sparktts.md) |

## Execution Providers

Every ONNX session phoonnx creates — the voice model and the auxiliary graphs an
engine loads (vocoders, speaker encoders, text encoders, diacritizers) — runs on a
resolved, ordered list of ONNX Runtime execution providers.

Pass the list explicitly:

```python
from phoonnx.voice import TTSVoice

voice = TTSVoice.load(
    "model.onnx", "config.json",
    providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
)
```

Or set it once for the process:

```bash
export PHOONNX_ONNX_PROVIDERS="ROCMExecutionProvider,CPUExecutionProvider"
export PHOONNX_ONNX_PROVIDERS="auto"   # the default: pick the best available
```

With neither, the best provider the installed runtime offers is auto-detected, in
this preference order:

`CUDAExecutionProvider` (NVIDIA) → `ROCMExecutionProvider` (AMD) →
`MIGraphXExecutionProvider` (AMD) → `DmlExecutionProvider` (DirectML) →
`CoreMLExecutionProvider` (Apple) → `OpenVINOExecutionProvider` (Intel) →
`CPUExecutionProvider`

A requested provider that the installed runtime does not offer is skipped with a
warning, and `CPUExecutionProvider` is always appended, so synthesis keeps working
on any machine.

`use_cuda=True` is a deprecated alias for `providers=["CUDAExecutionProvider"]`.

### Runtime packages

Which providers exist depends on the installed ONNX Runtime build, not on phoonnx.
The default `onnxruntime` wheel is CPU-only (plus a few platform providers); GPU
providers need a matching build, and only one of them can be installed at a time:

| Hardware | Package | Provider |
|---|---|---|
| NVIDIA | `onnxruntime-gpu` | `CUDAExecutionProvider`, `TensorrtExecutionProvider` |
| AMD (ROCm) | `onnxruntime-rocm` | `ROCMExecutionProvider`, `MIGraphXExecutionProvider` |
| Windows (any DX12 GPU) | `onnxruntime-directml` | `DmlExecutionProvider` |
| Intel | `onnxruntime-openvino` | `OpenVINOExecutionProvider` |
| Apple | `onnxruntime` | `CoreMLExecutionProvider` |

Check what a machine actually offers:

```python
import onnxruntime
print(onnxruntime.get_available_providers())
```
