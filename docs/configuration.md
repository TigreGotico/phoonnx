# Configuration Reference

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
| `add_diacritics` | `bool` | `False` | Auto-add diacritics (Arabic/Hebrew) |
| `phonemizer_model` | `str` \| `None` | `None` | Model path/id for neural phonemizers (e.g. ByT5) |
| `speaker_id_map` | `dict` | `{}` | Maps speaker names to integer IDs |
| `tokenizer` | `TTSTokenizer` | — | Tokenizer instance |

### Loading from a Config File

`VoiceConfig` is not usually constructed directly. It is loaded automatically from a `model.json` file via `TTSVoice.load()`. To load manually:

```python
config = VoiceConfig.from_dict(my_config_dict)
```

### Engine Auto-Detection

`VoiceConfig.from_dict()` automatically detects the training engine from the config structure:

- **Piper** — presence of `piper_version` or `phoneme_id_map` + `phoneme_type: "espeak"|"text"`
- **Mimic3** — presence of `phonemizer` + `phonemes` dict
- **Coqui VITS** — presence of `characters` dict
- **Transformers** — HuggingFace-style tokenizer config
- **phoonnx** — default fallback

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
| `add_diacritics` | `bool` | `True` | Add vowel diacritics before phonemization |
| `speaker_reference` | `str` \| `(audio, sr)` \| `None` | `None` | Reference clip for zero-shot [voice cloning](cloning.md) |
| `speaker_reference_text` | `str` \| `None` | `None` | Reference transcription, required by in-context cloning engines (ZipVoice) |
| `speaker_reference_lang` | `str` \| `None` | `None` | Language of the transcription, for cross-lingual cloning (defaults to the voice's `lang_code`) |

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
    "noise_w": 0.8
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

```python
from phoonnx.config import Engine

Engine.PHOONNX       # "phoonnx"
Engine.PIPER         # "piper"
Engine.MIMIC3        # "mimic3"
Engine.COQUI         # "coqui"
Engine.TRANSFORMERS  # "transformers"
```

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
