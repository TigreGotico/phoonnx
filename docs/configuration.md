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
