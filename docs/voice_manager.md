# Voice Manager

The `TTSModelManager` handles discovery, caching, downloading, and loading of TTS voice models from multiple upstream sources.

## Supported Upstream Sources

| Source | Description |
|--------|-------------|
| **OpenVoiceOS (OVOS)** | Community voices hosted on HuggingFace |
| **Proxectonos** | Galician voices |
| **Phonikud** | Hebrew voices |
| **Piper** | Voices from the rhasspy/piper project |
| **Mimic3** | Voices from the Mycroft Mimic3 project |

## Basic Usage

```python
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.load()              # load local cache
manager.merge_default_voices()  # also load bundled default voices
```

### Listing Available Voices

```python
# All voices
for voice in manager.all_voices:
    print(voice.voice_id, voice.lang)

# Voices for a specific language
pt_voices = manager.get_lang_voices("pt-PT")

# All supported language codes
print(manager.supported_langs)
```

### Getting a Specific Voice

```python
voice_info = manager.voices["OpenVoiceOS/phoonnx_eu-ES_dii_espeak"]
```

### Loading a Voice for Inference

```python
voice_info = manager.voices["OpenVoiceOS/phoonnx_ar_miro_espeak_V2"]
tts_voice = voice_info.load()   # downloads model if not cached, returns TTSVoice
```

## Updating the Cache

Fetch the latest voice lists from all upstream sources:

```python
manager.clear()                  # clear old cache
manager.get_piper_voice_list()
manager.get_mimic3_voice_list()
manager.get_ovos_voice_list()
manager.get_proxectonos_voice_list()
manager.get_phonikud_voice_list()
manager.save()
```

## Cache Location

By default, model files are cached in the XDG cache directory:

```
~/.cache/phoonnx/voices/<voice_id>/
    model.onnx
    model.json
    tokens.txt        (if applicable)
    vocab.json        (if applicable)
    tokenizer_config.json  (if applicable)
```

You can specify a custom cache path:

```python
manager = TTSModelManager(cache_path="/my/custom/cache/voices.json")
```

## TTSModelInfo

Each registered voice is represented by a `TTSModelInfo` dataclass:

```python
from phoonnx.model_manager import TTSModelInfo

info = TTSModelInfo(
    voice_id="my-custom-voice",
    lang="en-US",
    model_url="https://example.com/model.onnx",
    config_url="https://example.com/model.json",
    phoneme_type="espeak",       # optional override
    alphabet="ipa",              # optional override
    engine="piper",              # optional: phoonnx, piper, mimic3, coqui, transformers
)

# Access config (lazy-loaded from URL)
print(info.config.sample_rate)

# Load the voice
voice = info.load()
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `voice_id` | `str` | Unique identifier |
| `lang` | `str` | BCP-47 language tag (e.g. `en-US`) |
| `model_url` | `str` | URL to the `.onnx` file |
| `config_url` | `str` \| `None` | URL to the JSON config |
| `tokens_url` | `str` \| `None` | URL to `tokens.txt` (Mimic3/Sherpa style) |
| `vocab_url` | `str` \| `None` | URL to `vocab.json` (Transformers style) |
| `phoneme_type` | `PhonemeType` \| `None` | Override for phonemizer type |
| `alphabet` | `Alphabet` \| `None` | Override for phoneme alphabet |
| `engine` | `Engine` \| `None` | TTS engine the voice was trained with |
| `vocab_override` | `dict` \| `None` | Custom token-to-ID mapping |
