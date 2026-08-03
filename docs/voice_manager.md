# Voice Manager

This page is for developers using `phoonnx` from Python to discover, download, and load TTS voices. It documents the `TTSModelManager` and `TTSModelInfo` API in `phoonnx/model_manager.py`.

`TTSModelManager` handles discovery, caching, downloading, and loading of TTS voice models. Its catalog is assembled from bundled voice-index files that ship inside the package, so no network access is needed just to enumerate what is available.

## Where voices come from

The catalog is not fetched from a handful of live endpoints. `merge_default_voices()` loads a set of bundled `voice_index/*.json` files packaged with `phoonnx`, one per source. The sources are:

`OVOS`, `MMS`, `proxectonos`, `piper`, `phonikud`, `neurlang`, `mimic3`,
`transformers_community`, `piper_community`, `optispeech`, `glowtts`, `mixertts`,
`fastpitch`, `coqui_community`, `vits2`, `styletts2`, `f5tts`, `coqui_vits`, `BSC`,
`shami`, `chatterbox`, `supertonic`, `neutts`, `pockettts`, `sparktts`, `qwen3tts`,
`outetts`.

Each JSON entry becomes a `TTSModelInfo`. Model/config/vocoder files are only fetched from their URLs when a voice is actually downloaded or loaded.

## Basic Usage

```python
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.load()                  # load the on-disk cache into memory
manager.merge_default_voices()  # merge the bundled voice indexes in
```

`TTSModelManager(cache_path=...)` overrides the cache file location; by default the
cache is a `JsonStorageXDG` under `phoonnx/voices` in the XDG cache directory.

### Listing Available Voices

```python
# All voices currently in memory
for voice in manager.all_voices:
    print(voice.voice_id, voice.lang)

# Voices for a specific language (fuzzy language match)
pt_voices = manager.get_lang_voices("pt-PT")

# All language codes present in the loaded catalog
print(manager.supported_langs)
```

### Listing without downloading

`get_available_voice_ids_by_source()` reads the bundled index JSON files directly
(plain `json.load`, no `TTSModelInfo` construction, no network) and returns a
`{source: [voice_id, ...]}` mapping — a quick "what's available" listing before
committing to a download:

```python
by_source = manager.get_available_voice_ids_by_source()
for source, ids in by_source.items():
    print(source, len(ids))
```

### Getting a Specific Voice

```python
voice_info = manager.voices["OpenVoiceOS/phoonnx_eu-ES_dii_espeak"]
```

### Downloading a single voice

`download_voice_by_id(voice_id)` fetches one voice's model on demand. It looks the ID
up in the in-memory registry first, then falls back to the bundled indexes, so a voice
can be fetched without loading the whole catalog. Returns `True` if the voice was found:

```python
manager.download_voice_by_id("OpenVoiceOS/pipertts_es-ES_dii")
```

### Loading a Voice for Inference

```python
voice_info = manager.voices["OpenVoiceOS/phoonnx_ar_miro_espeak_V2"]
tts_voice = voice_info.load()   # downloads the model if not cached, returns TTSVoice

# optional: pin ONNX Runtime execution providers
tts_voice = voice_info.load(providers=["ROCMExecutionProvider", "CPUExecutionProvider"])
```

## Updating the Cache

Merging the bundled indexes and persisting the result to the on-disk cache is done with
`merge_default_voices(store=True)` — this is exactly what the `phoonnx-voices
update-cache` command calls:

```python
manager = TTSModelManager()
manager.clear()                         # optional: wipe the old cache first
manager.merge_default_voices(store=True) # merge bundled indexes and persist
```

`clear()` empties both the on-disk cache and the in-memory registry. `load()` reloads
the on-disk cache; `save()` writes the current in-memory voices back out.

## Cache Location

Downloaded files for a voice are cached per voice under the XDG cache directory:

```
~/.cache/phoonnx/voices/<voice_id>/
    model.onnx
    model.json             (if the voice has a config_url)
    tokens.txt             (Mimic3/Sherpa style, if applicable)
    vocab.json             (Transformers style, if applicable)
    tokenizer_config.json  (if applicable)
    vocoder.onnx           (two-stage engines, if applicable)
```

`TTSModelInfo.voice_path` returns this directory
(`~/.cache/phoonnx/voices/<voice_id>`). The voice catalog cache itself is a separate
`JsonStorageXDG` under `phoonnx/voices`.

## TTSModelInfo

Each voice in the catalog is a `TTSModelInfo` dataclass. It carries the URLs and
metadata for one voice and lazily resolves its `VoiceConfig` on first access.

```python
from phoonnx.model_manager import TTSModelInfo

info = TTSModelInfo(
    voice_id="my-custom-voice",
    lang="en-US",
    model_url="https://example.com/model.onnx",
    config_url="https://example.com/model.json",
    phoneme_type="espeak",       # optional override
    alphabet="ipa",              # optional override
    engine="piper",              # optional override
)

# Access the (lazily loaded) VoiceConfig
print(info.config.sample_rate)

# Fetch just the primary ONNX graph, everything needed to run offline,
# or load a full TTSVoice
info.download_model()
info.download_all()
voice = info.load(providers=None)
```

### Fields

`TTSModelInfo` has around two dozen fields; the important ones are below. Many are only
relevant to specific engine families (see [engines.md](engines.md), [cloning.md](cloning.md),
[vocoders.md](vocoders.md)).

| Field | Type | Description |
|-------|------|-------------|
| `voice_id` | `str` | Unique identifier |
| `lang` | `str` | BCP-47 language tag (e.g. `en-US`); may be absent/wrong in the source config |
| `model_url` | `str` | URL to the primary `.onnx` file |
| `config_url` | `str` \| `None` | URL to the JSON config (some voices ship only `tokens.txt`) |
| `vocab_url` | `str` \| `None` | URL to `vocab.json` (Transformers style) |
| `tokens_url` | `str` \| `None` | URL to `tokens.txt` (Mimic3/Sherpa style) |
| `tokenizer_config_url` | `str` \| `None` | Transformers tokenizer config; also the Chatterbox BPE `tokenizer.json` |
| `phoneme_type` | `PhonemeType` \| `None` | Override for phonemizer type |
| `alphabet` | `Alphabet` \| `None` | Override for phoneme alphabet |
| `engine` | `Engine` \| `None` | TTS engine the voice was trained with |
| `vocoder_url` | `str` \| `None` | Separate vocoder ONNX (two-stage engines) |
| `vocoder_config_url` | `str` \| `None` | Vocoder `vocoder.json` parameters |
| `vocoder_type` | `str` \| `None` | Vocoder implementation (`vocos`, `wavenext`, `hifigan`, `melgan`, `raw`, `griffinlim`) |
| `style_url` | `str` \| `None` | Per-voice StyleTTS2/Kokoro style embedding |
| `speaker_encoder_url` / `speaker_encoder_type` | `str` \| `None` | Cloning speaker-encoder ONNX (reference audio → d-vector) |
| `aux_model_urls` | `dict` \| `None` | Extra ONNX graphs/files for multi-graph engines (F5-TTS; SuperTonic's `duration_predictor`/`text_encoder`/`vocoder` graphs plus its `tts.json` config, `unicode_indexer.json` and per-speaker `style.json`; NeuTTS's NeuCodec decoder graph plus its `tokenizer.json` and `voices.json`), keyed by engine-param name |
| `engine_options` | `dict` \| `None` | Plain per-voice engine settings that are not downloadable files, merged into `engine_params` as-is (e.g. NeuTTS's `voice` preset name) |
| `display_name` | `str` \| `None` | Friendly name for UIs/CLIs; may contain `{engine}`/`{phoneme_type}` placeholders |
| `vocab_override` | `dict` \| `None` | Custom token-to-ID mapping |

Key methods and properties: `config` (lazy `VoiceConfig`), `voice_path`,
`download_model()`, `download_all()` (model + config + tokenizer + vocoder/style/
speaker-encoder/aux graphs), `download_vocoder()`, and `load(providers=None)`.
