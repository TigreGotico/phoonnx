# Voice manager

<!-- agent-hint
covers: how to list, look up, download and load voices from Python with TTSModelManager and TTSModelInfo
source: phoonnx/model_manager.py:TTSModelManager, TTSModelInfo; phoonnx/voice_index/*.json
generated: no
verify: python -m pytest tests/test_model_manager.py; the examples below, run with the espeak extra
see: choose_voice.md, precision.md, offline.md, cli.md
-->

This page shows how to find, download and load voices from Python. It is for developers who use phoonnx as a library.

`TTSModelManager` holds the voice catalog. The catalog comes from voice index files inside the package, so listing the voices needs no network.

## Where voices come from

The catalog is not fetched from live endpoints. `merge_default_voices()` loads the bundled
`phoonnx/voice_index/*.json` files, one file per source. The package holds 39 sources:

`arktts`, `BSC`, `chatterbox`, `coqui_community`, `coqui_vits`, `f5tts`, `fastpitch`,
`ghananlp`, `glowtts`, `indic_parler`, `inflect`, `llasa`, `magpie`, `mimic3`, `mixertts`,
`MMS`, `mosstts`, `neurlang`, `neutts`, `omnivoice`, `optispeech`, `orpheus`, `outetts`,
`OVOS`, `parlertts`, `phonikud`, `piper`, `piper_community`, `pockettts`, `proxectonos`,
`qwen3tts`, `rapido`, `shami`, `sparktts`, `styletts2`, `supertonic`,
`transformers_community`, `vits2`, `vosk`.

`phoonnx-voices list-available` prints the voice ids of each source.

Each JSON entry becomes a `TTSModelInfo`. Model/config/vocoder files are only fetched from their URLs when a voice is actually downloaded or loaded.

## Basic Usage

```python
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.load()                  # load the on-disk cache into memory
manager.merge_default_voices()  # merge the bundled voice indexes in
```

`TTSModelManager(cache_path=...)` overrides the cache file location. By default the
cache is a `JsonStorageXDG` under `phoonnx/voices` in the XDG cache directory.

### Listing Available Voices

```python
# All voices in memory
for voice in manager.all_voices:
    print(voice.voice_id, voice.lang)

# Voices for a language, nearest first
pt_voices = manager.get_lang_voices("pt-PT")

# All language codes present in the loaded catalog
print(manager.supported_langs)
```

`get_lang_voices("pt-PT")` gives 64 voices: every Portuguese voice, `pt-PT` first. The
`phoonnx-voices list-voices --lang pt-PT` command keeps only the 51 `pt-PT` voices.

### Listing without downloading

`get_available_voice_ids_by_source()` reads the bundled index JSON files directly
(plain `json.load`, no `TTSModelInfo` construction, no network) and returns a
`{source: [voice_id, ...]}` mapping. Use it for a quick list of what is available before
you download anything:

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
manager.download_voice_by_id("OpenVoiceOS/phoonnx_es-ES_dii_espeak")
```

### Loading a Voice for Inference

```python
voice_info = manager.voices["OpenVoiceOS/phoonnx_ar_miro_espeak_V2"]
tts_voice = voice_info.load()   # downloads the model if not cached, returns TTSVoice

# optional: pin ONNX Runtime execution providers
tts_voice = voice_info.load(providers=["ROCMExecutionProvider", "CPUExecutionProvider"])
```

This Arabic voice uses the eSpeak phonemizer, so it needs `pip install --pre "phoonnx[espeak]"`.

Some voices also have int8 files, which are smaller and run faster on a CPU. Load them with
`precision`:

```python
# Pocket TTS needs: pip install --pre "phoonnx[pockettts]"
tts_voice = manager.voices["pockettts/it/alba"].load(precision="int8")
```

`precision` is `auto` (the default), `int8` or `full`. A voice with no int8 files raises
`PrecisionUnavailable` for `precision="int8"`. See [Precision variants](precision.md).

## Updating the Cache

`merge_default_voices(store=True)` merges the bundled indexes and writes the result to the
on-disk cache. The `phoonnx-voices update-cache` command calls it:

```python
manager = TTSModelManager()
manager.clear()                         # optional: wipe the old cache first
manager.merge_default_voices(store=True) # merge bundled indexes and persist
```

`clear()` empties the in-memory registry only. The on-disk cache changes when `save()`, or
`merge_default_voices(store=True)`, writes the registry back out. `load()` reads the on-disk
cache.

## Cache Location

Voice files are not kept in a phoonnx-owned directory. They are fetched with
`huggingface_hub` and used exactly where it puts them: the shared HuggingFace hub
cache (`~/.cache/huggingface/hub/` by default, or wherever `HF_HOME`/`HF_HUB_CACHE`
points). A model named by many voices, or already pulled by another program on the
machine, is stored once and reused rather than downloaded again. The model file, its
config, the tokenizer artifacts and any vocoder, style, speaker-encoder or auxiliary graphs
all go into the same hub snapshot directory.

A voice whose URL is not a HuggingFace file URL (a self-hosted or mirrored voice)
cannot be served by the hub client, so it still gets a private copy under the hub
cache instead. That copy is not shared with other programs.

`TTSModelInfo.voice_path` returns the directory that holds the primary graph of the voice:
the hub snapshot directory, or the private-copy directory for a non-hub URL. The file names
in it change from voice to voice, so load a voice with `load()`, not by file name.

phoonnx does not read voice files from `~/.cache/phoonnx/voices/`. If that folder exists on a
machine, you can delete it.

The voice catalog cache is a separate thing from these model files: it is a
`JsonStorageXDG` under `phoonnx/voices` in the XDG cache directory (see
[`Basic Usage`](#basic-usage) above), and only holds the voice index, not any
downloaded model.

## TTSModelInfo

Each voice in the catalog is a `TTSModelInfo` dataclass. It carries the URLs and
metadata for one voice and lazily resolves its `VoiceConfig` on first access.

A `TTSModelInfo` can also describe a voice that is not in the index. This one points at the
files of an index voice:

```python
from phoonnx.model_manager import TTSModelInfo

base = "https://huggingface.co/Phoonnx/phoonnx_pt-PT_miro_unicode/resolve/eb9a020da900534d7232f11a6cf5d1ae1c545219"
info = TTSModelInfo(
    voice_id="my-copy-of-miro",
    lang="pt-PT",
    model_url=f"{base}/miro_pt-PT.onnx",
    config_url=f"{base}/miro_pt-PT.json",
    engine="phoonnx",  # optional override
)

print(info.config.sample_rate)   # the VoiceConfig, read on first access
print(info.download_model().name)
voice = info.load()
print(type(voice).__name__)
```

```
22050
miro_pt-PT.onnx
TTSVoice
```

### Fields

`TTSModelInfo` has 31 fields. The important ones are below. Many are only
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
| `dictionary_url` | `str` \| `None` | Vosk pronunciation dictionary (word → phonemes); downloaded as the voice's `phonemizer_model` |
| `speaker_encoder_url` / `speaker_encoder_type` | `str` \| `None` | Cloning speaker-encoder ONNX (reference audio → d-vector) |
| `aux_model_urls` | `dict` \| `None` | Extra ONNX graphs/files for multi-graph engines (F5-TTS; SuperTonic's `duration_predictor`/`text_encoder`/`vocoder` graphs plus its `tts.json` config, `unicode_indexer.json` and per-speaker `style.json`; NeuTTS's NeuCodec decoder graph plus its `tokenizer.json` and `voices.json`; Indic Parler's `text_encoder`/`decoder_decode`/`dac_decoder` graphs plus its two tokenizers; MOSS-TTS-Nano's decode-step / local-decoder / codec graphs plus its SentencePiece `tokenizer.model` and the `.data` external-weight blobs those graphs reference by filename), keyed by engine-param name |
| `engine_options` | `dict` \| `None` | Plain per-voice engine settings that are not downloadable files, merged into `engine_params` as-is (e.g. NeuTTS's `voice` preset name) |
| `display_name` | `str` \| `None` | Friendly name for UIs/CLIs; may contain `{engine}`/`{phoneme_type}` placeholders |
| `vocab_override` | `dict` \| `None` | Custom token-to-ID mapping |
| `precision` | `str` \| `None` | `int8` for an entry that has only int8 files, unset for full files |
| `variants` | `dict` \| `None` | `{"int8": {...}}`: the URL fields of the int8 files that differ, and `"auto": true` when `auto` may pick them |

### Hub revisions in the bundled index

Each Hugging Face URL in a bundled index file names a commit sha, never a branch:
`https://huggingface.co/<repo>/resolve/<40-hex sha>/<path>`. An installed phoonnx loads the
files of that commit. The index schema check rejects a Hub URL that names a branch or a tag.
[Use phoonnx offline](offline.md#why-a-cached-voice-keeps-working-after-an-upgrade) shows what
happens when the cache holds another commit of the voice.

Key methods and properties: `config` (lazy `VoiceConfig`), `voice_path`,
`download_model()`, `download_all()` (model + config + tokenizer + vocoder/style/
speaker-encoder/aux graphs), `download_vocoder()`, `load(providers=None, precision=None)`,
and `with_precision(precision)`, which gives the entry with the files that precision loads.

---

[Docs index](index.md) · Previous: [Choose a voice](choose_voice.md) · Next: [CLI reference](cli.md)
