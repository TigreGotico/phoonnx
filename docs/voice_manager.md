# Voice Manager

This page is for developers using `phoonnx` from Python to discover, download, and load TTS voices. It documents the `TTSModelManager` and `TTSModelInfo` API in `phoonnx/model_manager.py`.

`TTSModelManager` handles discovery, caching, downloading, and loading of TTS voice models. Its catalog is assembled from bundled voice-index files that ship inside the package, so no network access is needed just to enumerate what is available.

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
`PrecisionUnavailable` for `precision="int8"`. See
[configuration.md](configuration.md#precision).

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

If you upgraded from an older phoonnx, voices already downloaded under the old
`~/.cache/phoonnx/voices/<voice_id>/` layout are not reused. Each voice is fetched
once more into the hub cache. That old directory is safe to delete if you want the
space back.

The voice catalog cache is a separate thing from these model files: it is a
`JsonStorageXDG` under `phoonnx/voices` in the XDG cache directory (see
[`Basic Usage`](#basic-usage) above), and only holds the voice index, not any
downloaded model.

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

`TTSModelInfo` has about thirty fields. The important ones are below. Many are only
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

Each Hugging Face URL in a bundled index file names a commit sha, never `main`:
`https://huggingface.co/<repo>/resolve/<40-hex sha>/<path>`. An installed phoonnx then
loads the files it was released with, and an upload to a voice repo reaches users only
through a new index. The index schema check rejects a Hub URL that names a branch or a tag.

A hub cache keeps one snapshot folder per commit. Online, phoonnx fetches the pinned
commit: the client downloads only the files that changed and reuses the others. Offline, or
when the hub does not answer, a voice that is not cached at the pinned commit loads from
another cached snapshot of the same repo: the newest one that holds the file. phoonnx does not
read `refs/main` for this, because a fetch at a commit does not move it.

phoonnx logs one warning with the repo, the pinned commit and the snapshot it reads. The files
that one folder of the repo gives through this fallback all come from one snapshot, so a graph
and its external-data files match. When the pinned commit holds some files of a folder but not
all of them, as after an interrupted download, the missing files do not fall back: the load
fails, so that a graph and its config never come from two commits.

Key methods and properties: `config` (lazy `VoiceConfig`), `voice_path`,
`download_model()`, `download_all()` (model + config + tokenizer + vocoder/style/
speaker-encoder/aux graphs), `download_vocoder()`, `load(providers=None, precision=None)`,
and `with_precision(precision)`, which gives the entry with the files that precision loads.
