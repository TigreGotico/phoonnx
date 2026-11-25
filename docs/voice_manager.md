# Voice manager

<!-- agent-hint
covers: how to list, look up, download and load voices from Python with TTSModelManager and TTSModelInfo
source: phoonnx/model_manager.py:TTSModelManager, phoonnx/model_manager.py:TTSModelInfo, phoonnx/voice_index/*.json
generated: no
verify: XDG_CACHE_HOME=.venv/verify-cache HF_HUB_OFFLINE=1 .venv/bin/python -m doctest -o ELLIPSIS -o NORMALIZE_WHITESPACE docs/voice_manager.md
see: choose_voice.md, precision.md, offline.md, cli.md
-->

This page shows how to find, download and load voices from Python. It is for developers who use phoonnx as a library.

`TTSModelManager` holds the voice catalog. The catalog comes from voice index files inside the package, so listing the voices needs no network.

## Where voices come from

`merge_default_voices()` loads the bundled `phoonnx/voice_index/*.json` files, one file per
source. It fetches nothing from the network. The package holds 39 sources:

```python
>>> from phoonnx.model_manager import TTSModelManager
>>> sorted((path.stem for path in TTSModelManager.voice_index_files()), key=str.lower)
['arktts', 'BSC', 'chatterbox', 'coqui_community', 'coqui_vits', 'f5tts', 'fastpitch',
 'ghananlp', 'glowtts', 'indic_parler', 'inflect', 'llasa', 'magpie', 'mimic3', 'mixertts',
 'MMS', 'mosstts', 'neurlang', 'neutts', 'omnivoice', 'optispeech', 'orpheus', 'outetts',
 'OVOS', 'parlertts', 'phonikud', 'piper', 'piper_community', 'pockettts', 'proxectonos',
 'qwen3tts', 'rapido', 'shami', 'sparktts', 'styletts2', 'supertonic',
 'transformers_community', 'vits2', 'vosk']

```

`phoonnx-voices list-available` prints the voice ids of each source.

Each JSON entry becomes a `TTSModelInfo`. phoonnx fetches the model, config and vocoder
files of a voice only when you download it, load it or read its `config`.

## Load the catalog

```python
>>> from phoonnx.model_manager import TTSModelManager
>>> manager = TTSModelManager()
>>> manager.load()                  # read the catalog file into memory
>>> manager.merge_default_voices()  # merge the bundled voice indexes in
>>> "OpenVoiceOS/phoonnx_pt-PT_miro_unicode" in manager.voices
True

```

`TTSModelManager(cache_path=...)` sets the location of the catalog file. By default the
catalog is `phoonnx/voices.json` in the XDG cache directory, which is
`~/.cache/phoonnx/voices.json` on most Linux systems.

### List the voices

```python
>>> for voice in manager.all_voices[:2]:
...     print(voice.voice_id, voice.lang)
OpenVoiceOS/phoonnx_oc_miro_unicode oc-FR
OpenVoiceOS/phoonnx_oc_dii_unicode oc-FR
>>> pt_voices = manager.get_lang_voices("pt-PT")  # the voices for a language, nearest first
>>> pt_voices[0].voice_id
'OpenVoiceOS/phoonnx_pt-PT_miro_unicode'
>>> "pt-PT" in manager.supported_langs           # the language codes of the catalog
True

```

`get_lang_voices("pt-PT")` gives every Portuguese voice, with the `pt-PT` voices first. The
`phoonnx-voices list-voices --lang pt-PT` command keeps only the `pt-PT` voices.

### List without the catalog

`get_available_voice_ids_by_source()` reads the bundled index files directly. It builds no
`TTSModelInfo` and reads no network. It returns a `{source: [voice_id, ...]}` mapping, with
each source name in lowercase. Use it for a quick list of what is available before you
download anything:

```python
>>> by_source = manager.get_available_voice_ids_by_source()
>>> len(by_source), "OpenVoiceOS/phoonnx_an_dii_unicode" in by_source["ovos"]
(39, True)

```

### Get one voice

```python
>>> voice_info = manager.voices["OpenVoiceOS/phoonnx_eu-ES_dii_espeak"]
>>> voice_info.engine.value, voice_info.phoneme_type.value
('phoonnx', 'espeak')

```

An id that is not in the catalog raises `UnknownVoice`, a `KeyError`. `manager.get_voice(voice_id)`
gives `None` instead.

### Download one voice

`download_voice_by_id(voice_id)` downloads every file of one voice. It looks the id up in
the catalog first, then in the bundled indexes, so it works before you load the catalog. It
returns `True` when it finds the voice, and `False` when the id is unknown:

```python
manager.download_voice_by_id("OpenVoiceOS/phoonnx_es-ES_dii_espeak")
```

### Load a voice for inference

```python
voice_info = manager.voices["OpenVoiceOS/phoonnx_ar_miro_espeak_V2"]
tts_voice = voice_info.load()   # downloads the model if it is not cached, returns a TTSVoice

# optional: set the ONNX Runtime execution providers
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

## Update the catalog file

`merge_default_voices(store=True)` merges the bundled indexes and writes the result to the
catalog file. The `phoonnx-voices update-cache` command calls it:

```python
manager = TTSModelManager()
manager.clear()                          # optional: empty the catalog first
manager.merge_default_voices(store=True) # merge the bundled indexes and write the file
```

`clear()` empties the catalog in memory and in the storage object, but it does not write the
file. The file changes when `save()` or `merge_default_voices(store=True)` writes the catalog.
`load()` reads the file.

## Where the voice files go

phoonnx does not keep voice files in a folder of its own. It fetches them with
`huggingface_hub` and uses them where the hub client puts them: the shared Hugging Face hub
cache. That is `~/.cache/huggingface/hub/` by default, or the folder that `HF_HOME` or
`HF_HUB_CACHE` names. A file that many voices name, or that another program already
downloaded, is stored once.

The model file, its config, the tokenizer files and any vocoder, style, speaker encoder or
other graph of a voice go into the same hub snapshot folder.

A voice whose URL is not a Hugging Face file URL, such as a self-hosted voice, gets a
private copy under the hub cache. Other programs do not share that copy.

`TTSModelInfo.voice_path` gives the folder that holds the main graph of the voice: the hub
snapshot folder, or the private-copy folder for a URL that is not on the Hub. Getting the
path downloads the model. The file names in the folder change from voice to voice, so load a
voice with `load()`, not by file name.

The catalog file holds only the voice index. It holds no model files.

## TTSModelInfo

Each voice in the catalog is a `TTSModelInfo` dataclass. It holds the URLs and the metadata
of one voice, and it reads the `VoiceConfig` of the voice on first access.

A `TTSModelInfo` can also describe a voice that is not in the index. This one points at the
files of an index voice:

```python
from phoonnx.model_manager import TTSModelInfo

base = "https://huggingface.co/Phoonnx/phoonnx_pt-PT_miro_unicode/resolve/aec14a0428e6b67a47eb54b2baa4a604d42c0063"
info = TTSModelInfo(
    voice_id="my-copy-of-miro",
    lang="pt-PT",
    model_url=f"{base}/miro_pt-PT.onnx",
    config_url=f"{base}/phoonnx.json",
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

The methods and properties that you use most are `config` (the `VoiceConfig`, read on first
access), `voice_path`, `download_model()`, `download_vocoder()`, `download_all()`,
`load(providers=None, precision=None, threads=None)`, `disk_size()` and
`with_precision(precision)`. `download_all()` downloads the model, the config, the tokenizer
files and every vocoder, style, speaker encoder and auxiliary graph. `with_precision()` gives
the entry with the files that a precision loads.

### Hub revisions in the bundled index

Each Hugging Face URL in a bundled index file names a commit sha, never a branch:
`https://huggingface.co/<repo>/resolve/<40-hex sha>/<path>`. An installed phoonnx loads the
files of that commit. The index schema check rejects a Hub URL that names a branch or a tag.
[Use phoonnx offline](offline.md#why-a-cached-voice-keeps-working-after-an-upgrade) shows what
happens when the cache holds another commit of the voice.

### Fields

`TTSModelInfo` has 31 fields:

```python
>>> from dataclasses import fields
>>> from phoonnx.model_manager import TTSModelInfo
>>> len(fields(TTSModelInfo))
31

```

The table lists the fields that you use most. Many of them apply only to some engine
families. See [Engine architecture](engines.md), [Voice cloning](cloning.md) and
[Vocoders](vocoders.md).

| Field | Type | Description |
|-------|------|-------------|
| `voice_id` | `str` | Unique identifier |
| `lang` | `str` | BCP-47 language tag, for example `en-US`. The index value wins over the source config, which can lack the tag or hold a wrong one. |
| `model_url` | `str` | URL to the primary `.onnx` file |
| `config_url` | `str` \| `None` | URL to the JSON config. Some voices ship only `tokens.txt`. |
| `vocab_url` | `str` \| `None` | URL to `vocab.json` (Transformers style) |
| `tokens_url` | `str` \| `None` | URL to `tokens.txt` (Mimic3/Sherpa style) |
| `tokenizer_config_url` | `str` \| `None` | Transformers tokenizer config, or the Chatterbox BPE `tokenizer.json` |
| `phoneme_type` | `PhonemeType` \| `None` | Phonemizer type. A voice with a `config_url` reads it from its config, and the index value is for search only. A voice with no config reads it here. |
| `alphabet` | `Alphabet` \| `None` | Phoneme alphabet. The same rule as `phoneme_type` applies. |
| `engine` | `Engine` \| `None` | TTS engine the voice was trained with |
| `vocoder_url` | `str` \| `None` | Separate vocoder ONNX (two-stage engines) |
| `vocoder_config_url` | `str` \| `None` | Vocoder `vocoder.json` parameters |
| `vocoder_type` | `str` \| `None` | Vocoder implementation (`vocos`, `wavenext`, `hifigan`, `melgan`, `raw`, `griffinlim`) |
| `style_url` | `str` \| `None` | Per-voice StyleTTS2/Kokoro style embedding |
| `dictionary_url` | `str` \| `None` | Vosk pronunciation dictionary (word → phonemes), downloaded as the voice's `phonemizer_model` |
| `speaker_encoder_url` / `speaker_encoder_type` | `str` \| `None` | Cloning speaker-encoder ONNX (reference audio → d-vector) |
| `aux_model_urls` | `dict` \| `None` | The other graphs and files of a multi-graph engine, keyed by the engine parameter that receives the local path. Examples are the F5-TTS graphs, the SuperTonic graphs and `tts.json`, and the MOSS-TTS-Nano graphs with their `.data` weight files. |
| `engine_options` | `dict` \| `None` | Per-voice engine settings that are not files, merged into `engine_params` as they are, for example the NeuTTS `voice` preset name |
| `display_name` | `str` \| `None` | Name for UIs and the CLI. It can hold `{engine}` and `{phoneme_type}` placeholders. |
| `vocab_override` | `dict` \| `None` | Custom token-to-ID mapping |
| `precision` | `str` \| `None` | `int8` for an entry that has only int8 files, unset for full files |
| `variants` | `dict` \| `None` | `{"int8": {...}}`: the URL fields of the int8 files that differ, and `"auto": true` when `auto` may pick them |

---

[Docs index](index.md) · Previous: [Choose a voice](choose_voice.md) · Next: [CLI reference](cli.md)
