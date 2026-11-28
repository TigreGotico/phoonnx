# OVOS plugin

<!-- agent-hint
covers: how to configure ovos-tts-plugin-phoonnx in mycroft.conf, every config key it reads, and how it picks, caches and loads voices
source: phoonnx/opm.py:PhoonnxTTSPlugin, phoonnx/voice_cache.py:VoiceCache, pyproject.toml [project.entry-points."opm.tts"]
generated: no
verify: HF_HUB_OFFLINE=1 .venv/bin/python -m doctest -o ELLIPSIS -o NORMALIZE_WHITESPACE docs/ovos_plugin.md && .venv/bin/python -c "import re,sys; p=open('docs/ovos_plugin.md').read(); src=open('phoonnx/opm.py').read(); q=chr(96); t=p.split(chr(10)+'### Synthesis and cloning config keys',1)[1].split(chr(10)+'## ',1)[0]+p.split(chr(10)+'### Voice caching',1)[1].split(chr(10)+'### ',1)[0]; keys=[k for line in t.splitlines() if line.startswith('| ') for k in re.findall(q+'([a-z_-]+)'+q, line.split('|')[2] if line.count('|')>3 else line.split('|')[1])]; bad=[k for k in keys if chr(34)+k+chr(34) not in src]; print(len(keys), bad); sys.exit(1 if bad or not keys else 0)"
see: docker.md, deployment.md, precision.md, cloning.md
-->

This page shows how to use phoonnx as the text-to-speech engine of OpenVoiceOS. It is for OpenVoiceOS users and integrators who configure the `ovos-tts-plugin-phoonnx` plugin.

phoonnx registers the plugin class `PhoonnxTTSPlugin` under the `opm.tts` entry point `ovos-tts-plugin-phoonnx`. For grapheme-to-phoneme (G2P) conversion without speech, use the separate [ovos-scriptconv-g2p-plugin](https://github.com/TigreGotico/ovos-scriptconv-g2p-plugin). See [G2P for OVOS](phonemizers.md#g2p-for-ovos).

## Configuration

In your OpenVoiceOS `mycroft.conf` or skills config, set:

```json
{
  "tts": {
    "module": "ovos-tts-plugin-phoonnx",
    "ovos-tts-plugin-phoonnx": {
      "lang": "en-US",
      "voice": "OpenVoiceOS/pipertts_en-US_miro"
    }
  }
}
```

The module name and the config sub-key are both the entry point name, `ovos-tts-plugin-phoonnx`. This voice uses the eSpeak phonemizer, so the machine that runs the plugin needs the `espeak` extra:

```bash
pip install --pre "phoonnx[espeak]"
```

If `voice` is omitted or set to `"default"`, the plugin selects a default voice for the configured language.

To check that the plugin speaks before you restart OVOS, run it with the same settings from Python:

```python
from phoonnx.opm import PhoonnxTTSPlugin

config = {"lang": "en-US", "voice": "OpenVoiceOS/pipertts_en-US_miro"}
plugin = PhoonnxTTSPlugin(config=config)
plugin.get_tts("Hello from phoonnx.", "check.wav", voice=config["voice"])
```

An SSML document (`<speak>...</speak>`, with or without attributes) reaches the voice as it was written: the plugin reads the markup itself, as [SSML](ssml.md) describes. `get_tts` returns the path of the WAV file and `None`. The first call downloads the voice. Play `check.wav` to hear it. A call without `voice` speaks the configured `voice` when its language is the language of the plugin. A call in another language uses the default voice of that language. If the plugin stops at start, read [Errors you may see](installation.md#errors-you-may-see).

### Multi-speaker voices

Some models hold several speakers in one voice. For example, the Catalan multiaccent matxa model `OpenVoiceOS/matxa-cat-multiaccent-wavenext` holds four dialects with two speakers each. Select a speaker with `speaker_id`, an integer index, or with `speaker`, a name that the plugin finds in the `speaker_id_map` of the model. The name can be bare (`elia`) or carry its accent (`central/elia`). `speaker_id` wins when both are set. Single-speaker voices ignore both.

```json
{
  "tts": {
    "module": "ovos-tts-plugin-phoonnx",
    "ovos-tts-plugin-phoonnx": {
      "voice": "OpenVoiceOS/matxa-cat-multiaccent-wavenext",
      "speaker_id": 3
    }
  }
}
```

The config of the Catalan multiaccent model has no `speaker_id_map`, so select its speaker with `speaker_id`. A `speaker` name logs a warning, and the voice uses its default speaker. The upstream model card, `projecte-aina/matxa-tts-cat-multiaccent`, gives the indices:
`quim` 0 / `olga` 1 (balear), `grau` 2 / `elia` 3 (central), `pere` 4 / `emma` 5
(nord-occidental), `lluc` 6 / `gina` 7 (valencià).

### Execution providers

`onnx_providers` takes an ordered list of ONNX Runtime execution providers. The first provider that the machine has runs the models, and `CPUExecutionProvider` is always the last fallback. When the key is unset, the providers come from the `PHOONNX_ONNX_PROVIDERS` environment variable, or from detection.

```json
{
  "tts": {
    "module": "ovos-tts-plugin-phoonnx",
    "ovos-tts-plugin-phoonnx": {
      "onnx_providers": ["ROCMExecutionProvider", "CPUExecutionProvider"]
    }
  }
}
```

`onnx_providers` can also be one string, for one provider. `providers` is an alias of the key. A GPU needs the matching ONNX Runtime build, such as `onnxruntime-gpu`, `onnxruntime-rocm` or `onnxruntime-directml`. Read [Execution providers](configuration.md#execution-providers).

### Thread count

`onnx_threads` sets the number of threads that each graph operator of a voice runs on. When the key is unset, the value comes from the `PHOONNX_ORT_THREADS` environment variable. Without both, ONNX Runtime uses one thread per core. Set it on a host that serves several voices or shares the CPU with other jobs. Read [Thread count](configuration.md#thread-count).

```json
{
  "tts": {
    "module": "ovos-tts-plugin-phoonnx",
    "ovos-tts-plugin-phoonnx": {
      "onnx_threads": 2
    }
  }
}
```

### Precision

`precision` selects the files of a voice that has full files and int8 files: `auto` (the default), `int8` or `full`. When the key is unset, `null` or an empty string, the value comes from the `PHOONNX_PRECISION` environment variable.

```json
{
  "tts": {
    "module": "ovos-tts-plugin-phoonnx",
    "ovos-tts-plugin-phoonnx": {
      "precision": "int8"
    }
  }
}
```

A value that is not a string, or not one of the three, stops the plugin at start. A voice that has no files for the value raises `PrecisionUnavailable` at its first synthesis. The plugin caches the audio of int8 files apart from the audio of full files. [Precision variants](precision.md) tells what `auto` selects.

### Audio super-resolution

The `super_resolution` switch upscales the synthesized audio to 48 kHz with [`audiosronnx`](https://github.com/TigreGotico/audiosronnx), which runs on ONNX with no torch. It is off by default. When it is off, the audio and its sample rate are what the voice made, and phoonnx does not import `audiosronnx`.

```json
{
  "tts": {
    "module": "ovos-tts-plugin-phoonnx",
    "ovos-tts-plugin-phoonnx": {
      "super_resolution": true,
      "super_resolution_model": "novasr"
    }
  }
}
```

`super_resolution_model` names the `audiosronnx` engine. Without it, phoonnx uses `novasr`. The [audiosronnx](https://github.com/TigreGotico/audiosronnx) engine table lists the other engines. Super-resolution helps most on voices with a low sample rate, which have a missing high band.

Install the `audiosr` extra. It installs `audiosronnx`, which downloads its models from Hugging Face on first use:

```bash
pip install --pre "phoonnx[audiosr]"
```

When `super_resolution` is on and the extra is not installed, the first synthesis raises an `ImportError` that names the extra.

`TTSVoice` does the upscaling, so it also works when you use the library directly. It is a `SynthesisConfig` field, so you can turn it on for one call:

```python
from phoonnx.config import AudioPost, SynthesisConfig
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.load()
manager.merge_default_voices()

voice = manager.voices["OpenVoiceOS/pipertts_en-US_miro"].load()
cfg = SynthesisConfig(audio=AudioPost(super_resolution=True, super_resolution_model="novasr"))
for chunk in voice.synthesize("hello", cfg):
    print(chunk.sample_rate)
```

Each chunk reports the sample rate that the super-resolution engine gives. `synthesize_wav` takes its format from the first chunk, so the WAV header also has the upscaled rate.

### Synthesis and cloning config keys

The plugin reads these synthesis options from its config. Each option has an underscore name and can have aliases. The first key in the list that the config holds wins:

| Option | Accepted keys | Meaning |
|--------|---------------|---------|
| Phonetic spellings | `enable_phonetic_spellings`, `enable_phonetic_spelling` | Turn the phonetic spelling replacements on or off |
| Diacritics | `add_diacritics` | Add diacritics, for Arabic and Hebrew voices |
| Generator noise | `noise_scale`, `noise-scale` | VITS noise scale |
| Phoneme length | `length_scale`, `length-scale` | VITS length scale |
| Phoneme width noise | `noise_w_scale`, `noise_w`, `noise-w` | VITS noise W scale |
| Speaker | `speaker_id`, `speaker` | Multi-speaker selection, see above |
| Reference clip | `speaker_reference`, `ref_wav`, `clone_voice` | Cloning reference audio |
| Reference text | `speaker_reference_text`, `ref_text` | In-context reference transcription |
| Reference lang | `speaker_reference_lang`, `ref_lang` | In-context reference language |
| Providers | `onnx_providers`, `providers` | ONNX Runtime execution providers |
| Threads | `onnx_threads` | Threads per graph operator, see above |
| Precision | `precision` | `auto`, `int8` or `full`, see above |
| Super-resolution | `super_resolution` | Turn on 48 kHz upscaling, a bool that is off by default |
| Super-resolution model | `super_resolution_model` | The `audiosronnx` engine name, `novasr` by default |
| Pace stretch | `stretch_pace` | Make an engine that cannot change its pace (Orpheus, Llasa, Pocket TTS and the other codec language models, and Shami) follow `length_scale` and SSML rate with a time stretch, a bool that is off by default |
| Chunk | `chunk` | Text per engine call: `sentence` (default), `clause`, `paragraph` or `none` |

[Voice cloning](cloning.md) describes the cloning keys.

### Piper voice names

A Piper client names a voice `en_US-lessac-medium` or `lessac-medium`. The catalog calls it `piper/en_US-lessac-medium`. The plugin accepts all three spellings for a Piper voice, so a client that was set up for `ovos-tts-plugin-piper` or a Piper server needs no change of voice names. A Piper speaker suffix (`lessac-medium#3`, `piper/en_US-lessac-medium#3`) is ignored: `speaker_id` or `speaker` selects the speaker.

A short name that several languages share, such as `mls-medium`, goes to the voice of the language of the call (`lang`, or the language of the plugin when the call has none): first the same locale, then the same language. A short name that no language of the call fits and that two or more languages share is an error, not a guess. The voice is loaded once, whichever spelling asks for it. A pin in `pinned_voices` can use any of these spellings, and a short name of a pin goes to the language of the plugin.

## Per-language default voices

`voice` sets one voice for the requests in the language of the plugin. To serve many languages from one server, map languages to voices with `lang2voice`. The plugin then selects the voice for the language of each request, and downloads and caches it on first use:

```json
{
  "tts": {
    "module": "ovos-tts-plugin-phoonnx",
    "ovos-tts-plugin-phoonnx": {
      "lang2voice": {
        "gl": "proxectonos/celtia",
        "ca": "OpenVoiceOS/matxa-cat-multispeaker-wavenext",
        "pt-br": "OpenVoiceOS/pipertts_pt-BR_miro",
        "pt": "OpenVoiceOS/phoonnx_pt-PT_miro_unicode",
        "en-GB": "OpenVoiceOS/pipertts_en-GB_miro"
      }
    }
  }
}
```

Keys are BCP-47 tags. The plugin tries the language of a request in three forms, in this order:

1. Its standard form: a request for `pt-br` matches a `pt-BR` key.
2. Its default region, which is the place where the language first appeared
   and is still spoken: a bare `pt` matches a `pt-PT` key, and a bare `en`
   matches an `en-GB` key.
3. Its primary subtag: a request for `pt-BR` matches a `pt` key.

So a regional key can differ from the base language, and a bare key serves every region. This example uses the `lang2voice` table above:

```python
>>> from phoonnx.opm import PhoonnxTTSPlugin
>>> plugin = PhoonnxTTSPlugin(config={"lang": "pt-PT", "lang2voice": {
...     "gl": "proxectonos/celtia",
...     "ca": "OpenVoiceOS/matxa-cat-multispeaker-wavenext",
...     "pt-br": "OpenVoiceOS/pipertts_pt-BR_miro",
...     "pt": "OpenVoiceOS/phoonnx_pt-PT_miro_unicode",
...     "en-GB": "OpenVoiceOS/pipertts_en-GB_miro"}})
>>> for lang in ("pt_br", "pt-AO", "gl-ES", "en", "en-US"):
...     print(lang, plugin.configured_voice_for_lang(lang))
pt_br OpenVoiceOS/pipertts_pt-BR_miro
pt-AO OpenVoiceOS/phoonnx_pt-PT_miro_unicode
gl-ES proxectonos/celtia
en OpenVoiceOS/pipertts_en-GB_miro
en-US None

```

`en-US` has no key, so the default voice of the voice index for `en-US` speaks it.

When no `lang2voice` key and no environment variable matches, the plugin reads the bundled table `phoonnx/default_voices.json`. Each key is a BCP-47 tag, and each entry gives a `voice_id` and a `source`. The `source` names the QA record that selected the voice, `docs/data/voice_qa.json` (see [voices.md](voices.md)). A key matches a request at language distance 0, the same match that the voice index uses, so the `da-DK` key also serves `da`. When no key matches, the first voice of the voice index for the language speaks it.

The same table can come from the environment. This is often easier for a container whose `mycroft.conf` has no `lang2voice` entry for a language. Set `PHOONNX_DEFAULT_VOICE_<LANG>`, with underscores in place of dashes:

```bash
docker run -e PHOONNX_DEFAULT_VOICE_GL=proxectonos/celtia \
           -e PHOONNX_DEFAULT_VOICE_PT_BR=OpenVoiceOS/pipertts_pt-BR_miro \
           -p 9666:9666 ghcr.io/tigregotico/phoonnx:dev
```

The plugin tries the three forms of the tag against the whole `lang2voice` table before it reads the environment. So a `lang2voice` entry for `pt` wins over a more specific `PHOONNX_DEFAULT_VOICE_PT_BR` variable. The config wins for each language that it names, and the environment variables add only the languages that the config leaves out. When neither names a voice, the default of the voice index for the language applies. The plugin compares keys and tags after normalization, so `pt-br` and `pt_BR` agree. A configured voice id that the index does not have logs a warning, and the default of the voice index applies.

## How It Works

`PhoonnxTTSPlugin` is a subclass of the `TTS` base class of `ovos-plugin-manager`. When it starts, it does these steps:

1. Creates a `TTSModelManager` and loads the local voice cache.
2. Merges in the bundled voice indexes.
3. Finds the configured voice, or the default voice for the language, in the catalog. It
   does not load the voice. An unknown `voice` or a `lang` with no voice stops the plugin
   here.
4. Loads the voices in `pinned_voices`, if any.

The first synthesis downloads and loads the voice. A missing extra or a failed download shows at that point, not at start.

### Voice caching

The plugin keeps loaded voices in memory, in `plugin.voice_cache`, so it does not load a voice again for each sentence. The first call for a voice that is not loaded downloads it when needed. The memory budget counts each model one time, not each voice: index entries that name the same graph, such as the `omnivoice` entries, share one ONNX Runtime session.

These config keys limit what stays in memory:

| Option | Meaning |
|--------|---------|
| `max_loaded_voices` | The most voices in memory at one time, pinned voices included. When it is unset, the plugin removes no voice. |
| `max_loaded_bytes` | The memory budget for the voice weights in memory, as a byte count or a size string such as `"3GB"`, `"512 MB"` or `"1.5GiB"`. `KB`, `MB`, `GB` and `TB` are powers of 1000, and `KiB`, `MiB`, `GiB` and `TiB` are powers of 1024. When it is unset, the plugin removes no voice. |
| `pinned_voices` | A voice id, or a list of voice ids, that the plugin loads at start and never removes. A [Piper voice name](#piper-voice-names) counts as its catalog id. A pin that is not in the catalog is logged as an error at start and is not pinned. |
| `load_wait_timeout` | The seconds that a load waits for memory before it loads anyway. The default is 300, so a memory limit never hangs a request. |

A voice whose weights are larger than the whole `max_loaded_bytes` budget raises `VoiceExceedsMemoryBudget` and does not load. A voice that fits alone, but not beside the voices in memory, waits for room. After `load_wait_timeout` seconds it loads anyway, and the plugin logs a warning that memory use goes above the budget. A voice stays in the budget until each request that uses it finishes, so for some time the plugin can remove nothing. [Deployment](deployment.md#set-a-memory-budget) tells how to size the budget for the memory limit of a container.

### Refresh the voices

Call `refresh_voices()` to merge the bundled voice index into the voice list again:

```python
plugin.refresh_voices(force=True)
```

## Plugin Class Reference

```python
from phoonnx.opm import PhoonnxTTSPlugin

plugin = PhoonnxTTSPlugin(config={
    "lang": "gl-ES",
    "voice": "OpenVoiceOS/phoonnx_gl-ES_miro_unicode",
})
```

## Key methods

| Method | Description |
|--------|-------------|
| `get_tts(sentence, wav_file, lang=None, voice=None, ...)` | Synthesize `sentence`, write it to the WAV file path, and return `(wav_file, None)`. `lang` and `voice` select the voice for one call. Without `voice`, a call in the language of the plugin speaks the configured `voice`, and a call in another language uses the default voice of that language. The other keyword arguments set cloning (`speaker_reference`, `ref_wav` and the others), the speaker (`speaker_id`, `speaker`) and the scales (`length_scale`, `noise_scale`, `noise_w_scale`) for one call |
| `get_default_voice(lang)` | Return the default `TTSModelInfo` for a language, from `lang2voice`, the environment or the voice index |
| `refresh_voices(force=False)` | Merge the bundled voice index into the voice list, when the list is empty or `force` is set |

## Supported Voices

The plugin can use each voice of `TTSModelManager`. The [`phoonnx-voices`](cli.md) command lists them:

```bash
phoonnx-voices update-cache
phoonnx-voices list-voices --lang en-US
```

---

[← Alphabet conversion model](alphabet_model.md) · [Home](index.md) · [Docker TTS server →](docker.md)
