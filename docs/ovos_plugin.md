# OVOS Plugin

This page is for OpenVoiceOS users and integrators configuring phoonnx as their TTS
engine. phoonnx ships a TTS plugin for the [OpenVoiceOS](https://openvoiceos.com/)
ecosystem via `PhoonnxTTSPlugin`, registered under the `opm.tts` entry point
`ovos-tts-plugin-phoonnx`.

## Per-language default voices

`voice` pins one voice for every request. To serve **many languages from one
server**, leave it unset and map languages to voices instead — the plugin then
picks the voice for each request's language, downloading and caching it on
first use:

```json
{
  "tts": {
    "module": "ovos-tts-plugin-phoonnx",
    "ovos-tts-plugin-phoonnx": {
      "lang2voice": {
        "gl": "proxectonos/celtia",
        "ca": "OpenVoiceOS/matxa-cat-multispeaker-wavenext",
        "pt-br": "<a brazilian voice>",
        "pt": "<a european portuguese voice>"
      }
    }
  }
}
```

Keys are BCP-47 tags: the full tag (`pt-br`) is tried before the primary
subtag (`pt`), so regional variants can differ from the base language.

The same mapping can come from the environment, which is usually easier for
containers whose `mycroft.conf` carries no `lang2voice` entry for a given
language — `PHOONNX_DEFAULT_VOICE_<LANG>`, with underscores standing in for
dashes:

```bash
docker run -e PHOONNX_DEFAULT_VOICE_GL=proxectonos/celtia \
           -e PHOONNX_DEFAULT_VOICE_PT_BR=<voice-id> \
           -p 9666:9666 phoonnx
```

Resolution is table-major, not tag-major: every tag `lang2voice` might match
(full tag, then primary subtag) is tried before the env table is looked at at
all, so a `lang2voice` entry for the primary subtag (`pt`) still beats a more
specific `PHOONNX_DEFAULT_VOICE_PT_BR` env var — the config wins outright for
any language it names anything for, and the env vars only fill in languages
the config leaves unconfigured. Below both, the voice index's own default for
the language applies. Keys and tags are matched after normalisation, so
`gl-ES`/`gl`, `pt-br`/`pt_BR` and similar spellings agree.

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

The module name and the config sub-key are both the entry-point name,
`ovos-tts-plugin-phoonnx`.

If `voice` is omitted or set to `"default"`, the plugin selects a default voice for the configured language.

### Multi-speaker voices

Some models bundle several speakers in one voice (e.g. the Catalan multiaccent
matxa model `OpenVoiceOS/matxa-cat-multiaccent-wavenext`, which carries the four
dialects × two genders). Pick a speaker with `speaker_id` (an integer index) or
`speaker` (a name resolved against the model's `speaker_id_map`, given bare or
accent-qualified). `speaker_id` wins if both are set; single-speaker voices
ignore both.

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

For the Catalan multiaccent model the indices are: `quim` 0 / `olga` 1 (balear),
`grau` 2 / `elia` 3 (central), `pere` 4 / `emma` 5 (nord-occidental),
`lluc` 6 / `gina` 7 (valencià).

### Execution providers

`onnx_providers` takes an ordered list of ONNX Runtime execution providers; the
first one available on the machine runs the models, with `CPUExecutionProvider`
always kept as a final fallback. Unset, providers come from the
`PHOONNX_ONNX_PROVIDERS` environment variable or are auto-detected.

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

`onnx_providers` may also be given as a bare string for a single provider; `providers`
is accepted as an alias. Running on a GPU needs the matching ONNX Runtime build
(`onnxruntime-rocm`, `onnxruntime-gpu`, `onnxruntime-directml`, ...) — see
[configuration.md](configuration.md#execution-providers).

### Audio super-resolution

The optional `super_resolution` switch upscales synthesized audio to 48 kHz after
synthesis, using [`audiosronnx`](https://github.com/TigreGotico/audiosronnx)
(pure-ONNX, no torch at runtime). It is **off by default** — when disabled the
audio and its sample rate are exactly what the voice produced, and `audiosronnx`
is never imported.

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

`super_resolution_model` names the `audiosronnx` engine; omitting it selects
`novasr`, which extends the high band in a way that tracks
the speech harmonics and sounds natural on TTS output. Other engines are
available (`lavasr`, `hifiganbwe`, `apbwe`) — see the
[audiosronnx](https://github.com/TigreGotico/audiosronnx) engine table — but
`lavasr` tends to add audible high-frequency noise on already-wideband (>=22 kHz)
TTS voices, so it is better suited to genuinely low-rate inputs. Super-resolution
gives the biggest quality win on low-sample-rate voices, where there is a real
missing band to reconstruct.

Install the extra (pulls `audiosronnx`, which fetches its models from HuggingFace
on first use):

```bash
pip install phoonnx[audiosr]
```

Enabling `super_resolution` without that extra installed raises an `ImportError`
naming it, at the first synthesis.

The upscaling lives in the core `TTSVoice`, so it also applies to direct library
use — and, being a `SynthesisConfig` field, it can be switched on per call:

```python
from phoonnx.config import SynthesisConfig
from phoonnx.voice import TTSVoice

voice = TTSVoice.load("model.onnx", "model.json")
cfg = SynthesisConfig(super_resolution=True, super_resolution_model="novasr")
for chunk in voice.synthesize("hello", cfg):
    ...  # chunk.sample_rate is 48000
```

`synthesize_wav` takes its format from the first chunk, so the written WAV header
carries the upscaled rate too.

### Synthesis and cloning config keys

The plugin reads synthesis options from its config, accepting the documented
underscore name plus legacy aliases (resolved by `_cfg_opt` in `phoonnx/opm.py`):

| Option | Accepted keys | Meaning |
|--------|---------------|---------|
| Phonetic spellings | `enable_phonetic_spellings`, `enable_phonetic_spelling` | Toggle phonetic-spelling substitutions |
| Diacritics | `add_diacritics` | Add diacritics (Arabic/Hebrew) |
| Generator noise | `noise_scale`, `noise-scale` | VITS noise scale |
| Phoneme length | `length_scale`, `length-scale` | VITS length scale |
| Phoneme width noise | `noise_w_scale`, `noise_w`, `noise-w` | VITS noise-W scale |
| Speaker | `speaker_id`, `speaker` | Multi-speaker selection (see above) |
| Reference clip | `speaker_reference`, `ref_wav`, `clone_voice` | Cloning reference audio |
| Reference text | `speaker_reference_text`, `ref_text` | In-context reference transcription |
| Reference lang | `speaker_reference_lang`, `ref_lang` | In-context reference language |
| Providers | `onnx_providers`, `providers` | ONNX Runtime execution providers |
| Super-resolution | `super_resolution` | Enable 48 kHz upscaling (bool, off by default) |
| Super-resolution model | `super_resolution_model` | `audiosronnx` engine name (default `novasr`) |

See [cloning.md](cloning.md) for the cloning keys in detail.

## How It Works

`PhoonnxTTSPlugin` extends the `ovos-plugin-manager` `TTS` base class. On initialization, it:

1. Creates a `TTSModelManager` and loads the local voice cache
2. Merges in default bundled voices
3. Loads the configured voice (or the default voice for the language) into memory

### Voice Caching

Loaded voices are cached in memory (`plugin.voice_cache`) to avoid re-loading on every utterance. The first call for a new voice ID triggers a download if needed. Memory is charged per model rather than per voice: a family of catalog entries that name the same graph (for example the `omnivoice` or `qwen3tts` voice-config variants) shares one ONNX Runtime session and counts against the budget once, however many of its voices are loaded.

Two config keys bound how much stays resident:

| Option | Meaning |
|--------|---------|
| `max_loaded_voices` | Maximum number of voices resident at once (pinned ones included). Unset never evicts anything. |
| `max_loaded_bytes` | Memory budget for resident voice weights, as a plain byte count or a size string (`"3GB"`, `"512 MB"`, `"1.5GiB"` — `KB/MB/GB/TB` are powers of 1000, `KiB/MiB/GiB/TiB` powers of 1024). Unset never evicts anything. |
| `pinned_voices` | Voice id or list of voice ids to load at startup and never evict, regardless of the other limits. |
| `load_wait_timeout` | Seconds a cold load waits for memory in use to come back before loading anyway. Defaults to 300; a memory bound must never become a hang. |

A voice whose own weights are bigger than the whole `max_loaded_bytes` budget is refused with `VoiceExceedsMemoryBudget` rather than loaded, because loading it is a guaranteed OOM kill. A voice that fits on its own but not beside what is already in memory waits for room, and takes it anyway once `load_wait_timeout` seconds pass, logging a warning that memory use will exceed the budget: a voice a request is still synthesizing with is charged to the budget until that request finishes, so there are moments when nothing can be evicted. See [deployment.md](deployment.md#memory-budgeting) for sizing this against the container's memory limit.

### Refreshing Voices

Call `refresh_voices()` to reload the voice list from the model manager:

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
| `get_tts(sentence, wav_file, lang=None, voice=None)` | Synthesize `sentence` and write to the WAV file path; `lang`/`voice` override the configured defaults per call |
| `get_default_voice(lang)` | Returns the default `TTSModelInfo` for a language |
| `refresh_voices(force=False)` | Refresh the voice catalog from the model manager |

## Supported Voices

All voices available through `TTSModelManager` are accessible to the plugin. Use the
[`phoonnx-voices`](cli.md) command to see available options:

```bash
phoonnx-voices update-cache
phoonnx-voices list-voices --lang en-US
```
