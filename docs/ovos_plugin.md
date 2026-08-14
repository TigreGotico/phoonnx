# OVOS Plugin

This page is for OpenVoiceOS users and integrators configuring phoonnx as their TTS
engine. phoonnx ships a TTS plugin for the [OpenVoiceOS](https://openvoiceos.com/)
ecosystem via `PhoonnxTTSPlugin`, registered under the `opm.tts` entry point
`ovos-tts-plugin-phoonnx`.

Looking for standalone grapheme-to-phoneme (G2P) instead of TTS? See
[phonemizers.md](phonemizers.md#g2p-for-ovos) — that's
[ovos-scriptconv-g2p-plugin](https://github.com/TigreGotico/ovos-scriptconv-g2p-plugin),
a separate OPM plugin.

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

Loaded voices are cached in memory (`self.voices` dict) to avoid re-loading on every utterance. The first call for a new voice ID triggers a download if needed.

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
