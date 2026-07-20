# OVOS Plugin

This page is for OpenVoiceOS users and integrators configuring phoonnx as their TTS
engine. phoonnx ships a TTS plugin for the [OpenVoiceOS](https://openvoiceos.com/)
ecosystem via `PhoonnxTTSPlugin`, registered under the `opm.tts` entry point
`ovos-tts-plugin-phoonnx`.

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

## Key methods:

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
