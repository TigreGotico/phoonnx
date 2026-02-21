# OVOS Plugin

phoonnx ships with a TTS plugin for the [OpenVoiceOS](https://openvoiceos.com/) / Mycroft ecosystem via `PhoonnxTTSPlugin`.

## Installation

```bash
pip install phoonnx[ovos]
```

This installs `ovos-plugin-manager` and `ovos-utils` as additional dependencies.

## Configuration

In your OVOS/Mycroft `mycroft.conf` or skills config, set:

```json
{
  "tts": {
    "module": "phoonnx",
    "phoonnx": {
      "lang": "en-US",
      "voice": "en_US-lessac-medium"
    }
  }
}
```

If `voice` is omitted or set to `"default"`, the plugin selects the default voice for the configured language.

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
    "lang": "pt-PT",
    "voice": "pt_PT-tugão-medium",
})
```

### Methods inherited from OVOS TTS

The plugin integrates with the standard OVOS TTS interface. Key methods:

| Method | Description |
|--------|-------------|
| `get_tts(utterance, wav_file)` | Synthesize utterance and write to WAV file path |
| `get_default_voice(lang)` | Returns the default `TTSModelInfo` for a language |
| `refresh_voices(force=False)` | Refresh the voice catalog from the model manager |

## Supported Voices

All voices available through `TTSModelManager` are accessible to the plugin. Run the CLI to see available options:

```bash
phoonnx_cli.py update-cache
phoonnx_cli.py list-voices --lang en-US
```
