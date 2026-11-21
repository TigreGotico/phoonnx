# Vocoders

<!-- agent-hint
covers: which vocoder types exist, how a voice selects one, and how to add a vocoder type
source: phoonnx/engines/vocoders/__init__.py, phoonnx/engines/vocoders/base.py:BaseVocoder, phoonnx/model_manager.py:TTSModelInfo
generated: no
verify: pytest tests/test_vocoders.py
see: engines.md, training/engines/glowtts.md, training/engines/matcha.md
-->

This page describes the vocoder registry in `phoonnx/engines/vocoders/`, for developers who pair a mel-producing acoustic model with a vocoder or add a vocoder type.

Two-stage acoustic models — **GlowTTS**, **Matcha-TTS**, **OptiSpeech** (when it
emits a mel) — produce a **mel spectrogram** and rely on a separate **vocoder** to
turn that mel into a waveform. `phoonnx` keeps vocoders behind a small registry so
any two-stage adapter can drive any vocoder without knowing its internals.

```
text ──[acoustic model]──▶ mel ──[vocoder]──▶ waveform
```

## Vocoder families

| `vocoder_type` | Class | ONNX output | Notes |
|---|---|---|---|
| `vocos` | `VocosVocoder` | STFT mag + real + imag (3) | inverse-STFT reconstruction; supports `denoise` |
| `wavenext` | `WavenextVocoder` | waveform (1) | Vocos with a trained ISTFT-replacement layer |
| `hifigan` | `HiFiGANVocoder` | waveform (1) | GAN vocoder |
| `melgan` | `HiFiGANVocoder` (alias) | waveform (1) | (multiband-)MelGAN — same 1-output mel→audio contract |
| `raw` | `RawWaveformVocoder` | waveform (1) | generic single-output mel→audio |
| `griffinlim` | `GriffinLimVocoder` | — (no model) | parametric, no ONNX; works with any mel config; needs `librosa`, which the `train` and `all` extras install |

All live in `phoonnx/engines/vocoders/`.

## How a voice selects its vocoder

A two-stage voice declares its vocoder in the **voice index** entry; the model
manager resolves it at load time and hands it to the adapter via `engine_params`:

```json
{
  "voice_id": "coqui-tr-common-voice-glow-tts",
  "engine": "glowtts",
  "model_url": ".../model.onnx",
  "config_url": ".../config.json",
  "vocoder_url": ".../phoonnx-vocoders/tr-common-voice-hifigan/model.onnx",
  "vocoder_config_url": ".../tr-common-voice-hifigan/vocoder.json",
  "vocoder_type": "hifigan"
}
```

- **`vocoder_url`** — the vocoder ONNX (downloaded alongside the model). Omit for
  a parametric vocoder (Griffin-Lim).
- **`vocoder_type`** — selects the implementation explicitly. If absent, the
  vocoder is **auto-detected** from the ONNX output layout: 3+ outputs → Vocos,
  a single output → a raw-waveform vocoder.
- **`vocoder_config_url`** — a small `vocoder.json` of vocoder parameters
  (sample rate, mel settings, preprocessing flags — see below).

Nothing in user code changes: `voice.synthesize(...)` downloads the model + the
linked vocoder and produces audio.

## Mel preprocessing (config-driven flags)

Different vocoders expect the mel in different conventions. Rather than bake that
into each ONNX, a converted vocoder **declares its input convention with flags**
in its `vocoder.json`, and `BaseVocoder._preprocess_mel` applies them before the
ONNX runs. Flags are **opt-in** — a vocoder that needs nothing is untouched.

| Flag | Params | Effect |
|---|---|---|
| `stats_norm` | `mel_mean`, `mel_std` (per-channel) | standard-scale the mel `(mel − mean) / std` — required by Coqui stats-normalized vocoders (e.g. multiband-MelGAN), whose `scale_stats.npy` differs from the acoustic model's dB-mel domain |

Example `vocoder.json` for a stats-normalized MelGAN:

```json
{ "vocoder_type": "melgan", "sample_rate": 22050, "stats_norm": true,
  "mel_mean": [ ... 80 floats ... ], "mel_std": [ ... 80 floats ... ] }
```

Add a step the same way: a flag in the config and a branch in `_preprocess_mel`. The
flag ships with the data, not the code.

## Using a vocoder directly

```python
from phoonnx.engines.vocoders import build_vocoder, list_vocoders

print(list_vocoders())  # ['griffinlim', 'wavenext', 'hifigan', 'melgan', 'vocos', 'raw']

voc = build_vocoder(model_path="hifigan.onnx", vocoder_type="hifigan")
audio = voc.mel_to_audio(mel)               # mel [B, n_mels, T] -> waveform [N]

# parametric Griffin-Lim — no model file, all params in config
gl = build_vocoder(vocoder_type="griffinlim",
                   config={"sample_rate": 22050, "n_fft": 1024, "hop_length": 256, "num_mels": 80})
```

`build_vocoder(vocoder_type=..., config=...)` uses the named vocoder. With no type, it
probes the ONNX and picks the first match. When nothing matches, it logs a warning and
uses Vocos.

## Replacing / swapping a vocoder

Because the vocoder is just a link in the index entry, you can swap it without
touching the acoustic model:

- **Point at a different ONNX** — change `vocoder_url` / `vocoder_type` to another
  vocoder trained on the **same mel config** (sr, fft, hop, n_mels, fmin/fmax). A
  mismatched mel config produces garbage; if the only gap is normalization, add a
  `stats_norm` preprocessing flag, or an equivalent one.
- **Fall back to Griffin-Lim** — drop `vocoder_url`, set `vocoder_type:
  "griffinlim"`, and point `vocoder_config_url` at a `vocoder.json` with the mel
  params of the acoustic model. It needs no neural vocoder. The audio sounds robotic.
- **At runtime** — construct the adapter with an explicit vocoder instance:
  `GlowTTSAdapter(vocoder=build_vocoder(...))`.

## Adding a vocoder type

Subclass `BaseVocoder`, implement `mel_to_audio`, and register it:

```python
from phoonnx.engines.vocoders.base import BaseVocoder
from phoonnx.engines.vocoders import register_vocoder

class MyVocoder(BaseVocoder):
    name = "myvoc"
    def mel_to_audio(self, mel, denoise=False):
        mel = self._preprocess_mel(mel)             # honour config flags
        return self.session.run(None, {self.input_name: mel})[0].squeeze()

register_vocoder("myvoc", MyVocoder, detect_priority=35)
```

`detect_priority` orders auto-detection (lower runs first); set a high value (or
return `False` from `detect`) for vocoders that should only be chosen explicitly.

## Converting vocoders (provenance)

The ONNX vocoders under `Phoonnx/phoonnx-vocoders` are produced by standalone
exporters that vendor only the pure-torch generator code (no `coqui-tts` / Larynx
package dependency):

- **HiFi-GAN** — coqui `HifiganGenerator` → ONNX (`mel → audio`).
- **Multiband-MelGAN** — coqui `MultibandMelganGenerator` + PQMF synthesis → ONNX;
  the `scale_stats.npy` becomes the `stats_norm` flag in `vocoder.json`.
- **Larynx HiFi-GAN** — mirrored as-is.

A vocoder only works with an acoustic model whose **mel features match** it
(sample rate, FFT/hop, n_mels, fmin/fmax, and normalization). Where no matched
neural vocoder exists, the voice uses Griffin-Lim.

---

[Docs index](index.md) · Previous: [Engine architecture](engines.md) · Next: [Python API](api.md)
