# Export to ONNX

<!-- agent-hint
covers: how to convert a trained checkpoint to ONNX, which files each engine writes, and how to check the result
source: phoonnx_train/export_onnx.py:cli, phoonnx_train/engines/*.py:export_onnx, phoonnx_train/engines/vits.py:write_sidecars, phoonnx_train/styletts2/export.py
generated: no
verify: python -m phoonnx_train.export_onnx --help lists each flag on the page; pytest tests/test_export_native_config.py; the files each engine writes: none, checked by reading phoonnx_train/engines/*.py:export_onnx
see: ../usage.md, ../alignment.md, ../vocoders.md
-->

This page shows how to convert a trained checkpoint to an ONNX voice and check it, for anyone who has a `.ckpt` from [training](training.md).

## Command

```bash
python -m phoonnx_train.export_onnx CHECKPOINT --config CONFIG [options]
```

`CHECKPOINT` is the **single positional argument**. There is no second positional for the
output file — the output location is set with `-o/--output-dir`.

| Option | Default | Description |
|---|---|---|
| `CHECKPOINT` | required | Path to the trained `.ckpt` (positional) |
| `-c, --config PATH` | required | The `config.json` produced by preprocess |
| `-o, --output-dir PATH` | current dir | Directory to write the ONNX model into |
| `--engine NAME` | `vits` | Architecture used for training (must match how it was trained) |
| `-t, --generate-tokens` | off | Also write `tokens.txt` (needed by some engines, e.g. sherpa) |
| `-p, --piper` | off | Also write a Piper-compatible `.json` |
| `-a, --add-phoneme-alignment` | off | Expose the phoneme duration tensor as an extra ONNX output (VITS only); see [Phoneme alignment](../alignment.md#exporting-a-model-with-alignment-support) |

`--engine` selects the export procedure and the metadata format of that engine. The output
file names depend on the engine. `<checkpoint>` is the checkpoint file name, and `<stem>` is
that name without `.ckpt`:

| Engine | Model files | Config sidecar |
|---|---|---|
| `vits` | `<checkpoint>.onnx`, for example `epoch=999-step=250000.ckpt.onnx` | `<stem>.json` |
| `yourtts` | `<checkpoint>.onnx` | `<stem>.json` |
| `glowtts`, `fastpitch`, `speedyspeech`, `mixer` | `<stem>.onnx` | `<stem>.json` |
| `matcha`, `optispeech` | `model.onnx` | `model.json` |
| `zipvoice` | `text_encoder.onnx` + `fm_decoder.onnx` | `config.json` |
| `styletts2` | `model.onnx` + `style_encoder.onnx` | `config.json` |

With `--generate-tokens`, the `vits`, `glowtts`, `fastpitch`, `speedyspeech`, `mixer`, `matcha`
and `optispeech` exports also write `<stem>.tokens.txt`. With `--piper`, they also write
`<stem>.piper.json`. `yourtts` writes `tokens.txt` and has no Piper config. `zipvoice` and
`styletts2` ignore both flags.

## Example

```bash
python -m phoonnx_train.export_onnx \
  train_out/runs/lightning_logs/version_0/checkpoints/epoch=999-step=250000.ckpt \
  --config train_out/config.json \
  --engine vits \
  --output-dir exported \
  --generate-tokens \
  --piper
```

**Expected output** in `exported/`:

```
exported/
├── epoch=999-step=250000.ckpt.onnx    # the model
├── epoch=999-step=250000.json         # phoonnx config, always written
├── epoch=999-step=250000.tokens.txt   # from --generate-tokens
└── epoch=999-step=250000.piper.json   # from --piper
```

> `--piper` is only meaningful for voices with `phoneme_type=espeak` and `alphabet=ipa`, since
> the Piper runtime expects that phonemization.

## Validating the exported voice

Load the exported model with the config written beside it and synthesize a sentence:

```python
import wave
from phoonnx.voice import TTSVoice

voice = TTSVoice.load("exported/epoch=999-step=250000.ckpt.onnx",
                      "exported/epoch=999-step=250000.json")
with wave.open("check.wav", "wb") as wav_file:
    voice.synthesize_wav("Testing the exported voice.", wav_file)
```

If `check.wav` plays back intelligible speech, the export is good. Common issues:

- **Silence or noise** — the `--engine` at export time did not match how the model was trained,
  or the wrong `config.json` was passed.
- **`ImportError: phonemizer 'espeak' needs a missing backing package`** — the config's
  `phoneme_type` names a backend whose [install extra](../installation.md#which-extra-a-voice-needs)
  is missing.
- **Two-stage engine sounds buzzy** — it needs its separate vocoder; see [Vocoders](../vocoders.md).

---

[Docs index](../index.md) · Previous: [Training reference](training.md) · Next: [Matcha-TTS](engines/matcha.md)
