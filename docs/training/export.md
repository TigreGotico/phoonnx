# Export to ONNX

This page covers converting a trained checkpoint to an ONNX voice and confirming it works. It
is for anyone who has a `.ckpt` from [training](training.md) and wants a deployable voice.

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

The export is engine-aware: passing `--engine` uses that engine's export procedure and
metadata format. The output filenames depend on the engine — VITS writes `model.onnx`, while
two-graph engines write their pair (ZipVoice: `text_encoder.onnx` + `fm_decoder.onnx`;
StyleTTS2: `model.onnx` + `style_encoder.onnx`).

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

**Expected output** in `exported/`: `model.onnx`, plus `tokens.txt` and a Piper `.json` when
those flags are set.

> `--piper` is only meaningful for voices with `phoneme_type=espeak` and `alphabet=ipa`, since
> the Piper runtime expects that phonemization.

## Validating the exported voice

Load the exported model with the same `config.json` and synthesize a sentence:

```python
import wave
from phoonnx.voice import TTSVoice

voice = TTSVoice.load("exported/model.onnx", "train_out/config.json")
with wave.open("check.wav", "wb") as wav_file:
    voice.synthesize_wav("Testing the exported voice.", wav_file)
```

If `check.wav` plays back intelligible speech, the export is good. Common issues:

- **Silence or noise** — the `--engine` at export time did not match how the model was trained,
  or the wrong `config.json` was passed.
- **`invalid phonemizer` at load** — the config's `phoneme_type`/`alphabet` names a backend
  whose [install extra](../installation.md#language-extras) is missing.
- **Two-stage engine sounds buzzy** — it needs its separate vocoder; see [Vocoders](../vocoders.md).

Ship the exported voice through the [OVOS plugin](../ovos_plugin.md) or load it directly as in
[Usage](../usage.md).
