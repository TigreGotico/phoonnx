# Export to ONNX

<!-- agent-hint
covers: how to convert a trained checkpoint to ONNX, which files each engine writes, how to check the result, and how to export a voice for Piper or sherpa-onnx
source: phoonnx_train/export_onnx.py:cli, phoonnx/compat.py:export_piper, phoonnx/compat.py:export_sherpa, phoonnx_train/engines/*.py:export_onnx, phoonnx_train/engines/vits.py:write_sidecars, phoonnx_train/styletts2/export.py
generated: no
verify: python -m phoonnx_train.export_onnx --help lists each flag on the page; pytest tests/test_export_native_config.py tests/test_export_piper_sherpa.py; the Piper and sherpa-onnx outputs: a run on pl_PL-bass-high with piper-tts 1.8.0 and sherpa-onnx 2.0.2; the files each engine writes: none, checked by reading phoonnx_train/engines/*.py:export_onnx
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
| `-t, --generate-tokens` | off | Also write `tokens.txt` in the `<symbol> <id>` format of sherpa-onnx |
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
> the Piper runtime expects that phonemization. The two side-cars hold the ids of the training
> config as they are. To get a voice that Piper or sherpa-onnx reads with the same ids as phoonnx,
> use `phoonnx-voices export`, which checks the ids. See
> [Run a voice in Piper or sherpa-onnx](#run-a-voice-in-piper-or-sherpa-onnx).

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

## Run a voice in Piper or sherpa-onnx

`phoonnx-voices export` writes a voice for one of the two runtimes. It copies the model into a new folder and writes the config that the runtime reads. The source model does not change. The ids that the runtime feeds the model are the ids that phoonnx feeds it. The command refuses a voice when that is not possible, and gives the reason.

This example uses the Polish voice `pl_PL-bass-high`, with `model.onnx` and `config.json` in the folder `pl_PL-bass-high`.

1. Export the voice for Piper:

   ```
   $ phoonnx-voices export pl_PL-bass-high/model.onnx pl-piper --to piper --config pl_PL-bass-high/config.json
   Wrote pl-piper/model.onnx
   Wrote pl-piper/model.onnx.json
   ```

2. Load it with the `piper-tts` package:

   ```python
   import wave

   from piper import PiperVoice

   voice = PiperVoice.load("pl-piper/model.onnx")
   with wave.open("piper.wav", "wb") as wav:
       voice.synthesize_wav("Wszyscy puścili się biegiem z miejsca wypadku.", wav)
   print(voice.config.espeak_voice, voice.config.sample_rate)
   ```

   ```
   pl 22050
   ```

3. Export the voice for sherpa-onnx. This step needs the `onnx` package from the `streaming` extra:

   ```
   $ phoonnx-voices export pl_PL-bass-high/model.onnx pl-sherpa --to sherpa --config pl_PL-bass-high/config.json
   Wrote pl-sherpa/model.onnx
   Wrote pl-sherpa/tokens.txt
   ```

   The model in `pl-sherpa` holds the metadata that sherpa-onnx reads (`model_type`, `comment`, `voice`, `n_speakers`, `sample_rate` and others). The source model has no change.

4. Load it with sherpa-onnx. sherpa-onnx 2.x has no espeak-ng, so it reads phonemes, one list of code points for each sentence. This example gets them from phoonnx:

   ```python
   import unicodedata

   import sherpa_onnx
   from phoonnx.voice import TTSVoice

   text = "Wszyscy puścili się biegiem z miejsca wypadku."
   sentences = ["".join(u.symbols) for u in TTSVoice.load("pl-piper/model.onnx").prepare(text)]
   print(sentences)

   vits = sherpa_onnx.OfflineTtsVitsModelConfig(model="pl-sherpa/model.onnx", tokens="pl-sherpa/tokens.txt")
   tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(vits=vits)))
   config = sherpa_onnx.GenerationConfig()
   config.phoneme_codepoints = [[ord(c) for c in unicodedata.normalize("NFD", s)] for s in sentences]
   audio = tts.generate("", config)
   sherpa_onnx.write_wave("sherpa.wav", audio.samples, audio.sample_rate)
   print(audio.sample_rate)
   ```

   ```
   ['fʃˈɨstsɨ puɕtɕˈili ɕɛ bjˈɛɡʲɛm z mʲˈɛjstsa vɨpˈatku.']
   22050
   ```

   sherpa-onnx 1.x has espeak-ng. To use it, give the folder of its `espeak-ng-data` as `data_dir` and pass the text to `tts.generate`.

The command refuses a voice that a runtime cannot read with the same ids:

```
$ phoonnx-voices export de_DE-thorsten-medium/model.onnx thorsten-sherpa --to sherpa --config de_DE-thorsten-medium/config.json
Error: sherpa-onnx reads one code point per symbol; this voice reads 'aɪ' (2 code points) as the ids [161]
```

The same voice exports for Piper, because Piper merges `a` and `ɪ` into `aɪ` through the `vowel_clusters` that the export writes. The refusals are these:

- A phonemizer other than espeak-ng with the IPA alphabet. Piper and sherpa-onnx phonemize with espeak-ng only.
- A voice trained without punctuation, with a text casing other than `ignore`, or with added diacritics. Piper and sherpa-onnx always read punctuation and the text as espeak-ng gives it.
- For Piper only, an Arabic voice. Piper adds diacritics to Arabic text before espeak-ng reads it.
- A tokenizer layout other than `^ _ (p _)* $`, for example a word blank or another blank token.
- A symbol with more than one id.
- For sherpa-onnx only, a symbol of more than one code point after NFD decomposition.
- A graph whose inputs are not `input`, `input_lengths` and `scales`, plus `sid` when the voice has more than one speaker.

---

[Docs index](../index.md) · Previous: [Training reference](training.md) · Next: [Matcha-TTS](engines/matcha.md)
