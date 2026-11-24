# Export to ONNX

<!-- agent-hint
covers: how to convert a trained checkpoint to ONNX, which files each engine writes, how to check the result, and how to export a voice for Piper or sherpa-onnx
source: phoonnx_train/export_onnx.py:cli, phoonnx/compat.py:export_piper, phoonnx/compat.py:export_sherpa, phoonnx_train/engines/*.py:export_onnx, phoonnx_train/engines/vits.py:write_sidecars, phoonnx_train/styletts2/export.py, phoonnx_train/rapido/export_onnx.py, phoonnx/cli.py:export_voice
generated: no
verify: ! comm -23 <(grep -oP -- '(?<![\w-])--[a-z][a-z0-9-]*' docs/training/export.md | sort -u) <({ .venv/bin/python -m phoonnx_train.export_onnx --help; .venv/bin/phoonnx-voices export --help; } | grep -oP -- '(?<![\w-])--[a-z][a-z0-9-]*' | sort -u) | grep . && .venv/bin/python -W ignore -c "import re, sys; from phoonnx.config import PhonemeType, Alphabet, _voice_phonemizer, voice_frontend; from phoonnx_train.engines import list_engines; p = open('docs/training/export.md').read(); fe = voice_frontend(_voice_phonemizer(PhonemeType.ESPEAK, Alphabet.IPA, keep_punctuation=True), 'pl'); ipa = str([''.join(fe.phonemize_flat('Wszyscy puścili się biegiem z miejsca wypadku.'))]); table = p.split(chr(10) + '| Engine |')[1].split(chr(10) + chr(10))[0]; names = set(re.findall(r'^\| ([^|]+) \|', table, re.M)[1:]); names = {n for cell in names for n in re.findall(r'\x60([a-z0-9-]+)\x60', cell)}; bad = ([] if ipa in p else [ipa]) + sorted(names - set(list_engines())); print(bad); sys.exit(bool(bad))"
see: ../usage.md, ../alignment.md, ../vocoders.md
-->

This page shows how to convert a trained checkpoint to an ONNX voice, check it, and export it for Piper or sherpa-onnx. Read it when you have a `.ckpt` file from [training](training.md).

## Example

Export a VITS checkpoint with its tokens file and a Piper config:

```bash
python -m phoonnx_train.export_onnx \
  train_out/runs/lightning_logs/version_0/checkpoints/epoch=999-step=250000.ckpt \
  --config train_out/config.json \
  --engine vits \
  --output-dir exported \
  --generate-tokens \
  --piper
```

The output in `exported/` is:

```
exported/
├── epoch=999-step=250000.ckpt.onnx    # the model
├── epoch=999-step=250000.json         # phoonnx config, always written
├── epoch=999-step=250000.tokens.txt   # from --generate-tokens
└── epoch=999-step=250000.piper.json   # from --piper
```

The Piper config is useful only for a voice with `phoneme_type=espeak` and `alphabet=ipa`, because the Piper runtime phonemizes with espeak-ng. The two extra files hold the ids of the training config as they are. To get a voice that Piper or sherpa-onnx reads with the same ids as phoonnx, use `phoonnx-voices export`, which checks the ids. See [Run a voice in Piper or sherpa-onnx](#run-a-voice-in-piper-or-sherpa-onnx).

## Check the exported voice

Load the exported model with the config beside it and synthesize a sentence:

```python
import wave
from phoonnx.voice import TTSVoice

voice = TTSVoice.load("exported/epoch=999-step=250000.ckpt.onnx",
                      "exported/epoch=999-step=250000.json")
with wave.open("check.wav", "wb") as wav_file:
    voice.synthesize_wav("Testing the exported voice.", wav_file)
```

If `check.wav` plays clear speech, the export is good. These are the common problems:

- Silence or noise. The `--engine` of the export is not the engine of the training, or the `config.json` is the wrong one.
- `ImportError: phonemizer 'espeak' needs a missing backing package`. The `phoneme_type` of the config names a backend whose [install extra](../installation.md#which-extra-a-voice-needs) is missing.
- A two-stage engine sounds buzzy. It needs its separate vocoder. See [Vocoders](../vocoders.md).

## Command reference

```bash
python -m phoonnx_train.export_onnx CHECKPOINT --config CONFIG [options]
```

`CHECKPOINT` is the only positional argument. `-o/--output-dir` sets the output folder.

| Option | Default | Description |
|---|---|---|
| `CHECKPOINT` | required | The path of the trained `.ckpt` file (positional) |
| `-c, --config PATH` | required | The `config.json` that preprocess wrote |
| `-o, --output-dir PATH` | the current directory | The folder for the ONNX model |
| `--engine NAME` | `vits` | The training engine. It must be the engine that trained the checkpoint |
| `-t, --generate-tokens` | off | Also write a `tokens.txt` in the `<symbol> <id>` format of sherpa-onnx |
| `-p, --piper` | off | Also write a Piper config |
| `-a, --add-phoneme-alignment` | off | Add the phoneme duration tensor as an extra ONNX output (VITS only). See [Phoneme alignment](../alignment.md#exporting-a-model-with-alignment-support) |

`--engine` selects the export procedure and the metadata format of that engine. The output file names depend on the engine. `<checkpoint>` is the file name of the checkpoint, and `<stem>` is that name without `.ckpt`:

| Engine | Model files | Config file |
|---|---|---|
| `vits` | `<checkpoint>.onnx`, for example `epoch=999-step=250000.ckpt.onnx` | `<stem>.json` |
| `yourtts` | `<checkpoint>.onnx` | `<stem>.json` |
| `glowtts`, `fastpitch`, `speedyspeech`, `mixer` | `<stem>.onnx` | `<stem>.json` |
| `matcha`, `optispeech` | `model.onnx` | `model.json` |
| `zipvoice` | `text_encoder.onnx`, `fm_decoder.onnx` | `config.json` |
| `styletts2` | `model.onnx`, `style_encoder.onnx` | `config.json` |
| `rapido` | `model.onnx`, `text_encoder.onnx`, `aligner.onnx`, `decoder.onnx` | `config.json` |

With `--generate-tokens`, the `vits`, `glowtts`, `fastpitch`, `speedyspeech`, `mixer`, `matcha` and `optispeech` exports also write `<stem>.tokens.txt`. With `--piper`, they also write `<stem>.piper.json`. `yourtts` writes `tokens.txt` and has no Piper config. `zipvoice`, `styletts2` and `rapido` ignore both flags.

A SuperTonic model has three checkpoints, and this command passes only one. Export it with `phoonnx_train.supertonic.export_onnx`, as the [SuperTonic](engines/supertonic.md) page shows.

## Run a voice in Piper or sherpa-onnx

`phoonnx-voices export` writes a voice for one of the two runtimes. It copies the model into a new folder and writes the config that the runtime reads. The source model does not change. The runtime gives the model the same ids that phoonnx gives it. When that is not possible, the command refuses the voice and gives the reason.

This example uses the Polish Piper voice `pl_PL-bass-high`, with `model.onnx` and `config.json` in the folder `pl_PL-bass-high`.

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

   The model in `pl-sherpa` holds the metadata that sherpa-onnx reads, such as `model_type`, `comment`, `voice`, `n_speakers` and `sample_rate`.

4. Load it with sherpa-onnx. sherpa-onnx 2.x has no espeak-ng, so it reads phonemes: one list of code points for each sentence. This example gets the phonemes from phoonnx:

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

The same voice exports for Piper, because Piper joins `a` and `ɪ` into `aɪ` through the `vowel_clusters` that the export writes. The command refuses:

- A phonemizer other than espeak-ng with the IPA alphabet. Piper and sherpa-onnx phonemize with espeak-ng only.
- A voice trained without punctuation, with a text casing other than `ignore`, or with added diacritics. Piper and sherpa-onnx always read the punctuation, and read the text as espeak-ng gives it.
- For Piper only, an Arabic voice. Piper adds diacritics to Arabic text before espeak-ng reads it.
- A tokenizer layout other than `^ _ (p _)* $`, for example a word blank or another blank token.
- A symbol with more than one id.
- For sherpa-onnx only, a symbol of more than one code point after NFD decomposition.
- A graph whose inputs are not `input`, `input_lengths` and `scales`, plus `sid` when the voice has more than one speaker.

---

[Docs index](../index.md) · Previous: [Training reference](training.md) · Next: [Matcha-TTS](engines/matcha.md)
