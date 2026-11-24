# Choose a voice

<!-- agent-hint
covers: how to find the voices for a language, which extra each voice needs, how large its download is, and how to hear it first
source: phoonnx/cli.py:voices_for_lang, phoonnx/model_manager.py:TTSModelManager.get_lang_voices, phoonnx/voice_index/*.json, pyproject.toml [project.optional-dependencies]
generated: no
verify: phoonnx-voices list-voices --lang pt-PT; phoonnx-voices list-voices --verbose | grep -A6 'ID:          OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone'
see: installation.md, voice_manager.md, precision.md, offline.md
-->

This page helps you pick a voice for your language. It shows how to list the voices, what each voice needs installed, how large its download is, and how to hear it before you decide.

The steps use the `phoonnx-voices` command and the base install from the [Quickstart](quickstart.md).

## 1. List the voices for your language

Fill the local catalog, then list the voices for a language tag:

```
$ phoonnx-voices update-cache
Clearing local voice cache...
Loading bundled voice indexes...

Cache updated successfully!
Total voices available: 3191
Total languages supported: 1817
$ phoonnx-voices list-voices --lang pt-PT

Found 51 voices for language 'pt-PT':
* OpenVoiceOS/phoonnx_pt-PT_miro_unicode (pt-PT)
* OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone (pt-PT)
* OpenVoiceOS/phoonnx_pt-PT_dii_tugaphone (pt-PT)
* OpenVoiceOS/pipertts_pt-PT_miro (pt-PT)
* OpenVoiceOS/pipertts_pt-PT_dii (pt-PT)
* piper/pt_PT-tugão-medium (pt-PT)
...
```

The tag decides which voices you see:

| Tag | Result |
|---|---|
| A tag with a region, `pt-PT` | The voices of that region only. |
| A tag with no region, `pt` | The voices of every region, nearest first. |
| A region that has no voice, `en-AU` | The voices of the nearest regions, nearest first. |
| An Arabic tag with a region, `ar-EG` | The Arabic voices, nearest first. A dialect voice of that region, such as Egyptian Arabic (`arz-EG`), ranks as a voice of that region. |
| `ar` | The Arabic voices of every region, nearest first. A dialect voice (`arz-EG`) answers only its own region or its own dialect tag. |

The `en-AU` list starts with British English, because the default region of English is `en-GB`:

```
$ phoonnx-voices list-voices --lang en-AU

Found 256 voices for language 'en-AU':
* OpenVoiceOS/pipertts_en-GB_miro (en-GB)
* OpenVoiceOS/pipertts_en-GB_dii (en-GB)
* facebook/mms-tts-eng-English (en)
...
```

Voices at the same distance keep the order of the bundled index files. The OpenVoiceOS voices come first.

## 2. Read what the voice needs

Add `--verbose` to see the engine and the phonemizer of each voice:

```
$ phoonnx-voices list-voices --verbose | grep -A6 'ID:          OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone'
  ID:          OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone
  Language:    pt-PT
  Engine:      phoonnx
  Phoneme Type: Phonemizer.TUGAPHONE
  Model URL:   https://huggingface.co/Phoonnx/phoonnx_pt-PT_miro_tugaphone/resolve/b081738cde545e32a8c27fd780c5a2d948f1fb1f/miro_pt-PT.onnx
  Config URL:  https://huggingface.co/Phoonnx/phoonnx_pt-PT_miro_tugaphone/resolve/b081738cde545e32a8c27fd780c5a2d948f1fb1f/miro_pt-PT.json
----------------------------------------
```

The `Phoneme Type` line tells you which extra to install. Most voices in the index need no extra:

| Phoneme Type | Voices in the index | Extra to install |
|---|---|---|
| `UNICODE` | 1331 | none |
| `GRAPHEMES` | 1211 | none |
| `ESPEAK` | 341 | `espeak` |
| `MISAKI_ZH`, `MISAKI_EN`, `MISAKI_JA` | 108, 43, 5 | `zh`, `en`, `ja` |
| `GRUUT` | 78 | the language extra, for example `de` |
| `COTOVIA` | 19 | `gl` |
| `AHOTTS` | 12 | `eu` |
| `F5TTS_CHAR`, `F5TTS_PINYIN` | 11, 5 | none |
| `TUGAPHONE` | 6 | `pt` |
| `EPITRAN` | 4 | the language extra, for example `yo` |
| `GORUUT` | 4 | `all` |
| `MANTOQ` | 3 | `ar` |
| `VOSK` | 3 | none |
| `PHONIKUD` | 2 | `he` |
| `SHAMI` | 1 | none |

Two voices show `None`. Each one names its phonemizer in its own config file.

A voice that needs a missing extra stops at load with an `ImportError`. The message names the package. [Installation](installation.md#errors-you-may-see) shows these errors and the fix for each one.

Some voices clone a speaker and need a reference clip. For those voices, `--verbose` adds the line `Requires reference: yes (needs a SynthesisConfig.reference / a style path)`. These are the Chatterbox voices and some StyleTTS2 voices. See [Voice cloning](cloning.md).

## 3. Check the download size

The engine of a voice sets the size of its download. The table gives the download of one voice of each engine family, read from the Hugging Face Hub. Voices of one family share some files, so a second voice of the same family downloads less.

| Engine | Voices in the index | Example voice | Download |
|---|---|---|---|
| `transformers` | 1166 | `facebook/mms-tts-abi-Abidji` | 114 MB |
| `omnivoice` | 646 | `omnivoice/aae` | 3204 MB |
| `supertonic` | 310 | `supertonic/F1/en` | 399 MB |
| `piper` | 244 | `OpenVoiceOS/pipertts_en-GB_miro` | 64 MB |
| `styletts2` | 228 | `ddatt/en-styletts2` | 330 MB |
| `pockettts` | 156 | `pockettts/en/alba` | 171 MB |
| `coqui` | 64 | `proxectonos/brais` | 132 MB |
| `glowtts` | 58 | `larynx/de-de-eva_k-glow_tts` | 120 MB |
| `chatterbox` | 42 | `chatterbox/base/en` | 1548 MB |
| `mimic3` | 40 | `mimic3/af_ZA/google-nwu_low` | 76 MB |
| `indic_parler` | 33 | `indic_parler/amit/as` | 5183 MB |
| `magpie` | 30 | `magpie/Aria/ar` | 1476 MB |
| `arktts` | 25 | `arktts/zortzi-maider/eu` | 1501 MB |
| `phoonnx` | 24 | `OpenVoiceOS/phoonnx_oc_miro_unicode` | 63 MB |
| `outetts` | 23 | `outetts/0.6B/en` | 2649 MB |
| `mosstts` | 19 | `mosstts/nano/zh` | 762 MB |
| `f5tts` | 16 | `f5tts/v1-base` | 1453 MB |
| `qwen3tts` | 15 | `qwen3tts/vivian/zh` | 4417 MB |
| `matcha` | 13 | `proxectonos/icia-matcha-phonemes` | 131 MB |
| `neutts` | 9 | `neutts/akiti/twi/abena` | 1307 MB |
| `orpheus` | 8 | `orpheus/canopylabs/en/tara` | 13339 MB |
| `mixertts` | 5 | `nipponjo/mixer-tts-ljspeech-80` | 61 MB |
| `fastpitch` | 4 | `nipponjo/tts-arabic-fastpitch` | 244 MB |
| `sparktts` | 4 | `sparktts/female/en` | 3432 MB |
| `llasa` | 4 | `llasa/HKUST/en/en_female_a` | 6275 MB |
| `optispeech` | 3 | `hf_community/mush42/optispeech-lightspeech-en-us-emily` | 40 MB |
| `vosk` | 3 | `vosk/tamara` | 171 MB |
| `shami` | 1 | `Tushe/shami-tts` | 113 MB |
| `rapido` | 1 | `edwixx/pt-BR-rapido` | 267 MB |

The download size is also a guide to the memory that the loaded voice takes, because the weights fill most of both. Some large voices have smaller int8 files. See [Precision variants](precision.md).

## 4. Hear the voices before you decide

Speak the same sentence with each voice on your short list. This script writes one WAV file for each voice and prints what it loaded. The second voice uses eSpeak, so install the `espeak` extra first.

```python
import wave
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.load()
manager.merge_default_voices()

sentence = "Olá, isto é um teste."
for voice_id in ["OpenVoiceOS/phoonnx_pt-PT_miro_unicode", "OpenVoiceOS/pipertts_pt-PT_miro"]:
    info = manager.voices[voice_id]
    voice = info.load()
    out = voice_id.split("/")[-1] + ".wav"
    with wave.open(out, "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file)
    print(f"{out}: {info.engine.value}, {info.phoneme_type.value}, {info.disk_size() / 1e6:.0f} MB")
```

```
phoonnx_pt-PT_miro_unicode.wav: phoonnx, unicode, 63 MB
pipertts_pt-PT_miro.wav: piper, espeak, 64 MB
```

The first load of each voice downloads it. Play the WAV files and keep the voice you like.

To see only the voices that the base install runs, filter on the phoneme type:

```python
for info in manager.get_lang_voices("pt-PT"):
    if info.lang == "pt-PT" and info.phoneme_type and info.phoneme_type.value in ("unicode", "graphemes"):
        print(info.voice_id, info.engine.value)
```

```
OpenVoiceOS/phoonnx_pt-PT_miro_unicode phoonnx
chatterbox/multilingual/pt chatterbox
outetts/1B/pt outetts
```

The list shows that one small voice serves `pt-PT` on the base install. The other two are large voices from step 3.

---

[Docs index](index.md) · Previous: [Quickstart](quickstart.md) · Next: [Voice manager](voice_manager.md)
