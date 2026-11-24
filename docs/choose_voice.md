# Choose a voice

<!-- agent-hint
covers: how to find the voices for a language, which extra each voice needs, how large its download is, and how to hear it first
source: phoonnx/cli.py:voices_for_lang, phoonnx/model_manager.py:TTSModelManager.get_lang_voices, phoonnx/model_manager.py:TTSModelInfo.disk_size, phoonnx/voice_index/*.json, pyproject.toml
generated: no
verify: XDG_CACHE_HOME=.venv/verify-cache HF_HUB_OFFLINE=1 .venv/bin/python -c "import re, sys, tomllib; from phoonnx.model_manager import TTSModelManager; m = TTSModelManager(); m.merge_default_voices(); q = chr(96); names = lambda s: re.findall(q + '([^' + q + ']+)' + q, s); x = tomllib.load(open('pyproject.toml', 'rb'))['project']['optional-dependencies']; p = open('docs/choose_voice.md').read().split(chr(10) + '-' + '->', 1)[1]; sec = lambda a, b: [l.split('|')[1:-1] for l in p.split(a, 1)[1].split(b, 1)[0].splitlines() if l.startswith('| ' + q)]; prow = sec('## 2.', '## 3.'); erow = sec('## 3.', '## 4.'); ptypes = {n for r in prow for n in names(r[0])}; extras = [e for r in prow for e in names(r[1]) if e not in x]; engines = {names(r[0])[0] for r in erow}; wrong = [names(r[1])[0] for r in erow if m.voices.get(names(r[1])[0]) is None or m.voices[names(r[1])[0]].engine.value != names(r[0])[0]]; pt = {v.phoneme_type.name for v in m.all_voices if v.phoneme_type}; en = {v.engine.value for v in m.all_voices}; bad = [ptypes ^ pt, engines ^ en, extras, wrong]; print(bad); sys.exit(1 if any(bad) else 0)" && diff <(XDG_CACHE_HOME=.venv/verify-cache HF_HUB_OFFLINE=1 .venv/bin/phoonnx-voices update-cache | sed -E 's/(: |^Found )[0-9]+/\1.../') <(sed -n '/^\$ phoonnx-voices update-cache$/,/^\x60\x60\x60$/{//!p}' docs/choose_voice.md) && [ $(sed -n '/^\$ phoonnx-voices list-voices --lang pt-PT$/,/^\x60\x60\x60$/{//!p}' docs/choose_voice.md | wc -l) -gt 3 ] && ! grep -vxF -e '...' -f <(XDG_CACHE_HOME=.venv/verify-cache HF_HUB_OFFLINE=1 .venv/bin/phoonnx-voices list-voices --lang pt-PT | sed -E 's/(: |^Found )[0-9]+/\1.../') <(sed -n '/^\$ phoonnx-voices list-voices --lang pt-PT$/,/^\x60\x60\x60$/{//!p}' docs/choose_voice.md) && [ $(sed -n '/^\$ phoonnx-voices list-voices --lang en-AU$/,/^\x60\x60\x60$/{//!p}' docs/choose_voice.md | wc -l) -gt 3 ] && ! grep -vxF -e '...' -f <(XDG_CACHE_HOME=.venv/verify-cache HF_HUB_OFFLINE=1 .venv/bin/phoonnx-voices list-voices --lang en-AU | sed -E 's/(: |^Found )[0-9]+/\1.../') <(sed -n '/^\$ phoonnx-voices list-voices --lang en-AU$/,/^\x60\x60\x60$/{//!p}' docs/choose_voice.md) && XDG_CACHE_HOME=.venv/verify-cache HF_HUB_OFFLINE=1 .venv/bin/python -m doctest -o ELLIPSIS -o NORMALIZE_WHITESPACE docs/choose_voice.md
see: installation.md, voice_manager.md, precision.md, offline.md
-->

This page helps you pick a voice for your language. It shows how to list the voices, what each voice needs installed, how large its download is, and how to hear it before you decide.

The steps use the `phoonnx-voices` command and the base install from the [Quickstart](quickstart.md).

## 1. List the voices for your language

Fill the local catalog:

```
$ phoonnx-voices update-cache
Clearing local voice cache...
Loading bundled voice indexes...

Cache updated successfully!
Total voices available: ...
Total languages supported: ...
```

The last two lines give the number of voices and language tags in the bundled index.

Then list the voices for a language tag. The list starts like this:

```
$ phoonnx-voices list-voices --lang pt-PT

Found ... voices for language 'pt-PT':
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

Found ... voices for language 'en-AU':
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

The `Phoneme Type` line tells you which extra to install. The table lists each phoneme type in the index:

| Phoneme Type | Extra to install |
|---|---|
| `UNICODE` | none |
| `GRAPHEMES` | none |
| `ESPEAK` | `espeak` |
| `MISAKI_ZH`, `MISAKI_EN`, `MISAKI_JA` | `zh`, `en`, `ja` |
| `GRUUT` | the language extra, for example `de` |
| `COTOVIA` | `gl` |
| `AHOTTS` | `eu` |
| `F5TTS_CHAR`, `F5TTS_PINYIN` | none |
| `TUGAPHONE` | `pt` |
| `EPITRAN` | the language extra, for example `yo` |
| `GORUUT` | `all` |
| `MANTOQ` | `ar` |
| `VOSK` | none |
| `ORTHOGRAPHY2IPA` | `o2i` |
| `PHONIKUD` | `he` |
| `SHAMI` | none |

To count the voices of one type, filter the verbose list. For example, `phoonnx-voices list-voices --verbose | grep -c 'Phonemizer.ESPEAK$'` counts the eSpeak voices.

A voice that shows `None` names its phonemizer in its own config file.

A voice that needs a missing extra stops at load with an `ImportError`. The message names the package. [Installation](installation.md#errors-you-may-see) shows these errors and the fix for each one.

Some voices clone a speaker and need a reference clip. For those voices, `--verbose` adds the line `Requires reference: yes (needs a SynthesisConfig.reference / a style path)`. These are the Chatterbox voices and the zero-shot BSC StyleTTS2 voices, such as `bsc/es-styletts2`. To list them, run `phoonnx-voices list-voices --verbose | grep -B6 'Requires reference' | grep 'ID:'`. See [Voice cloning](cloning.md).

## 3. Check the download size

The engine of a voice sets the size of its download. The table gives the download of one voice of each engine family, read from the Hugging Face Hub. Voices of one family share some files, so a second voice of the same family downloads less.

| Engine | Example voice | Download |
|---|---|---|
| `transformers` | `facebook/mms-tts-abi-Abidji` | 114 MB |
| `omnivoice` | `omnivoice/aae` | 3204 MB |
| `supertonic` | `supertonic/F1/en` | 399 MB |
| `piper` | `OpenVoiceOS/pipertts_en-GB_miro` | 64 MB |
| `styletts2` | `ddatt/en-styletts2` | 330 MB |
| `pockettts` | `pockettts/en/alba` | 171 MB |
| `coqui` | `proxectonos/brais` | 132 MB |
| `glowtts` | `larynx/de-de-eva_k-glow_tts` | 120 MB |
| `chatterbox` | `chatterbox/base/en` | 1548 MB |
| `mimic3` | `mimic3/af_ZA/google-nwu_low` | 76 MB |
| `indic_parler` | `indic_parler/amit/as` | 5183 MB |
| `magpie` | `magpie/Aria/ar` | 1476 MB |
| `arktts` | `arktts/zortzi-maider/eu` | 1501 MB |
| `phoonnx` | `OpenVoiceOS/phoonnx_oc_miro_unicode` | 63 MB |
| `outetts` | `outetts/0.6B/en` | 2649 MB |
| `mosstts` | `mosstts/nano/zh` | 762 MB |
| `f5tts` | `f5tts/v1-base` | 1453 MB |
| `qwen3tts` | `qwen3tts/vivian/zh` | 4417 MB |
| `matcha` | `proxectonos/icia-matcha-phonemes` | 131 MB |
| `neutts` | `neutts/akiti/twi/abena` | 1307 MB |
| `orpheus` | `orpheus/canopylabs/en/tara` | 13339 MB |
| `mixertts` | `nipponjo/mixer-tts-ljspeech-80` | 61 MB |
| `fastpitch` | `nipponjo/tts-arabic-fastpitch` | 244 MB |
| `sparktts` | `sparktts/female/en` | 3432 MB |
| `llasa` | `llasa/HKUST/en/en_female_a` | 6275 MB |
| `optispeech` | `hf_community/mush42/optispeech-lightspeech-en-us-emily` | 40 MB |
| `vosk` | `vosk/tamara` | 171 MB |
| `shami` | `Tushe/shami-tts` | 113 MB |
| `rapido` | `edwixx/pt-BR-rapido` | 267 MB |

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

To see only the voices that the base install runs, filter on the phoneme type. This step
reads the catalog only, so it downloads nothing:

```python
>>> from phoonnx.model_manager import TTSModelManager
>>> manager = TTSModelManager()
>>> manager.merge_default_voices()
>>> base = [info.voice_id for info in manager.get_lang_voices("pt-PT")
...         if info.lang == "pt-PT" and info.phoneme_type and info.phoneme_type.value in ("unicode", "graphemes")]
>>> "OpenVoiceOS/phoonnx_pt-PT_miro_unicode" in base, "chatterbox/multilingual/pt" in base
(True, True)

```

`base` holds the `pt-PT` voices of the base install. Print it to see them all. The small `phoonnx` voice is one of them. The list also holds large voices from step 3, such as the Chatterbox voice, which also needs a reference clip.

---

[Docs index](index.md) · Previous: [Quickstart](quickstart.md) · Next: [Voice manager](voice_manager.md)
