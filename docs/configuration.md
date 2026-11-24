# Configuration Reference

<!-- agent-hint
covers: every VoiceConfig and SynthesisConfig field, the native config.json schema, phonemizer provenance, the Engine enum, execution providers and precision
source: phoonnx/config.py:VoiceConfig, phoonnx/config.py:SynthesisConfig, phoonnx/config.py:Engine, phoonnx/compat.py:LOADERS, phoonnx/providers.py, phoonnx/model_manager.py:PRECISIONS
generated: no
verify: .venv/bin/python -W ignore -c "import re,sys,dataclasses as d; from phoonnx.config import VoiceConfig,SynthesisConfig,Engine,AudioPost,SpeakerReference,_1X_NAMES; from scriptconv.frontend import Stages; from phoonnx.compat import LOADERS; p=open('docs/configuration.md').read(); q=chr(96); n=chr(10); rows=lambda h: [[re.findall(q+'([^'+q+']*)'+q,c) for c in re.split('(?<!'+chr(92)*2+')[|]',l)[1:4]] for l in p.split(n+h,1)[1].split(n+'#',1)[0].splitlines() if l.startswith('| '+q)]; vc=VoiceConfig.from_dict({'phoonnx_version':'1.0','phoneme_type':'graphemes','alphabet':'unicode','phoneme_id_map':{'_':[0],'^':[1],'$':[2],' ':[3]}}); sc=SynthesisConfig(); T=[('### VoiceConfig fields',vc,VoiceConfig),('### SynthesisConfig fields',sc,SynthesisConfig),('### stages fields',Stages(),Stages),('### reference fields',SpeakerReference(),SpeakerReference),('### audio fields',AudioPost(),AudioPost)]; same=lambda t,v: t.strip(chr(34))==str(v) or t==getattr(v,'name',None); bad=[]; [bad.append(h) for h,o,C in T if sorted(r[0][0] for r in rows(h))!=sorted(f.name for f in d.fields(C))]; [bad.append(h+' '+r[0][0]) for h,o,C in T for r in rows(h) if r[2] and not same(r[2][0],getattr(o,r[0][0]))]; bad+=[] if {r[0][0]:r[1][0] for r in rows('### 1.x names')}==_1X_NAMES else ['1.x names']; eng=[r[0][0] for r in rows('## Engine Enum') if r[0][0]!='Engine']; bad+=[] if sorted(eng)==sorted(e.value for e in Engine) else ['engine table']; ld=[l.__name__ for l in LOADERS]; bad+=[x for x in re.findall(q+'([A-Za-z0-9]+Loader)'+q,p) if x not in ld]; print(bad); sys.exit(1 if bad else 0)"
see: usage.md, phonemizers.md, precision.md
-->

This page is the reference for advanced users. It lists each `VoiceConfig` and `SynthesisConfig` field, the native `config.json` format, engine detection, execution providers and precision. For a walk-through, read [Usage](usage.md).

## VoiceConfig

`VoiceConfig` holds the configuration of a model, read from the `config.json` of the voice.

```python
from phoonnx.config import VoiceConfig
```

### VoiceConfig fields

The default is the value that `VoiceConfig.from_dict` gives when a native config does not have the key.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_symbols` | `int` | `256` | Number of phoneme/token IDs in the vocabulary |
| `num_speakers` | `int` | `1` | Number of speakers (1 for single-speaker) |
| `num_langs` | `int` | `1` | Number of languages (1 for monolingual) |
| `sample_rate` | `int` | `16000` | Output audio sample rate in Hz |
| `lang_code` | `str` \| `None` | `"und"` | BCP-47 language code (e.g. `"en-US"`) |
| `phonemizer_lang` | `str` \| `None` | `None` | The language the phonemizer reads when it differs from `lang_code`. Text normalisation keeps `lang_code`. A Piper config sets it from `espeak.voice`, so an en_US voice with the eSpeak voice `en` reads dates as en-US and gets the phonemes of `en` |
| `phoneme_type` | `PhonemeType` | — | Phonemizer backend |
| `alphabet` | `Alphabet` | — | Phoneme representation |
| `engine` | `Engine` | `PHOONNX` | The file format or the architecture of the model. See [Engine Enum](#engine-enum) |
| `noise_scale` | `float` \| `None` | `None` | Generator noise scale from `inference.noise_scale`. `None` gives the engine's default |
| `length_scale` | `float` \| `None` | `None` | Phoneme length scale from `inference.length_scale`. `None` gives the engine's default |
| `noise_w_scale` | `float` \| `None` | `None` | Phoneme width noise scale from `inference.noise_w`. `None` gives the engine's default |
| `hop_length` | `int` | `256` | Model frames → audio samples factor for [phoneme alignment](alignment.md) |
| `add_diacritics` | `bool` | Depends on the config format, see the description | Add diacritics before phonemization. A native config reads it from `inference.add_diacritics`. Without the key, an Arabic `lang_code` gives `True`. A Piper config ignores the key, and an Arabic `lang_code` gives `True`. A Chatterbox config gives `True` for Arabic and Hebrew. A config with no format marker reads the key and defaults to `False`. Every other format gives `False` |
| `diacritizer_model` | `str` \| `None` | `None` | Diacritizer model name, read from the config's `inference` block. `None` lets scriptconv pick the default of each backend, `rawi-ensemble` for Arabic |
| `phonemizer_model` | `str` \| `None` | `None` | Model path/id or variant selector for phonemizers that take one (ByT5, AhoTTS, Cotovia, arbtok) |
| `keep_punctuation` | `bool` | `False` (`True` for Piper configs) | The model was trained with clause punctuation (`, . ? ! : ;`), so synthesis keeps it. A phoonnx config without the key, or a Piper side-car phoonnx exported (`piper_version` starting with `phoonnx`, or an `alphabet` key), was trained without it |
| `text_casing` | `str` | `"ignore"` | The casing training applied to the text: `lower`, `upper`, `casefold` or `ignore`. Synthesis applies the same casing after normalisation. A backend that normalises its own text, such as tugaphone, euskaphone, dizione or tltk, receives no casing, and scriptconv logs a warning. A config without the key keeps the text as written |
| `phonemizer_identity` | `dict` \| `None` | `None` | The phonemizer that made the training phonemes. See [Phonemizer provenance](#phonemizer-provenance) |
| `speaker_id_map` | `dict` | `{}` | Maps speaker names to integer IDs |
| `default_speaker_id` | `int` \| `None` | `None` | The speaker of a multi-speaker voice when the call names none. `None` is speaker 0 |
| `lang_id_map` | `dict` | `{}` | Maps language codes to integer IDs (multi-lang voices) |
| `lang_tokens` | `dict` | `{}` | Optional BCP-47 → model language-token override (dialect models) |
| `engine_params` | `dict` | `{}` | Adapter-specific parameters (vocoder/speaker-encoder/style paths, etc.) |
| `tokenizer` | `TTSTokenizer` \| `None` | — | The tokenizer that `from_dict` builds from the vocabulary |
| `blank_at_start` | `bool` | `True` | Insert a blank token before the sequence |
| `blank_at_end` | `bool` | `True` | Insert a blank token after the sequence |
| `pad_token` | `str` \| `None` | `_` | Padding token |
| `blank_token` | `str` \| `None` | `_` | Blank token inserted between tokens |
| `bos_token` | `str` \| `None` | `^` | Beginning-of-sequence token |
| `eos_token` | `str` \| `None` | `$` | End-of-sequence token |
| `word_sep_token` | `str` \| `None` | `" "` | Token inserted between words. `from_dict` reads `word_sep_token`, then `blank_word` |
| `blank_between` | `BlankBetween` | `TOKENS_AND_WORDS` | Where blank tokens are inserted |

### Loading from a Config File

`TTSVoice.load(model_path, config_path)` reads the config file and builds the `VoiceConfig`. Without `config_path`, it reads `<model_path>.json`, such as `model.onnx.json`. To build a `VoiceConfig` from a dict:

```python
config = VoiceConfig.from_dict(my_config_dict)
```

`TTSVoice.load()` raises `phoonnx.errors.VoiceLoadError` when the config file does not exist. It raises the same error for a VITS graph whose vocabulary uses an id that is not below `num_symbols`.

### Engine Auto-Detection

`VoiceConfig.from_dict()` gives the config to the loaders of `phoonnx.compat.LOADERS`, in this order, and uses the first loader that accepts it:

1. `ChatterboxLoader` and the raw-text loaders accept a config whose `engine` is `chatterbox`, `llasa`, `neutts`, `orpheus`, `mosstts`, `supertonic`, `pockettts`, `sparktts`, `indic_parler`, `omnivoice`, `magpie`, `arktts`, `qwen3tts` or `outetts`. These engines read text with their own tokenizer, and use no phonemizer. Chatterbox needs a BPE `tokenizer.json`.
2. `PhoonnxLoader` accepts a config with `phoonnx_version`.
3. `VoskLoader` accepts a config with `engine: "vosk"`, or a `phoneme_id_map` with the vosk symbols `a0` and `sch` and no `phoneme_type`.
4. `PiperLoader` accepts a config with `piper_version`. It also accepts a config whose `phoneme_type` is a Piper phonemizer, such as `espeak` or `text`, and whose `phoneme_id_map` values are all lists.
5. `Mimic3Loader` accepts a config with a `phonemes` section and a `phonemizer` of `symbols`, `gruut`, `espeak` or `epitran`. It needs the `phonemes.txt` file of the voice.
6. `CoquiLoader` accepts a config with a `characters` section and a known `characters_class`.
7. `TransformersLoader` accepts the `vocab` argument of `from_dict()`. The voice manager gives it from `vocab.json`.
8. `TokensTxtLoader` accepts the `tokens_txt` argument, for sherpa-onnx models that have only a tokens file.
9. `CanonicalLoader` accepts every other config. It uses the `phoneme_id_map`, `engine`, `phoneme_type` and `alphabet` of the config as written.

A config that names its `engine` is not given to the loader of another engine. A config without `phoonnx_version` loads with a `DeprecationWarning`. `phoonnx.compat.convert_config` writes the native config one time.

## SynthesisConfig

`SynthesisConfig` holds the settings of one synthesis call. Each field is optional and has the default in the table below. A `None` default has a different effect for each field, and the Description column gives it. For the speaker, the language and the three scales, `None` uses the value of the voice. The config is frozen: build a changed copy with `dataclasses.replace`.

```python
from phoonnx.config import AudioPost, SynthesisConfig
from scriptconv.frontend import Stages

syn_config = SynthesisConfig(
    stages=Stages(diacritize=True),
    speaker=None,
    language=None,
    length_scale=None,
    noise_scale=None,
    noise_w_scale=None,
    audio=AudioPost(normalize=True, volume=1.0),
    phonetic_spellings=True,
)
```

### SynthesisConfig fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `stages` | `Stages` | empty | The text stages of the call, from `scriptconv.frontend`. The stages table below lists its fields |
| `speaker` | `int` \| `str` \| `None` | `None` | Speaker of a multi-speaker voice: an id, or a name in the voice's `speaker_id_map`. A name that the map lacks goes to the engine, which may hold its own names (Magpie). `None` uses the speaker of the voice, or speaker 0 |
| `language` | `int` \| `str` \| `None` | `None` | Language of a multi-language voice: an id, or a code in the voice's `lang_id_map`. A code also sets `stages.lang`. `None` uses the language of the voice, or language 0 |
| `length_scale` | `float` \| `None` | `None` | Speed control: < 1 faster, > 1 slower. `None` uses the value of the voice |
| `noise_scale` | `float` \| `None` | `None` | Generator noise: affects naturalness. `None` uses the value of the voice |
| `noise_w_scale` | `float` \| `None` | `None` | Phoneme duration noise. `None` uses the value of the voice |
| `exaggeration` | `float` \| `None` | `None` | Expressiveness/intensity for engines that support it. `None` uses the engine default (Chatterbox, ~0.5) |
| `temperature` | `float` \| `None` | `None` | Sampling temperature for autoregressive engines. `None` uses the engine default, about 0.8 for Chatterbox. `0` is greedy decoding |
| `top_p` | `float` \| `None` | `None` | Nucleus sampling cutoff for autoregressive engines. `None` uses the engine default (Chatterbox, ~0.95) |
| `reference` | `SpeakerReference` \| `None` | `None` | The speaker that a cloning engine imitates. The reference table below lists its fields. `None` means no clip |
| `extra` | `dict` | `{}` | Engine-specific per-call parameters (e.g. `d_factor`, `p_factor`, `e_factor`) |
| `audio` | `AudioPost` | the defaults below | What happens to the audio after the engine call. The audio table below lists its fields |
| `phonetic_spellings` | `bool` | `True` | Apply word-level pronunciation overrides |
| `input_alphabet` | `Alphabet` \| `None` | `None` | The alphabet of the input text. `None` means plain text. Set it to pass text that is already phonemic, such as IPA, so that scriptconv converts it and does not phonemize it |

A scale comes from the first of these that sets it: the call's `SynthesisConfig`, the voice's `engine_params`, the `inference` block of the voice's `config.json`, the engine's `default_params()`. The keys of `extra` are applied last and win over all four.

### stages fields

`stages` is a `Stages` from `scriptconv.frontend`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `diacritize` | `bool` \| `None` | `None` | Add vowel diacritics before phonemization. `None` uses the value of the voice config |
| `diacritizer_model` | `str` \| `None` | `None` | The diacritizer model for this call. `None` uses the `diacritizer_model` of the voice config |
| `normalize` | `bool` \| `None` | `None` | `False` skips the text normalisation hook |
| `chunk` | `str` \| `None` | `None` | How much text one engine call reads: `sentence`, `clause`, `paragraph` (the text between blank lines) or `none` (the whole text). `None` reads one sentence per call. The text is read whole before it is cut. An engine that reads text, not phonemes, gets the whole text. Phonemic input (`input_alphabet` set) is cut per sentence whatever the value. Any other value raises `ValueError` when the config is built |
| `lang` | `str` \| `None` | `None` | The language of the text |
| `casing` | callable \| `None` | `None` | A function that sets the text casing. `to_dict()` raises `ValueError` when it is set, because a function has no plain-data form |

### reference fields

`reference` is a `SpeakerReference` from `phoonnx.config`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `audio` | `str` \| `(audio, sr)` \| `None` | `None` | Reference clip for zero-shot [voice cloning](cloning.md) |
| `text` | `str` \| `None` | `None` | Reference transcription, required by in-context cloning engines (ZipVoice) |
| `lang` | `str` \| `None` | `None` | Language of the transcription, for cross-lingual cloning (defaults to the voice's `lang_code`) |

### audio fields

`audio` is an `AudioPost` from `phoonnx.config`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `normalize` | `bool` | `True` | Scale the peak of the audio to full scale |
| `target_loudness` | `float` \| `None` | `None` | Bring each sentence to this integrated loudness in LUFS, such as `-16.0`, with peaks limited to -1 dBFS. `None` keeps the peak normalisation. Needs `scipy`. Without it, phoonnx logs one warning and uses peak normalisation |
| `volume` | `float` | `1.0` | Volume multiplier |
| `super_resolution` | `bool` | `False` | Upscale each chunk to 48 kHz with `audiosronnx` (the `audiosr` extra). The chunk reports the upscaled sample rate |
| `super_resolution_model` | `str` \| `None` | `None` | The `audiosronnx` engine, such as `"novasr"` or `"lavasr"`. `None` selects `"novasr"`. Ignored unless `super_resolution` is `True` |
| `voice_conversion` | `SpeakerReference` \| `None` | `None` | Target-speaker clip, in its `audio` field, for [voice conversion](voice_conversion.md) after synthesis. It works with every engine. `None` means no conversion |
| `voice_conversion_engine` | `str` \| `None` | `None` | The voiceclonnx engine for `voice_conversion`. `None` selects `"openvoice"` |

### Plain data

`SynthesisConfig.to_dict()` gives the config as plain data, and `SynthesisConfig.from_dict()` reads it back. The JSON Schema `phoonnx/synthesis_config.schema.json`, in the wheel, describes that data. A reference in plain data is a path or URL. A reference that holds samples, or a `stages.casing` callable, makes `to_dict()` raise `ValueError`. The constructor also accepts the groups as dicts, so `SynthesisConfig(**dataclasses.asdict(config))` builds the same config.

### 1.x names

The flat names of phoonnx 1.x still work as keywords, attributes, `dataclasses.replace()` keywords and `SynthesisConfig.from_dict` keys. Positional arguments in the 1.x field order work too. Each use raises a `DeprecationWarning` that names the new field and points at the caller's line. A string in `speaker_id` or `lang_id` goes to the engine as it is, as in 1.x: no voice map and no text stage reads it. Most engines read it with `int()`, so `speaker_id="40"` is the id 40, and Magpie reads it as a voice name. In the new `speaker` and `language` fields a string is a name or a code first, so `speaker="40"` is the speaker named `40`. The names and the positional form stop working in phoonnx 2.0.0.

| Flat name | Field |
|-----------|-------|
| `alphabet` | `input_alphabet` |
| `enable_phonetic_spellings` | `phonetic_spellings` |
| `speaker_id` | `speaker` |
| `lang_id` | `language` |
| `extra_params` | `extra` |
| `speaker_reference` | `reference.audio` |
| `speaker_reference_text` | `reference.text` |
| `speaker_reference_lang` | `reference.lang` |
| `add_diacritics` | `stages.diacritize` |
| `diacritizer_model` | `stages.diacritizer_model` |
| `chunk` | `stages.chunk` |
| `normalize_audio` | `audio.normalize` |
| `target_loudness` | `audio.target_loudness` |
| `volume` | `audio.volume` |
| `super_resolution` | `audio.super_resolution` |
| `super_resolution_model` | `audio.super_resolution_model` |
| `vc_reference` | `audio.voice_conversion.audio` |
| `vc_engine` | `audio.voice_conversion_engine` |

## config.json Format

This is a config in the native phoonnx format:

```json
{
  "phoonnx_version": "1.0",
  "engine": "phoonnx",
  "phoneme_type": "espeak",
  "alphabet": "ipa",
  "lang_code": "en-US",
  "num_symbols": 256,
  "num_speakers": 1,
  "num_langs": 1,
  "phoneme_id_map": {"_": [0], "^": [1], "$": [2], " ": [3], "a": [4], "b": [5]},
  "audio": {"sample_rate": 22050},
  "inference": {
    "noise_scale": 0.667,
    "length_scale": 1.0,
    "noise_w": 0.8,
    "add_diacritics": false,
    "diacritizer_model": null
  },
  "blank": "_",
  "pad": "_",
  "bos": "^",
  "eos": "$",
  "blank_at_start": true,
  "blank_at_end": true,
  "add_blank_char": true,
  "speaker_id_map": {}
}
```

The wheel ships a JSON Schema of the native format as `phoonnx/config.schema.json`. `VoiceConfig.to_dict()` writes a config that the schema accepts. See [Writing the native config](api.md#writing-the-native-config).

phoonnx also reads the config formats of [Engine Auto-Detection](#engine-auto-detection).

### Tokenizer keys

These keys of the native format control how the phonemes become ids.

| Key | Default | Effect |
|-----|---------|--------|
| `phoneme_id_map` | — | Symbol to id. A value is an id or a list of ids. A list of one id is that id. A longer list gives each id in turn, and an empty list gives no id. |
| `pad` | `_` | The padding symbol. `null` gives a vocabulary with no padding symbol. |
| `add_blank_char` | `true` | Put the `blank` symbol between all ids, and at the start and end as `blank_at_start` and `blank_at_end` say. |
| `add_blank_word` | `false` | Put the `blank_word` symbol after each word that gives an id. With `blank_at_end`, the last word gets one too. |
| `blank_word` | none | The symbol that `add_blank_word` puts between words. Without it, a space gives no id. |
| `use_eos_bos` | `true` | Put `bos` before the ids and `eos` after them. |
| `fold_compounds` | `true` | Read the longest map key that starts at each symbol, so that `t͡ʃ` gives one id. A Piper config with no `vowel_clusters` whose only multi-character keys are Piper's default diphthongs (`aɪ`, `aʊ`, `ɔɪ`, `eɪ`, `oʊ` at ids 161 to 165) loads with `false`, because Piper reads `aɪ` as `a` and `ɪ` for such a voice. |
| `blank_after` | `null` | A list of symbols. When it is set, the `blank` symbol follows only these symbols, and `add_blank_char` and `blank_at_start` have no effect. `blank_at_end` then adds only the trailing `blank_word` of `add_blank_word`. Piper's pinyin voices (`zh_CN-chaowen-medium`, `zh_CN-xiao_ya-medium`) set it to the tone digits `1` to `5`, the pause marks and the space. |
| `keep_punctuation` | `false` | Keep the clause punctuation `, . ? ! : ;` for the map to read. |

A Mimic 3 voice has its tokenizer rules in a `phonemes` section and in a
`phoneme_map.txt` file. Its native config has the same rules in the keys above:

- `"add_blank_word": true` and `"blank_word": "#"` put `#` between words.
- Each `phoneme_map.txt` line becomes a list value. The line `‖ · ·` gives `"‖": [3, 3]`.
- `"keep_punctuation": true` keeps the punctuation, and the map gives its ids.
  A gruut voice writes a break as a word of its own, so its entry starts with
  the `#` id: `".": [4, 3, 3]`.

```json
{
  "phoonnx_version": "1.0",
  "phoneme_type": "gruut",
  "alphabet": "ipa",
  "lang_code": "en-US",
  "phoneme_id_map": {"_": 0, "^": 1, "$": 2, "·": 3, "#": 4, "ˈ": 5, "‖": [3, 3], ".": [4, 3, 3], ",": [4, 3]},
  "pad": "_", "blank": "_", "bos": "^", "eos": "$",
  "add_blank_char": true,
  "add_blank_word": true,
  "blank_word": "#",
  "keep_punctuation": true
}
```

Two Mimic 3 steps are not tokenizer rules, and a native config does not hold
them. Mimic 3 adds `auto_append_text` (usually `.`) to the end of the text when
the text does not end with it. With `minor_break_ms` set, Mimic 3 also sends
each clause to the model in a separate call. phoonnx sends one call for each
sentence.

## Phonemizer provenance

A voice learns the phonemes of one phonemizer. When a later release of that
phonemizer gives other phonemes for the same text, the voice can mispronounce
words. The `phonemizer_identity` block records the phonemizer that made the
training phonemes, so that phoonnx can tell you when the phonemizer changed.

Preprocessing (`phoonnx_train/preprocess.py`, see the
[preprocess reference](training/preprocess.md)) writes the block into the
`config.json` of the dataset:

```json
{
  "phoneme_type": "orthography2ipa",
  "lang_code": "pt-PT",
  "phonemizer_identity": {
    "backend": "orthography2ipa",
    "distribution": "orthography2ipa",
    "version": "7.124.36a2",
    "lang": "pt-PT",
    "revision": "sc2:4e4965fdc13657a68e80b5b5a9f8710f470efd0965d86bd881868cdb72ac419a",
    "frontend": null
  }
}
```

When `TTSVoice` loads a voice, it starts the comparison of the block with the
phonemizer that runs. The comparison runs in a background thread, and the load
does not wait for it. Thus the warning can come after `load` returns.

When a field is different, phoonnx logs one warning. The warning names the
voice, each field that is different, and the trained and the running value.
The voice loads and speaks as usual. This example is a voice that was trained
on another phonemizer revision:

```text
WARNING - Voice pt_PT-tugão was trained with another phonemizer: revision: trained 'sc2:0000000000000000000000000000000000000000000000000000000000000000', running 'sc2:4e4965fdc13657a68e80b5b5a9f8710f470efd0965d86bd881868cdb72ac419a'. The voice loads, but its pronunciation can differ from training.
```

The comparison follows these rules:

- `revision` is the hash that scriptconv records for the phonemes. An `sc2:`
  revision hashes the scriptconv source that makes the phonemes, the data files
  next to that source, the engine release with its language extra (such as
  `gruut[fr]`), and the versions of the packages that the backend's extras
  require, at any depth. A revision from the older `sc1:` scheme always
  differs from an `sc2:` revision, so a voice trained with it warns on each
  load. A later release of one of those
  packages changes the revision, even when the phonemes stay the same. A block
  with a `frontend` holds the frontend revision, which also covers the frontend
  stages. When both revisions are set and equal, a different `version` gives no
  warning, because the phonemes did not change.
- `lang` is the lect that the phonemizer resolved. A voice trained on `es-ES`
  that runs on `pt-PT` gives a warning.
- `frontend` is `null` when preprocess called the backend directly. A `null`
  value is not compared.
- A voice without the block, or with an empty block, gives no warning.
- A block that phoonnx cannot read, such as a string or a block without
  `lang`, gives one warning. A phonemizer that cannot report its identity also
  gives one warning. In both cases the voice loads without the check.
- An engine that reads text and not phonemes (`"alphabet": "graphemes"`) is not
  checked.

To clear the warning, install the phonemizer release in the trained block, or
train the voice again with the running phonemizer.

## Engine Enum

`Engine` in `phoonnx.config` names the file format or the architecture of a voice. The table lists each member. `phoonnx`, `piper`, `mimic3`, `coqui` and `transformers` name a file format, and all of them run through the VITS adapter, `vits`. Each other member names an adapter.

The adapter registry itself is described in [Engines](engines.md).

`Engine("kokoro")` gives `Engine.STYLETTS2`, because sherpa-onnx and kokoro-onnx write
`kokoro` for the StyleTTS2 graph.

| `Engine` value | Kind | Per-engine guide |
|---|---|---|
| `phoonnx` | VITS (shared adapter), native config | [Engines](engines.md) |
| `vits` | Single-graph VITS: the shared adapter | [Engines](engines.md) |
| `vits_streaming` | VITS split at the flow/HiFi-GAN boundary | [Streaming](streaming.md) |
| `vits_prior_split` | VITS split at the prior boundary | [Engines](engines.md) |
| `piper` | VITS (shared adapter) | [Engines](engines.md) |
| `mimic3` | VITS (shared adapter) | [Engines](engines.md) |
| `coqui` | VITS (shared adapter) | [Engines](engines.md) |
| `transformers` | HuggingFace VITS/MMS (shared adapter) | [Engines](engines.md) |
| `matcha` | Two-stage flow-matching mel | [Matcha](training/engines/matcha.md) |
| `optispeech` | FastSpeech2-style + GAN vocoder | [OptiSpeech](training/engines/optispeech.md) |
| `glowtts` | Flow-based mel + vocoder | [GlowTTS](training/engines/glowtts.md) |
| `mixertts` | MLP-Mixer mel + vocoder | [MixerTTS](training/engines/mixertts.md) |
| `fastpitch` | FastSpeech2-style mel + vocoder | [FastPitch](training/engines/fastpitch.md) |
| `styletts2` | StyleTTS2 / Kokoro end-to-end | [Training](training/training.md) |
| `yourtts` | Multilingual VITS + d-vector cloning | [Cloning](cloning.md) |
| `zipvoice` | Flow-matching in-context cloning | [ZipVoice](training/engines/zipvoice.md) |
| `shami` | Levantine Arabic / English (HamsVITS) | [Shami](training/engines/shami.md) |
| `tinyvoca` | Source-filter acoustic model on dizione tokens and stress | [Engines](engines.md) |
| `f5tts` | DiT flow-matching (F5-TTS / Habibi) | [F5-TTS](training/engines/f5tts.md) |
| `chatterbox` | Autoregressive codec-LM cloning | [Chatterbox](training/engines/chatterbox.md) |
| `vosk` | VITS adapter, vosk-tts Russian front-end | [Engines](engines.md) |
| `supertonic` | Flow-matching, fixed per-speaker style | [Engines](engines.md) |
| `neutts` | Autoregressive NeuCodec LM, preset cloning @24 kHz | [Cloning](cloning.md) |
| `pockettts` | Flow-matching codec LM with explicit stream state, raw-text (no phonemizer) | [Pocket TTS](pockettts.md) |
| `sparktts` | Autoregressive codec-LM (Qwen2 + BiCodec), en/zh | [Spark-TTS](training/engines/sparktts.md) |
| `qwen3tts` | Two-stage codec LM (talker + code predictor), 10 languages | [Qwen3-TTS](training/engines/qwen3tts.md) |
| `outetts` | Autoregressive codec-LM (Qwen3 or Llama-3.2 + DAC.speech), raw-text, 23 languages | [OuteTTS](training/engines/outetts.md) |
| `arktts` | Two-stage codec LM (slow AR + fast AR), Basque or 11 languages, 44.1 kHz | [ArkTTS](training/engines/arktts.md) |
| `llasa` | Autoregressive XCodec2 LM (LLaMA-3.2), preset cloning, en/zh @16 kHz | [Cloning](cloning.md) |
| `omnivoice` | Masked-diffusion codec LM (Qwen3 + Higgs Audio V2), raw-text, 600+ languages, 24 kHz | [OmniVoice](training/engines/omnivoice.md) |
| `indic_parler` | Encoder-decoder codec LM (Flan-T5 + AR DAC decoder), description-controlled voices, 20 Indic languages + English, 44.1 kHz | [Indic Parler-TTS](training/engines/indic_parler.md) |
| `mosstts` | Autoregressive RVQ-16 codec-LM, zero-shot cloning @48 kHz | [Cloning](cloning.md) |
| `orpheus` | Llama codec-LM + SNAC decoder, emotive tags | [Orpheus](orpheus.md) |
| `magpie` | Encoder-decoder codec LM (NVIDIA Magpie-TTS), 8 codebooks | [Engines](engines.md) |
| `rapido` | Flow-matching DiT over VoxCPM2 AudioVAE latents @48 kHz | [Rápido](training/engines/rapido.md) |
| `xvits` | Piper/VITS with flow-matching prosody and word emphasis, 5 graphs | [x-vits](training/engines/xvits.md) |

## Execution Providers

Each ONNX session of phoonnx runs on one ordered list of ONNX Runtime execution providers. This includes the voice model and the auxiliary graphs of an engine, such as vocoders, speaker encoders, text encoders and diacritizers. The decoder of a
[streaming VITS voice](streaming.md) uses the same list as its encoder.

Give the list to the load:

```python
from phoonnx.voice import TTSVoice

voice = TTSVoice.load(
    "model.onnx", "config.json",
    providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
)
```

Or set it once for the process:

```bash
export PHOONNX_ONNX_PROVIDERS="ROCMExecutionProvider,CPUExecutionProvider"
export PHOONNX_ONNX_PROVIDERS="auto"   # the default: detect the best provider
```

Without both, phoonnx selects the best provider of the installed runtime, in this order:

`CUDAExecutionProvider` (NVIDIA) → `ROCMExecutionProvider` (AMD) →
`MIGraphXExecutionProvider` (AMD) → `DmlExecutionProvider` (DirectML) →
`CoreMLExecutionProvider` (Apple) → `OpenVINOExecutionProvider` (Intel) →
`CPUExecutionProvider`

phoonnx skips a requested provider that the installed runtime does not have, and logs a warning. It always adds `CPUExecutionProvider` at the end, so synthesis works on each machine.

`use_cuda=True` is a deprecated alias for `providers=["CUDAExecutionProvider"]`.

### Runtime packages

The installed ONNX Runtime build sets the available providers, not phoonnx. The default `onnxruntime` wheel has the CPU provider and some platform providers. GPU providers need a matching build, and you can install only one of these builds at a time:

| Hardware | Package | Provider |
|---|---|---|
| NVIDIA | `onnxruntime-gpu` | `CUDAExecutionProvider`, `TensorrtExecutionProvider` |
| AMD (ROCm) | `onnxruntime-rocm` | `ROCMExecutionProvider`, `MIGraphXExecutionProvider` |
| Windows (any DX12 GPU) | `onnxruntime-directml` | `DmlExecutionProvider` |
| Intel | `onnxruntime-openvino` | `OpenVINOExecutionProvider` |
| Apple | `onnxruntime` | `CoreMLExecutionProvider` |

To see the providers of a machine:

```python
import onnxruntime
print(onnxruntime.get_available_providers())
```

### Thread count

ONNX Runtime runs each graph operator on one thread per core. Several voices in
one process, or a voice beside other jobs, then compete for every core. To limit
a voice, pass `threads`. The voice model and every auxiliary graph of the voice
run each operator on that many threads (`intra_op_num_threads`):

```python
voice = TTSVoice.load("model.onnx", "config.json", threads=2)
```

Or set it once for the process:

```bash
export PHOONNX_ORT_THREADS=2
```

The `threads` argument comes first, then `PHOONNX_ORT_THREADS`. With neither,
ONNX Runtime keeps its default. `threads=0` asks for that default too, also when
`PHOONNX_ORT_THREADS` is set. `TTSModelInfo.load` takes the same
argument, and the OVOS plugin reads the `onnx_threads` key. A voice with a
different thread count gets its own session, so voices share a session only when
they ask for the same count. The decoder of a [streaming VITS voice](streaming.md)
reads its own `intra_op_num_threads` key first.

## Precision

Some voices have two sets of files: full files and smaller int8 files. The
`precision` setting picks the set that loads. It has three values:

| Value | Files that load |
|---|---|
| `auto` | The default. Full files, except where the int8 files are marked for `auto` and no GPU provider runs the voice. |
| `int8` | The int8 files. A voice that has no int8 files raises `PrecisionUnavailable`. |
| `full` | The full files. A voice that has only int8 files raises `PrecisionUnavailable`. |

Pass it to a load:

```python
voice = manager.get_voice("edwixx/pt-BR-rapido").load(precision="int8")
```

Or set it once for the process:

```bash
export PHOONNX_PRECISION=int8
```

The load argument wins over the environment variable. In the OVOS plugin, the
`precision` key wins over the environment variable. An argument or key that is
`None`, `null` or an empty string counts as unset, so the environment variable
applies. With nothing set, the value is `auto`. Case and surrounding spaces do
not matter. A value that is not a string, or not one of the three values,
raises `ValueError`.

`PrecisionUnavailable` is a `ValueError` and a `phoonnx.errors.VoiceLoadError`.
Its message names the voice and the value that loads it.

`auto` picks the int8 files only when both of these are true:

1. The index marks the int8 files with `"auto": true`. The mark goes on a voice
   whose int8 files pass the quality check and run faster than its full files.
2. No GPU provider is in the resolved providers. The GPU providers are
   `CUDAExecutionProvider`, `ROCMExecutionProvider`, `MIGraphXExecutionProvider`,
   `DmlExecutionProvider` and `CoreMLExecutionProvider`.

In every other case `auto` loads the full files. A voice that has only int8
files, such as a Pocket TTS voice in any language but Italian, loads its int8
files under `auto`.

### Precision in the voice index

An index entry names the full files of a voice. The int8 files go in the
optional `variants` field, which holds only the URL fields that differ.
`aux_model_urls` merges key by key, so the int8 entry below loads
`flow_int8.onnx` and the full `text_encoder.onnx`:

```json
{
  "model_url": "https://huggingface.co/Phoonnx/demo/resolve/0123456789abcdef0123456789abcdef01234567/model.onnx",
  "aux_model_urls": {
    "text_encoder_path": "https://huggingface.co/Phoonnx/demo/resolve/0123456789abcdef0123456789abcdef01234567/text_encoder.onnx",
    "flow_path": "https://huggingface.co/Phoonnx/demo/resolve/0123456789abcdef0123456789abcdef01234567/flow.onnx"
  },
  "variants": {
    "int8": {
      "model_url": "https://huggingface.co/Phoonnx/demo/resolve/0123456789abcdef0123456789abcdef01234567/model_int8.onnx",
      "aux_model_urls": {
        "flow_path": "https://huggingface.co/Phoonnx/demo/resolve/0123456789abcdef0123456789abcdef01234567/flow_int8.onnx"
      },
      "auto": true
    }
  }
}
```

An entry that has only int8 files carries `"precision": "int8"` and no variant.

The voice id `edwixx/pt-BR-rapido-int8` is deprecated. It loads the int8 files
of `edwixx/pt-BR-rapido` and logs a warning that names the release that removes
it. Use `edwixx/pt-BR-rapido` with `precision="int8"`.

---

[Docs index](index.md) · Previous: [Phoneme alignment](alignment.md) · Next: [Phonemizers](phonemizers.md)
