# Python API

<!-- agent-hint
covers: the Python API that the code exports, as a walk-through and as a name list, plus the design of a target API
source: phoonnx/__init__.py, phoonnx/voice.py:TTSVoice, phoonnx/encoders.py, phoonnx/audio.py:write_wav, phoonnx/config.py, phoonnx/errors.py, phoonnx/engines/__init__.py, phoonnx/config.schema.json
generated: no
verify: diff <(awk '/^## Part 3/{exit} /^\x60\x60\x60python/{f=1;next} /^\x60\x60\x60/{f=0} f' docs/api.md | .venv/bin/python -I -) <(awk '/^## Part 3/{exit} /^\x60\x60\x60text/{f=1;next} /^\x60\x60\x60/{f=0} f' docs/api.md) && .venv/bin/python -c "import inspect,re,sys; import phoonnx,phoonnx.errors as e; from phoonnx.voice import TTSVoice; p=open('docs/api.md').read(); q=chr(96); row=re.search('^[|] '+q+'TTSVoice.load'+q+' [|].*$',p,re.M).group(0); doc=re.findall(q+'([a-z_]+)'+q,row); w=q+'([A-Za-z_]+)'+q; errs=re.search('the classes below it:\n\n((?:- .*\n)+)',p).group(1); root=p.split(chr(10)+'The root package exports',1)[1].split(chr(10))[0]; ok=[doc==list(inspect.signature(TTSVoice.load).parameters), sorted(re.findall(w,errs))==sorted(n for n in e.__all__ if n!='PhoonnxError'), sorted(re.findall(w,root))==sorted(phoonnx.__all__)]; print(ok); sys.exit(0 if all(ok) else 1)"
see: usage.md, architecture.md, configuration.md, engines.md
-->

This page describes the Python API of phoonnx, for developers who use it as a library.

Part 1 is a walk-through of the API that exists in the code. Part 2 lists each name with its module. Part 3 is a design for a target API. For every target object it names the existing name that it replaces.

Every example in Parts 1 and 2 runs offline. None of them downloads a model. The output shown is the real output. Run the examples in order in one Python session.

## Part 1: Walk-through

### 1. Read a voice config

A voice has a native `config.json`. `VoiceConfig.from_dict` turns the parsed JSON into a `VoiceConfig`. The example below uses a tiny config with a 6-symbol vocabulary.

```python
import phoonnx
from phoonnx.config import VoiceConfig

native = {
    "phoonnx_version": "1.0",
    "audio": {"sample_rate": 22050},
    "num_symbols": 6,
    "num_speakers": 1,
    "lang_code": "en-US",
    "phoneme_type": "graphemes",
    "alphabet": "unicode",
    "phoneme_id_map": {"_": [0], "^": [1], "$": [2], " ": [3], "h": [4], "i": [5]},
}
config = VoiceConfig.from_dict(native)
print(config.engine, config.sample_rate, config.lang_code)
print(config.phoneme_type, config.alphabet)
print(config.tokenizer.encode("hi"))
```

```text
Engine.PHOONNX 22050 en-US
Phonemizer.GRAPHEMES Alphabet.UNICODE
[4, 5]
```

`engine` names the file format here (`PHOONNX`), not the model architecture. [Part 3](#part-3-target-api-design) describes a design where `engine` names the architecture.

### 2. Turn text into model input

`TTSVoice.prepare` is the text half of synthesis. It reads text and yields one `Utterance` for each engine call. An `Utterance` holds the text, the symbols that the tokenizer read, the token ids, the side streams, and the params that the adapter read from the text.

A `TTSVoice` normally comes from `TTSVoice.load("model.onnx", "model.json")`, which also opens the ONNX session. `prepare` and `phonemize` use no session, so this example passes `session=None`.

```python
from phoonnx.voice import TTSVoice

voice = TTSVoice(session=None, config=config)
print(voice.phonemize("hi"))
for utterance in voice.prepare("hi hi"):
    print(utterance)
```

```text
[['h', 'i']]
Utterance(text='hi hi', symbols=('h', 'i', ' ', 'h', 'i'), ids=(1, 0, 4, 0, 5, 0, 3, 0, 4, 0, 5, 0, 2), streams={}, params={})
```

The ids hold the blank id `0` between symbols and the start and end ids `1` and `2`. The tokenizer adds them.

### 3. Render audio and write a WAV file

`TTSVoice.render` is the audio half. It takes one `Utterance` and gives one `AudioChunk`. `TTSVoice.synthesize(text)` is `render` over `prepare` and yields the chunks. This needs a real model, so the example below builds two silent `AudioChunk` objects by hand and writes them with `phoonnx.audio.write_wav`. That function accepts any stream of chunks.

```python
import io
import wave

import numpy as np

from phoonnx.audio import write_wav
from phoonnx.voice import AudioChunk

audio = np.zeros(2205, dtype=np.float32)
chunk = AudioChunk(sample_rate=22050, sample_width=2, sample_channels=1,
                   audio_float_array=audio)

buffer = io.BytesIO()
with wave.open(buffer, "wb") as wav_file:
    write_wav([chunk, chunk], wav_file)
buffer.seek(0)
with wave.open(buffer) as wav_file:
    print(wav_file.getframerate(), wav_file.getnframes())
```

```text
22050 4410
```

With a loaded voice, the same call is `write_wav(voice.synthesize("Hello"), wav_file)`. The convenience methods `synthesize_wav` and `synthesize_ssml_wav` call it for you.

### 4. Change one call

`SynthesisConfig` holds the settings of one call. Each engine setting is `None` until you set it. A `None` scale takes the value of the voice's `engine_params`, then of the `inference` block in the voice's `config.json`, and then the engine's default.

```python
from phoonnx.config import SynthesisConfig

print(config.length_scale, SynthesisConfig().length_scale)
print(SynthesisConfig(speaker=2, length_scale=1.2).speaker)
```

```text
None None
2
```

This config sets no scales and the call sets none, so the engine's own default applies. The order of precedence is:

1. A scale in the call.
2. The same key in the voice's `engine_params`.
3. The `inference` block of `config.json`.
4. The engine's default.

The keys of `SynthesisConfig.extra` come last and win over all four.

## Part 2: Name list

This part lists the names that exist in the code. The package root exports eleven names, listed below. Import every other name from its own module.

### Voice and audio

| Name | Module | Role |
| --- | --- | --- |
| `TTSVoice` | `phoonnx.voice` | A loaded voice. A dataclass built by `TTSVoice.load`. |
| `TTSVoice.load` | `phoonnx.voice` | Takes `model_path`, `config_path`, `vocab_path`, `tokenizer_config_path`, `phonemes_txt`, `phoneme_map`, `lang_code`, `phoneme_type_str`, `alphabet_str`, `engine_params`, `providers`, the deprecated `use_cuda`, `warmup`, `phonemizer_lang` and `threads`. |
| `TTSVoice.prepare` | `phoonnx.voice` | Text to `Iterator[Utterance]`. |
| `TTSVoice.render` | `phoonnx.voice` | One `Utterance` to one `AudioChunk`. |
| `TTSVoice.synthesize` | `phoonnx.voice` | Text to `Iterable[AudioChunk]`. Argument `syn_config`, flag `include_alignments`. |
| `TTSVoice.stream` | `phoonnx.voice` | Text to `Iterator[AudioChunk]`, yielded as the engine makes the audio. See [Streaming the audio of a sentence](#streaming-the-audio-of-a-sentence). |
| `TTSVoice.synthesize_ssml` | `phoonnx.voice` | SSML to `Iterable[AudioChunk]`. Arguments `syn_config`, `lang`, `include_alignments`, `on_mark` and `seed`. |
| `TTSVoice.synthesize_wav`, `synthesize_ssml_wav` | `phoonnx.voice` | Write a WAV file through `write_wav`. |
| `TTSVoice.frontend` | `phoonnx.voice` | The scriptconv `TextFrontend` of the voice. |
| `TTSVoice.phonemize` | `phoonnx.voice` | Text to phonemes. |
| `TTSVoice.encoder` | `phoonnx.voice` | The `Encoder` that turns text into the ids of the voice's engine. |
| `Utterance` | `phoonnx.encoders` | Frozen: `text`, `symbols`, `ids`, `streams`, `params`. `phoonnx.voice` re-exports it. |
| `AudioChunk` | `phoonnx.voice` | Audio plus `phoneme_ids`, `phoneme_id_samples` and `phoneme_alignments`. |
| `write_wav` | `phoonnx.audio` | Writes any chunk stream to a WAV writer. |

`TTSVoice.phonemes_to_ids` and `TTSVoice.phoneme_ids_to_audio` work, but they are deprecated. Each warns with a removal version.

### Configuration

| Name | Module | Role |
| --- | --- | --- |
| `VoiceConfig` | `phoonnx.config` | What the model is. `from_dict` reads a config and `to_dict` writes the native format. |
| `SynthesisConfig` | `phoonnx.config` | One call. Frozen. The groups `stages`, `reference` and `audio` hold the text stages, the cloning reference and the audio steps. The other fields are flat. |
| `AudioPost`, `SpeakerReference` | `phoonnx.config` | The `audio` group of `SynthesisConfig`, and the reference of `reference` and `audio.voice_conversion`. |
| `Engine` | `phoonnx.config` | Names a file format or an architecture. The alias `kokoro` gives `Engine.STYLETTS2`. |
| `Alphabet`, `PhonemeType` | `phoonnx.config` | Re-exports of scriptconv enums. `PhonemeType` is scriptconv `Phonemizer`. |
| `Chunk` | `phoonnx.config` | Re-export of the scriptconv enum behind `SynthesisConfig.stages.chunk`. |
| `TTSModelManager`, `TTSModelInfo` | `phoonnx.model_manager` | The voice index and the index entry. |

### Engines and errors

The engine registry in `phoonnx.engines` holds the adapter names, and `phoonnx.engines.list_engines()` returns them. The `Engine` enum has more members than the registry, because it also names the file formats that load through the VITS adapter. This example prints the two counts and the names that are in only one of them:

```python
from phoonnx.config import Engine
from phoonnx.engines import list_engines

print(len(Engine), len(list_engines()))
print(sorted(set(e.value for e in Engine) ^ set(list_engines())))
```

```text
36 31
['coqui', 'mimic3', 'phoonnx', 'piper', 'transformers']
```

An adapter subclasses `BaseOnnxAdapter` in `phoonnx.engines.base`. It receives an `AdapterSynthesisRequest` and returns an `AdapterSynthesisResult`. `phoonnx.errors` defines `PhoonnxError` and the classes below it:

- `VoiceLoadError`, with the subclasses `FrontendMismatchError`, `UnsupportedConfigError` and `PrecisionUnavailable`.
- `ExportNotSupported`.
- `NonFiniteAudioError`.
- `UnsupportedTokenizer`, `UnsupportedSentencePieceModel`, `UnsupportedVoiceLanguage` and `VoiceExceedsMemoryBudget`.
- `UnknownVoice`.

`PrecisionUnavailable` and `UnsupportedVoiceLanguage` also subclass `ValueError`. `NonFiniteAudioError` and `VoiceExceedsMemoryBudget` also subclass `RuntimeError`. `UnknownVoice` also subclasses `KeyError`: `TTSModelManager.voices` raises it for a voice id that the index does not have.

`TTSVoice.load` raises `VoiceLoadError` when the config file does not exist. It also raises it for a VITS graph whose vocabulary uses an id that is not below `num_symbols`. Synthesis raises `NonFiniteAudioError` when the model returns NaN or infinite samples, before the audio is levelled. As 16-bit samples, such audio is silence. `phoonnx.compat.export_piper` and `export_sherpa` raise `ExportNotSupported` when Piper or sherpa-onnx cannot read the ids of the voice. Nothing in the code raises `FrontendMismatchError` or `UnsupportedConfigError`. Part 3 describes where the design raises them.

The root package exports `TTSVoice`, `AudioChunk`, `Utterance`, `VoiceConfig`, `SynthesisConfig`, `AudioPost`, `SpeakerReference`, `TTSModelManager`, `Stages`, `FrontendConfig` and `Chunk`. Each name loads on first access, so `import phoonnx` stays cheap.

### Writing the native config

`VoiceConfig.to_dict()` writes the native format, and `VoiceConfig.from_dict` reads it back. The `phoonnx_version` key holds the installed package version.

```python
from phoonnx.version import VERSION_STR

d = voice.config.to_dict()
print(sorted(d))
print(d["phoonnx_version"] == VERSION_STR)
print(VoiceConfig.from_dict(d).tokenizer.encode("hi"))
```

```text
['add_blank_char', 'add_blank_word', 'alphabet', 'audio', 'blank', 'blank_at_end', 'blank_at_start', 'blank_between', 'blank_word', 'bos', 'default_speaker_id', 'engine', 'engine_params', 'eos', 'fold_compounds', 'hop_length', 'inference', 'keep_punctuation', 'lang_code', 'lang_id_map', 'lang_tokens', 'num_langs', 'num_speakers', 'num_symbols', 'pad', 'phoneme_id_map', 'phoneme_type', 'phonemizer_model', 'phoonnx_version', 'speaker_id_map', 'text_casing', 'use_eos_bos', 'word_sep_token']
True
[4, 5]
```

`to_native_dict()` is the deprecated name. It warns and returns the same dict.

The wheel ships a JSON Schema of this format as `phoonnx/config.schema.json`. A validator such as `jsonschema` can check a `config.json` against it.

```python
import json
from importlib.resources import files

schema = json.loads(files("phoonnx").joinpath("config.schema.json").read_text())
print(schema["required"])
```

```text
['phoonnx_version', 'engine', 'phoneme_type', 'alphabet', 'lang_code', 'audio', 'num_symbols', 'phoneme_id_map']
```

### The encoder

The encoder is the step between the text frontend and the engine. It turns text into the ids that the model reads, and it runs no neural network. `TTSVoice.prepare` reads text through `TTSVoice.encoder`.

```python
from phoonnx.config import SynthesisConfig

print(type(voice.encoder).__name__)
print(next(voice.encoder.encode("hi", SynthesisConfig())).ids)
```

```text
SymbolEncoder
(1, 0, 4, 0, 5, 0, 2)
```

`phoonnx.encoders` holds the `Encoder` protocol and its two classes:

- `SymbolEncoder` gives the symbols of the frontend to the voice's vocabulary tokenizer. Phoneme and grapheme engines use it.
- `TextEncoder` gives the text of each chunk to the engine's own tokenizer, such as a BPE.

The adapter builds the encoder of a voice in `BaseOnnxAdapter.make_encoder`. An adapter that overrides `encode_text` gets a `TextEncoder`, and every other adapter gets a `SymbolEncoder`.

### Streaming the audio of a sentence

`TTSVoice.stream(text, config)` yields the audio as the engine makes it. An engine that makes audio in steps, such as Pocket TTS, yields parts of a sentence. Every other engine yields one chunk for each sentence. Joined, the parts give the audio of `synthesize`. The first part of a sentence carries its phonemes and ids.

Peak normalisation, a loudness target, voice conversion and super-resolution need the whole sentence. When one of them is on, each sentence is one chunk. Peak normalisation is on by default, so pass `SynthesisConfig(audio=AudioPost(normalize=False))` to get parts of a sentence.

An adapter streams by overriding `BaseOnnxAdapter.synthesize_stream(request, session)`. The default yields the one result of `synthesize`.

## Part 3: Target API design

This part is a design, not a description of the code. Its objects do not exist, except where a row says that a name exists in the code. The examples in Parts 1 and 2 describe the code.

### Design rules

1. One concept has one name and one home.
2. Three objects have three lifetimes. `VoiceConfig` is what the model is. It is read one time and frozen. `TextFrontend` is how text becomes symbols. It is built one time from `VoiceConfig.frontend`. `SynthesisConfig` is what one call asks for. Every field is optional, and `None` means the value of the voice.
3. Every config is plain data with `to_dict()` and `from_dict()`, and a JSON Schema ships in the wheel. A config holds no callable and no instance.
4. One precedence rule: call, then voice injection, then `config.json`, then engine default.
5. A loaded voice does not change. A different frontend makes another voice that shares the ONNX session.
6. The runtime reads one config format: the native `config.json`. Other formats are converted once, offline, by `phoonnx.compat`.

### Target objects and the names they replace

| Target | Replaces in the code |
| --- | --- |
| `VoiceConfig` frozen, with `engine` as the architecture, `frontend`, `tokenizer`, `speakers`, `languages`, `defaults`, `assets` | Mutable `VoiceConfig`, where `engine` can mean a file format such as `piper` |
| `VoiceConfig.frontend: FrontendConfig` | `lang_code`, `phoneme_type`, `alphabet`, `phonemizer_model`, `keep_punctuation`, `add_diacritics`, `diacritizer_model`, `lang_tokens` |
| `VoiceConfig.tokenizer: TokenizerConfig` | The copies of tokenizer fields on `VoiceConfig` (`blank_at_start`, `blank_at_end`, `pad_token`, `blank_token`, `bos_token`, `eos_token`, `word_sep_token`, `blank_between`) and the format-specific tokenizer constructors |
| `VoiceConfig.speakers`, `languages` | `speaker_id_map`, `lang_id_map` |
| `VoiceConfig.defaults` | `length_scale`, `noise_scale`, `noise_w_scale`, which are `None` when `config.json` omits them, so the engine's default applies |
| `VoiceConfig.assets` | `engine_params`, without `providers` |
| `SynthesisConfig.stages: Stages` | Exists in the code. Replaces `add_diacritics`, `diacritizer_model`, `chunk`, which work with a `DeprecationWarning` |
| `SynthesisConfig.speaker`, `language`, `reference` | Exist in the code. Replace `speaker_id`, `lang_id`, `speaker_reference`, `speaker_reference_text`, `speaker_reference_lang`, which work with a `DeprecationWarning` |
| `SynthesisConfig.audio: AudioPost` | Exists in the code. Replaces `normalize_audio`, `target_loudness`, `volume`, `super_resolution`, `super_resolution_model`, `vc_reference`, `vc_engine`, which work with a `DeprecationWarning` |
| `SynthesisConfig.extra` | Exists in the code. Replaces `extra_params`, which works with a `DeprecationWarning` |
| `TTSVoice.load(model, config=None, *, frontend=None, providers=None, runtime=None, warmup=False)` | `TTSVoice.load(model_path, config_path, vocab_path, tokenizer_config_path, phonemes_txt, phoneme_map, lang_code, phoneme_type_str, alphabet_str, engine_params, providers, use_cuda, warmup)` |
| `voice.with_frontend(frontend)` | Reassignment of `voice.phonemizer`, `phoneme_type` and `alphabet` after load, as the model manager does |
| `voice.prepare`, `voice.render` | Exist in the code with this signature |
| `voice.synthesize(text, config=None, *, alignments=False)` | `syn_config` and `include_alignments` |
| `AudioChunk.durations` | `phoneme_id_samples`, kept as a property |
| `EngineRequest`, `EngineResult` | `AdapterSynthesisRequest`, `AdapterSynthesisResult` |
| Adapter hooks `build_feed`, `parse_outputs`, `default_params`, `detect` | `build_feed_dict`, and the optional `param_labels` and `parse_config`, which are deleted |
| `Runtime(fetch, open_session)` | Private names that phoonnx.js patches: `_resolve`, `_fetch_onnx`, `make_session`, `detect_engine` |
| `TTSModelInfo` as the index entry only | Index fields `phonemizer_lang`, `phoneme_type`, `phonemizer_model`, `alphabet`, `vocab_override`, the format-valued `engine`, `phoneme_map_url`, and the `info.config` property |
| `phoonnx.compat.convert_config` | The loaders of `config_loaders.py` and the `from_piper_config`, `from_vosk_config`, `from_mimic3_config`, `from_coqui_config` and `from_tokens_txt` constructors |
| `phoonnx.compat.export_piper`, `export_sherpa` | Exist in the code. The training side-cars use their writers, `to_piper_config` and `to_sherpa_tokens` |
| `OPTION_ALIASES` in the OPM plugin | The per-key handling of `mycroft.conf` options |
| Raise sites for `FrontendMismatchError` and `UnsupportedConfigError` | The classes exist in the code. Nothing raises them |

### The three configs

`VoiceConfig` holds `engine` (the architecture, such as `vits` or `matcha`), `sample_rate`, `num_symbols`, `frontend`, `tokenizer`, `speakers`, `languages`, `defaults`, `assets` and `phonemizer_identity`. Its `lang` property reads `frontend.lang`. The native `config.json` keeps its keys. `VoiceConfig.from_dict` builds `frontend` from `lang_code`, `phoneme_type`, `alphabet`, `phonemizer_model`, `keep_punctuation`, `inference.add_diacritics` and `inference.diacritizer_model`. It builds `tokenizer` from `phoneme_id_map` and the blank, pad, BOS and EOS keys. A fact that the format lacks gets its own optional key. No existing key changes its meaning or place.

`SynthesisConfig` is frozen. Its groups are:

- `stages`: the scriptconv `Stages` class, with `diacritize`, `diacritizer_model`, `normalize`, `chunk` and `lang`.
- Flat engine fields: `speaker` (a name or an id), `language`, `length_scale`, `noise_scale`, `noise_w_scale`, `temperature`, `top_p`, `exaggeration`, `reference` and `extra`.
- `audio`: `normalize`, `target_loudness`, `volume`, `super_resolution`, `super_resolution_model`, `voice_conversion`, `voice_conversion_engine` and `stretch_pace`.
- Text options: `phonetic_spellings` and `input_alphabet` (the alphabet of the caller's text).

The 1.x flat names, such as `add_diacritics`, `speaker_id` and `normalize_audio`, work as keywords, as attributes and in `SynthesisConfig.from_dict`, with a `DeprecationWarning`. [Configuration](configuration.md) has the table of each 1.x name and its field.

### Voice steps

The target form of the call sequence is:

```python
# Target API design. This code does not run.
voice = TTSVoice.load("model.onnx")
other = voice.with_frontend(frontend)        # another voice, same ONNX session
for utterance in voice.prepare("Hello"):
    chunk = voice.render(utterance, SynthesisConfig(speaker="anna"))
```

`with_frontend` checks `keep_punctuation` and the alphabet against the voice, and probes the backend symbols against the vocabulary. A mismatch raises `FrontendMismatchError`. `synthesize(text)` stays as `render` over `prepare`, and `synthesize_ssml` does the same for each SSML piece. `write_wav` exists in the code.

### Foreign formats and export

At runtime `TTSVoice.load` reads only the native `config.json`. For any other format it raises `UnsupportedConfigError`, and the message names `phoonnx.compat.convert_config`. The runtime never imports `phoonnx.compat`. The command `phoonnx-voices convert-config` exposes the conversion.

`phoonnx.compat` also writes. `export_piper(voice, dest)` and `export_sherpa(voice, dest)` write a Piper or sherpa-onnx voice from a loaded native voice. An export that the target cannot run raises `ExportNotSupported` with the reason. Both exist in the code, and the command `phoonnx-voices export` runs them.

### Engine adapters

An adapter must define `detect(session, config)` (static), `default_params()`, `build_feed(request)` and `parse_outputs(outputs)`. It may define `configure`, `encode_text` (with the class flag `reads_text = True`), `max_input_tokens` and `synthesize` for engines with more than one graph. `EngineRequest` holds `ids`, `streams`, `speaker`, `language` and `params`. `EngineResult` holds `audio`, `durations` and `extras`. The engine enum equals the registry names, and a test checks it. The one runtime alias is `kokoro` to `styletts2`.

### Package root and errors

The root of the code exports `TTSVoice`, `VoiceConfig`, `SynthesisConfig`, `AudioPost`, `SpeakerReference`, `Utterance`, `AudioChunk`, `TTSModelManager`, `Stages`, `FrontendConfig` and `Chunk`. The target adds `Runtime`.

The hierarchy of `phoonnx.errors` exists in the code, and `VoiceLoadError` and `ExportNotSupported` have their raise sites. The target is the other raise sites. `with_frontend` raises `FrontendMismatchError`. A foreign config raises `UnsupportedConfigError`.

### Model manager

`TTSModelInfo` keeps the voice id, the display name, the language, `requires_reference`, the model URLs and a voice preset. Text facts live only in `config.json`. A model with many voices gets one native `config.json` for the model repo. Each index entry names its preset. `Runtime(fetch, open_session)` lets a host such as phoonnx.js supply its own downloads and ONNX sessions. `save()` takes the registry lock.

### Training

`phoonnx_train` builds its `TextFrontend` from the same `FrontendConfig` that `export_onnx` writes into `config.json` through `VoiceConfig.to_dict()`. That is the one writer of the format.

---

[← Vocoders](vocoders.md) · [Home](index.md)
