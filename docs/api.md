# Python API

This page is for developers who use phoonnx as a library. Part 1 is a walk-through of the API that exists in the code. Part 2 lists each name with its module. Part 3 describes the target API for phoonnx 2.0. For every target object it names the 1.x name that it replaces.

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

`engine` names the file format here (`PHOONNX`), not the model architecture. See [Part 3](#part-3-the-target-api-for-phoonnx-20) for the planned change.

### 2. Turn text into model input

`TTSVoice.prepare` is the text half of synthesis. It reads text and yields one `Utterance` for each engine call. An `Utterance` holds the text, the symbols that the tokenizer read, the token ids, and the side streams.

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
Utterance(text='hi hi', symbols=('h', 'i', ' ', 'h', 'i'), ids=(1, 0, 4, 0, 5, 0, 3, 0, 4, 0, 5, 0, 2), streams={})
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

`SynthesisConfig` holds the settings of one call. Each engine setting is `None` until you set it. A `None` scale takes the value of the `inference` block in the voice's `config.json`, and then the engine's default.

```python
from phoonnx.config import SynthesisConfig

print(config.length_scale, SynthesisConfig().length_scale)
print(SynthesisConfig(speaker_id=2, length_scale=1.2).speaker_id)
```

```text
None None
2
```

This config sets no `inference` scales and the call sets none, so the engine's own default applies. A scale in the call wins over one in `config.json`, and that wins over the engine's default.

## Part 2: What exists in 1.x

This part lists the names that exist in the 1.x code. The package root exports nine names, listed below. Import every other name from its own module.

### Voice and audio

| Name | Module | Role |
| --- | --- | --- |
| `TTSVoice` | `phoonnx.voice` | A loaded voice. A dataclass built by `TTSVoice.load`. |
| `TTSVoice.load` | `phoonnx.voice` | Takes `model_path`, `config_path`, `vocab_path`, `tokenizer_config_path`, `phonemes_txt`, `phoneme_map`, `lang_code`, `phoneme_type_str`, `alphabet_str`, `engine_params`, `providers`, `warmup` and the deprecated `use_cuda`. |
| `TTSVoice.prepare` | `phoonnx.voice` | Text to `Iterator[Utterance]`. |
| `TTSVoice.render` | `phoonnx.voice` | One `Utterance` to one `AudioChunk`. |
| `TTSVoice.synthesize` | `phoonnx.voice` | Text to `Iterable[AudioChunk]`. Argument `syn_config`, flag `include_alignments`. |
| `TTSVoice.synthesize_ssml` | `phoonnx.voice` | SSML to `Iterable[AudioChunk]`. Arguments `lang` and `on_mark`. |
| `TTSVoice.synthesize_wav`, `synthesize_ssml_wav` | `phoonnx.voice` | Write a WAV file through `write_wav`. |
| `TTSVoice.frontend` | `phoonnx.voice` | The scriptconv `TextFrontend` of the voice. |
| `TTSVoice.phonemize` | `phoonnx.voice` | Text to phonemes. |
| `Utterance` | `phoonnx.voice` | Frozen: `text`, `symbols`, `ids`, `streams`. |
| `AudioChunk` | `phoonnx.voice` | Audio plus `phoneme_ids`, `phoneme_id_samples` and `phoneme_alignments`. |
| `write_wav` | `phoonnx.audio` | Writes any chunk stream to a WAV writer. |

`TTSVoice.phonemes_to_ids` and `TTSVoice.phoneme_ids_to_audio` still work. They are deprecated, and each warns with a removal version.

### Configuration

| Name | Module | Role |
| --- | --- | --- |
| `VoiceConfig` | `phoonnx.config` | What the model is. `from_dict` reads a config and `to_native_dict` writes the native format. |
| `SynthesisConfig` | `phoonnx.config` | One call. A flat list of fields. |
| `Engine` | `phoonnx.config` | 32 members. Names a file format or an architecture. |
| `Alphabet`, `PhonemeType` | `phoonnx.config` | Re-exports of scriptconv enums. `PhonemeType` is scriptconv `Phonemizer`. |
| `Chunk` | `phoonnx.config` | Re-export of the scriptconv enum behind `SynthesisConfig.chunk`. |
| `TTSModelManager`, `TTSModelInfo` | `phoonnx.model_manager` | The voice index and the index entry. |

### Engines and errors

The engine registry in `phoonnx.engines` holds 30 names, and `phoonnx.engines.list_engines()` returns them. The `Engine` enum has 32 members. The two lists differ by these names:

```python
from phoonnx.config import Engine
from phoonnx.engines import list_engines

print(len(Engine), len(list_engines()))
print(sorted(set(e.value for e in Engine) ^ set(list_engines())))
```

```text
32 30
['coqui', 'mimic3', 'phoonnx', 'piper', 'transformers', 'vits', 'vits_prior_split', 'vits_streaming']
```

An adapter subclasses `BaseOnnxAdapter` in `phoonnx.engines.base`. It receives an `AdapterSynthesisRequest` and returns an `AdapterSynthesisResult`. `phoonnx.errors` defines `PhoonnxError` and the classes below it:

- `VoiceLoadError`, with the subclasses `FrontendMismatchError`, `UnsupportedConfigError` and `PrecisionUnavailable`.
- `ExportNotSupported`.
- `UnsupportedTokenizer`, `UnsupportedSentencePieceModel`, `UnsupportedVoiceLanguage` and `VoiceExceedsMemoryBudget`.

`PrecisionUnavailable` and `UnsupportedVoiceLanguage` also subclass `ValueError`. `VoiceExceedsMemoryBudget` also subclasses `RuntimeError`. Only `PrecisionUnavailable` and the four classes that predate the hierarchy are raised in 1.x. `VoiceLoadError`, `FrontendMismatchError`, `UnsupportedConfigError` and `ExportNotSupported` have no raise site until the 2.0 steps in Part 3 land.

The root package exports `TTSVoice`, `AudioChunk`, `Utterance`, `VoiceConfig`, `SynthesisConfig`, `TTSModelManager`, `Stages`, `FrontendConfig` and `Chunk`. Each name loads on first access, so `import phoonnx` stays cheap.

### Writing the native config

`VoiceConfig.to_native_dict()` writes the native format.

```python
d = voice.config.to_native_dict()
print(sorted(d))
print(d["phoonnx_version"])
```

```text
['add_blank_char', 'add_blank_word', 'alphabet', 'audio', 'blank', 'blank_at_end', 'blank_at_start', 'blank_between', 'blank_word', 'bos', 'engine', 'engine_params', 'eos', 'fold_compounds', 'hop_length', 'inference', 'keep_punctuation', 'lang_code', 'lang_id_map', 'lang_tokens', 'num_langs', 'num_speakers', 'num_symbols', 'pad', 'phoneme_id_map', 'phoneme_type', 'phonemizer_model', 'phoonnx_version', 'speaker_id_map', 'use_eos_bos', 'word_sep_token']
1.0
```

The `phoonnx_version` value is a fixed string. It is not the installed package version. The 2.0 `to_dict()` writes the real version.

## Part 3: The target API for phoonnx 2.0

The objects in this part are targets, except where a row says that a name exists in 1.x. Each object is the target of a task in the roadmap. The code examples in Parts 1 and 2 keep working through the 1.x shims until 2.0.

### Design rules

1. One concept has one name and one home.
2. Three objects have three lifetimes. `VoiceConfig` is what the model is: read once, frozen. `TextFrontend` is how text becomes symbols: built once from `VoiceConfig.frontend`. `SynthesisConfig` is what one call asks for: every field optional, `None` means the voice's value.
3. Every config is plain data with `to_dict()` and `from_dict()`, and a JSON Schema ships in the wheel. A config holds no callable and no instance.
4. One precedence rule: call, then voice injection, then `config.json`, then engine default.
5. A loaded voice does not change. A different frontend makes a new voice that shares the ONNX session.
6. The runtime reads one config format: the native `config.json`. Other formats are converted once, offline, by `phoonnx.compat`.

### Target objects and the names they replace

| Target | Replaces in 1.x |
| --- | --- |
| `VoiceConfig` frozen, with `engine` as the architecture, `frontend`, `tokenizer`, `speakers`, `languages`, `defaults`, `assets` | Mutable `VoiceConfig`, where `engine` can mean a file format such as `piper` |
| `VoiceConfig.frontend: FrontendConfig` | `lang_code`, `phoneme_type`, `alphabet`, `phonemizer_model`, `keep_punctuation`, `add_diacritics`, `diacritizer_model`, `lang_tokens` |
| `VoiceConfig.tokenizer: TokenizerConfig` | The copies of tokenizer fields on `VoiceConfig` (`blank_at_start`, `blank_at_end`, `pad_token`, `blank_token`, `bos_token`, `eos_token`, `word_sep_token`, `blank_between`) and the format-specific tokenizer constructors |
| `VoiceConfig.speakers`, `languages` | `speaker_id_map`, `lang_id_map` |
| `VoiceConfig.defaults` | `length_scale`, `noise_scale`, `noise_w_scale`, which are `None` when `config.json` omits them, so the engine's default applies |
| `VoiceConfig.assets` | `engine_params`, without `providers` |
| `VoiceConfig.to_dict()` and a JSON Schema | `to_native_dict()` |
| `SynthesisConfig.stages: Stages` | `add_diacritics`, `diacritizer_model`, `chunk` |
| `SynthesisConfig.speaker`, `language`, `reference` | `speaker_id`, `lang_id`, `speaker_reference`, `speaker_reference_text`, `speaker_reference_lang` |
| `SynthesisConfig.audio: AudioPost` | `normalize_audio`, `target_loudness`, `volume`, `super_resolution`, `super_resolution_model`, `vc_reference`, `vc_engine` |
| `SynthesisConfig.extra` | `extra_params` |
| `TTSVoice.load(model, config=None, *, frontend=None, providers=None, runtime=None, warmup=False)` | `TTSVoice.load(model_path, config_path, vocab_path, tokenizer_config_path, phonemes_txt, phoneme_map, lang_code, phoneme_type_str, alphabet_str, engine_params, providers, use_cuda, warmup)` |
| `voice.with_frontend(frontend)` | Reassignment of `voice.phonemizer`, `phoneme_type` and `alphabet` after load, as the model manager does |
| `voice.prepare`, `voice.render` | Exist in 1.x with this signature |
| `voice.synthesize(text, config=None, *, alignments=False)` | `syn_config` and `include_alignments` |
| `AudioChunk.durations` | `phoneme_id_samples`, kept as a property |
| `EngineRequest`, `EngineResult` | `AdapterSynthesisRequest`, `AdapterSynthesisResult` |
| Adapter hooks `build_feed`, `parse_outputs`, `default_params`, `detect` | `build_feed_dict`, and the optional `param_labels` and `parse_config`, which are deleted |
| `Runtime(fetch, open_session)` | Private names that phoonnx.js patches: `_resolve`, `_fetch_onnx`, `make_session`, `detect_engine` |
| `TTSModelInfo` as the index entry only | Index fields `phonemizer_lang`, `phoneme_type`, `phonemizer_model`, `alphabet`, `vocab_override`, the format-valued `engine`, `phoneme_map_url`, and the `info.config` property |
| `phoonnx.compat.convert_config` | The loaders of `config_loaders.py` and the `from_piper_config`, `from_vosk_config`, `from_mimic3_config`, `from_coqui_config` and `from_tokens_txt` constructors |
| `phoonnx.compat.export_piper`, `export_sherpa` | The writers in `phoonnx_train/engines/vits.py` |
| `OPTION_ALIASES` in the OPM plugin | The per-key handling of `mycroft.conf` options |
| Raise sites for `VoiceLoadError`, `FrontendMismatchError`, `UnsupportedConfigError` and `ExportNotSupported` | The classes exist in 1.x. Nothing raises them |

### The three configs

`VoiceConfig` holds `engine` (the architecture, such as `vits` or `matcha`), `sample_rate`, `num_symbols`, `frontend`, `tokenizer`, `speakers`, `languages`, `defaults`, `assets` and `phonemizer_identity`. Its `lang` property reads `frontend.lang`. The native `config.json` keeps its keys. `VoiceConfig.from_dict` builds `frontend` from `lang_code`, `phoneme_type`, `alphabet`, `phonemizer_model`, `keep_punctuation`, `inference.add_diacritics` and `inference.diacritizer_model`. It builds `tokenizer` from `phoneme_id_map` and the blank, pad, BOS and EOS keys. A new fact gets a new optional key. No existing key changes its meaning or place.

`SynthesisConfig` is frozen. Its groups are:

- `stages`: the scriptconv `Stages` class, with `diacritize`, `diacritizer_model`, `normalize`, `chunk` and `lang`.
- Flat engine fields: `speaker` (a name or an id), `language`, `length_scale`, `noise_scale`, `noise_w_scale`, `temperature`, `top_p`, `exaggeration`, `reference` and `extra`.
- `audio`: `normalize`, `target_loudness`, `volume`, `super_resolution` and `voice_conversion`.

`SynthesisConfig.from_dict` also reads the flat 1.x names, with a `DeprecationWarning`. The names are `add_diacritics`, `speaker_id`, `lang_id`, `normalize_audio`, `vc_reference`, `extra_params` and `alphabet`.

### Voice steps

The target form of the call sequence is:

```python
# Target API. Not available in 1.x.
voice = TTSVoice.load("model.onnx")
other = voice.with_frontend(frontend)        # new voice, same ONNX session
for utterance in voice.prepare("Hello"):
    chunk = voice.render(utterance, SynthesisConfig(speaker="anna"))
```

`with_frontend` checks `keep_punctuation` and the alphabet against the voice, and probes the backend symbols against the vocabulary. A mismatch raises `FrontendMismatchError`. `synthesize(text)` stays as `render` over `prepare`, and `synthesize_ssml` does the same for each SSML piece. `write_wav` already exists.

### Foreign formats and export

At runtime `TTSVoice.load` reads only the native `config.json`. For any other format it raises `UnsupportedConfigError`, and the message names `phoonnx.compat.convert_config`. The runtime never imports `phoonnx.compat`. The command `phoonnx-voices convert-config` exposes the conversion.

`phoonnx.compat` also writes. `export_piper(voice, dest)` and `export_sherpa(voice, dest)` write a Piper or sherpa-onnx voice from a loaded native voice. An export that the target cannot run raises `ExportNotSupported` with the reason.

### Engine adapters

An adapter must define `detect(session, config)` (static), `default_params()`, `build_feed(request)` and `parse_outputs(outputs)`. It may define `configure`, `encode_text` (with the class flag `reads_text = True`), `max_input_tokens` and `synthesize` for engines with more than one graph. `EngineRequest` holds `ids`, `streams`, `speaker`, `language` and `params`. `EngineResult` holds `audio`, `durations` and `extras`. The engine enum equals the registry names, and a test checks it. The one runtime alias is `kokoro` to `styletts2`.

### Package root and errors

The root already exports `TTSVoice`, `VoiceConfig`, `SynthesisConfig`, `Utterance`, `AudioChunk`, `TTSModelManager`, `Stages`, `FrontendConfig` and `Chunk`. The target adds `AudioPost`, `SpeakerReference` and `Runtime`.

The hierarchy of `phoonnx.errors` exists in 1.x. The target is the raise sites. `VoiceLoadError` covers a missing config and a `num_symbols` mismatch. `with_frontend` raises `FrontendMismatchError`. A foreign config raises `UnsupportedConfigError`. An export that the target cannot run raises `ExportNotSupported`.

### Model manager

`TTSModelInfo` keeps the voice id, the display name, the language, `requires_reference`, the model URLs and a voice preset. Text facts live only in `config.json`. A model with many voices gets one native `config.json` for the model repo, and each index entry names its preset. `Runtime(fetch, open_session)` lets a host such as phoonnx.js supply its own downloads and ONNX sessions. `save()` takes the registry lock.

### Training

`phoonnx_train` builds its `TextFrontend` from the same `FrontendConfig` that `export_onnx` writes into `config.json` through `VoiceConfig.to_dict()`. That is the one writer of the format.
