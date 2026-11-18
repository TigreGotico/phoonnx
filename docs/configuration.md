# Configuration Reference

This page is the exhaustive reference for advanced users: every `VoiceConfig` and
`SynthesisConfig` field, the `model.json` schema, engine detection, execution providers and precision.
For a task-oriented walkthrough see [Usage](usage.md) instead.

## VoiceConfig

`VoiceConfig` holds all model-level configuration parsed from the voice's `model.json`.

```python
from phoonnx.config import VoiceConfig
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_symbols` | `int` | — | Number of phoneme/token IDs in the vocabulary |
| `num_speakers` | `int` | — | Number of speakers (1 for single-speaker) |
| `num_langs` | `int` | — | Number of languages (1 for monolingual) |
| `sample_rate` | `int` | `16000` | Output audio sample rate in Hz |
| `lang_code` | `str` \| `None` | `"und"` | BCP-47 language code (e.g. `"en-US"`) |
| `phoneme_type` | `PhonemeType` | — | Phonemizer backend |
| `alphabet` | `Alphabet` | — | Phoneme representation |
| `engine` | `Engine` | `PHOONNX` | Training framework the model was built with |
| `noise_scale` | `float` | `0.667` | Default generator noise scale |
| `length_scale` | `float` | `1.0` | Default phoneme length scale |
| `noise_w_scale` | `float` | `0.8` | Default phoneme width noise scale |
| `hop_length` | `int` | `256` | Model frames → audio samples factor for [phoneme alignment](alignment.md) |
| `add_diacritics` | `bool` | `False` | Auto-add diacritics (Arabic/Hebrew) |
| `diacritizer_model` | `str` | `rawi-ensemble` | Diacritizer model name (Arabic `text2tashkeel`); round-tripped through the config's `inference` block |
| `phonemizer_model` | `str` \| `None` | `None` | Model path/id or variant selector for phonemizers that take one (ByT5, AhoTTS, Cotovia, arbtok) |
| `keep_punctuation` | `bool` | `False` (`True` for Piper configs) | The model was trained with clause punctuation (`, . ? ! : ;`), so synthesis keeps it. A phoonnx config without the key, or a Piper side-car phoonnx exported (`piper_version` starting with `phoonnx`, or an `alphabet` key), was trained without it |
| `phonemizer_identity` | `dict` \| `None` | `None` | The phonemizer that made the training phonemes. See [Phonemizer provenance](#phonemizer-provenance) |
| `speaker_id_map` | `dict` | `{}` | Maps speaker names to integer IDs |
| `lang_id_map` | `dict` | `{}` | Maps language codes to integer IDs (multi-lang voices) |
| `lang_tokens` | `dict` | `{}` | Optional BCP-47 → model language-token override (dialect models) |
| `engine_params` | `dict` | `{}` | Adapter-specific parameters (vocoder/speaker-encoder/style paths, etc.) |
| `tokenizer` | `TTSTokenizer` | — | Tokenizer instance |
| `blank_at_start` | `bool` | `True` | Insert a blank token before the sequence |
| `blank_at_end` | `bool` | `True` | Insert a blank token after the sequence |
| `pad_token` | `str` \| `None` | `<pad>` default | Padding token |
| `blank_token` | `str` \| `None` | pad default | Blank token inserted between tokens |
| `bos_token` | `str` \| `None` | `<bos>` default | Beginning-of-sequence token |
| `eos_token` | `str` \| `None` | `<eos>` default | End-of-sequence token |
| `word_sep_token` | `str` \| `None` | word-blank default | Token inserted between words |
| `blank_between` | `BlankBetween` | `TOKENS_AND_WORDS` | Where blank tokens are inserted |

### Loading from a Config File

`VoiceConfig` is not usually constructed directly. It is loaded automatically from a `model.json` file via `TTSVoice.load()`. To load manually:

```python
config = VoiceConfig.from_dict(my_config_dict)
```

### Engine Auto-Detection

`VoiceConfig.from_dict()` automatically detects the training engine from the config structure:

- **Chatterbox** — an explicit `engine: "chatterbox"` (needs a BPE `tokenizer.json`)
- **NeuTTS** — an explicit `engine: "neutts"` (the adapter loads its own BPE `tokenizer.json` and `voices.json` from `engine_params`)
- **Llasa** — an explicit `engine: "llasa"` (the adapter loads its own BPE `tokenizer.json` and `voices.json` from `engine_params`)
- **Pocket TTS** — an explicit `engine: "pockettts"` (multi-graph flow-matching codec LM with explicit stream state; raw-text, no phonemizer)
- **MOSS-TTS-Nano** — an explicit `engine: "mosstts"` (needs a SentencePiece `tokenizer.model`)
- **phoonnx** — presence of `phoonnx_version`
- **Piper** — presence of `piper_version`, or a list-valued `phoneme_id_map` + `phoneme_type: "espeak"|"text"` (an explicitly declared non-piper engine wins over shape-sniffing)
- **Mimic3** — presence of `phonemizer` + `phonemes` dict (requires an external `phonemes.txt`)
- **Coqui VITS** — presence of a `characters` dict with a recognized `characters_class`
- **Transformers** — driven by an external `vocab` argument passed to `from_dict()` (the voice manager supplies it from `vocab.json`), not by the ONNX shape
- **Canonical / fallback** — a config that ships its own `phoneme_id_map` and declares its `engine`/`phoneme_type`/`alphabet` is honored as-is

A `config.json` that explicitly declares its `engine` is trusted over shape-sniffing.

## SynthesisConfig

`SynthesisConfig` controls synthesis behavior per request. All fields are optional and fall back to model defaults when `None`.

```python
from phoonnx.config import SynthesisConfig

syn_config = SynthesisConfig(
    speaker_id=None,
    lang_id=None,
    length_scale=None,
    noise_scale=None,
    noise_w_scale=None,
    normalize_audio=True,
    volume=1.0,
    enable_phonetic_spellings=True,
    add_diacritics=True,
)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `speaker_id` | `int` \| `None` | `None` | Speaker index for multi-speaker models |
| `lang_id` | `int` \| `None` | `None` | Language index for multi-language models |
| `length_scale` | `float` \| `None` | `None` | Speed control: < 1 faster, > 1 slower |
| `noise_scale` | `float` \| `None` | `None` | Generator noise: affects naturalness |
| `noise_w_scale` | `float` \| `None` | `None` | Phoneme duration noise |
| `normalize_audio` | `bool` | `True` | Scale audio to full amplitude range |
| `volume` | `float` | `1.0` | Volume multiplier |
| `enable_phonetic_spellings` | `bool` | `True` | Apply word-level pronunciation overrides |
| `add_diacritics` | `bool` \| `None` | `None` | Add vowel diacritics before phonemization; `None` defers to the voice config |
| `diacritizer_model` | `str` \| `None` | `None` | Override the diacritizer model for this call; `None` inherits the voice config's `diacritizer_model` |
| `chunk` | `str` \| `None` | `None` | How much text one engine call reads: `sentence`, `clause`, `paragraph` (the text between blank lines) or `none` (the whole text). `None` reads one sentence per call. The text is read whole before it is cut. An engine that reads text, not phonemes, gets the whole text |
| `speaker_reference` | `str` \| `(audio, sr)` \| `None` | `None` | Reference clip for zero-shot [voice cloning](cloning.md) |
| `speaker_reference_text` | `str` \| `None` | `None` | Reference transcription, required by in-context cloning engines (ZipVoice) |
| `speaker_reference_lang` | `str` \| `None` | `None` | Language of the transcription, for cross-lingual cloning (defaults to the voice's `lang_code`) |
| `exaggeration` | `float` \| `None` | `None` | Expressiveness/intensity for engines that support it (Chatterbox, ~0.5) |
| `temperature` | `float` \| `None` | `None` | Sampling temperature for autoregressive engines (Chatterbox, ~0.8; `0` = greedy) |
| `top_p` | `float` \| `None` | `None` | Nucleus sampling cutoff for autoregressive engines (Chatterbox, ~0.95) |
| `vc_reference` | `str` \| `(audio, sr)` \| `None` | `None` | Target-speaker clip for post-synthesis [voice conversion](voice_conversion.md); works with every engine |
| `vc_engine` | `str` \| `None` | `None` | voiceclonnx engine for `vc_reference`; `None` selects `"openvoice"` |
| `extra_params` | `dict` | `{}` | Engine-specific per-call parameters (e.g. `d_factor`, `p_factor`, `e_factor`) |

## model.json Format

phoonnx supports multiple JSON config schemas. The native phoonnx format looks like:

```json
{
  "phoneme_type": "espeak",
  "alphabet": "ipa",
  "num_symbols": 256,
  "num_speakers": 1,
  "num_langs": 1,
  "phoneme_id_map": { "a": [1], "b": [2], ... },
  "audio": { "sample_rate": 22050 },
  "inference": {
    "noise_scale": 0.667,
    "length_scale": 1.0,
    "noise_w": 0.8,
    "add_diacritics": false,
    "diacritizer_model": "rawi-ensemble"
  },
  "blank": "_",
  "pad": "<pad>",
  "bos": "<bos>",
  "eos": "<eos>",
  "blank_at_start": true,
  "blank_at_end": true,
  "add_blank": true,
  "speaker_id_map": {},
  "lang_code": "en-US"
}
```

Piper, Mimic3, and Coqui config formats are also parsed automatically.

### Tokenizer keys

These keys of the native format control how the phonemes become ids.

| Key | Default | Effect |
|-----|---------|--------|
| `phoneme_id_map` | — | Symbol to id. A value is an id or a list of ids. A list of one id is that id. A longer list gives each id in turn, and an empty list gives no id. |
| `add_blank_char` | `true` | Put the `blank` symbol between all ids, and at the start and end as `blank_at_start` and `blank_at_end` say. |
| `add_blank_word` | `false` | Put the `blank_word` symbol after each word that gives an id. With `blank_at_end`, the last word gets one too. |
| `blank_word` | none | The symbol that `add_blank_word` puts between words. Without it, a space gives no id. |
| `use_eos_bos` | `true` | Put `bos` before the ids and `eos` after them. |
| `fold_compounds` | `true` | Read the longest map key that starts at each symbol, so that `t͡ʃ` gives one id. |
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
    "version": "7.123.11a2",
    "lang": "pt-PT",
    "revision": "sha256:7188b00e1bd7f3f8400628ce46e037ea038a5ad8312f74fa45d69c2f728ef1c3",
    "frontend": null
  }
}
```

When `TTSVoice` loads a voice, it compares the block with the phonemizer that
runs. When a field is different, it logs one warning. The warning names the
voice, each field that is different, and the trained and the running value.
The voice loads and speaks as usual. This example is a voice that was trained
on another orthography2ipa spec revision:

```text
WARNING - Voice pt_PT-tugão was trained with another phonemizer: revision: trained 'sha256:0000000000000000000000000000000000000000000000000000000000000000', running 'sha256:7188b00e1bd7f3f8400628ce46e037ea038a5ad8312f74fa45d69c2f728ef1c3'. The voice loads, but its pronunciation can differ from training.
```

The comparison follows these rules:

- `revision` is the orthography2ipa spec revision. When the two revisions are
  equal, a different `version` gives no warning, because the spec that makes
  the phonemes did not change.
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

`Engine` (in `phoonnx.config`) records which framework a voice was built with. It has 17
members. `piper`, `mimic3` and `coqui` all run through the single VITS adapter; the rest map
to a dedicated adapter. The adapter registry itself is described in [Engines](engines.md).

| `Engine` value | Kind | Per-engine guide |
|---|---|---|
| `phoonnx` | Native VITS | [Engines](engines.md) |
| `piper` | VITS (shared adapter) | [Engines](engines.md) |
| `mimic3` | VITS (shared adapter) | [Engines](engines.md) |
| `coqui` | VITS (shared adapter) | [Engines](engines.md) |
| `transformers` | HuggingFace VITS/MMS | [Engines](engines.md) |
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
| `vosk` | VITS (shared adapter), vosk-tts Russian front-end | [Engines](engines.md) |
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

## Execution Providers

Every ONNX session phoonnx creates — the voice model and the auxiliary graphs an
engine loads (vocoders, speaker encoders, text encoders, diacritizers) — runs on a
resolved, ordered list of ONNX Runtime execution providers.

Pass the list explicitly:

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
export PHOONNX_ONNX_PROVIDERS="auto"   # the default: pick the best available
```

With neither, the best provider the installed runtime offers is auto-detected, in
this preference order:

`CUDAExecutionProvider` (NVIDIA) → `ROCMExecutionProvider` (AMD) →
`MIGraphXExecutionProvider` (AMD) → `DmlExecutionProvider` (DirectML) →
`CoreMLExecutionProvider` (Apple) → `OpenVINOExecutionProvider` (Intel) →
`CPUExecutionProvider`

A requested provider that the installed runtime does not offer is skipped with a
warning, and `CPUExecutionProvider` is always appended, so synthesis keeps working
on any machine.

`use_cuda=True` is a deprecated alias for `providers=["CUDAExecutionProvider"]`.

### Runtime packages

Which providers exist depends on the installed ONNX Runtime build, not on phoonnx.
The default `onnxruntime` wheel is CPU-only (plus a few platform providers); GPU
providers need a matching build, and only one of them can be installed at a time:

| Hardware | Package | Provider |
|---|---|---|
| NVIDIA | `onnxruntime-gpu` | `CUDAExecutionProvider`, `TensorrtExecutionProvider` |
| AMD (ROCm) | `onnxruntime-rocm` | `ROCMExecutionProvider`, `MIGraphXExecutionProvider` |
| Windows (any DX12 GPU) | `onnxruntime-directml` | `DmlExecutionProvider` |
| Intel | `onnxruntime-openvino` | `OpenVINOExecutionProvider` |
| Apple | `onnxruntime` | `CoreMLExecutionProvider` |

Check what a machine actually offers:

```python
import onnxruntime
print(onnxruntime.get_available_providers())
```

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

`PrecisionUnavailable` is a `ValueError`. Its message names the voice and the
value that loads it.

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
