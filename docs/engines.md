# Engine Architecture

<!-- agent-hint
covers: how inference adapters and training engines register, detect and run, and how to add one
source: phoonnx/engines/__init__.py, phoonnx/engines/base.py:BaseOnnxAdapter, phoonnx_train/engines/__init__.py, phoonnx_train/engines/base.py:BaseTrainingEngine, phoonnx_train/train.py, phoonnx_train/export_onnx.py
generated: no
verify: pytest tests/test_engine_coverage.py
see: architecture.md, vocoders.md, streaming.md, training/training.md
-->

This page describes the inference adapter and training engine registries, for developers who add or integrate a TTS back end in phoonnx.

phoonnx runs each TTS back end through a small adapter framework. Inference (`phoonnx/engines/`) and training (`phoonnx_train/engines/`) are both pluggable. To add an architecture, such as VITS, OptiSpeech or Matcha-TTS, you implement a few methods.

---

## Inference Adapters

### Responsibilities

An inference adapter lives in `phoonnx/engines/` and implements `BaseOnnxAdapter`.  Its job is to translate between the generic `phoonnx` voice layer and a specific ONNX model layout:

* **Input names** – VITS models accept `input`/`input_lengths`/`scales`; other engines may use `input_ids`/`attention_mask` or entirely different names.
* **Scale / control parameters** – VITS uses `noise_scale`/`length_scale`/`noise_w_scale`; OptiSpeech uses `d_factor`/`p_factor`/`e_factor`.
* **Output layout** – some engines return raw waveform, others return mel-spectrograms that still need a vocoder.

### Adapter Lifecycle

In this sketch, `cfg` is the voice config dict, `sess` is the `onnxruntime.InferenceSession`
of the voice, and `ids` and `lengths` are int64 arrays of shape `(1, T)` and `(1,)`.

```python
from phoonnx.engines import detect_engine, get_adapter
from phoonnx.engines.base import AdapterSynthesisRequest

# 1. Auto-detect the right adapter from config + ONNX session
adapter = detect_engine(config=cfg, session=sess)

# 2. Or look one up explicitly
adapter = get_adapter("vits")

# 3. Build a feed dict and run ONNX
request = AdapterSynthesisRequest(
    phoneme_ids=ids,
    phoneme_lengths=lengths,
    params={"noise_scale": 0.667, "length_scale": 1.0}
)
feed = adapter.build_feed_dict(request, sess)
outputs = sess.run(None, feed)

# 4. Parse outputs back to audio
result = adapter.parse_outputs(outputs, request)
```

### Registration

Add an adapter by subclassing `BaseOnnxAdapter` and calling `register_engine`:

```python
# phoonnx/engines/my_engine.py
from phoonnx.engines.base import BaseOnnxAdapter, AdapterSynthesisRequest, AdapterSynthesisResult
from phoonnx.engines import register_engine

class MyEngineAdapter(BaseOnnxAdapter):
    def build_feed_dict(self, request, session):
        ...

    def parse_outputs(self, outputs, request):
        ...

    def default_params(self):
        return {"speed": 1.0}

    @staticmethod
    def detect(config=None, session=None):
        if config and config.get("model_type") == "my_engine":
            return True
        return False

register_engine("my_engine", MyEngineAdapter, detect_priority=50)
```

Auto-detection probes the lower `detect_priority` values first. A voice config that names
a registered engine in its `engine` key skips detection, and phoonnx uses that adapter. When
no adapter matches, phoonnx logs a warning and uses the VITS adapter.

### Optional adapter hooks

The base class gives a default for each hook below. Override one only when the engine
needs it.

| Hook | Default | Override it when |
|---|---|---|
| `make_encoder(voice_config, frontend, tokenizer)` | A `SymbolEncoder` (frontend symbols through the voice's tokenizer). An adapter that overrides `encode_text` gets a `TextEncoder` that gives `encode_text` the whole prepared text. | The engine reads text through its own tokenizer, such as a BPE. See [The encoder](api.md#the-encoder). |
| `max_input_tokens` | `None`: a request is never cut. | The model reads a limited number of ids. A longer request runs as several pieces whose audio is joined. A piece ends after the last clause mark that fits, else at the last word gap that fits. Each piece gets `bos` and `eos` again. A word longer than the limit raises `ValueError`. An adapter that sets `max_input_tokens` also sets `bos`, `eos`, `word_gap` and `clause_marks` in `configure`. |
| `synthesize(request, session)` | `build_feed_dict`, `session.run`, `parse_outputs`, once for each piece of the request. | The engine runs more than one graph or a sampling loop. |
| `synthesize_stream(request, session)` | Yields the one result of `synthesize`. | The engine makes audio in steps, and `TTSVoice.stream` should yield each part when it exists. Joined, the parts must give the audio of `synthesize`. |
| `configure(voice_config)` | Does nothing. | The engine loads auxiliary graphs or reads `engine_params` once at load. |

### Existing Adapters

| Adapter | File | Detection |
|---|---|---|
| VITS / Piper / Mimic3 / Coqui / VITS2 / YourTTS-VITS | `phoonnx/engines/vits.py` | `engine` in (`vits`, `piper`, `mimic3`, `coqui`, `phoonnx`), a Piper, phoonnx, Coqui or Mimic3 config shape, or a `scales` input |
| [Streaming VITS](streaming.md) | `phoonnx/engines/vits_streaming.py` | `streaming: true` **and** a decoder graph (split encoder/decoder) |
| Prior-split VITS (Inflect-v2 Micro/Nano) | `phoonnx/engines/vits.py` (`VitsPriorSplitAdapter`) | `engine_params.decode_path` **and** the duration graph's outputs ⊇ `{m_p_exp, logs_p_exp, y_mask}` — split at the flow **prior**, not the flow/HiFiGAN boundary Streaming VITS uses |
| [Matcha](training/engines/matcha.md) | `phoonnx/engines/matcha.py` | `engine == "matcha"` (flow-matching mel + separate vocoder) |
| [GlowTTS](training/engines/glowtts.md) | `phoonnx/engines/glowtts.py` | `engine == "glowtts"` |
| [OptiSpeech](training/engines/optispeech.md) | `phoonnx/engines/optispeech.py` | `engine == "optispeech"` (wav + durations outputs) |
| [MixerTTS](training/engines/mixertts.md) | `phoonnx/engines/mixertts.py` | `engine == "mixertts"` |
| [FastPitch](training/engines/fastpitch.md) / SpeedySpeech | `phoonnx/engines/fastpitch.py` | `engine == "fastpitch"` |
| StyleTTS2 / Kokoro | `phoonnx/engines/styletts2.py` | `engine in ("styletts2", "kokoro")` — supports d-vector [cloning](cloning.md); Galician voices in [galician.md](galician.md), BSC Spanish/Catalan speakers in [bsc_multispeaker.md](bsc_multispeaker.md) |
| YourTTS | `phoonnx/engines/yourtts.py` | `engine == "yourtts"` — d-vector [cloning](cloning.md) |
| [ZipVoice](training/engines/zipvoice.md) | `phoonnx/engines/zipvoice.py` | `engine == "zipvoice"` — **iterative** engine (flow-matching ODE loop), in-context [cloning](cloning.md) |
| [F5-TTS](training/engines/f5tts.md) | `phoonnx/engines/f5tts.py` | `engine == "f5tts"` — multi-graph engine (auxiliary ONNX graphs via `aux_model_urls`) |
| Shami / HamsVITS | `phoonnx/engines/shami.py` | `engine in ("shami", "hams")` — VITS variant with per-phoneme `language_ids` for Levantine Arabic / English code-switching |
| tiny-voca | `phoonnx/engines/tinyvoca.py` | `engine == "tinyvoca"`, or a graph with `tokens`, `stress`, `voice` and `pitch_range` inputs — 4.5M-parameter American English model with a source-filter waveform head; reads dizione tokens with a stress value each, one engine call per sentence as dizione cuts the text, two voices (F2, M2), 24 kHz. The adapter adds a full stop to a sentence that ends without a mark, cuts a sentence over 500 tokens after a clause mark or between words, and scales the audio to a peak of 0.99. It reads plain text only. An inline `[[...]]` block gets one warning that names the voice and the block, and the voice leaves the block out of the text it reads. Phonetic input (`SynthesisConfig.alphabet`) is not supported: the voice logs one warning, and the tokenizer drops the characters that have no id |
| [Chatterbox](training/engines/chatterbox.md) | `phoonnx/engines/chatterbox.py` | `engine == "chatterbox"` — **autoregressive** engine (codec-LM), d-vector [cloning](cloning.md) + exaggeration |
| Vosk | `phoonnx/engines/vits.py` (shared) | `engine == "vosk"` — alphacep vosk-tts Russian voices: plain VITS exports whose front-end is scriptconv's dictionary/rule `VoskPhonemizer` |
| SuperTonic | `phoonnx/engines/supertonic.py` | `engine == "supertonic"` — multi-graph flow-matching engine (4 ONNX graphs via `aux_model_urls`), raw-text (no phonemizer), fixed per-speaker style instead of cloning |
| NeuTTS (NeuTTS Air / VieNeu / Akiti) | `phoonnx/engines/neutts.py` | `engine == "neutts"` — autoregressive single-codebook codec-LM (Qwen3 or Llama backbone + NeuCodec decoder), raw text phonemized with espeak-ng in the checkpoint's language ([front-end keys](#neutts-front-end)), an inline `[[...]]` block going into the phone string as its IPA written, in-context [cloning](cloning.md) from pre-encoded voice presets, 24 kHz |
| [Pocket TTS](pockettts.md) | `phoonnx/engines/pockettts.py` | `engine == "pockettts"` — 5-graph flow-matching codec LM with explicit stream state, raw-text (no phonemizer), state-based voices and reference [cloning](cloning.md) |
| [Spark-TTS](training/engines/sparktts.md) | `phoonnx/engines/sparktts.py` | `engine == "sparktts"` — autoregressive codec-LM (Qwen2 + BiCodec), raw-text, en/zh; preset 32-token speakers or zero-shot [cloning](cloning.md) |
| [Qwen3-TTS](training/engines/qwen3tts.md) | `phoonnx/engines/qwen3tts.py` | `engine == "qwen3tts"` — two autoregressive stages (28-layer talker + 5-layer code predictor) writing 16 code groups per 12.5 Hz frame, raw-text, 10 languages, nine fixed timbres, 24 kHz |
| [OuteTTS](training/engines/outetts.md) | `phoonnx/engines/outetts.py` | `engine == "outetts"` — autoregressive two-codebook codec-LM (Qwen3 backbone + DAC.speech decoder), raw text in 23 languages, in-context [cloning](cloning.md) from pre-encoded speaker profiles, 24 kHz |
| [ArkTTS (Zortzi + Audio8)](training/engines/arktts.md) | `phoonnx/engines/arktts.py` | `engine == "arktts"` — two autoregressive stages (24-layer slow AR + 4-layer fast AR) writing 10 codebooks per 21.5 Hz frame, raw-text, Basque or 11 languages, reference-clip voices, 44.1 kHz |
| Llasa (Llasa-1B / XCodec2) | `phoonnx/engines/llasa.py` | `engine == "llasa"` — autoregressive single-codebook codec-LM (LLaMA-3.2 backbone + XCodec2 decoder), raw text through the LLaMA chat template, in-context [cloning](cloning.md) from pre-encoded voice presets, English and Mandarin, 16 kHz |
| [OmniVoice](training/engines/omnivoice.md) | `phoonnx/engines/omnivoice.py` | `engine == "omnivoice"` — **masked-diffusion** engine: a Qwen3 backbone unmasks 8 Higgs-codec streams over 32 bidirectional full-sequence steps, raw-text, 600+ languages, in-context [cloning](cloning.md), 24 kHz |
| [Indic Parler-TTS](training/engines/indic_parler.md) | `phoonnx/engines/indic_parler.py` | `engine == "indic_parler"` — **encoder-decoder** engine: a frozen Flan-T5 encoder turns a natural-language voice description into cross-attention states for a 24-layer AR decoder over 9 delayed DAC codebooks, raw-text, 20 Indic languages + English, 44.1 kHz |
| [Orpheus](orpheus.md) | `phoonnx/engines/orpheus.py` | `engine == "orpheus"` — autoregressive codec-LM (Llama-3.2-3B backbone + SNAC decoder), raw text with emotive tags, eight named English voices, 24 kHz. **GPU engine**: 36-46x slower than real time on a 12-core CPU in fp32 ([measurement](orpheus.md#read-this-first-orpheus-is-a-gpu-engine)) |
| Magpie-TTS | `phoonnx/engines/magpie.py` | `engine == "magpie"` — encoder-decoder codec-LM (6-layer causal text encoder, 12-layer causal decoder + 2-layer local transformer refiner, NanoCodec decoder), 8 codebooks over 2 stacked frames per step, attention-prior alignment (no learned monotonic alignment), five fixed baked-context voices (no cloning), byte (ByT5) and character sub-tokenizers encoded as NeMo encodes them, picked by language or by the voice ([sub-tokenizer keys](#magpie-sub-tokenizers)); English (`english_phoneme`) through scriptconv's `nemo_ipa` phonemizer; the other G2P sub-tokenizers (de/es/pt/hi/zh/ja) need a G2P that scriptconv does not have and are refused with `NotImplementedError`, 22.05 kHz |
| MOSS-TTS-Nano | `phoonnx/engines/mosstts.py` | `engine == "mosstts"` — autoregressive RVQ-16 codec-LM (6 ONNX graphs via `aux_model_urls`), raw text via SentencePiece, in-context [cloning](cloning.md) from a reference clip alone, native 48 kHz |
| [x-vits](training/engines/xvits.md) | `phoonnx/engines/xvits.py` | `engine == "xvits"` — Piper/VITS with a flow-matching prosody model: 5 ONNX graphs (encoder, prosody predictor, duration, acoustic, prominence; the encoder is the voice's own graph, the others load from `aux_model_urls`), six styles, `*word*` emphasis, espeak ids from the voice pipeline, 22.05 kHz |
| Rápido | `phoonnx/engines/rapido.py` | `engine == "rapido"` — flow-matching DiT (4 ONNX graphs: text encoder, aligner, velocity model, VoxCPM2 AudioVAE decoder), token ids from the voice pipeline with no text path of its own, midpoint ODE with classifier-free guidance, 42 preset speakers, 48 kHz. Export with `python -m phoonnx_train.rapido.export_onnx --output-dir <dir>` |

#### Engines with no index voice

Each other registered adapter is reached by at least one voice in `phoonnx/voice_index/`.
The voices in the `phoonnx`, `piper`, `mimic3`, `coqui` and `transformers` formats reach
the VITS adapter, and the Inflect entries reach the prior-split VITS adapter. Five
adapters have no voice of their own:

| Adapter | Why no index voice |
| --- | --- |
| `vits_streaming` | It is a mode of a VITS voice, not a voice. A config with `"streaming": true` selects it, and phoonnx splits the model on first load. See [streaming](streaming.md). |
| `yourtts` | `Phoonnx/yourtts-multi-graphemes` holds the model and six fixed-speaker configs (English and Portuguese). The index has no entry for them. |
| `zipvoice` | It clones from a reference clip and its transcript on each request, and the Phoonnx organisation has no converted ZipVoice model. |
| `tinyvoca` | The mirror `Phoonnx/tinyvoca-en_us` is not public. |
| `xvits` | The weights of `fdemelo/x-vits` are CC BY-NC 4.0, and the Phoonnx organisation has no mirror of them. |

`tests/test_engine_coverage.py` fails when an adapter has no voice and is not in this table.

#### Magpie sub-tokenizers

A Magpie checkpoint joins one sub-tokenizer per language into one id space. Two keys
decide which sub-tokenizer encodes a voice's text and how:

| Key | Where | What it sets |
| --- | --- | --- |
| `tokenizer_kinds` | `tokenizer.json`, written by `scripts/conversion/magpie/dump_assets.py` | The encoder of each sub-tokenizer. `{"kind": "byt5"}` is NeMo's ByT5 byte tokenizer: each UTF-8 byte, then `</s>`. `{"kind": "chars", ...}` is NeMo's `BaseCharsTokenizer` (`"accept": "alnum"`) or `ArabicCharsTokenizer` (`"accept": "any"`), with its punctuation list, `pad_with_space` and text preprocessing. `{"kind": "IPATokenizer"}` is NeMo's `IPATokenizer` over `IpaG2p`; phoonnx reads `english_phoneme` through scriptconv's `nemo_ipa` phonemizer, which gives one character per vocabulary symbol. Any other kind, and an `IPATokenizer` of another language, needs a G2P that scriptconv does not have, and phoonnx refuses it. |
| `tokenizer_name` | the voice's `engine_params` | The sub-tokenizer to use, for example `text_ce_tokenizer` or `swahili_chartokenizer`. Without it, the voice uses the sub-tokenizer that the checkpoint's `language_to_tokenizer` gives for its exact tag, such as `ar-SA`. A tag with no entry of its own takes the first entry with the same base language: on the base checkpoint, `ar` takes `ar-AE`. |

A `tokenizer.json` with `tokenizer_kinds` encodes as NeMo's `AggregatedTTSTokenizer`
does, and a voice can name any byte or character sub-tokenizer in it, or `english_phoneme`.
A voice that names a sub-tokenizer the asset does not have, or another G2P one, does not load.

A `tokenizer.json` without `tokenizer_kinds` uses fixed encoders: fr, it, vi and ko
as bytes with no `</s>`, and ar as every known character, with no space collapse and no
space pad. It can only encode a language with a sub-tokenizer that the checkpoint maps to
the same base language. A voice in `ar` can name `arabic_SA_chartokenizer`, but a voice in
`fr` that names `english_phoneme` does not load.

A fine-tune whose checkpoint has no language map names its sub-tokenizer:

```json
"engine_params": {"lang": "ka", "tokenizer_name": "text_ce_tokenizer"}
```

---

## Training Engines

### Responsibilities

A training engine lives in `phoonnx_train/engines/` and implements `BaseTrainingEngine`.  It encapsulates everything that differs between architectures:

* **Model creation** – build the PyTorch Lightning module with the right hyper-parameters.
* **ONNX export** – convert a checkpoint to ONNX and embed any architecture-specific metadata.
* **Quality presets** – map tier names (`x-low`, `medium`, `high`) to hyper-parameter overrides.
* **Checkpoint loading** – custom resume logic (e.g. encoder size mismatch tolerance).

### Engine Lifecycle

```python
from pathlib import Path

from phoonnx_train.engines import get_engine
from phoonnx_train.engines.base import TrainingEngineConfig

engine = get_engine("vits")

# 1. Create model
cfg = TrainingEngineConfig(
    num_symbols=133,
    num_speakers=1,
    sample_rate=22050,
    extra={"inter_channels": 192, "hidden_channels": 192}
)
model = engine.create_model(cfg, dataset_paths=[Path("/data")])

# 2. Train with PyTorch Lightning …

# 3. Export to ONNX
onnx_path = engine.export_onnx(
    checkpoint_path=Path("epoch=100.ckpt"),
    config_path=Path("config.json"),
    output_dir=Path("./exported"),
)
```

### Registration

Add a training engine by subclassing `BaseTrainingEngine` and calling `register_engine`:

```python
# phoonnx_train/engines/my_engine.py
from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig
from phoonnx_train.engines import register_engine

class MyTrainingEngine(BaseTrainingEngine):
    def create_model(self, config, dataset_paths, **kwargs):
        ...

    def export_onnx(self, checkpoint_path, config_path, output_dir, **kwargs):
        ...

    def quality_presets(self):
        return {
            "x-low":  {"hidden": 64},
            "medium": {"hidden": 128},
            "high":   {"hidden": 256},
        }

register_engine("my_engine", MyTrainingEngine)
```

### Existing Engines

| Engine | File | Quality Presets |
|---|---|---|
| VITS | `phoonnx_train/engines/vits.py` | `x-low`, `medium`, `high` |
| [GlowTTS](training/engines/glowtts.md) | `phoonnx_train/engines/glowtts.py` | see page |
| [Matcha](training/engines/matcha.md) | `phoonnx_train/engines/matcha.py` | see page |
| [OptiSpeech](training/engines/optispeech.md) | `phoonnx_train/engines/optispeech.py` | see page |
| [FastPitch](training/engines/fastpitch.md) / `speedyspeech` | `phoonnx_train/engines/fastpitch.py` | see page |
| [ZipVoice](training/engines/zipvoice.md) | `phoonnx_train/engines/zipvoice.py` | see page |
| `mixer` or `mixertts` / [MixerTTS](training/engines/mixertts.md) | `phoonnx_train/engines/mixer.py` | see page |
| StyleTTS2 (`styletts2`, `styletts2-aligner`, `styletts2-plbert`, `styletts2-pitch`) | `phoonnx_train/engines/styletts2*.py` | see page |
| YourTTS | `phoonnx_train/engines/yourtts.py` | see page |
| [Rápido](training/engines/rapido.md) | `phoonnx_train/engines/rapido.py` | `base`, `low` |
| [SuperTonic](training/engines/supertonic.md) | `phoonnx_train/engines/supertonic.py` | see page |

`Chatterbox`, `F5-TTS` and `Pocket TTS` have inference adapters (`phoonnx/engines/`) and
no `phoonnx_train` training engine.

---

## Engine-Aware CLI

### Training

`train.py` accepts `--engine` and delegates model creation / checkpoint loading to the selected engine:

```bash
python -m phoonnx_train.train \
  --dataset-dir /data \
  --engine vits \
  --quality medium \
  --max-epochs 1000
```

`train.py` reads the `--quality` tier from the engine's `quality_presets()`, so each engine
defines its own tier names. An unknown tier logs a warning, and training uses `medium` or,
if the engine has no `medium`, its first tier.

### ONNX Export

`export_onnx.py` accepts `--engine`. It passes the checkpoint, `--config`, `--output-dir`,
`--generate-tokens` and `--piper` to the engine's `export_onnx()` method:

```bash
python -m phoonnx_train.export_onnx epoch=500.ckpt \
  --config config.json \
  --engine vits \
  --output-dir ./exported \
  --generate-tokens \
  --piper
```

`--add-phoneme-alignment` runs after the engine export and adds a phoneme-alignment output
to the exported graph.

---

## Configuration

### VoiceConfig engine params

`VoiceConfig` stores engine-specific metadata in `engine_params` (parsed from JSON config at load time):

```python
from phoonnx.config import VoiceConfig

cfg = VoiceConfig.from_dict(config, engine_params={"noise_scale": 0.5})
```

These are forwarded to the adapter as `request.params` at synthesis time.

#### NeuTTS front end

A NeuTTS fine-tune reads phonemes in the form its own training code made them. Three
`engine_params` keys describe that front end, so a fine-tune needs a voice config and no
code. A voice without them gets the Akiti-TTS front end.

| Key | Default | What it sets |
| --- | --- | --- |
| `espeak_voice` | `lfn` | The espeak-ng language the checkpoint was trained with, for example `pl` or `en-us`. |
| `prompt_template` | `vieneu` | `vieneu` is the bare prompt of VieNeu-TTS and Akiti-TTS, with at most 200 reference codes. `neuphonic` is the chat turn of Neuphonic's NeuTTS Air and Nano (`user: Convert the text to speech:` ... `assistant:`), with every reference code. |
| `phoneme_substitutions` | none | A list of `[pattern, replacement]` regular-expression pairs. They run on the phones in list order. The voice does not load if the value is not a list of string pairs. It also does not load if a pattern does not compile, or if a replacement names a group that is not in its pattern. |

After the substitutions, runs of whitespace become one space for every voice. The
newline that espeak-ng writes at a clause break, such as an en dash, also becomes a
space, as in the `phonemizer` package that VieNeu-TTS and Akiti-TTS use.

The voice config for `shefowl/neutts-nano-polish`:

```json
"engine_params": {
  "espeak_voice": "pl",
  "prompt_template": "neuphonic",
  "phoneme_substitutions": [["([a-zA-Z0-9])", " \\1"]]
}
```

### SynthesisConfig extra params

Per-call overrides go through `SynthesisConfig.extra_params`:

```python
from phoonnx.config import SynthesisConfig

syn = SynthesisConfig(
    length_scale=1.2,
    extra_params={"d_factor": 0.9}
)
```

---

## Adding an Engine (Checklist)

1. **Inference**
   - Subclass `BaseOnnxAdapter`.
   - Implement `build_feed_dict`, `parse_outputs`, `default_params`.
   - Implement `detect()` for auto-discovery.
   - Call `register_engine()` in your module.

2. **Training**
   - Subclass `BaseTrainingEngine`.
   - Implement `create_model`, `export_onnx`, `quality_presets`.
   - Optionally override `load_checkpoint()` for custom resume logic.
   - Call `register_engine()` in your module.

3. **CLI / packaging**
   - Ensure your module is imported somewhere (e.g. in `__init__.py`) so registration runs.
   - Add tests that exercise `detect_engine()` and `get_engine("your_engine")`.

---

## Testing

The test suite covers the engine-independent paths. For an added engine, check these:

```python
from phoonnx.engines import detect_engine, get_adapter, list_engines
from phoonnx_train.engines import get_engine, list_engines as list_train_engines

assert "your_engine" in list_engines()
assert "your_engine" in list_train_engines()

# Auto-detection
adapter = detect_engine(config={"model_type": "your_engine"})
assert adapter.__class__.__name__ == "YourEngineAdapter"
```

---

[Docs index](index.md) · Previous: [Architecture](architecture.md) · Next: [Vocoders](vocoders.md)
