# Engine Architecture

<!-- agent-hint
covers: how inference adapters and training engines register, detect and run, and how to add one
source: phoonnx/engines/__init__.py, phoonnx/engines/base.py:BaseOnnxAdapter, phoonnx_train/engines/__init__.py, phoonnx_train/engines/base.py:BaseTrainingEngine, phoonnx_train/train.py, phoonnx_train/export_onnx.py, scripts/engine_matrix.py
generated: yes, by scripts/engine_matrix.py for the table between the engine-matrix markers; edit the generator, not that table
verify: .venv/bin/python -c "import re,sys,inspect; from phoonnx.engines import list_engines, get_adapter; p=open('docs/engines.md').read(); t=p.split(chr(10)+'### Existing adapters',1)[1].split('####',1)[0]; q=chr(96); rows=[[c.strip().strip(q) for c in r.split('|')[1:4]] for r in t.splitlines() if r.startswith('| '+q)]; got={n: (type(get_adapter(n)).__name__, inspect.getfile(type(get_adapter(n))).split('phoonnx/engines/')[1]) for n in list_engines()}; doc={n.strip(q): (a, f) for ns,a,f in rows for n in ns.split(q+', '+q)}; bad=sorted(n for n in set(got)|set(doc) if got.get(n)!=tuple(x.replace('phoonnx/engines/','') for x in doc.get(n,('',''))) ); print(bad); sys.exit(1 if bad else 0)" && .venv/bin/python -m pytest -q tests/test_engine_matrix.py tests/test_engine_coverage.py
see: architecture.md, vocoders.md, streaming.md, training/training.md
-->

This page describes the inference adapter registry and the training engine registry. It is for developers who add a TTS back end to phoonnx or who integrate one.

phoonnx runs each TTS back end through an adapter. Inference adapters live in `phoonnx/engines/`, and training engines live in `phoonnx_train/engines/`. To add an architecture, you write a class with a few methods and register it.

---

## Inference Adapters

### Responsibilities

An inference adapter is a subclass of `BaseOnnxAdapter` in `phoonnx/engines/`. It translates between the voice layer of phoonnx and the layout of one family of ONNX models:

* **Input names.** VITS models take `input`, `input_lengths` and `scales`. Other engines take other names, such as `x` and `x_lengths`, or `token_ids`.
* **Control parameters.** VITS uses `noise_scale`, `length_scale` and `noise_w_scale`. OptiSpeech uses `d_factor`, `p_factor` and `e_factor`.
* **Output layout.** Some engines return a waveform. Others return a mel spectrogram, which a [vocoder](vocoders.md) turns into a waveform.

A subclass must write three methods: `build_feed_dict`, `parse_outputs` and `default_params`.

### Adapter lifecycle

In this sketch, `cfg` is the voice config dict, `sess` is the `onnxruntime.InferenceSession` of the voice, and `ids` and `lengths` are int64 arrays of shape `(1, T)` and `(1,)`.

```python
from phoonnx.engines import detect_engine, get_adapter
from phoonnx.engines.base import AdapterSynthesisRequest

# 1. Detect the adapter from the config and the ONNX session
adapter = detect_engine(config=cfg, session=sess)

# 2. Or get one by name
adapter = get_adapter("vits")

# 3. Build a feed dict and run ONNX
request = AdapterSynthesisRequest(
    phoneme_ids=ids,
    phoneme_lengths=lengths,
    params={"noise_scale": 0.667, "length_scale": 1.0}
)
feed = adapter.build_feed_dict(request, sess)
outputs = sess.run(None, feed)

# 4. Turn the outputs into audio
result = adapter.parse_outputs(outputs, request)
```

### Registration

To add an adapter, make a subclass of `BaseOnnxAdapter` and call `register_engine`:

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

A voice config that names a registered engine in its `engine` key gets that adapter, with no detection. For any other voice, `detect_engine` asks the adapters in order of `detect_priority`, lowest first, and the default priority is 100. When no adapter accepts the voice, phoonnx logs a warning and uses the VITS adapter.

### Optional adapter hooks

The base class gives a default for each hook below. Override a hook only when the engine needs it.

| Hook | Default | Override it when |
|---|---|---|
| `make_encoder(voice_config, frontend, tokenizer)` | A `SymbolEncoder`: the symbols of the front end go through the tokenizer of the voice. An adapter that overrides `encode_text` gets a `TextEncoder` that gives `encode_text` the whole prepared text. | The engine reads text through its own tokenizer, such as a BPE. See [The encoder](api.md#the-encoder). |
| `max_input_tokens` | `None`: phoonnx never cuts a request. | The model reads a limited number of ids. A longer request runs as several pieces, and phoonnx joins their audio. A piece ends after the last clause mark that fits, or else at the last word gap that fits. Each piece gets `bos` and `eos` again. A word longer than the limit raises `ValueError`. An adapter that sets `max_input_tokens` also sets `bos`, `eos`, `word_gap` and `clause_marks` in `configure`. |
| `synthesize(request, session)` | `build_feed_dict`, `session.run` and `parse_outputs`, one time for each piece of the request. | The engine runs more than one graph or a sampling loop. |
| `synthesize_stream(request, session)` | Yields the one result of `synthesize`. | The engine makes audio in steps, and `TTSVoice.stream` must yield each part when it exists. Joined, the parts must give the audio of `synthesize`. |
| `configure(voice_config)` | Does nothing. | The engine loads auxiliary graphs or reads `engine_params` one time at load. |
| `length_scale_params(length_scale)` | Returns nothing: the engine reads `length_scale` itself. | The engine has a speed key of its own, such as `speed` or `pace`. Return that key for the length scale, scaled the way the engine reads it. |
| `PACE_CONTROL` | `True` | The engine cannot change its pace at all: it reads neither `length_scale` nor a speed key, as a codec language model does. Set it to `False`, and `audio.stretch_pace` then time-stretches its audio. See [SSML](ssml.md#engines-without-pace-control). |

### Existing adapters

Each row is a name in the inference registry. A voice selects the adapter when its config names the engine, or when the condition in the last column holds. The engine pages and the [architecture matrix](#architecture-matrix) give the details.

| Name | Adapter | File | Also selected by |
|---|---|---|---|
| `vits` | `VitsAdapter` | `phoonnx/engines/vits.py` | `engine` set to `piper`, `mimic3`, `coqui` or `phoonnx`, a Piper, phoonnx, Coqui or Mimic3 config shape, or a `scales` input |
| `vosk` | `VitsAdapter` | `phoonnx/engines/vits.py` | the alphacep vosk-tts Russian voices, plain VITS exports that read the scriptconv `VoskPhonemizer` |
| `vits_streaming` | `VitsStreamingAdapter` | `phoonnx/engines/vits_streaming.py` | `streaming: true` and a `decoder_path` in `engine_params`. See [Streaming](streaming.md). |
| `vits_prior_split` | `VitsPriorSplitAdapter` | `phoonnx/engines/vits.py` | a `decode_path` in `engine_params`, and a duration graph with the outputs `m_p_exp`, `logs_p_exp` and `y_mask` (the Inflect voices) |
| `matcha` | `MatchaAdapter` | `phoonnx/engines/matcha.py` | `engine` set to `matcha-tts`, a `vocoder_path`, or the inputs `x`, `x_lengths` and `scales`. See [Matcha-TTS](training/engines/matcha.md). |
| `glowtts` | `GlowTTSAdapter` | `phoonnx/engines/glowtts.py` | `engine` set to `glow_tts` or `larynx`, or a `scales` input with a mel output. See [GlowTTS](training/engines/glowtts.md). |
| `optispeech` | `OptiSpeechAdapter` | `phoonnx/engines/optispeech.py` | OptiSpeech model metadata, or the inputs `x` and `x_lengths` with a `wav` or `durations` output. See [OptiSpeech](training/engines/optispeech.md). |
| `mixertts` | `MixerTTSAdapter` | `phoonnx/engines/mixertts.py` | `engine` set to `mixer_tts`, or a `token_ids` input with `pace` and `pitch_mul` or a `mel_spec` output. See [Mixer-TTS](training/engines/mixertts.md). |
| `fastpitch` | `FastPitchAdapter` | `phoonnx/engines/fastpitch.py` | `engine` set to `fast_pitch`. See [FastPitch](training/engines/fastpitch.md). |
| `styletts2` | `StyleTTS2Adapter` | `phoonnx/engines/styletts2.py` | `engine` set to `kokoro`. It supports d-vector [cloning](cloning.md). See [Galician](galician.md) and [BSC multispeaker](bsc_multispeaker.md). |
| `yourtts` | `YourTTSAdapter` | `phoonnx/engines/yourtts.py` | only the engine name. It supports d-vector [cloning](cloning.md). |
| `zipvoice` | `ZipVoiceAdapter` | `phoonnx/engines/zipvoice.py` | only the engine name. It clones in context from a clip and its transcript. See [ZipVoice](training/engines/zipvoice.md). |
| `f5tts` | `F5TTSAdapter` | `phoonnx/engines/f5tts.py` | `engine` set to `f5-tts` or `habibi`, a `preprocess_path` with `nfe`, or the F5-TTS transformer inputs. See [F5-TTS](training/engines/f5tts.md). |
| `shami` | `ShamiAdapter` | `phoonnx/engines/shami.py` | `engine` set to `hams`, or the inputs `phoneme_ids` and `language_ids`. See [Shami](training/engines/shami.md). |
| `tinyvoca` | `TinyVocaAdapter` | `phoonnx/engines/tinyvoca.py` | the inputs `tokens`, `stress`, `voice` and `pitch_range`. American English from dizione tokens, 24 kHz, at most 500 tokens per engine call. |
| `chatterbox` | `ChatterboxAdapter` | `phoonnx/engines/chatterbox.py` | only the engine name. See [Chatterbox](training/engines/chatterbox.md). |
| `supertonic` | `SuperTonicAdapter` | `phoonnx/engines/supertonic.py` | the SuperTonic flow-matching inputs (`noisy_latent`, `text_emb`, `style_ttl` and others). See [SuperTonic](training/engines/supertonic.md). |
| `neutts` | `NeuTTSAdapter` | `phoonnx/engines/neutts.py` | only the engine name. NeuTTS Air, VieNeu and Akiti-TTS checkpoints, 24 kHz. See [NeuTTS front end](#neutts-front-end). |
| `pockettts` | `PocketTTSAdapter` | `phoonnx/engines/pockettts.py` | the inputs `sequence`, `text_embeddings`, `state_0` and `state_1`. See [Pocket TTS](pockettts.md). |
| `sparktts` | `SparkTTSAdapter` | `phoonnx/engines/sparktts.py` | only the engine name. See [Spark-TTS](training/engines/sparktts.md). |
| `qwen3tts` | `Qwen3TTSAdapter` | `phoonnx/engines/qwen3tts.py` | only the engine name. See [Qwen3-TTS](training/engines/qwen3tts.md). |
| `outetts` | `OuteTTSAdapter` | `phoonnx/engines/outetts.py` | only the engine name. See [OuteTTS](training/engines/outetts.md). |
| `arktts` | `ArkTTSAdapter` | `phoonnx/engines/arktts.py` | only the engine name. See [ArkTTS](training/engines/arktts.md). |
| `llasa` | `LlasaAdapter` | `phoonnx/engines/llasa.py` | only the engine name. Llasa-1B with the XCodec2 decoder, 16 kHz. |
| `omnivoice` | `OmniVoiceAdapter` | `phoonnx/engines/omnivoice.py` | only the engine name. See [OmniVoice](training/engines/omnivoice.md). |
| `indic_parler` | `IndicParlerAdapter` | `phoonnx/engines/indic_parler.py` | the inputs `codec_input_ids`, `prompt_input_ids` and `encoder_hidden_states`. See [Indic Parler-TTS](training/engines/indic_parler.md). |
| `orpheus` | `OrpheusAdapter` | `phoonnx/engines/orpheus.py` | only the engine name. It needs a GPU for real-time speech. See [Orpheus](orpheus.md). |
| `magpie` | `MagpieAdapter` | `phoonnx/engines/magpie.py` | `model_type` set to `magpie-tts`, or the inputs `attn_prior`, `cross_k`, `self_k` and `cond_mask`. Five fixed voices and no cloning. See [Magpie sub-tokenizers](#magpie-sub-tokenizers). |
| `mosstts` | `MossTTSNanoAdapter` | `phoonnx/engines/mosstts.py` | only the engine name. 48 kHz, cloning from a reference clip alone. See [MOSS-TTS-Nano](training/engines/mosstts.md). |
| `xvits` | `XVitsAdapter` | `phoonnx/engines/xvits.py` | a graph with only the inputs `ids` and `lengths` and only the outputs `x` and `x_mask`. See [x-vits](training/engines/xvits.md). |
| `rapido` | `RapidoAdapter` | `phoonnx/engines/rapido.py` | only the engine name. See [Rápido](training/engines/rapido.md). |

#### Engines with no index voice

Each other registered adapter loads at least one voice of `phoonnx/voice_index/`. The voices in the `phoonnx`, `piper`, `mimic3`, `coqui` and `transformers` formats load with the VITS adapter, and the Inflect entries load with the prior-split VITS adapter. Five adapters have no voice of their own:

| Adapter | Why no index voice |
| --- | --- |
| `vits_streaming` | It is a mode of a VITS voice, not a voice. A config with `"streaming": true` selects it, and phoonnx splits the model on the first load. See [streaming](streaming.md). |
| `yourtts` | `Phoonnx/yourtts-multi-graphemes` holds the model and six fixed-speaker configs, in English and Portuguese. The index has no entry for them. |
| `zipvoice` | It clones from a reference clip and its transcript on each request, and the Phoonnx organisation has no converted ZipVoice model. |
| `tinyvoca` | The mirror `Phoonnx/tinyvoca-en_us` is not public. |
| `xvits` | The weights of `fdemelo/x-vits` are CC BY-NC 4.0, and the Phoonnx organisation has no mirror of them. |

`tests/test_engine_coverage.py` fails when an adapter has no voice and is not in this table.

#### Magpie sub-tokenizers

A Magpie checkpoint joins one sub-tokenizer per language into one id space. Two keys decide which sub-tokenizer encodes the text of a voice, and how:

| Key | Where | What it sets |
| --- | --- | --- |
| `tokenizer_kinds` | `tokenizer.json`, written by `scripts/conversion/magpie/dump_assets.py` | The encoder of each sub-tokenizer. `{"kind": "byt5"}` is the NeMo ByT5 byte tokenizer: each UTF-8 byte, then `</s>`. `{"kind": "chars", ...}` is the NeMo `BaseCharsTokenizer` (`"accept": "alnum"`) or `ArabicCharsTokenizer` (`"accept": "any"`), with its punctuation list, `pad_with_space` and text preprocessing. `{"kind": "IPATokenizer"}` is the NeMo `IPATokenizer` over `IpaG2p`. phoonnx reads `english_phoneme` through the scriptconv `nemo_ipa` phonemizer, which gives one character per vocabulary symbol. Any other kind, and an `IPATokenizer` of another language, needs a G2P that scriptconv does not have, and phoonnx refuses it with `NotImplementedError`. |
| `tokenizer_name` | the `engine_params` of the voice | The sub-tokenizer to use, for example `text_ce_tokenizer` or `swahili_chartokenizer`. Without it, the voice uses the sub-tokenizer that the `language_to_tokenizer` map of the checkpoint gives for its exact tag, such as `ar-SA`. A tag with no entry of its own takes the first entry with the same base language: on the base checkpoint, `ar` takes `ar-AE`. |

A `tokenizer.json` with `tokenizer_kinds` encodes as the NeMo `AggregatedTTSTokenizer` does. A voice can name any byte or character sub-tokenizer in it, or `english_phoneme`. A voice that names a sub-tokenizer that the file does not have, or another G2P sub-tokenizer, does not load.

A `tokenizer.json` without `tokenizer_kinds` uses fixed encoders: fr, it, vi and ko as bytes with no `</s>`, and ar as every known character, with no space collapse and no space pad. It can encode only a language with a sub-tokenizer that the checkpoint maps to the same base language. A voice in `ar` can name `arabic_SA_chartokenizer`, but a voice in `fr` that names `english_phoneme` does not load.

A fine-tune whose checkpoint has no language map names its sub-tokenizer:

```json
"engine_params": {"lang": "ka", "tokenizer_name": "text_ce_tokenizer"}
```

---

## Training Engines

### Responsibilities

A training engine is a subclass of `BaseTrainingEngine` in `phoonnx_train/engines/`. It holds everything that differs between architectures:

* **Model creation.** `create_model` builds the PyTorch Lightning module with the right hyperparameters.
* **ONNX export.** `export_onnx` turns a checkpoint into ONNX and adds the metadata of the architecture.
* **Quality presets.** `quality_presets` maps tier names, such as `x-low`, `medium` and `high`, to hyperparameter overrides.
* **Checkpoint loading.** `load_checkpoint` can hold custom resume logic, such as a tolerance for a different encoder size.

### Engine lifecycle

```python
from pathlib import Path

from phoonnx_train.engines import get_engine
from phoonnx_train.engines.base import TrainingEngineConfig

engine = get_engine("vits")

# 1. Create the model
cfg = TrainingEngineConfig(
    num_symbols=133,
    num_speakers=1,
    sample_rate=22050,
    extra={"inter_channels": 192, "hidden_channels": 192}
)
model = engine.create_model(cfg, dataset_paths=[Path("/data")])

# 2. Train with PyTorch Lightning

# 3. Export to ONNX
onnx_path = engine.export_onnx(
    checkpoint_path=Path("epoch=100.ckpt"),
    config_path=Path("config.json"),
    output_dir=Path("./exported"),
)
```

### Registration

To add a training engine, make a subclass of `BaseTrainingEngine` and call `register_engine`:

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

### Existing engines

| Engine | File | Quality presets |
|---|---|---|
| `vits` | `phoonnx_train/engines/vits.py` | `x-low`, `medium`, `high` |
| `glowtts` ([GlowTTS](training/engines/glowtts.md)) | `phoonnx_train/engines/glowtts.py` | `x-low`, `medium`, `high` |
| `matcha` ([Matcha-TTS](training/engines/matcha.md)) | `phoonnx_train/engines/matcha.py` | `x-low`, `medium`, `high` |
| `optispeech` ([OptiSpeech](training/engines/optispeech.md)) | `phoonnx_train/engines/optispeech.py` | `x-low`, `medium`, `high` |
| `fastpitch`, `speedyspeech` ([FastPitch](training/engines/fastpitch.md)) | `phoonnx_train/engines/fastpitch.py` | `x-low`, `medium`, `high` |
| `zipvoice` ([ZipVoice](training/engines/zipvoice.md)) | `phoonnx_train/engines/zipvoice.py` | `base`, `low` |
| `mixer`, `mixertts` ([Mixer-TTS](training/engines/mixertts.md)) | `phoonnx_train/engines/mixer.py` | `x-low`, `medium`, `high` |
| `styletts2` | `phoonnx_train/engines/styletts2.py` | `low`, `medium`, `high` |
| `styletts2-aligner` | `phoonnx_train/engines/styletts2_aligner.py` | `low`, `medium` |
| `styletts2-plbert` | `phoonnx_train/engines/styletts2_plbert.py` | `low`, `medium` |
| `styletts2-pitch` | `phoonnx_train/engines/styletts2_pitch.py` | `medium` |
| `yourtts` | `phoonnx_train/engines/yourtts.py` | `x-low`, `medium`, `high` |
| `supertonic` ([SuperTonic](training/engines/supertonic.md)) | `phoonnx_train/engines/supertonic.py` | `base`, `low` |
| `rapido` ([Rápido](training/engines/rapido.md)) | `phoonnx_train/engines/rapido.py` | `base`, `low` |

The [architecture matrix](#architecture-matrix) shows which architectures have an inference adapter and no training engine.

---

## Architecture Matrix

phoonnx is a training library and an inference library. Each architecture is to have both sides. A gap closes when somebody adds the missing side. An engine is never removed to close a gap.

The table has one row for each name in the inference registry (`phoonnx.engines.list_engines()`) and for each architecture in the training registry (`phoonnx_train.engines.list_engines()`). The columns are:

| Column | Content |
| --- | --- |
| Inference | The adapter class that `get_adapter` returns for the name. |
| Training | The training engines of the architecture. When there is none, the roadmap task that adds one. |
| Export | `export_onnx` is the export of the training engine. `export_piper` and `export_sherpa` in `phoonnx.compat` write a phoonnx VITS voice for Piper or sherpa-onnx. |
| Index voices | The entries of `phoonnx/voice_index/` that load with the adapter. An entry in the `phoonnx`, `piper`, `mimic3`, `coqui` or `transformers` format counts for `vits`, or for `vits_prior_split` when it has a `decode_path`. |
| Id fixture | `yes` when a voice of `tests/fixtures/id_parity.json` loads with the adapter. |
| Open task | The roadmap task that adds the first index voice of the adapter. |

`python scripts/engine_matrix.py` writes the table. `tests/test_engine_matrix.py` fails when the table differs from the registries, and when an inference engine has no index voice and no open task. The open tasks are in `TRAINING_TASKS` and `VOICE_TASKS` in the script.

<!-- engine-matrix:begin -->
| Architecture | Inference | Training | Export | Index voices | Id fixture | Open task |
| --- | --- | --- | --- | --- | --- | --- |
| `arktts` | `ArkTTSAdapter` | R4.8 | `scripts/conversion/arktts` | 25 | no |  |
| `chatterbox` | `ChatterboxAdapter` | R4.8 | none | 42 | no |  |
| `f5tts` | `F5TTSAdapter` | R4.8 | none | 16 | no |  |
| `fastpitch` | `FastPitchAdapter` | `fastpitch`, `speedyspeech` | `export_onnx`, `scripts/conversion/coqui_fastpitch_export` | 4 | no |  |
| `glowtts` | `GlowTTSAdapter` | `glowtts` | `export_onnx`, `scripts/conversion/coqui_glowtts_export` | 58 | no |  |
| `indic_parler` | `IndicParlerAdapter` | R4.8 | `scripts/conversion/indicparler` | 33 | no |  |
| `llasa` | `LlasaAdapter` | R4.8 | `scripts/conversion/llasa` | 4 | no |  |
| `magpie` | `MagpieAdapter` | R4.8 | `scripts/conversion/magpie` | 30 | no |  |
| `matcha` | `MatchaAdapter` | `matcha` | `export_onnx` | 13 | no |  |
| `mixertts` | `MixerTTSAdapter` | `mixer`, `mixertts` | `export_onnx` | 5 | no |  |
| `mosstts` | `MossTTSNanoAdapter` | R4.8 | none | 19 | no |  |
| `neutts` | `NeuTTSAdapter` | R4.8 | `scripts/conversion/neutts` | 9 | no |  |
| `omnivoice` | `OmniVoiceAdapter` | R4.8 | none | 646 | no |  |
| `optispeech` | `OptiSpeechAdapter` | `optispeech` | `export_onnx` | 3 | no |  |
| `orpheus` | `OrpheusAdapter` | R4.8 | none | 8 | no |  |
| `outetts` | `OuteTTSAdapter` | R4.8 | `scripts/conversion/outetts` | 23 | no |  |
| `pockettts` | `PocketTTSAdapter` | R4.8 | none | 156 | no |  |
| `qwen3tts` | `Qwen3TTSAdapter` | R4.8 | `scripts/conversion/qwen3tts` | 15 | no |  |
| `rapido` | `RapidoAdapter` | `rapido` | `export_onnx` | 1 | no |  |
| `shami` | `ShamiAdapter` | R4.8 | `scripts/conversion/shami_tts` | 1 | no |  |
| `sparktts` | `SparkTTSAdapter` | R4.8 | none | 4 | no |  |
| `styletts2` | `StyleTTS2Adapter` | `styletts2`, `styletts2-aligner`, `styletts2-plbert`, `styletts2-pitch` | `export_onnx`, `scripts/conversion/styletts2` | 228 | no |  |
| `supertonic` | `SuperTonicAdapter` | `supertonic` | `export_onnx` | 310 | no |  |
| `tinyvoca` | `TinyVocaAdapter` | R4.8 | none | 0 | no | R3.4 |
| `vits` | `VitsAdapter` | `vits` | `export_onnx`, `export_piper`, `export_sherpa`, `scripts/conversion/coqui_vits_export`, `scripts/conversion/inflect` | 1601 | yes |  |
| `vits_prior_split` | `VitsPriorSplitAdapter` | R4.8 | none | 2 | no |  |
| `vits_streaming` | `VitsStreamingAdapter` | R4.8 | `ensure_split_vits` on a `vits` export | 0 | no | R2.7 |
| `vosk` | `VitsAdapter` | R4.8 | none | 3 | no |  |
| `xvits` | `XVitsAdapter` | R4.8 | none | 0 | no | R2.7 |
| `yourtts` | `YourTTSAdapter` | `yourtts` | `export_onnx`, `scripts/conversion/yourtts` | 0 | no | R2.7 |
| `zipvoice` | `ZipVoiceAdapter` | `zipvoice` | `export_onnx` | 0 | no | R2.7 |
<!-- engine-matrix:end -->
---

## Engine-Aware CLI

### Training

`phoonnx_train.train` takes `--engine`, and the selected engine creates the model and loads the checkpoints:

```bash
python -m phoonnx_train.train \
  --dataset-dir /data \
  --engine vits \
  --quality medium \
  --max-epochs 1000
```

The `--quality` tier comes from the `quality_presets()` of the engine, so each engine defines its own tier names. An unknown tier logs a warning, and training uses `medium`, or the first tier when the engine has no `medium`. The [training reference](training/training.md) lists every flag.

### ONNX export

`phoonnx_train.export_onnx` takes `--engine`. It gives the checkpoint, `--config`, `--output-dir`, `--generate-tokens` and `--piper` to the `export_onnx()` method of the engine:

```bash
python -m phoonnx_train.export_onnx epoch=500.ckpt \
  --config config.json \
  --engine vits \
  --output-dir ./exported \
  --generate-tokens \
  --piper
```

`--add-phoneme-alignment` runs after the engine export. It adds a phoneme duration output to the exported VITS graph. See [Export to ONNX](training/export.md).

---

## Configuration

### Voice engine params

`VoiceConfig` keeps the engine-specific settings in `engine_params`, from the JSON config or from the `engine_params` option of `VoiceConfig.from_dict`:

```python
from phoonnx.config import VoiceConfig

cfg = VoiceConfig.from_dict(config, engine_params={"noise_scale": 0.5})
```

At synthesis time, phoonnx merges them into `request.params` of the adapter, after the adapter defaults and the voice scales.

#### NeuTTS front end

A NeuTTS fine-tune reads phonemes in the form that its own training code made them. Three `engine_params` keys describe that front end, so a fine-tune needs a voice config and no code. A voice without them gets the Akiti-TTS front end.

| Key | Default | What it sets |
| --- | --- | --- |
| `espeak_voice` | `lfn` | The espeak-ng language that the checkpoint was trained with, for example `pl` or `en-us`. |
| `prompt_template` | `vieneu` | `vieneu` is the bare prompt of VieNeu-TTS and Akiti-TTS, with at most 200 reference codes. `neuphonic` is the chat turn of the Neuphonic NeuTTS Air and Nano models (`user: Convert the text to speech:` ... `assistant:`), with every reference code. |
| `phoneme_substitutions` | none | A list of `[pattern, replacement]` regular expression pairs. They run on the phones in list order. The voice does not load if the value is not a list of string pairs, if a pattern does not compile, or if a replacement names a group that its pattern does not have. |

After the substitutions, each run of whitespace becomes one space, for every voice. The newline that espeak-ng writes at a clause break, such as an en dash, also becomes a space, as in the `phonemizer` package that VieNeu-TTS and Akiti-TTS use.

A voice config for the `shefowl/neutts-nano-polish` fine-tune:

```json
"engine_params": {
  "espeak_voice": "pl",
  "prompt_template": "neuphonic",
  "phoneme_substitutions": [["([a-zA-Z0-9])", " \\1"]]
}
```

### Synthesis extra params

A call gives engine settings through `SynthesisConfig.extra`. They win over the `engine_params` of the voice:

```python
from phoonnx.config import SynthesisConfig

syn = SynthesisConfig(
    length_scale=1.2,
    extra={"d_factor": 0.9}
)
```

---

## Add an engine

1. Inference:
   - Make a subclass of `BaseOnnxAdapter`.
   - Write `build_feed_dict`, `parse_outputs` and `default_params`.
   - Write `detect()`, so that voices that do not name the engine find it.
   - Call `register_engine()` in your module.

2. Training:
   - Make a subclass of `BaseTrainingEngine`.
   - Write `create_model`, `export_onnx` and `quality_presets`.
   - Override `load_checkpoint()` if the engine needs its own resume logic.
   - Call `register_engine()` in your module.

3. Packaging:
   - Make sure that registration runs. The built-in engines register in `_register_builtins()` of `phoonnx/engines/__init__.py` and `phoonnx_train/engines/__init__.py`. A module of your own registers when something imports it.
   - Add tests that call `detect_engine()` and `get_engine("your_engine")`.

These checks show that the registries hold an added engine:

```python
from phoonnx.engines import detect_engine, get_adapter, list_engines
from phoonnx_train.engines import get_engine, list_engines as list_train_engines

assert "your_engine" in list_engines()
assert "your_engine" in list_train_engines()

# Detection
adapter = detect_engine(config={"model_type": "your_engine"})
assert adapter.__class__.__name__ == "YourEngineAdapter"
```

---

[← Architecture](architecture.md) · [Home](index.md) · [Vocoders →](vocoders.md)
