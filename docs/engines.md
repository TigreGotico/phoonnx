# Engine Architecture

This page is for developers adding or integrating TTS back-ends in phoonnx. It describes
the inference adapter and training engine registries and the contract each must satisfy.

`phoonnx` supports multiple TTS back-ends through a small, engine-agnostic adapter framework.  Both **inference** (`phoonnx/engines/`) and **training** (`phoonnx_train/engines/`) are pluggable, so adding a new architecture (VITS, OptiSpeech, Matcha-TTS, …) only requires implementing a handful of methods.

---

## Inference Adapters

### Responsibilities

An inference adapter lives in `phoonnx/engines/` and implements `BaseOnnxAdapter`.  Its job is to translate between the generic `phoonnx` voice layer and a specific ONNX model layout:

* **Input names** – VITS models accept `input`/`input_lengths`/`scales`; other engines may use `input_ids`/`attention_mask` or entirely different names.
* **Scale / control parameters** – VITS uses `noise_scale`/`length_scale`/`noise_w_scale`; OptiSpeech uses `d_factor`/`p_factor`/`e_factor`.
* **Output layout** – some engines return raw waveform, others return mel-spectrograms that still need a vocoder.

### Adapter Lifecycle

```python
from phoonnx.engines import detect_engine, get_adapter

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
feed = adapter.build_feed_dict(request, session)
outputs = session.run(None, feed)

# 4. Parse outputs back to audio
result = adapter.parse_outputs(outputs, request)
```

### Registration

Add a new adapter by subclassing `BaseOnnxAdapter` and calling `register_engine`:

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

Lower `detect_priority` values are probed first during auto-detection.

### Existing Adapters

| Adapter | File | Detection |
|---|---|---|
| VITS / Piper / Mimic3 / Coqui / VITS2 / YourTTS-VITS | `phoonnx/engines/vits.py` | `model_type == "vits"`, `"scales"` input, piper/mimic3 signatures |
| [Streaming VITS](streaming.md) | `phoonnx/engines/vits_streaming.py` | `streaming: true` **and** a decoder graph (split encoder/decoder) |
| [Matcha](training/engines/matcha.md) | `phoonnx/engines/matcha.py` | `engine == "matcha"` (flow-matching mel + separate vocoder) |
| [GlowTTS](training/engines/glowtts.md) | `phoonnx/engines/glowtts.py` | `engine == "glowtts"` |
| [OptiSpeech](training/engines/optispeech.md) | `phoonnx/engines/optispeech.py` | `engine == "optispeech"` (wav + durations outputs) |
| [MixerTTS](training/engines/mixertts.md) | `phoonnx/engines/mixertts.py` | `engine == "mixertts"` |
| [FastPitch](training/engines/fastpitch.md) / SpeedySpeech | `phoonnx/engines/fastpitch.py` | `engine == "fastpitch"` |
| StyleTTS2 / Kokoro | `phoonnx/engines/styletts2.py` | `engine in ("styletts2", "kokoro")` — supports d-vector [cloning](cloning.md) |
| YourTTS | `phoonnx/engines/yourtts.py` | `engine == "yourtts"` — d-vector [cloning](cloning.md) |
| [ZipVoice](training/engines/zipvoice.md) | `phoonnx/engines/zipvoice.py` | `engine == "zipvoice"` — first **iterative** engine (flow-matching ODE loop), in-context [cloning](cloning.md) |
| [F5-TTS](training/engines/f5tts.md) | `phoonnx/engines/f5tts.py` | `engine == "f5tts"` — multi-graph engine (auxiliary ONNX graphs via `aux_model_urls`) |
| Shami / HamsVITS | `phoonnx/engines/shami.py` | `engine in ("shami", "hams")` — VITS variant with per-phoneme `language_ids` for Levantine Arabic / English code-switching |
| [Chatterbox](training/engines/chatterbox.md) | `phoonnx/engines/chatterbox.py` | `engine == "chatterbox"` — first **autoregressive** engine (codec-LM), d-vector [cloning](cloning.md) + exaggeration |
| SuperTonic | `phoonnx/engines/supertonic.py` | `engine == "supertonic"` — multi-graph flow-matching engine (4 ONNX graphs via `aux_model_urls`), raw-text (no phonemizer), fixed per-speaker style instead of cloning |

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

Add a new training engine by subclassing `BaseTrainingEngine` and calling `register_engine`:

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
| `mixer` / [MixerTTS](training/engines/mixertts.md) | `phoonnx_train/engines/mixer.py` | see page |
| StyleTTS2 (`styletts2`, `styletts2-aligner`, `styletts2-plbert`, `styletts2-pitch`) | `phoonnx_train/engines/styletts2*.py` | see page |
| YourTTS | `phoonnx_train/engines/yourtts.py` | see page |

`Chatterbox`, `F5-TTS` and `SuperTonic` currently ship inference-only adapters
(`phoonnx/engines/`) — there is no `phoonnx_train` training engine for them yet.

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

`--quality` is validated lazily against the engine’s `quality_presets()` so custom engines can define their own tier names.

### ONNX Export

`export_onnx.py` accepts `--engine` and passes all CLI flags through to the engine’s `export_onnx()` method:

```bash
python -m phoonnx_train.export_onnx epoch=500.ckpt \
  --config config.json \
  --engine vits \
  --output-dir ./exported \
  --generate-tokens \
  --piper
```

Engine-specific flags can be added by the engine via `extra_cli_options()`.

---

## Configuration

### VoiceConfig engine params

`VoiceConfig` stores engine-specific metadata in `engine_params` (parsed from JSON config at load time):

```python
from phoonnx.config import VoiceConfig

cfg = VoiceConfig.from_dict(config, engine_params={"noise_scale": 0.5})
```

These are forwarded to the adapter as `request.params` at synthesis time.

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

## Adding a New Engine (Checklist)

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

The built-in test suite should already cover engine-agnostic paths.  When adding a new engine, verify:

```python
from phoonnx.engines import detect_engine, get_adapter, list_engines
from phoonnx_train.engines import get_engine, list_engines as list_train_engines

assert "your_engine" in list_engines()
assert "your_engine" in list_train_engines()

# Auto-detection
adapter = detect_engine(config={"model_type": "your_engine"})
assert adapter.__class__.__name__ == "YourEngineAdapter"
```
