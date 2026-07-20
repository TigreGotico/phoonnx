# Installation

This page is for anyone setting up phoonnx. It covers the base install, exactly what each
optional extra unlocks, the GPU runtime packages, and the one system dependency you may need.

## Requirements

- Python **3.11+**
- An ONNX Runtime build (the base install pulls the CPU `onnxruntime` wheel)

## Base install

```bash
pip install phoonnx
```

The base package depends only on `numpy`, `onnxruntime`, `quebra-frases`, `langcodes`,
`ovos-number-parser`, `ovos-date-parser`, `scriptconv`, `click`, `requests` and
`json_database`. With nothing else installed you can:

- run ONNX voices that tokenize with **graphemes** or **unicode** codepoints,
- use the full `phoonnx-voices` CLI (`update-cache`, `list-langs`, `list-voices`,
  `list-available`, `download`).

It does **not** include the language-specific phonemizers, voice cloning, streaming
auto-split, or the training pipeline — those come from the extras below.

## Extras matrix

Install extras with `pip install "phoonnx[<name>]"`. Combine them with commas, e.g.
`pip install "phoonnx[en,cloning]"`.

### Feature extras

| Extra | Unlocks | Pulls in |
|---|---|---|
| `espeak` | The eSpeak phonemizer without a system binary | `espyak` (a pure-Python port of the eSpeak phonemizer, byte-for-byte parity) |
| `cloning` | Loading non-WAV reference clips for [voice cloning](cloning.md) | `soundfile`, `scipy` |
| `streaming` | Auto-splitting a monolithic VITS voice into a [streaming](streaming.md) encoder/decoder pair | `onnx` (graph surgery) |
| `matcha` | Matcha two-stage synthesis helpers | `scipy` |
| `chatterbox-multilingual` | Chatterbox per-language script transforms (ja/zh) | `pykakasi`, `spacy-pkuseg` |
| `o2i` | The multilingual data-driven IPA backend | `orthography2ipa` |
| `all` | Every language phonemizer + cloning (used by the Docker image) | see below |

### Language extras

Each language extra installs the phonemizer backends for that language. Most pull `epitran`;
several add higher-quality engines.

| Extra | Language | Notable backends |
|---|---|---|
| `ar` | Arabic | `epitran`, `arbtok`, `text2tashkeel` |
| `ca` | Catalan | `epitran` |
| `cs` | Czech | `epitran` |
| `de` | German | `epitran`, `gruut[de]` |
| `en` | English | `epitran`, `gruut[en]`, `misaki[en]`, `spacy` |
| `es` | Spanish | `epitran` |
| `eu` | Basque | `ahotts-g2p`, `euskaphone` |
| `fa` | Persian | `epitran` |
| `fr` | French | `epitran` |
| `gl` | Galician | `pycotovia` (see [Galician](galician.md)) |
| `he` | Hebrew | `epitran` |
| `it` | Italian | `epitran` |
| `ja` | Japanese | `epitran`, `gruut[ja]`, `misaki[ja]`, `spacy` |
| `ko` | Korean | `epitran` |
| `mwl` | Mirandese | `mwl_phonemizer` |
| `pt` | Portuguese | `epitran`, `g2p_barranquenho` |
| `ru` | Russian | `epitran` |
| `sv` | Swedish | `epitran` |
| `sw` | Swahili | `epitran` |
| `vi` | Vietnamese | `epitran`, `gruut[vi]`, `misaki[vi]`, `spacy` |
| `zh` | Chinese | `epitran`, `gruut[zh]`, `misaki[zh]`, `spacy` |

> **Arabic note.** The `ar` extra installs `epitran`, `arbtok` and `text2tashkeel`, **not**
> `mantoq`. The `mantoq` phonemizer has no extra of its own — install it separately if a voice
> requires it. Arabic diacritization is provided by `text2tashkeel`; without the `ar` extra it
> raises a clear `ImportError` when a voice needs diacritics.

### Training extras

| Extra | For |
|---|---|
| `train` | The core training pipeline (PyTorch Lightning, librosa, torch) |
| `train-eval` | Checkpoint evaluation loop (UTMOS, speaker similarity, quality filters) |
| `train-fastpitch` | FastPitch/SpeedySpeech F0 extraction (`pyworld`) |
| `train-mixer` | Mixer-TTS F0 sidecar (`pyworld`) |
| `train-styletts2` | The StyleTTS2 training engine |
| `train-resample` | Resampling for engines defined at a fixed sample rate (e.g. ZipVoice 24 kHz) |

See the [Training quickstart](training/quickstart.md) for how these fit together.

## GPU inference

Which execution providers exist depends on the installed ONNX Runtime build, not on phoonnx.
The default `onnxruntime` wheel is CPU-only; only one GPU build can be installed at a time:

| Hardware | Package | Providers |
|---|---|---|
| NVIDIA | `onnxruntime-gpu` | `CUDAExecutionProvider`, `TensorrtExecutionProvider` |
| AMD (ROCm) | `onnxruntime-rocm` | `ROCMExecutionProvider`, `MIGraphXExecutionProvider` |
| Windows (DX12 GPU) | `onnxruntime-directml` | `DmlExecutionProvider` |
| Intel | `onnxruntime-openvino` | `OpenVINOExecutionProvider` |
| Apple | `onnxruntime` | `CoreMLExecutionProvider` |

Select providers explicitly when loading a voice:

```python
voice = TTSVoice.load(
    "model.onnx", "model.json",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

or set `PHOONNX_ONNX_PROVIDERS` once for the process. `CPUExecutionProvider` is always
appended so synthesis keeps working. Full details in the
[Configuration reference](configuration.md#execution-providers).

## System dependency: espeak-ng

The eSpeak phonemizer can drive the system `espeak-ng` binary directly. Install it from your
package manager if a voice uses it and you are not using the `espeak` extra:

```bash
sudo apt-get install espeak-ng   # Debian/Ubuntu
sudo pacman -S espeak-ng          # Arch Linux
brew install espeak-ng            # macOS
```

The `phoonnx[espeak]` extra provides a pure-Python port of the eSpeak phonemizer (byte-for-byte
parity with the binary) and needs no system package.

## Docker

The published Docker image installs `phoonnx[all]` and runs a TTS server. See
[Docker / TTS server](docker.md).

## Installing from source

```bash
git clone https://github.com/TigreGotico/phoonnx
cd phoonnx
pip install -e .
# or with training extras:
pip install -e ".[train]"
```
