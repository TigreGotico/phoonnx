# Installation

<!-- agent-hint
covers: the base install, the version to install, every optional extra, GPU runtimes, system packages, and the errors a missing piece gives
source: pyproject.toml, phoonnx/providers.py:resolve_providers, phoonnx/errors.py, phoonnx/model_manager.py:_VoiceRegistry, phoonnx/opm.py:PhoonnxTTSPlugin.get_voice_info
generated: no
verify: .venv/bin/python -c "import re, sys, tomllib; q = chr(96); x = tomllib.load(open('pyproject.toml', 'rb'))['project']['optional-dependencies']; n = lambda r: re.split('[<>=; ]', r.strip().split(chr(91))[0])[0].lower().replace('_', '-'); names = lambda s: re.findall(q + '([^' + q + ']+)' + q, s); rows = re.findall('^[|] (' + q + '[a-z0-9-]+' + q + '(?:, ' + q + '[a-z0-9-]+' + q + ')*) [|][^|\\n]*[|]?([^|\\n]*)[|]?$', open('docs/installation.md').read(), re.M); bad = [(e, p) for a, c in rows for e in names(a) for p in ([None] if e not in x else [] if 'pyproject' in c else names(c)) if p is None or n(p) not in {n(r) for r in x[e]}]; print(len(rows), bad); sys.exit(1 if bad or len(rows) < 45 else 0)"
see: quickstart.md, choose_voice.md, offline.md, docker.md
-->

This page shows how to install phoonnx and the extras that your voices need. It is for anyone who sets up phoonnx on a machine.

## Requirements

- Python **3.11+**
- An ONNX Runtime build (the base install pulls the CPU `onnxruntime` wheel)

## Base install

```bash
pip install --pre phoonnx
```

With uv:

```bash
uv pip install --prerelease=allow phoonnx
```

The base install pulls in the CPU `onnxruntime` wheel, the text frontend (`scriptconv`,
`text2spoken`, `quebra-frases`, the OVOS number and date parsers), `huggingface_hub` for voice
downloads and `ovos-plugin-manager` for the OVOS plugin.

With nothing else installed, phoonnx runs the ONNX voices that tokenize with graphemes or
unicode code points. The `phoonnx-voices` commands `update-cache`, `list-langs`,
`list-voices`, `list-available` and `download` also work. `add-alignment` and
`export --to sherpa` need the `onnx` package from the `streaming` extra.

The language-specific phonemizers, voice cloning, the streaming auto-split and the training
pipeline come from the extras below.

## Which version to install

phoonnx ships as alpha releases, and these pages describe the alpha line. PyPI also holds an
old stable release, `1.3.3`. A plain `pip install phoonnx` picks that release, and its
`update-cache` command fails:

```
$ phoonnx-voices update-cache
Clearing local voice cache...
Fetching voice lists from upstream sources (this requires an internet connection)...
An unexpected error occurred while fetching voice lists: 'TTSModelManager' object has no attribute 'get_ovos_voice_list'
```

Always install with `--pre` (pip) or `--prerelease=allow` (uv). To see which version you
have:

```
$ python -c "from phoonnx.version import VERSION_STR; print(VERSION_STR)"
<major>.<minor>.<patch>a<n>
```

The command prints the installed version. An alpha version ends in `a` and a number, as in
the placeholder above. If you see `1.3.3`, upgrade. A second install
with `--pre` alone does not replace an installed release, so add `-U`:

```bash
pip install --pre -U phoonnx
# or, with uv:
uv pip install --prerelease=allow -U phoonnx
```

## Extras matrix

Install extras with `pip install --pre "phoonnx[<name>]"`. Combine them with commas, for
example `pip install --pre "phoonnx[en,cloning]"`.

### Which extra a voice needs

The phonemizer of a voice decides which extra it needs. `phoonnx-voices list-voices --verbose`
prints it on the `Phoneme Type` line of each voice:

```
$ phoonnx-voices list-voices --verbose | grep -A4 'ID:          OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone'
  ID:          OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone
  Language:    pt-PT
  Engine:      phoonnx
  Phoneme Type: Phonemizer.TUGAPHONE
  Model URL:   https://huggingface.co/Phoonnx/phoonnx_pt-PT_miro_tugaphone/resolve/b081738cde545e32a8c27fd780c5a2d948f1fb1f/miro_pt-PT.onnx
```

[Choose a voice](choose_voice.md#2-read-what-the-voice-needs) maps each phoneme type to its extra.
For example, `ESPEAK` needs `espeak` and `TUGAPHONE` needs `pt`. `UNICODE` and `GRAPHEMES` need no
extra.

A voice that shows `None` names its phonemizer in its own config file. If that phonemizer is
missing, the voice load stops with an `ImportError` that names the package (see
[Errors you may see](#errors-you-may-see)).

### Feature extras

| Extra | Adds | Pulls in |
|---|---|---|
| `espeak` | The eSpeak phonemizer, which every eSpeak voice needs | `scriptconv[espeak]`, which gives the eSpeak wrapper in `scriptconv-copyleft`, and `espyak` (a pure-Python port of the eSpeak phonemizer) |
| `cloning` | Loading non-WAV reference clips for [voice cloning](cloning.md) | `soundfile`, `scipy` |
| `streaming` | Auto-splitting a monolithic VITS voice into a [streaming](streaming.md) encoder/decoder pair, `phoonnx-voices add-alignment`, and alignment on demand | `onnx` (graph surgery) |
| `matcha` | Nothing that the Matcha adapter needs. The Matcha adapter and its vocoders import no `scipy`. | `scipy` |
| `pockettts` | Pocket TTS voice states | `safetensors` |
| `omnivoice` | Faster silence trimming for OmniVoice (a pure-numpy fallback runs without it) | `pydub` |
| `indic-parler` | The Indic Parler tokenizers | `tokenizers` |
| `chatterbox-multilingual` | Chatterbox per-language script transforms (ja/zh) | `pykakasi`, `spacy-pkuseg` |
| `cjk` | Same CJK script-transform deps as `chatterbox-multilingual`, under a shorter name | `pykakasi`, `spacy-pkuseg` |
| `o2i` | The multilingual data-driven IPA backend | `orthography2ipa`, `o2i-lexicons` |
| `audiosr` | Optional 48 kHz [audio super-resolution](ovos_plugin.md#audio-super-resolution) over synthesized audio | `audiosronnx` |
| `vc` | Optional [voice conversion](voice_conversion.md) of the output to a target speaker | `voiceclonnx` |
| `all` | The phonemizers of most language extras, cloning, super-resolution and voice conversion. The Docker image installs it. | see `pyproject.toml` |

### Language extras

Each language extra installs the phonemizer backends for that language.

Some wrappers are not in the Apache-2.0 `scriptconv` wheel. The eSpeak, Cotovia, AhoTTS,
Persian, Korean, pykakasi and misaki wrappers are in `scriptconv-copyleft`. The mantoq, Halabi,
Iqra and VIPhoneme wrappers are in `scriptconv-nonfree`, which is under a non-commercial licence.
The extras that use these wrappers install the package that holds them.

| Extra | Language | Notable backends |
|---|---|---|
| `ar` | Arabic | `epitran`, `gruut[ar]`, `scriptconv[ar-phonemizers]`, `arbtok`, `scriptconv-nonfree` (mantoq), `text2tashkeel` |
| `ca` | Catalan | `epitran`, `gruut[ca]` |
| `cs` | Czech | `epitran`, `gruut[cs]` |
| `de` | German | `epitran`, `gruut[de]` |
| `en` | English | `epitran`, `gruut[en]`, `misaki[en]`, `scriptconv-copyleft`, `spacy` |
| `es` | Spanish | `epitran`, `gruut[es]` |
| `eu` | Basque | `scriptconv[eu]`, `ahotts-g2p`, `euskaphone`, `epitran` |
| `fa` | Persian | `epitran`, `scriptconv[fa]`, which installs the Persian wrapper of scriptconv-copyleft |
| `fr` | French | `epitran`, `gruut[fr]` |
| `gl` | Galician | `scriptconv[gl]`, `pycotovia`, `epitran` (see [Galician](galician.md)) |
| `he` | Hebrew | `epitran`, `scriptconv[he]`, `phonikud_onnx`, `phonikud` |
| `it` | Italian | `epitran`, `gruut[it]` |
| `ja` | Japanese | `epitran`, `gruut[ja]`, `scriptconv[ja-phonemizers]`, `misaki`, `pyopenjtalk-plus`, `fugashi`, `cutlet`, `unidic-lite`, `pykakasi`, `spacy` |
| `ko` | Korean | `epitran`, `scriptconv[ko]`, `g2pk`, `misaki[ko]` |
| `lb`, `nl` | Luxembourgish, Dutch | `gruut[<lang>]` |
| `mwl` | Mirandese | `scriptconv[mwl]`, which gives the orthography2ipa Mirandese specs |
| `pt` | Portuguese | `epitran`, `scriptconv[o2i]`, `tugaphone`, `gruut[pt]` |
| `ru` | Russian | `epitran`, `gruut[ru]` |
| `sv` | Swedish | `epitran`, `gruut[sv]` |
| `sw` | Swahili | `epitran`, `gruut[sw]` |
| `vi` | Vietnamese | `epitran`, `gruut[vi]`, `scriptconv[vi]`, `misaki[vi]`, `scriptconv-copyleft`, `viphoneme`, `underthesea`, `spacy` |
| `zh` | Chinese | `scriptconv[zh-phonemizers]`, `epitran`, `gruut[zh]`, `misaki[zh]`, `scriptconv-copyleft`, `g2pM`, `pypinyin`, `jieba`, `spacy` |

Many more languages have an extra that installs `epitran` only. Its name is the language
code, for example `fi`, `hi`, `pl`, `tr` or `uk`. The full list is in `pyproject.toml`.

> **Arabic note.** The `ar` extra installs `scriptconv-nonfree`, which holds the mantoq
> phonemizer under a non-commercial licence. Arabic diacritization comes from `text2tashkeel`.
> When diacritization fails or its backend is missing, phoonnx logs a warning and speaks the
> text without diacritics (see [Deployment](deployment.md#prosody-enrichment-degrades-instead-of-failing)).

### Training extras

| Extra | For |
|---|---|
| `train` | The core training pipeline (PyTorch Lightning, librosa, torch) |
| `train-eval` | Checkpoint evaluation loop (UTMOS, speaker similarity, quality filters) |
| `train-pyworld` | The `pyworld` F0 extractor, which the FastPitch, Mixer-TTS, OptiSpeech and StyleTTS2 pitch code can use |
| `train-fastpitch` | The FastPitch/SpeedySpeech training engine (no extra deps beyond `[train]`) |
| `train-mixer` | The Mixer-TTS training engine (no extra deps beyond `[train]`) |
| `train-styletts2` | The StyleTTS2 training engine |
| `train-resample` | Resampling for engines defined at a fixed sample rate (e.g. ZipVoice 24 kHz) |
| `train-data` | Multi-format dataset loading for preprocessing (jsonl/parquet/HF repos with embedded audio bytes) |
| `train-optispeech` | The OptiSpeech training engine |
| `train-mosstts` | The MOSS-TTS-Nano training pipeline |
| `train-features` | The feature sources (phonematcher, orthography2ipa) of the features.json generator |

See the [Training quickstart](training/quickstart.md) for how these fit together.

## GPU inference

The installed ONNX Runtime build, not phoonnx, decides which execution providers exist.
The default `onnxruntime` wheel is CPU-only. Install only one GPU build at a time:

| Hardware | Package | Providers |
|---|---|---|
| NVIDIA | `onnxruntime-gpu` | `CUDAExecutionProvider`, `TensorrtExecutionProvider` |
| AMD (ROCm) | `onnxruntime-rocm` | `ROCMExecutionProvider`, `MIGraphXExecutionProvider` |
| Windows (DX12 GPU) | `onnxruntime-directml` | `DmlExecutionProvider` |
| Intel | `onnxruntime-openvino` | `OpenVINOExecutionProvider` |
| Apple | `onnxruntime` | `CoreMLExecutionProvider` |

Select providers explicitly when you load a voice:

```python
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.load()
manager.merge_default_voices()

voice = manager.voices["OpenVoiceOS/phoonnx_pt-PT_miro_unicode"].load(
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print(voice.session.get_providers())
```

On a machine with the CPU-only wheel, phoonnx skips the GPU provider with a warning, and the
voice runs on the CPU:

```
WARNING - 'CUDAExecutionProvider' is not available in this onnxruntime build (available: ['AzureExecutionProvider', 'CPUExecutionProvider']), skipping it. Install the matching onnxruntime package to enable it.
['CPUExecutionProvider']
```

You can also set `PHOONNX_ONNX_PROVIDERS` once for the process. phoonnx always appends
`CPUExecutionProvider`, so synthesis keeps working. Full details are in the
[Configuration reference](configuration.md#execution-providers).

Set `PHOONNX_ORT_CACHE_DIR` to cache the optimized ONNX Runtime graph across process
restarts. This makes the session start faster after the first run.

## System dependency: espeak-ng

The eSpeak phonemizer can drive the system `espeak-ng` binary directly. The wrapper itself comes
from the `espeak` extra, so an eSpeak voice needs that extra in every case. Install the binary
from your package manager if you want espeak-ng instead of the pure-Python port:

```bash
sudo apt-get install espeak-ng   # Debian/Ubuntu
sudo pacman -S espeak-ng          # Arch Linux
brew install espeak-ng            # macOS
```

The `phoonnx[espeak]` extra gives a pure-Python port of the eSpeak phonemizer and needs no
system package.

## System dependency: ffmpeg

phoonnx writes WAV and needs nothing extra for it. A server that offers other formats (mp3,
ogg, flac) encodes them with `pydub`, which runs `ffmpeg`. Without `ffmpeg` those requests
fall back to WAV. The symptom is a response in the wrong format, not an error.

```bash
sudo apt-get install ffmpeg   # Debian/Ubuntu
sudo pacman -S ffmpeg          # Arch Linux
brew install ffmpeg            # macOS
```

The published Docker image carries it.

## Docker

The published Docker image installs `phoonnx[all]` and runs a TTS server. See
[Docker / TTS server](docker.md).

## Installing from source

```bash
git clone https://github.com/TigreGotico/phoonnx
cd phoonnx
pip install --pre -e .
# or with training extras:
pip install --pre -e ".[train]"
```

## Errors you may see

Each error below is the last line that Python prints. The fix follows each one.

**A missing extra.** The voice needs a phonemizer that the install does not have, so the
voice load stops.

```
ImportError: tugaphone is required for the Tugaphone phonemizer. Install it with 'pip install tugaphone' (or 'pip install scriptconv[pt-phonemizers]').
```

Fix: `pip install --pre "phoonnx[pt]"`.

```
ImportError: phonemizer 'espeak' needs a missing backing package — install with `pip install scriptconv[espeak]`
```

Fix: `pip install --pre "phoonnx[espeak]"`.

**An unknown voice id.** The id is not in the catalog. Python raises `UnknownVoice`, a
`KeyError` that names the id:

```
phoonnx.errors.UnknownVoice: unknown voice 'OpenVoiceOS/phoonnx_pt-PT_miro'; run 'phoonnx-voices list-voices' to see the voice ids
```

The OVOS plugin stops at start with:

```
Exception: Unknown voice: OpenVoiceOS/phoonnx_pt-PT_miro
```

Fix: copy the exact id from `phoonnx-voices list-available`.

**A language with no voice.** The OVOS plugin stops at start when no voice in the catalog
serves the configured `lang`:

```
ValueError: No voices available for language: xx-YY
```

`TTSVoice.load(model_path, config_path, lang_code=...)` takes a language that overrides the config. When its phonemizer cannot read that language, the load stops. `manager.voices[id].load()` takes no `lang_code`. It uses the language of the index entry.

```
phoonnx.errors.UnsupportedVoiceLanguage: voice 'dii_es-ES': no phonemizer backend supports lang 'xx-YY' for phoneme_type 'espeak'
```

Fix: use a language tag from `phoonnx-voices list-langs`, or a voice for that language.

**A voice that is not downloaded, on a machine with no network.** With `HF_HUB_OFFLINE=1`,
or with no connection to the Hugging Face Hub, a voice that is not in the local cache cannot
load:

```
huggingface_hub.errors.LocalEntryNotFoundError: model.onnx of Phoonnx/vits-en_us-danny-low-espeak at commit aeaf5a3ab43f6d25a6798581793de0b97818e3ab is not in the hub cache, and the hub cannot be reached. Connect once so that phoonnx can download it. After a phoonnx upgrade, the voice index can name a file that no cached snapshot holds.
```

Fix: run `phoonnx-voices download <voice_id>` while the machine is online. After that, the
voice loads offline. See [Use phoonnx offline](offline.md).

---

[Docs index](index.md) · Previous: [Docs index](index.md) · Next: [Quickstart](quickstart.md)
