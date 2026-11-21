# AGENTS.md

phoonnx loads and runs text-to-speech voices from many ecosystems through ONNX Runtime, behind one `TTSVoice` API, a voice catalog and an OpenVoiceOS plugin. `phoonnx_train` trains voices and exports them to the native phoonnx format.

## Map

| Path | What it holds |
|---|---|
| `phoonnx/voice.py` | `TTSVoice`: load a voice, run the text frontend, synthesize, stream, align |
| `phoonnx/config.py` | `VoiceConfig`, `SynthesisConfig`, the text frontend built on `scriptconv.frontend` |
| `phoonnx/model_manager.py` | `TTSModelManager`, `TTSModelInfo`: the catalog, downloads, the offline fallback, precision |
| `phoonnx/voice_index/*.json` | The bundled voice index, one file per source; `phoonnx/index_schema.py` checks it |
| `phoonnx/config.schema.json` | The schema of the native `config.json` |
| `phoonnx/compat.py`, `phoonnx/config_loaders.py` | Readers for Piper, Mimic 3, Coqui and other config formats |
| `phoonnx/engines/` | One inference adapter per engine, and `vocoders/` |
| `phoonnx/tokenizer.py`, `phoonnx/providers.py`, `phoonnx/voice_cache.py` | Tokenization, ONNX Runtime sessions, the plugin's voice cache |
| `phoonnx/cli.py`, `phoonnx/opm.py` | The `phoonnx-voices` command and the `ovos-tts-plugin-phoonnx` plugin |
| `phoonnx_train/` | `preprocess.py`, `train.py`, `export_onnx.py`, `eval_loop.py` and one module per training engine |
| `scripts/` | `voice_sweep.py`, `bench_latency.py`, `model_card.py`, model conversion scripts |
| `docs/` | User and contributor docs; [docs/index.md](docs/index.md) lists every page |

Phonemizers do not live here. They come from `scriptconv` (Apache-2.0), `scriptconv-copyleft` (GPL: eSpeak, Cotovia, AhoTTS, misaki) and `scriptconv-nonfree` (non-commercial: mantoq, Halabi, Iqra, VIPhoneme). Change a phonemizer in its own repository.

## Setup and tests

`pyproject.toml` floors several dependencies on alpha releases, such as `text2spoken>=0.0.2a78`, so the install must allow prereleases.

```bash
uv venv -p 3.12 .venv
uv pip install --python .venv/bin/python --prerelease=allow -e . pytest jsonschema "onnx<1.18"
.venv/bin/python -m pytest tests/test_cli.py
```

Run one test file at a time. Many files need more than this base set:

- `tests/test_voice_config_to_dict.py` needs `jsonschema`.
- `tests/test_card_examples.py` imports `text2tashkeel` and `onnx`.
- 12 test files import `onnx`, and 20 import `torch`, mostly the training tests.
- The `test` extra installs all of these. CI installs `-e .[test,train,train-styletts2,train-eval,train-data,train-optispeech,train-pyworld]` on Python 3.11 to 3.14, with the `espeak-ng` system package.

## Rules the repository enforces

- Every Hugging Face URL in the voice index names a commit sha, never a branch or a tag. Test: `tests/test_voice_index_pinned.py`. See [Voice manager](docs/voice_manager.md#hub-revisions-in-the-bundled-index).
- Every index entry matches `phoonnx/index_schema.py`. Test: `tests/test_index_schema.py`.
- `VoiceConfig.to_dict()` gives a native `config.json` that validates against `phoonnx/config.schema.json`. Test: `tests/test_voice_config_to_dict.py`. See [Configuration reference](docs/configuration.md).
- Every registered inference adapter is used by an index voice, or it has a row in the "Engines with no index voice" table of [docs/engines.md](docs/engines.md). Test: `tests/test_engine_coverage.py`.
- phoonnx never depends on an espeak-ng wrapper library such as `piper-phonemize` or `phonemizer`. Test: `tests/test_no_wrapper_deps.py`. See [Installation](docs/installation.md#language-extras).
- The base install imports `phoonnx.cli` and `phoonnx.model_manager`. Test: `tests/test_base_deps.py`.
- Python 3.11 is the floor (`requires-python` in `pyproject.toml`).
- `phoonnx/version.py` is written by the release workflow (`.github/workflows/release_workflow.yml`, `version_file`).

## Agent hints in the docs

README.md and each page in `docs/` carry an HTML comment right after the H1, `<!-- agent-hint ... -->`. A rendered page does not show it. It names what the page covers, the source files that hold the truth for it, whether a script generates it, the command that checks it, and the related pages. When you change a file that a hint names as `source`, read that page and fix what is no longer true, the hint included. Each page ends with a footer that links the index and the previous and next page in the reading order of `docs/index.md`.

## Before a push

1. Run the test files of the code you changed, and the contract tests above when you touch the index, a config format or an engine.
2. For each docs page that you changed, or whose `source` files you changed, run the command in its agent-hint `verify` line.
