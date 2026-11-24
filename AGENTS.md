# AGENTS.md

phoonnx loads text-to-speech voices from many ecosystems and runs them through ONNX Runtime, behind one `TTSVoice` API, a voice catalog and an OpenVoiceOS plugin. The `phoonnx_train` package trains voices and exports them to the native phoonnx format.

## Map

| Path | What it holds |
|---|---|
| `phoonnx/voice.py` | `TTSVoice`: load a voice, run the text frontend, synthesize, stream, align |
| `phoonnx/config.py` | `VoiceConfig`, `SynthesisConfig` and the text frontend on `scriptconv.frontend` |
| `phoonnx/model_manager.py` | `TTSModelManager`, `TTSModelInfo`: the catalog, downloads, the offline fallback, precision |
| `phoonnx/voice_index/*.json` | The bundled voice index, one file per source, checked by `phoonnx/index_schema.py` |
| `phoonnx/config.schema.json` | The schema of the native `config.json` |
| `phoonnx/compat.py`, `phoonnx/config_loaders.py` | Readers for the Piper, Mimic 3, Coqui and other config formats |
| `phoonnx/engines/` | One inference adapter per engine, and the vocoders in `vocoders/` |
| `phoonnx/tokenizer.py`, `phoonnx/providers.py`, `phoonnx/voice_cache.py` | Tokenizers, ONNX Runtime sessions and providers, the voice cache of the plugin |
| `phoonnx/cli.py`, `phoonnx/opm.py` | The `phoonnx-voices` command and the `ovos-tts-plugin-phoonnx` plugin |
| `phoonnx_train/` | `preprocess.py`, `train.py`, `export_onnx.py`, `eval_loop.py` and one module per training engine |
| `scripts/` | `engine_matrix.py`, `voices_md.py`, `voice_sweep.py`, `bench_latency.py`, `model_card.py` and the model conversion scripts |
| `docs/` | User and contributor docs. [docs/index.md](docs/index.md) lists every page. |

The phonemizers are in other packages: `scriptconv` (Apache-2.0), `scriptconv-copyleft` (GPL) and `scriptconv-nonfree` (non-commercial). Change a phonemizer in its own repository. See [Phonemizers](docs/phonemizers.md).

## Setup and tests

`pyproject.toml` sets floors on alpha releases, so the install must allow prereleases. The `test` extra installs every library that the test files import, `torch` included. The `train` and `train-data` extras let the training commands and the `verify` lines of the training pages run.

```bash
uv venv .venv
uv pip install --python .venv/bin/python --prerelease=allow -e '.[test,train,train-data]'
.venv/bin/python -m pytest -q tests/test_cli.py
```

Run the test files one at a time. CI installs `.[test,train,train-styletts2,train-eval,train-data,train-optispeech,train-pyworld]` on Python 3.11 to 3.14, with the `espeak-ng` system package (`.github/workflows/build-tests.yml`).

## Rules the repository enforces

- Every Hugging Face URL in the voice index names a commit sha, not a branch or a tag. Test: `tests/test_voice_index_pinned.py`. See [Voice manager](docs/voice_manager.md).
- Every index entry matches `phoonnx/index_schema.py`. Test: `tests/test_index_schema.py`.
- `VoiceConfig.to_dict()` gives a native `config.json` that validates against `phoonnx/config.schema.json`. Test: `tests/test_voice_config_to_dict.py`. See [Configuration reference](docs/configuration.md).
- [Phonemizers](docs/phonemizers.md) has a row for each phoneme type that the schema lists. Test: `tests/test_voice_config_to_dict.py`.
- Each registered inference adapter has an index voice, or a row in the "Engines with no index voice" table of [Engine architecture](docs/engines.md). Test: `tests/test_engine_coverage.py`.
- The architecture table in [Engine architecture](docs/engines.md) is the output of `scripts/engine_matrix.py`. Edit the script, not the table. Test: `tests/test_engine_matrix.py`.
- `VOICES.md` is the output of `scripts/voices_md.py`. Test: `tests/test_voices_md.py`.
- The support table of [SSML](docs/ssml.md) has one row per case that `tests/test_ssml_audit.py` checks.
- phoonnx never depends on an espeak-ng wrapper library, such as `piper-phonemize` or `phonemizer`. Test: `tests/test_no_wrapper_deps.py`. See [Installation](docs/installation.md).
- The base install imports `phoonnx.cli` and `phoonnx.model_manager`. Test: `tests/test_base_deps.py`.
- Python 3.11 is the floor (`requires-python` in `pyproject.toml`).
- The release workflow writes `phoonnx/version.py` (`version_file` in `.github/workflows/release_workflow.yml`). Do not edit it.

## Agent hints in the docs

README.md and each page in `docs/` have an HTML comment after the H1, `<!-- agent-hint ... -->`. A rendered page does not show it. The hint names what the page covers, the source files that hold the truth for it, the command that checks it, and the related pages. Each page ends with a footer that links the index and the previous and next page in the order of `docs/index.md`.

## Before a push

1. Run the test files of the code that you changed. Also run the tests in the rules above when you change the index, a config format, an engine or a docs table.
2. For each docs page that you changed, or whose `source` files you changed, run the command on its `verify` line. It exits 0 when the page is true.
