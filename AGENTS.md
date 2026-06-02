# phoonnx

Multilingual phonemization and Text-to-Speech library running ONNX voice models, with a native OVOS TTS plugin and a voice-management CLI.

## Setup

```bash
pip install -e .
```

Per-language phonemizer backends are optional extras (each pulls `epitran`, and some pull `gruut`/`misaki`):

```bash
pip install -e .[en]      # english (espeak/gruut/misaki extras)
pip install -e .[pt]      # portuguese, etc.
pip install -e .[train]   # torch + pytorch-lightning for phoonnx_train
```

Core runtime deps: `numpy`, `onnxruntime`, `quebra-frases`, `langcodes`, `ovos-number-parser`, `ovos-date-parser`. Optional phonemizers (espeak, gruut, epitran, misaki, etc.) are imported lazily.

## Test

```bash
pytest tests
```

CI also runs with coverage: `pytest --cov=phoonnx --cov-report xml tests`.

Note: the gh-automations workflows pass `test_path: 'test'` / `'test/'`, but the actual test directory is `tests/`. Tests are network-light (`tests/test_util.py`, `tests/test_ar.py`); model download / synthesis paths require network access and ONNX models.

## Lint/Typecheck

Ruff via the shared `lint.yml` workflow (`ruff: true`). No type checker configured.

## Layout

- `phoonnx/voice.py` — `TTSVoice` (model load + `synthesize` / `synthesize_wav`), `PhoneticSpellings`, `AudioChunk`.
- `phoonnx/config.py` — `VoiceConfig`, and the `Engine`, `Alphabet`, `PhonemeType` enums. Engines supported: phoonnx, piper, mimic3, coqui, transformers.
- `phoonnx/phonemizers/` — phonemizer backends. `base.py` defines `BasePhonemizer`/`GraphemePhonemizer`/`UnicodeCodepointPhonemizer`; language modules (`en`, `pt`, `ja`, `ko`, `zh`, `ar`, `fa`, `he`, `gl`, `vi`, `mwl`, `mul`) wrap espeak/gruut/epitran/misaki/goruut/byt5/charsiu/transphone and language-specific engines. `__init__.py` exposes the `Phonemizer` union.
- `phoonnx/thirdparty/` — vendored converters (bw2ipa, arpa2ipa, hangul2ipa, zh_num) and engines (mantoq, tashkeel, phonikud, kog2p).
- `phoonnx/model_manager.py` — `TTSModelInfo` / `TTSModelManager`: fetch, cache, merge default voice catalogs (OpenVoiceOS, Proxectonos, Phonikud, Piper, Mimic3), download model/config/vocab.
- `phoonnx/opm.py` — `PhoonnxTTSPlugin`, the OVOS TTS plugin.
- `phoonnx/cli.py` — `phoonnx-voices` Click CLI (`update-cache`, voice listing/info).
- `phoonnx/tokenizer.py`, `phoonnx/util.py` — tokenization and text/number/date normalization helpers.
- `phoonnx_train/` — training side: `preprocess.py`, `train.py`, `export_onnx.py`, and a VITS implementation under `phoonnx_train/vits/`.
- `tests/` — pytest suite. `requirements/` — per-language pinned dep files.

Entry-point groups:
- `mycroft.plugin.tts` -> `ovos-tts-plugin-phoonnx = phoonnx.opm:PhoonnxTTSPlugin` (OVOS TTS plugin).
- `console_scripts` -> `phoonnx-voices = phoonnx.cli:cli`.

## Conventions

- Branches: `dev` (work) / `master` (stable). NEVER `main`.
- Never edit `phoonnx/version.py`; gh-automations bumps semver from conventional-commit prefixes (`feat:`, `fix:`, `feat!:`).
- New repos private by default.
- Commit identity: JarbasAi <jarbasai@mailfence.com>.
- Reference `OpenVoiceOS/gh-automations` reusable workflows at `@dev`.
- No Neon / `neon-*` references.
- No meta-commentary (no history, no dates) in docs, commits, code comments.
- CI is provided by OpenVoiceOS/gh-automations.

## Gotchas

- This declares an OVOS TTS plugin (`mycroft.plugin.tts`), so an OPM plugin-load check belongs in CI; none is wired in.
- `release_workflow.yml` and `publish_stable.yml` reference `TigreGotico/gh-automations@master` and a `setup.py`, but packaging is pyproject-only (no `setup.py`) and the standard reusable workflows live under `OpenVoiceOS/gh-automations@dev`.
- CI `test_path` is `test`/`test/`; the real directory is `tests/`.
- `phoonnx.egg-info/` is committed to the tree.
- Optional phonemizer/engine deps are heavy and language-scoped; importing a phonemizer without its extra installed will fail at runtime, not install time.
- `model_manager` and the CLI `update-cache` reach the network to fetch voice catalogs and models.
