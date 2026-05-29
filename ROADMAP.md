# Roadmap

`phoonnx` is the OVOS-native, ONNX-based multilingual TTS engine and the Lusophone
synthesis flagship. This roadmap sequences the work from CI/packaging hardening
through a best-in-class Portuguese frontend to serving and positioning. Backlog
items track in [TODO.md](./TODO.md).

## Phase 0 — Hardening & hygiene

Make CI trustworthy and packaging consistent with the org standard.

- Fix the CI `test_path`: it is `test`/`test/` in `build-tests.yml` and
  `coverage.yml`, but the test directory is `tests/`. Tests are not running until
  this is corrected.
- Align release and publish workflows to the `OpenVoiceOS/gh-automations@dev`
  reusable workflows. `release_workflow.yml` and `publish_stable.yml` reference
  `TigreGotico/gh-automations@master` and assume a `setup.py`; packaging is
  pyproject-only, so the bespoke `python setup.py` build/publish jobs go away.
- Untrack `phoonnx.egg-info/`: gitignore it and remove it from the tree.
- Consolidate duplicate CI: the hand-rolled `unit_tests.yml` overlaps the
  gh-automations `build-tests.yml`, and `conventional-label.yaml` duplicates
  `conventional-label.yml`. Keep one of each.
- Add an OPM (`mycroft.plugin.tts`) plugin-load smoke test so the entry point is
  exercised in CI.

## Phase 1 — Fix known bugs

Close the outstanding correctness issues.

- #63 piper Protobuf parse failure: some piper voices fail to load with a Protobuf
  parsing error; identify the offending model shape and handle it.
- #28 multiprocessing spawn-safety: the phonemizer must be initialized per worker
  rather than shared across the process boundary during preprocessing.
- #29 Lightning ≥2.0 `ModelCheckpoint`: the current configuration raises a
  `MisconfigurationException`; bring it in line with Lightning 2.x.
- `export_onnx` unsupported-espeak guard (`phoonnx_train/export_onnx.py:112`): the
  piper export hardcodes `phoneme_type: "espeak"`; validate against supported
  espeak languages and error out on unsupported ones.

## Phase 2 — Best-in-class Lusophone frontend

Make `phoonnx` the highest-quality Portuguese-family TTS frontend by promoting the
org's own G2P stack to first-class phonemizer backends. Today `pt.py` wraps
TugaPhone but leaves regional dialects as a TODO (`phoonnx/phonemizers/pt.py:31`).

- TugaPhone — Portuguese phonemization across pt-PT/BR/AO/MZ/TL; resolve the
  regional-dialect TODO so the declared variants actually differentiate.
- `orthography2ipa` backend — the org's validated G2P spanning 350+ languages,
  exposed as a general-purpose IPA phonemizer.
- `sotaque_forçado` — pt-PT regional-accent phonemization, layered on the Lusophone
  frontend.
- `g2p_barranquenho` — Barranquenho G2P.
- `mwl_phonemizer` — Mirandese (already wrapped as `MirandesePhonemizer`); keep it
  current.
- `cotovia` — Galician, completing the Iberian Romance coverage.

## Phase 3 — Training & voices

Modernize the training stack and make voice production repeatable.

- Modernize `phoonnx_train`: land the Lightning fixes from Phase 1 and add the
  config-vs-checkpoint params flag (`phoonnx_train/train.py:126`).
- A repeatable loop: synthetic data → train → ONNX export → publish to Hugging
  Face, runnable end to end.
- Keep `VOICES.md` and the voice index fresh, prioritizing Lusophone voices.

## Phase 4 — Serving & deployment

Run `phoonnx` as a service inside and alongside OVOS.

- #98 ovos-tts-server mode plus the wyoming-ovos-tts bridge, so `phoonnx` serves
  over both the OVOS and Wyoming protocols.
- Streaming synthesis, including the phonikud streaming model download
  (`phoonnx/thirdparty/phonikud/__init__.py:15`).

## Phase 5 — Positioning

`phoonnx` is a website flagship and the OVOS/Lusophone TTS engine.

- Keep `README.md` and the DeepWiki current.
- Frame the coherent stack: `orthography2ipa` (G2P) → `phoonnx` (synthesis) →
  custom voice training (service).
