# TODO

## Open issues

- [ ] #98 Starting as an ovos-tts-server
- [ ] #93 Dependency Dashboard
- [ ] #63 Some piper voices not working: Protobuf parsing failed (bug)
- [ ] #62 Feature requests
- [ ] #29 ModelCheckpoint misconfiguration causes MisconfigurationException in Lightning >=2.0
- [ ] #28 Multiprocessing spawn-safety: Phonemizer object should be initialized per worker

## CI & packaging

- [ ] CI `test_path` is `test`/`test/` (build-tests.yml, coverage.yml) but the test directory is `tests/`; fix the path so tests actually run.
- [ ] `release_workflow.yml` and `publish_stable.yml` point at `TigreGotico/gh-automations@master` and assume a `setup.py`; packaging is pyproject-only. Align with `OpenVoiceOS/gh-automations@dev` standard release/publish workflows.
- [ ] Two overlapping unit-test workflows (`unit_tests.yml` hand-rolled, `build-tests.yml` via gh-automations) and two conventional-label files (`conventional-label.yaml` + `conventional-label.yml`); consolidate.
- [ ] No OPM plugin-load check in CI even though the `mycroft.plugin.tts` entry point is declared; add a load smoke test.

## Hygiene

- [ ] `phoonnx.egg-info/` is committed; gitignore it and remove it from the tree.

## Code TODOs

- [ ] `phoonnx/phonemizers/pt.py:31` — support regional dialects
- [ ] `phoonnx/phonemizers/mul.py:243` — more models
- [ ] `phoonnx/thirdparty/phonikud/__init__.py:15` — streaming download
- [ ] `phoonnx_train/train.py:126` — add a flag to use params from config vs from checkpoint on mismatch
- [ ] `phoonnx_train/export_onnx.py:112` — check for supported espeak languages, throw error if unsupported
