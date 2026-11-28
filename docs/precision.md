# Precision variants

<!-- agent-hint
covers: which voices have int8 files, how the precision setting picks full or int8 files, and the errors a wrong value gives
source: phoonnx/model_manager.py:PRECISIONS, phoonnx/model_manager.py:resolve_precision, phoonnx/model_manager.py:choose_precision, phoonnx/model_manager.py:TTSModelInfo.with_precision, phoonnx/model_manager.py:TTSModelInfo.disk_size, phoonnx/errors.py:PrecisionUnavailable, phoonnx/cli.py:download_voice, phoonnx/opm.py:PhoonnxTTSPlugin._precision, phoonnx/voice_index/*.json
generated: no
verify: XDG_CACHE_HOME=.venv/verify-cache HF_HUB_OFFLINE=1 .venv/bin/python -m doctest -o ELLIPSIS -o NORMALIZE_WHITESPACE docs/precision.md
see: choose_voice.md, voice_manager.md, ovos_plugin.md, configuration.md
-->

This page shows how to load the smaller int8 files of a voice, and how phoonnx decides between full and int8 files. It is for anyone who runs large voices on a CPU or on a small disk.

## Which voices have int8 files

A voice entry in the bundled index has one of three forms:

| Form | What loads |
|---|---|
| Full files only | The full files, always. |
| Full files and an int8 variant | The full files, or the int8 files when you ask for them. |
| Int8 files only | The int8 files, always. |

This script sorts the entries of the bundled index by form, and prints the forms it finds:

```python
>>> from collections import Counter
>>> from phoonnx.model_manager import TTSModelManager
>>> manager = TTSModelManager()
>>> manager.merge_default_voices()
>>> def form(info):
...     if info.precision == "int8":
...         return "int8 only"
...     return "full and int8" if info.variants else "full only"
>>> forms = Counter(form(info) for info in manager.all_voices)
>>> sorted(forms)
['full and int8', 'full only']

```

`forms` holds the number of entries of each form. The entries with an int8 variant are
Piper voices, the Pocket TTS voices of every language (`pockettts/...`), the F5-TTS voices
(`f5tts/...`, `habibi/...`, `namaa/ar-sa-v2`, `silma/...`, `openbible/...`) and
`edwixx/pt-BR-rapido`. No bundled entry is int8-only: the int8 Pocket TTS graphs fall back to
the CPU on a GPU, because ONNX Runtime's CUDA provider does not run their operators, so every
Pocket TTS voice loads its full graphs unless int8 is asked for:

```python
>>> sorted({info.engine.value for info in manager.all_voices if info.variants})
['f5tts', 'piper', 'pockettts', 'rapido']
>>> sorted({info.voice_id.split("/")[1] for info in manager.all_voices
...         if info.engine.value == "pockettts" and info.variants})
['de', 'en', 'es', 'fr', 'it', 'pt']
>>> [info.voice_id for info in manager.all_voices if info.precision == "int8"]
[]

```

## 1. Look at the files of each precision

`with_precision()` gives the entry as it loads under a precision. It downloads nothing:

```python
>>> from phoonnx.model_manager import TTSModelManager
>>> manager = TTSModelManager()
>>> manager.merge_default_voices()
>>> voice = manager.voices["pockettts/it/alba"]
>>> print(voice.precision, list(voice.variants))
None ['int8']
>>> print(voice.with_precision("full").model_url.rsplit("/", 1)[-1])
flow_lm_main.onnx
>>> print(voice.with_precision("int8").model_url.rsplit("/", 1)[-1])
flow_lm_main_int8.onnx
>>> print(voice.with_precision("auto").model_url.rsplit("/", 1)[-1])
flow_lm_main.onnx

```

The entry has full files (`precision` is `None`) and an `int8` variant. The `auto` setting picks the full files.

## 2. Download and load the int8 files

Pocket TTS needs the `pockettts` extra:

```bash
pip install --pre "phoonnx[pockettts]"
```

Download the int8 files with `--precision`:

```
$ phoonnx-voices download pockettts/it/alba --precision int8
Attempting to download files for: pockettts/it/alba (it)
-> Downloading ONNX model and side files (this may take a while)...

Download complete. Files saved to: ~/.cache/huggingface/hub/models--Phoonnx--pockettts-it-kyutai-unicode/snapshots/8a5d2d456f99075eb69b84f4f5dc009ed683fa9f
```

Then load the voice with the same precision and speak:

```python
import wave
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.load()
manager.merge_default_voices()

info = manager.voices["pockettts/it/alba"]
print(f"int8 on disk: {info.with_precision('int8').disk_size() / 1e6:.0f} MB")
print(f"full on disk: {info.with_precision('full').disk_size() / 1e6:.0f} MB")

voice = info.load(precision="int8")
with wave.open("alba.wav", "wb") as wav_file:
    voice.synthesize_wav("Ciao, questa è una prova.", wav_file)
```

```
int8 on disk: 171 MB
full on disk: 62 MB
```

`disk_size()` counts only the files that are in the cache. The full files are not
downloaded, so the full size counts only the files that both precisions share.

## 3. Set the precision once

The precision comes from the first of these that is set:

1. The `precision` argument of `load()`, `download_all()` or `download_voice_by_id()`, the `--precision` option of `phoonnx-voices download`, or the `precision` key of the OVOS plugin.
2. The `PHOONNX_PRECISION` environment variable.
3. `auto`.

```
$ PHOONNX_PRECISION=int8 python -c "
from phoonnx.model_manager import TTSModelManager
m = TTSModelManager(); m.load()
print(m.voices['pockettts/it/alba'].with_precision().model_url.rsplit('/', 1)[-1])"
flow_lm_main_int8.onnx
```

Case and spaces around the value do not matter. An empty value counts as unset.

## What each value loads

| Value | Full files only | Full and int8 | Int8 only |
|---|---|---|---|
| `full` | full | full | `PrecisionUnavailable` |
| `int8` | `PrecisionUnavailable` | int8 | int8 |
| `auto` | full | full, or int8 when the variant sets `"auto": true` and no GPU provider runs the voice | int8 |

No entry in the bundled index sets `"auto": true`, so `auto` loads the full files of every
voice that has them:

```python
>>> [info.voice_id for info in manager.all_voices if (info.variants or {}).get("int8", {}).get("auto")]
[]

```

## Errors you may see

A precision that the voice has no files for raises `PrecisionUnavailable`. It is a `ValueError` and a `VoiceLoadError`:

```python
>>> manager.voices["OpenVoiceOS/phoonnx_oc_miro_unicode"].with_precision("int8")
Traceback (most recent call last):
  ...
phoonnx.errors.PrecisionUnavailable: voice 'OpenVoiceOS/phoonnx_oc_miro_unicode' has no int8 files; precision 'auto' or 'full' loads it

```

A value that is not `int8`, `full` or `auto` raises `ValueError`:

```python
>>> manager.voices["pockettts/it/alba"].with_precision("fp16")
Traceback (most recent call last):
  ...
ValueError: precision must be one of int8, full, auto, not 'fp16'

```

The OVOS plugin checks its `precision` key at start, so a wrong value stops the plugin before it loads a voice. A voice with no files for a correct value raises `PrecisionUnavailable` at its first synthesis.

## The deprecated int8 voice id

The id `edwixx/pt-BR-rapido-int8` loads the int8 files of `edwixx/pt-BR-rapido`. It logs a deprecation warning that names the release that removes it. Use `edwixx/pt-BR-rapido` with `precision="int8"`.

---

[← CLI reference](cli.md) · [Home](index.md) · [Use phoonnx offline →](offline.md)
