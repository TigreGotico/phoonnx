# Use phoonnx offline

<!-- agent-hint
covers: how to prepare voices online and then run phoonnx with no network, what works offline and what fails
source: phoonnx/model_manager.py:_hf_fetch, _offline_snapshot, _voice_snapshot, TTSModelManager.merge_default_voices; phoonnx/cli.py:download_voice; phoonnx/opm.py:PhoonnxTTSPlugin.__init__
generated: no
verify: python -m pytest tests/test_offline_model_load.py tests/test_offline_pinned_cache.py; HF_HUB_OFFLINE=1 python hello.py (the quickstart script, voice cached)
see: quickstart.md, deployment.md, voice_manager.md, docker.md
-->

This page shows how to download voices while a machine is online and then speak with no network. It is for anyone who runs phoonnx on a device or server without internet access.

The voice catalog ships inside the package, so phoonnx needs the network only to download voice files. Once the files of a voice are in the local cache, the voice loads and speaks offline.

## 1. Download the voices while online

Download each voice that the machine will use:

```
$ phoonnx-voices download OpenVoiceOS/phoonnx_pt-PT_miro_unicode
Attempting to download files for: OpenVoiceOS/phoonnx_pt-PT_miro_unicode (pt-PT)
-> Downloading ONNX model and side files (this may take a while)...

Download complete. Files saved to: ~/.cache/huggingface/hub/models--Phoonnx--phoonnx_pt-PT_miro_unicode/snapshots/eb9a020da900534d7232f11a6cf5d1ae1c545219
```

The files go into the Hugging Face hub cache, `~/.cache/huggingface/hub/` by default. Set `HF_HOME` or `HF_HUB_CACHE` to put the cache somewhere else. Use the same value online and offline.

For a voice with int8 files, download the precision that you will load. See [Precision variants](precision.md).

Then speak one sentence with each voice while the machine is still online. Some phonemizers and post-processing models fetch their own model files on first use, and this step puts those files in the cache too.

## 2. Turn off network access

Set `HF_HUB_OFFLINE=1`. The Hugging Face client then reads only the local cache and makes no network requests:

```bash
export HF_HUB_OFFLINE=1
```

Without this variable phoonnx also works offline, but the client tries the network first and waits for it to fail.

## 3. Speak

Run the `hello.py` script from the [Quickstart](quickstart.md):

```
$ HF_HUB_OFFLINE=1 python hello.py
$ python -c "import wave; w = wave.open('hello.wav'); print(w.getframerate(), w.getnchannels(), w.getsampwidth())"
22050 1 2
```

The catalog commands also work offline, because they read the bundled index files:

```
$ HF_HUB_OFFLINE=1 phoonnx-voices update-cache
Clearing local voice cache...
Loading bundled voice indexes...

Cache updated successfully!
Total voices available: 3189
Total languages supported: 1816
```

The OVOS plugin starts offline and speaks with a cached voice:

```python
from phoonnx.opm import PhoonnxTTSPlugin

config = {"lang": "pt-PT", "voice": "OpenVoiceOS/phoonnx_pt-PT_miro_unicode"}
plugin = PhoonnxTTSPlugin(config=config)
print(plugin.get_tts("Olá, isto é um teste.", "offline.wav", voice=config["voice"]))
```

```
('offline.wav', None)
```

## What fails offline

A voice that is not in the cache cannot load. Python stops with this error:

```
huggingface_hub.errors.LocalEntryNotFoundError: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.
```

The download command gives the same error and exits with status 1:

```
$ HF_HUB_OFFLINE=1 phoonnx-voices download piper/en_US-danny-low
Attempting to download files for: piper/en_US-danny-low (en-US)
-> Downloading ONNX model and side files (this may take a while)...
An unexpected error occurred during download: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.
```

Fix: connect the machine, download the voice, and go offline again.

The OVOS plugin finds a voice in the catalog at start, but it downloads the voice at the first synthesis. So a plugin with an uncached voice starts offline and fails at its first sentence.

## Why a cached voice keeps working after an upgrade

Each Hugging Face URL in the bundled index names a commit, not a branch. An installed phoonnx release loads the files of that commit.

A later phoonnx release can pin a later commit of the same voice. Online, phoonnx downloads the files that changed. Offline, the cache does not hold the pinned commit, so phoonnx loads the newest cached snapshot of the same repository that holds every file the voice needs. It logs one warning that names the repository, the pinned commit and the snapshot it reads:

```
<repo> is not cached at the pinned commit <commit> and the hub cannot be reached; loading the cached snapshot <snapshot> instead
```

All files of one voice come from the same snapshot, so a model and its config always match. When the pinned snapshot holds only part of the files, as after an interrupted download, phoonnx does not mix commits. The load fails, and a download while online repairs it.

For servers and containers, [Deployment](deployment.md) shows how to keep the cache on a volume and prefetch voices before the server starts.

---

[Docs index](index.md) · Previous: [Precision variants](precision.md) · Next: [Pocket TTS](pockettts.md)
