# Use phoonnx offline

<!-- agent-hint
covers: how to prepare voices online and then run phoonnx with no network, what works offline and what fails
source: phoonnx/model_manager.py:_hf_fetch, phoonnx/model_manager.py:_offline_snapshot, phoonnx/model_manager.py:_voice_snapshot, phoonnx/model_manager.py:TTSModelManager.merge_default_voices, phoonnx/cli.py:download_voice, phoonnx/opm.py:PhoonnxTTSPlugin.get_tts
generated: no
verify: diff <(XDG_CACHE_HOME=.venv/verify-cache HF_HUB_OFFLINE=1 .venv/bin/phoonnx-voices update-cache | sed -E 's/: [0-9]+$/: .../') <(sed -n '/^\$ HF_HUB_OFFLINE=1 phoonnx-voices update-cache$/,/^\x60\x60\x60$/{//!p}' docs/offline.md) && diff <(XDG_CACHE_HOME=.venv/verify-cache HF_HUB_CACHE=.venv/verify-empty-hub HF_HUB_OFFLINE=1 .venv/bin/phoonnx-voices download piper/en_US-danny-low 2>&1) <(sed -n '/^\$ HF_HUB_OFFLINE=1 phoonnx-voices download piper\/en_US-danny-low$/,/^\x60\x60\x60$/{//!p}' docs/offline.md)
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

Download complete. Files saved to: ~/.cache/huggingface/hub/models--Phoonnx--phoonnx_pt-PT_miro_unicode/snapshots/aec14a0428e6b67a47eb54b2baa4a604d42c0063
```

The files go into the Hugging Face hub cache, `~/.cache/huggingface/hub/` by default. Set `HF_HOME` or `HF_HUB_CACHE` to put the cache somewhere else. Use the same value online and offline.

For a voice with int8 files, download the precision that you will load. See [Precision variants](precision.md).

Then speak one sentence with each voice while the machine is still online. Some phonemizers and post-processing models fetch their own model files on first use, and this step puts those files in the cache too.

## 2. Turn off network access

Set `HF_HUB_OFFLINE=1`. The Hugging Face client then reads only the local cache and makes no network requests:

```bash
export HF_HUB_OFFLINE=1
```

Without this variable, phoonnx also works offline. The client then tries the network first and waits for the attempt to fail.

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
Total voices available: ...
Total languages supported: ...
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

A voice that is not in the cache cannot load. The error names the missing file, its repository and the pinned commit. Python stops with this error:

```
huggingface_hub.errors.LocalEntryNotFoundError: model.onnx of Phoonnx/vits-en_us-danny-low-espeak at commit aeaf5a3ab43f6d25a6798581793de0b97818e3ab is not in the hub cache, and the hub cannot be reached. Connect once so that phoonnx can download it. After a phoonnx upgrade, the voice index can name a file that no cached snapshot holds.
```

The download command gives the same error and exits with status 1:

```
$ HF_HUB_OFFLINE=1 phoonnx-voices download piper/en_US-danny-low
Attempting to download files for: piper/en_US-danny-low (en-US)
-> Downloading ONNX model and side files (this may take a while)...
An unexpected error occurred during download: model.onnx of Phoonnx/vits-en_us-danny-low-espeak at commit aeaf5a3ab43f6d25a6798581793de0b97818e3ab is not in the hub cache, and the hub cannot be reached. Connect once so that phoonnx can download it. After a phoonnx upgrade, the voice index can name a file that no cached snapshot holds.
```

Fix: connect the machine, download the voice, and go offline again.

The OVOS plugin finds a voice in the catalog at start, but it downloads the voice at the first synthesis. So a plugin with an uncached voice starts offline and fails at its first sentence.

## Why a cached voice keeps working after an upgrade

Each Hugging Face URL in the bundled index names a commit, not a branch. An installed phoonnx release loads the files of that commit.

A later phoonnx release can pin a later commit of the same voice. Online, phoonnx downloads the files that changed. Offline, the cache does not hold the pinned commit. phoonnx then loads the newest cached snapshot of the same repository that holds every file the voice needs. It logs one warning that names the repository, the pinned commit and the snapshot that it reads:

```
<repo> is not cached at the pinned commit <commit> and the hub cannot be reached; loading the cached snapshot <snapshot> instead
```

All files of one voice come from the same snapshot, so a model and its config always match. When the pinned snapshot holds only part of the files, as after an interrupted download, phoonnx does not mix commits. The load fails, and a download while online repairs it.

A later release can also name a file that no cached snapshot holds, for example the native `phoonnx.json` config of a voice. Offline, that voice then fails with the error of the previous section, which names the file. So after an upgrade, load each voice once while online. A download while online fetches only the files that changed.

For servers and containers, [Deployment](deployment.md) shows how to keep the cache on a volume and prefetch voices before the server starts.

---

[Docs index](index.md) · Previous: [Precision variants](precision.md) · Next: [Pocket TTS](pockettts.md)
