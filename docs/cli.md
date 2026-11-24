# CLI reference

<!-- agent-hint
covers: every phoonnx-voices subcommand, its options, its output and its exit status
source: phoonnx/cli.py, pyproject.toml [project.scripts]
generated: no
verify: diff <(XDG_CACHE_HOME=.venv/verify-cache HF_HUB_OFFLINE=1 .venv/bin/phoonnx-voices --help) <(sed -n '/^\$ phoonnx-voices --help$/,/^\x60\x60\x60$/{//!p}' docs/cli.md) && diff <(XDG_CACHE_HOME=.venv/verify-cache HF_HUB_OFFLINE=1 .venv/bin/phoonnx-voices update-cache | sed -E 's/(: |Languages \(|Voices \(|^Found )[0-9]+/\1.../') <(sed -n '/^\$ phoonnx-voices update-cache$/,/^\x60\x60\x60$/{//!p}' docs/cli.md) && [ $(sed -n '/^\$ phoonnx-voices list-langs | head -5$/,/^\x60\x60\x60$/{//!p}' docs/cli.md | wc -l) -gt 3 ] && ! grep -vxF -e '...' -f <(XDG_CACHE_HOME=.venv/verify-cache HF_HUB_OFFLINE=1 .venv/bin/phoonnx-voices list-langs | sed -E 's/(: |Languages \(|Voices \(|^Found )[0-9]+/\1.../') <(sed -n '/^\$ phoonnx-voices list-langs | head -5$/,/^\x60\x60\x60$/{//!p}' docs/cli.md) && [ $(sed -n '/^\$ phoonnx-voices list-voices --lang pt-PT | head -4$/,/^\x60\x60\x60$/{//!p}' docs/cli.md | wc -l) -gt 3 ] && ! grep -vxF -e '...' -f <(XDG_CACHE_HOME=.venv/verify-cache HF_HUB_OFFLINE=1 .venv/bin/phoonnx-voices list-voices --lang pt-PT | sed -E 's/(: |Languages \(|Voices \(|^Found )[0-9]+/\1.../') <(sed -n '/^\$ phoonnx-voices list-voices --lang pt-PT | head -4$/,/^\x60\x60\x60$/{//!p}' docs/cli.md) && [ $(sed -n '/^\$ phoonnx-voices list-available | head -6$/,/^\x60\x60\x60$/{//!p}' docs/cli.md | wc -l) -gt 3 ] && ! grep -vxF -e '...' -f <(XDG_CACHE_HOME=.venv/verify-cache HF_HUB_OFFLINE=1 .venv/bin/phoonnx-voices list-available | sed -E 's/(: |Languages \(|Voices \(|^Found )[0-9]+/\1.../') <(sed -n '/^\$ phoonnx-voices list-available | head -6$/,/^\x60\x60\x60$/{//!p}' docs/cli.md) && ! comm -3 <(sed '/<!-- agent-hint/,/-->/d' docs/cli.md | grep -oP -- '(?<![\w-])--[a-z][a-z-]*' | sort -u) <(for c in update-cache list-langs list-voices list-available download add-alignment export; do .venv/bin/phoonnx-voices $c --help; done | grep -oP -- '(?<![\w-])--[a-z][a-z-]*' | sort -u) | grep .
see: quickstart.md, choose_voice.md, voice_manager.md, offline.md
-->

This page shows how to manage voices from the shell with the `phoonnx-voices` command. It is for anyone who finds, downloads or prepares voices without Python code.

## A first session

Fill the local catalog from the index files that ship with phoonnx:

```
$ phoonnx-voices update-cache
Clearing local voice cache...
Loading bundled voice indexes...

Cache updated successfully!
Total voices available: ...
Total languages supported: ...
```

The last two lines give the number of voices and language tags in the bundled index.

List the languages:

```
$ phoonnx-voices list-langs | head -5

Supported Languages (...):
 - aae
 - aal
 - aao
```

Find the voices for a language:

```
$ phoonnx-voices list-voices --lang pt-PT | head -4

Found ... voices for language 'pt-PT':
* OpenVoiceOS/phoonnx_pt-PT_miro_unicode (pt-PT)
* OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone (pt-PT)
```

Download one voice:

```
$ phoonnx-voices download OpenVoiceOS/phoonnx_pt-PT_miro_unicode
Attempting to download files for: OpenVoiceOS/phoonnx_pt-PT_miro_unicode (pt-PT)
-> Downloading ONNX model and side files (this may take a while)...

Download complete. Files saved to: ~/.cache/huggingface/hub/models--Phoonnx--phoonnx_pt-PT_miro_unicode/snapshots/eb9a020da900534d7232f11a6cf5d1ae1c545219
```

An id that is not in the catalog stops the command with exit status 1:

```
$ phoonnx-voices download no/such-voice
Voice ID 'no/such-voice' not found in local cache, looking it up in bundled indexes...
Error: Voice ID 'no/such-voice' not found. Run 'phoonnx-voices list-available' to see the voice IDs.
$ echo $?
1
```

The command also logs an `ERROR` line from the voice manager before the last line.

## Commands

```
$ phoonnx-voices --help
Usage: phoonnx-voices [OPTIONS] COMMAND [ARGS]...

  CLI to manage and download phoonnx TTS voice models.

Options:
  --help  Show this message and exit.

Commands:
  add-alignment   Expose a model's per-phoneme duration tensor as a graph...
  download        Downloads the model, config, and token files for a...
  export          Write the voice of MODEL_PATH as a Piper or sherpa-onnx...
  list-available  Lists all voice IDs bundled with phoonnx, grouped by...
  list-langs      Lists all supported language codes available in the...
  list-voices     Lists all available voice models.
  update-cache    Clears the local voice cache, loads the bundled voice...
```

### `update-cache`

Clears the local catalog, merges the bundled voice index files into it, and saves it. It reads the index files of the package, not the network. See [Where voices come from](voice_manager.md#where-voices-come-from).

| Option | Effect |
|---|---|
| `--no-clear` | Keep the existing catalog and add the bundled voices to it. |

### `list-langs`

Lists the language codes in the local catalog. When the catalog is empty, it prints `No languages found. Run 'update-cache' first.`

### `list-voices`

Lists the voices in the local catalog. When the catalog is empty, it prints `No voices found in cache. Run 'update-cache' first.`

| Option | Effect |
|---|---|
| `--lang LANG` | List the voices for a language. A tag with a region (`pt-PT`) keeps the voices of that region. A tag with no region (`pt`), or a region with no voice (`en-AU`), lists the nearest voices first. [Choose a voice](choose_voice.md) shows the output of each case. |
| `-v`, `--verbose` | Print the id, name, language, engine, phoneme type, model URL and config URL of each voice. A voice that needs a reference clip also gets the line `Requires reference: yes (needs a SynthesisConfig.reference / a style path)`. |

Each line starts with the voice id. A voice with a display name shows that name after the language, for example `* pockettts/it/alba (it) - Pocket TTS Alba`.

### `list-available`

Lists every voice id in the bundled index files, grouped by source. It reads the files directly, so it works before `update-cache` and downloads nothing.

```
$ phoonnx-voices list-available | head -6

Total available voices found (bundled indexes): ...

--- OVOS Voices (...) ---
  OpenVoiceOS/phoonnx_an_dii_unicode
  OpenVoiceOS/phoonnx_an_miro_unicode
```

### `download`

Downloads every file that a voice needs: the ONNX model, the config, the tokenizer files, and any vocoder, style, speaker encoder or other graph of the voice.

| Option | Effect |
|---|---|
| `--precision auto\|int8\|full` | Download the files that load with this precision. When unset, the value comes from `PHOONNX_PRECISION`, else `auto`. See [Precision variants](precision.md). |

The command looks the id up in the local catalog first. When the id is not there, it reads the bundled index files, so you can download a voice without `update-cache`.

The files go into the Hugging Face hub cache, `~/.cache/huggingface/hub/` by default, or the folder that `HF_HOME` or `HF_HUB_CACHE` names. Other programs share this cache. A file that another voice or program already downloaded is not downloaded again.

The command exits with status 1 when the id is unknown, when the network fails, and when the machine is offline and the voice is not cached. The last line on stderr names the error. [Use phoonnx offline](offline.md) shows the offline case.

### `add-alignment`

Writes a copy of a model with its per-phoneme duration tensor exposed as a graph output. With that copy, `synthesize(include_alignments=True)` reports the timing of each phoneme. See [Phoneme alignment](alignment.md).

```bash
phoonnx-voices add-alignment MODEL_PATH
```

| Option | Effect |
|---|---|
| `--output`, `-o` | Write the copy to this path. |

Without `--output`, the copy goes to `<model>.alignment.onnx` next to the model, or into the folder that `PHOONNX_ORT_CACHE_DIR` names. phoonnx looks for the copy there when a voice is asked for alignments.

The command needs the `onnx` package from the `streaming` extra. Without it, the command stops with exit status 1:

```
$ phoonnx-voices add-alignment model.onnx
Error: the 'onnx' package is required for graph surgery; install it with `pip install phoonnx[streaming]`.
```

It also exits with status 1 when the graph has no duration tensor that it can find.

### `export`

Writes a voice as a Piper voice or a sherpa-onnx voice in the folder DEST. The command copies the model and writes the config that the runtime reads. The source model does not change.

```bash
phoonnx-voices export MODEL_PATH DEST --to piper|sherpa [--config CONFIG]
```

| Option | Effect |
|---|---|
| `--to piper` | Write `<model>` and `<model>.json`, which `piper.PiperVoice.load` reads. |
| `--to sherpa` | Write `<model>` with the sherpa-onnx metadata in it, and `tokens.txt`. This needs the `onnx` package from the `streaming` extra. |
| `--config` | The voice config. Without it, the command reads `<model>.json`. |

The command refuses a voice that the runtime cannot read with the ids that phoonnx gives it, and exits with status 1. The last line names the reason. For example, sherpa-onnx reads one code point per symbol, so it cannot read a Piper 1.5 voice that merges diphthongs into one id:

```
$ phoonnx-voices export de_DE-thorsten-medium/model.onnx thorsten-sherpa --to sherpa --config de_DE-thorsten-medium/config.json
Error: sherpa-onnx reads one code point per symbol; this voice reads 'aɪ' (2 code points) as the ids [161]
```

[Run a voice in Piper or sherpa-onnx](training/export.md#run-a-voice-in-piper-or-sherpa-onnx) shows a full export and the runtime that reads it.

---

[Docs index](index.md) · Previous: [Voice manager](voice_manager.md) · Next: [Precision variants](precision.md)
