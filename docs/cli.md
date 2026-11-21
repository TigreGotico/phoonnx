# CLI reference

<!-- agent-hint
covers: every phoonnx-voices subcommand, its options, its output and its exit status
source: phoonnx/cli.py, pyproject.toml [project.scripts]
generated: no
verify: python -m pytest tests/test_cli.py; phoonnx-voices --help
see: quickstart.md, choose_voice.md, voice_manager.md, offline.md
-->

This page shows how to manage voices from the shell with the `phoonnx-voices` command. It is for anyone who finds, downloads or prepares voices without writing Python.

## A first session

Fill the local catalog from the index files that ship with phoonnx:

```
$ phoonnx-voices update-cache
Clearing local voice cache...
Loading bundled voice indexes...

Cache updated successfully!
Total voices available: 3189
Total languages supported: 1816
```

List the languages:

```
$ phoonnx-voices list-langs | head -5

Supported Languages (1816):
 - aae
 - aal
 - aao
```

Find the voices for a language:

```
$ phoonnx-voices list-voices --lang pt-PT | head -4

Found 51 voices for language 'pt-PT':
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
| `-v`, `--verbose` | Print the id, name, language, engine, phoneme type, model URL and config URL of each voice. A voice that needs a reference clip also gets the line `Requires reference: yes (needs speaker_reference / a style path)`. |

A voice with a display name shows that name in place of its id, for example `* Pocket TTS Alba (it)`. The index holds 1288 such voices. To get the id of one of them, use `--verbose` or `list-available`.

### `list-available`

Lists every voice id in the bundled index files, grouped by source. It reads the files directly, so it works before `update-cache` and downloads nothing.

```
$ phoonnx-voices list-available | head -6

Total available voices found (bundled indexes): 3189

--- OVOS Voices (37) ---
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

---

[Docs index](index.md) · Previous: [Voice manager](voice_manager.md) · Next: [Precision variants](precision.md)
