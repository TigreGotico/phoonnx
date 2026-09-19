# CLI Reference

This page is for anyone managing phoonnx voices from the shell. It documents the
`phoonnx-voices` command (the `phoonnx.cli:cli` entry point declared in
`pyproject.toml`).

## Usage

```bash
phoonnx-voices <command> [options]
```

---

## Commands

### `update-cache`

Merges the bundled voice indexes (Piper, Mimic3, OpenVoiceOS, and the rest — see
[voice_manager.md](voice_manager.md#where-voices-come-from)) into the local cache and
saves it. This reads packaged JSON index files, not live network endpoints.

```bash
phoonnx-voices update-cache
```

**Options:**

| Flag | Description |
|------|-------------|
| `--no-clear` | Do not wipe the existing cache before updating. Only adds new voices. |

**Example:**

```bash
# Full refresh (default)
phoonnx-voices update-cache

# Incremental update — keep existing entries
phoonnx-voices update-cache --no-clear
```

---

### `list-langs`

Lists all language codes available in the local voice cache.

```bash
phoonnx-voices list-langs
```

Run `update-cache` first if no languages appear.

---

### `list-voices`

Lists all available voice models, optionally filtered by language.

```bash
phoonnx-voices list-voices [--lang LANG] [--verbose]
```

**Options:**

| Flag | Description |
|------|-------------|
| `--lang LANG` | Filter by language code (e.g. `en-US`, `pt-PT`) |
| `-v`, `--verbose` | Show full details for each voice |

**Examples:**

```bash
# All voices
phoonnx-voices list-voices

# Portuguese voices only
phoonnx-voices list-voices --lang pt-PT

# With detailed info
phoonnx-voices list-voices --verbose
```

---

### `list-available`

Lists every voice ID bundled with phoonnx, grouped by source (Piper, Mimic3, OVOS, …),
by reading the packaged index files directly. It downloads nothing — no config or model
files are fetched — so it works before `update-cache` has ever been run.

```bash
phoonnx-voices list-available
```

Use `download <VOICE_ID>` to fetch a specific voice from this list on demand.

---

### `download`

Downloads everything a voice needs to run offline: the ONNX model, the config,
the tokenizer artifacts (`vocab.json`/`tokens.txt`/`tokenizer.json`) and any
vocoder, style embedding, speaker encoder or auxiliary graph the voice uses.

```bash
phoonnx-voices download VOICE_ID
```

The voice is looked up in the local cache first; if it is not cached, the download
falls back to the bundled indexes, so a single voice can be fetched without running
`update-cache` first.

**Example:**

```bash
phoonnx-voices download OpenVoiceOS/pipertts_es-ES_dii
```

Files are fetched with `huggingface_hub` straight into the shared HuggingFace
cache (`~/.cache/huggingface/hub/` by default, or wherever `HF_HOME`/`HF_HUB_CACHE`
points), the same place any other program on the machine keeps hub files. A model
named by more than one voice, or already pulled by another tool, is stored once
and reused, not downloaded again.

---

### `add-alignment`

Exposes a voice model's per-phoneme duration tensor as a graph output, so
`synthesize(include_alignments=True)` can report per-phoneme timings for it.

```bash
phoonnx-voices add-alignment MODEL_PATH
```

The patched copy is written to `<model>.alignment.onnx` next to the original,
which is where phoonnx looks for it when a voice is asked for alignments.
Running this once, offline, saves every later load from doing the same surgery
on its first request. See [alignment.md](alignment.md) for what alignments are
and which models can produce them.

**Options:**

| Flag | Description |
|------|-------------|
| `--output`, `-o` | Write the patched model somewhere other than the default sibling path. |

It needs the `onnx` package (`pip install phoonnx[streaming]`) and a model
whose graph has a locatable duration tensor; it says which of the two is
missing and exits non-zero otherwise.

---

## Examples: Full Workflow

```bash
# 1. Fetch the voice catalog
phoonnx-voices update-cache

# 2. Browse available languages
phoonnx-voices list-langs

# 3. Find a voice for Portuguese
phoonnx-voices list-voices --lang pt-PT

# 4. Download a specific voice
phoonnx-voices download OpenVoiceOS/phoonnx_eu-ES_dii_espeak
```
