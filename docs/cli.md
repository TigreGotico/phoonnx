# CLI Reference

phoonnx ships with a command-line interface for managing voices.

## Usage

```bash
phoonnx_cli.py <command> [options]
```

---

## Commands

### `update-cache`

Fetches the latest voice lists from all upstream sources (Piper, Mimic3, OpenVoiceOS, Proxectonos, Phonikud) and saves them to the local cache.

```bash
phoonnx_cli.py update-cache
```

**Options:**

| Flag | Description |
|------|-------------|
| `--no-clear` | Do not wipe the existing cache before updating. Only adds new voices. |

**Example:**

```bash
# Full refresh (default)
phoonnx_cli.py update-cache

# Incremental update — keep existing entries
phoonnx_cli.py update-cache --no-clear
```

---

### `list-langs`

Lists all language codes available in the local voice cache.

```bash
phoonnx_cli.py list-langs
```

Run `update-cache` first if no languages appear.

---

### `list-voices`

Lists all available voice models, optionally filtered by language.

```bash
phoonnx_cli.py list-voices [--lang LANG] [--verbose]
```

**Options:**

| Flag | Description |
|------|-------------|
| `--lang LANG` | Filter by language code (e.g. `en-US`, `pt-PT`) |
| `-v`, `--verbose` | Show full details for each voice |

**Examples:**

```bash
# All voices
phoonnx_cli.py list-voices

# Portuguese voices only
phoonnx_cli.py list-voices --lang pt-PT

# With detailed info
phoonnx_cli.py list-voices --verbose
```

---

### `download`

Downloads the ONNX model (and config/tokens files) for a specific voice ID.

```bash
phoonnx_cli.py download VOICE_ID
```

The `VOICE_ID` must exist in the local cache. Run `update-cache` first if needed.

**Example:**

```bash
phoonnx_cli.py download en_US-lessac-medium
```

Files are saved to the XDG cache directory: `~/.cache/phoonnx/voices/<voice_id>/`

---

## Examples: Full Workflow

```bash
# 1. Fetch the voice catalog
phoonnx_cli.py update-cache

# 2. Browse available languages
phoonnx_cli.py list-langs

# 3. Find a voice for Portuguese
phoonnx_cli.py list-voices --lang pt-PT

# 4. Download a specific voice
phoonnx_cli.py download pt_PT-tugão-medium
```
