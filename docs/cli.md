# CLI Reference

phoonnx ships with a command-line interface for managing voices.

## Usage

```bash
phoonnx-voices <command> [options]
```

---

## Commands

### `update-cache`

Fetches the latest voice lists from all upstream sources (Piper, Mimic3, OpenVoiceOS, Proxectonos, Phonikud) and saves them to the local cache.

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

### `download`

Downloads the ONNX model (and config/tokens files) for a specific voice ID.

```bash
phoonnx-voices download VOICE_ID
```

The `VOICE_ID` must exist in the local cache. Run `update-cache` first if needed.

**Example:**

```bash
phoonnx-voices download OpenVoiceOS/pipertts_es-ES_dii
```

Files are saved to the XDG cache directory: `~/.cache/phoonnx/voices/<voice_id>/`

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
