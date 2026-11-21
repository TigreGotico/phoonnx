# Phoneme Alignment

<!-- agent-hint
covers: how to get per-phoneme timing from synthesis, at runtime or from an exported model
source: phoonnx/voice.py:TTSVoice.synthesize, phoonnx/voice.py:TTSVoice.render, phoonnx/alignment.py, phoonnx/onnx_surgery.py:add_phoneme_alignment_output, phoonnx/cli.py:add_alignment
generated: no
verify: pytest tests/test_alignment.py tests/test_runtime_alignment.py, plus the runtime alignment and add-alignment examples on the page
see: streaming.md, cli.md, training/export.md
-->

This page shows how to get the timing of each phoneme in synthesized audio, for developers who build lip-sync, karaoke highlighting or subtitles.

Phoneme alignment tells you how many audio samples each phoneme takes in the output.

> **Model support required.** Alignment is an optional second output of the ONNX
> model. Standard exported models do **not** include it. For a VITS-family model,
> phoonnx adds that output at runtime when the `onnx` package is installed. See
> [Runtime alignment](#runtime-alignment-for-models-exported-without-the-flag) and
> [Exporting a model with alignment support](#exporting-a-model-with-alignment-support).

---

## Getting alignments from `synthesize()`

Pass `include_alignments=True` to `TTSVoice.synthesize()`:

```python
from phoonnx.voice import TTSVoice

voice = TTSVoice.load("model.onnx", "config.json")

for chunk in voice.synthesize("Olá mundo.", include_alignments=True):
    print(f"phonemes : {chunk.phonemes}")
    print(f"phoneme ids: {chunk.phoneme_ids}")

    if chunk.phoneme_alignments:
        for align in chunk.phoneme_alignments:
            duration_ms = align.num_samples / chunk.sample_rate * 1000
            print(f"  {align.phoneme!r:6s}  {duration_ms:6.1f} ms")
    else:
        # Model does not expose alignment output, or reconstruction failed
        print("  (no alignment available)")
```

With the files of `OpenVoiceOS/phoonnx_pt-PT_miro_unicode` and the `streaming` extra
(which installs `onnx`), the output is:

```text
phonemes : ['O', 'l', 'a', '́', ' ', 'm', 'u', 'n', 'd', 'o', '.']
phoneme ids: [1, 0, 26, 0, 23, 0, 12, 0, 41, 0, 3, 0, 24, 0, 32, 0, 25, 0, 15, 0, 26, 0, 8, 0, 2]
  'o'      104.5 ms
  'l'       46.4 ms
  'a'       58.0 ms
  '́'       34.8 ms
  ' '       34.8 ms
  'm'       46.4 ms
  'u'       69.7 ms
  'n'       81.3 ms
  'd'       69.7 ms
  'o'       46.4 ms
  '.'      534.1 ms
```

The alignment shows `o` where `phonemes` shows `O`. The vocabulary of this voice holds no
`O`, so the tokenizer gives the id of `o`, and the alignment reads each symbol back from
the vocabulary. The first call also writes `model.alignment.onnx` next to `model.onnx`.

### `AudioChunk` alignment fields

| Field | Type | Description |
|---|---|---|
| `phonemes` | `list[str]` | Phoneme tokens for this sentence |
| `phoneme_ids` | `list[int]` | Integer IDs passed to the ONNX model |
| `phoneme_id_samples` | `np.ndarray \| None` | Raw sample counts per phoneme ID (from model) |
| `phoneme_alignments` | `list[PhonemeAlignment] \| None` | Reconstructed per-phoneme timings |

`phoneme_alignments` is `None` when:
- `include_alignments=False` (default)
- The model has only one output (does not support alignment)
- The alignment reconstruction fails (ID sequence mismatch)

`phonemes` and `phoneme_ids` are always populated regardless of `include_alignments`.

### `PhonemeAlignment`

```python
@dataclass
class PhonemeAlignment:
    phoneme: str      # the phoneme token, e.g. "h", "ɛ", "l"
    num_samples: int  # number of PCM samples occupied by this phoneme
```

Convert `num_samples` to milliseconds: `num_samples / chunk.sample_rate * 1000`.

---

## Lower-level: `render()`

If you work with token IDs, put them in an `Utterance` and render it with `alignments=True`:

```python
from phoonnx.voice import Utterance

chunk = voice.render(Utterance(text="", symbols=None, ids=tuple(phoneme_ids)), alignments=True)
samples = chunk.phoneme_id_samples  # samples for each ID, or None if the model has no durations
```

`render()` always returns an `AudioChunk`. When the model gives durations,
`chunk.phoneme_alignments` also holds the samples of each phoneme, read from the voice's
vocabulary as in `synthesize()`.

---

## `hop_length`

The raw model output is a duration in frames; phoonnx converts frames to samples
using `hop_length` (default **256**, matching the standard VITS vocoder hop size).

Override it in the voice config JSON:

```json
{
  "hop_length": 256
}
```

Or via `VoiceConfig`:

```python
voice.config.hop_length = 256
```

---

## Engine support matrix

Alignment works uniformly across every duration-predictor engine adapter — not
just VITS. Each adapter's `parse_outputs` looks for a per-phoneme duration
tensor among the model's ONNX outputs by name (see `DURATION_OUTPUT_NAMES` on
each adapter class), and `TTSVoice.render` converts whatever it
finds to samples the same way for every engine: `samples = frames * hop_length`.
This holds for single-stage engines (the "frames" are the internal
duration-predictor's frame rate, matching the decoder's implicit hop) and for
two-stage engines (the "frames" are literal mel frames, and `hop_length` is
the mel→waveform vocoder's hop size).

| Engine family | Native durations? | Units | Conversion rule |
|---|---|---|---|
| VITS (piper/mimic3/coqui/phoonnx) | Yes, in phoonnx's own `--add-phoneme-alignment` exports (`phoneme_id_samples`) | duration-predictor frames | `samples = frames * hop_length` |
| YourTTS | Same VITS-family contract; yes if the export adds a second output | duration-predictor frames | `samples = frames * hop_length` |
| OptiSpeech | Yes — `durations` is a standard third output (`[wav, wav_lengths, durations]`) | duration-predictor frames | `samples = frames * hop_length` |
| FastPitch / Mixer-TTS | No — standard exports emit only `mel_spec` | mel frames (for an export that adds one) | `samples = frames * hop_length` |
| GlowTTS (Larynx/Coqui) | No — standard exports emit only the mel tensor(s) | mel frames (for an export that adds one) | `samples = frames * hop_length` |
| Matcha-TTS | No — standard two-stage exports emit `[mel, mel_lengths]`; end-to-end fused exports have no separate mel/durations at all | mel frames (for an export that adds one) | `samples = frames * hop_length` |
| StyleTTS2 / Kokoro | No — standard exports emit only the waveform | model frames (for an export that adds one) | `samples = frames * hop_length` |
| ZipVoice | **Not supported.** In-context flow-matching with no discrete duration predictor — there is nothing to align to a token. `synthesize()` never populates `phoneme_id_samples`. | — | — |

For the "No" rows, the adapter still implements name-based detection
(`DURATION_OUTPUT_NAMES` lists plausible output names such as `durations`,
`dur`, `w_ceil`, `logw`, `pred_dur`) so a model re-exported with that tensor
exposed gives alignments with no change to the adapter code. For a model without
that tensor, `include_alignments=True` gives `None` for `phoneme_id_samples` and
`phoneme_alignments`, as for an unpatched VITS model. Synthesis itself does not change.

ZipVoice and the other in-context or autoregressive engines have no discrete duration
predictor, so they cannot give native alignment this way. phoonnx has no forced
aligner for their audio.

---

## Exporting a model with alignment support

Standard VITS models expose only the audio tensor. To expose the phoneme-duration
tensor as a second output, use the `--add-phoneme-alignment` flag when exporting:

```bash
python -m phoonnx_train.export_onnx checkpoint.ckpt -c config.json --add-phoneme-alignment
```

This changes the exported `.onnx` graph so that the `Ceil` node output (the phoneme
durations) is a named model output. `add_phoneme_alignment_output()` in
`phoonnx/onnx_surgery.py` makes the change.

You can also apply it post-hoc to a model you already have, without the
training package installed:

```bash
phoonnx-voices add-alignment model.onnx
```

```text
Wrote alignment-capable model to: model.alignment.onnx
```

The command writes `model.alignment.onnx` next to the original. The runtime path looks for
that name, so every later load finds it and no request pays for the surgery. Pass `-o`
to write somewhere else. The command fails with an error that names the reason when the
model has no unique duration tensor or `onnx` is not installed.

The same thing from Python:

```python
from phoonnx.alignment import export_alignment_model

export_alignment_model("model.onnx", "model-aligned.onnx")
```

> **Compatibility note.** Adding the alignment output may break third-party
> frameworks (e.g. Piper) that expect a single output tensor. Keep a separate
> copy of the model for standard TTS use.

---

## Runtime alignment for models exported without the flag

You don't have to re-export a model to get alignments from it.
``TTSVoice.synthesize(include_alignments=True)`` (and
``render(utterance, alignments=True)``) retrofit the duration
output automatically the first time they are asked for one and the loaded
session doesn't already have it:

1. Locate the duration tensor in the model's graph (the same ``Ceil``-node
   autodetection ``--add-phoneme-alignment`` uses, via
   `phoonnx.onnx_surgery.add_phoneme_alignment_output`).
2. Write a patched copy next to the original model, `<model>.alignment.onnx`
   (or under `PHOONNX_ORT_CACHE_DIR` if that env var is set — the model's own
   directory may be read-only, e.g. a shared voice cache).
3. Rebuild an ONNX Runtime session from the patched copy, on the same
   execution providers as the original, and retry inference on it.

The surgery for a given destination runs under a lock and the patched model
is moved into place atomically, so concurrent requests on a shared voice
either wait for the copy being written or reuse the one already there.

This runs **at most once per `TTSVoice` instance** — the outcome (including a
negative one: no locatable duration tensor, `onnx` not installed, or the
derived file couldn't be written) is cached on the voice, so later
`include_alignments=True` calls either reuse the already-open patched session
or go straight back to `None` without retrying. Across process restarts, the
`<model>.alignment.onnx` file itself is the cache: if it already exists and is
newer than the model, it's reused as-is, no surgery repeated.

`include_alignments=False` never runs any of this and costs nothing.

```python
from phoonnx.voice import TTSVoice

# model.onnx was exported without --add-phoneme-alignment
voice = TTSVoice.load("model.onnx", "config.json")

for chunk in voice.synthesize("Hello world.", include_alignments=True):
    # first call: locates the duration tensor, writes model.alignment.onnx
    # next to model.onnx, and synthesizes from the patched session.
    print(chunk.phoneme_alignments)
```

If the model genuinely has no discrete duration predictor (e.g. ZipVoice) or
the `onnx` package isn't installed, this degrades exactly like any other
unsupported model: `phoneme_id_samples` / `phoneme_alignments` stay `None`,
and a single debug/info-level log line explains why.

---

## Use cases

### Visemes / lip-sync

Map each phoneme to a viseme index, then schedule face-rig blend-shape changes
at the sample offset accumulated from `num_samples`:

```python
VISEME = {"p": 0, "b": 0, "m": 0, "f": 1, "v": 1, ...}  # your mapping

offset = 0
for align in chunk.phoneme_alignments or []:
    t = offset / chunk.sample_rate
    viseme = VISEME.get(align.phoneme, -1)
    schedule_viseme(t, viseme)
    offset += align.num_samples
```

### Karaoke / subtitle highlighting

Add up the sample offsets to get the start time of each word. Use these times to move
the text highlight with the audio.

---

[Docs index](index.md) · Previous: [Streaming](streaming.md) · Next: [Configuration reference](configuration.md)
