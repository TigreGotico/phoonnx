# Phoneme Alignment

Phoneme alignment gives you per-phoneme timing: how many audio samples each phoneme
occupies in the synthesized output. This is the foundation for visemes (lip-sync),
karaoke-style word highlighting, and subtitle generation.

> **Model support required.** Alignment is an optional second output of the ONNX
> model. Standard exported models do **not** include it. See
> [Exporting a model with alignment support](#exporting-a-model-with-alignment-support)
> below.

---

## Getting alignments from `synthesize()`

Pass `include_alignments=True` to `TTSVoice.synthesize()`:

```python
from phoonnx.voice import TTSVoice

voice = TTSVoice.load("model-aligned.onnx")

for chunk in voice.synthesize("Hello world.", include_alignments=True):
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

## Lower-level: `phoneme_ids_to_audio()`

If you are working at the phoneme-ID level you can call the method directly:

```python
audio_or_tuple = voice.phoneme_ids_to_audio(phoneme_ids, include_alignments=True)

if isinstance(audio_or_tuple, tuple):
    audio, phoneme_id_samples = audio_or_tuple
    # phoneme_id_samples[i] = samples for phoneme_ids[i], or None if unsupported
else:
    audio = audio_or_tuple  # include_alignments=False
```

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
each adapter class), and `TTSVoice.phoneme_ids_to_audio` converts whatever it
finds to samples the same way for every engine: `samples = frames * hop_length`.
This holds for single-stage engines (the "frames" are the internal
duration-predictor's frame rate, matching the decoder's implicit hop) and for
two-stage engines (the "frames" are literal mel frames, and `hop_length` is
the mel→waveform vocoder's hop size).

| Engine family | Native durations today? | Units | Conversion rule |
|---|---|---|---|
| VITS (piper/mimic3/coqui/phoonnx) | Yes, in phoonnx's own `--add-phoneme-alignment` exports (`phoneme_id_samples`) | duration-predictor frames | `samples = frames * hop_length` |
| YourTTS | Same VITS-family contract; yes if the export adds a second output | duration-predictor frames | `samples = frames * hop_length` |
| OptiSpeech | Yes — `durations` is a standard third output (`[wav, wav_lengths, durations]`) | duration-predictor frames | `samples = frames * hop_length` |
| FastPitch / Mixer-TTS | No — standard exports emit only `mel_spec` | mel frames (if a future export adds one) | `samples = frames * hop_length` |
| GlowTTS (Larynx/Coqui) | No — standard exports emit only the mel tensor(s) | mel frames (if a future export adds one) | `samples = frames * hop_length` |
| Matcha-TTS | No — standard two-stage exports emit `[mel, mel_lengths]`; end-to-end fused exports have no separate mel/durations at all | mel frames (if a future export adds one) | `samples = frames * hop_length` |
| StyleTTS2 / Kokoro | No — standard exports emit only the waveform | model frames (if a future export adds one) | `samples = frames * hop_length` |
| ZipVoice | **Not supported.** In-context flow-matching with no discrete duration predictor — there is nothing to align to a token. `synthesize()` never populates `phoneme_id_samples`. | — | — |

For the "No" rows, the adapter still implements name-based detection
(`DURATION_OUTPUT_NAMES` lists plausible output names such as `durations`,
`dur`, `w_ceil`, `logw`, `pred_dur`) so a model re-exported with that tensor
exposed lights up alignment automatically, with no adapter code changes. Until
then, `include_alignments=True` degrades gracefully: `phoneme_id_samples` and
`phoneme_alignments` come back `None`, exactly like an unpatched VITS model —
synthesis itself is never affected.

**Future work — forced alignment.** ZipVoice and any other in-context /
autoregressive engine without a discrete duration predictor cannot expose
native alignment this way. A forced-alignment fallback (e.g. running a
phoneme-to-audio aligner such as CTC-segmentation or MFA over the generated
audio) would be needed to support those engines; it is not implemented here.

---

## Exporting a model with alignment support

Standard VITS models expose only the audio tensor. To expose the phoneme-duration
tensor as a second output, use the `--add-phoneme-alignment` flag when exporting:

```bash
python -m phoonnx_train.export_onnx checkpoint.ckpt -c config.json --add-phoneme-alignment
```

This modifies the exported `.onnx` graph to surface the `Ceil` node output (phoneme
durations) as a named model output. The modification is done by
`add_phoneme_alignment_output()` in `phoonnx_train/export_onnx.py`.

You can also apply it post-hoc to an already-exported model:

```python
from phoonnx_train.export_onnx import add_phoneme_alignment_output
from pathlib import Path

add_phoneme_alignment_output(
    model_path=Path("model.onnx"),
    output_path=Path("model-aligned.onnx"),  # omit to overwrite in place
    tensor_name="autodetect",                # or pass the tensor name explicitly
)
```

> **Compatibility note.** Adding the alignment output may break third-party
> frameworks (e.g. Piper) that expect a single output tensor. Keep a separate
> copy of the model for standard TTS use.

---

## Runtime alignment for models exported without the flag

You don't have to re-export a model to get alignments from it.
``TTSVoice.synthesize(include_alignments=True)`` (and
``phoneme_ids_to_audio(include_alignments=True)``) retrofit the duration
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

This runs **at most once per `TTSVoice` instance** — the outcome (including a
negative one: no locatable duration tensor, `onnx` not installed, or the
derived file couldn't be written) is cached on the voice, so later
`include_alignments=True` calls either reuse the already-open patched session
or go straight back to `None` without retrying. Across process restarts, the
`<model>.alignment.onnx` file itself is the cache: if it already exists and is
newer than the model, it's reused as-is, no surgery repeated.

`include_alignments=False` never touches any of this — it is the exact same
zero-cost no-op it always was.

```python
from phoonnx.voice import TTSVoice

# model.onnx was exported without --add-phoneme-alignment
voice = TTSVoice.load("model.onnx")

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

Accumulate sample offsets to get word-start timestamps, then use them to
synchronise text highlights with audio playback.
