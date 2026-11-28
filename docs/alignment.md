# Phoneme Alignment

<!-- agent-hint
covers: how to get the timing of each phoneme from synthesis, which engines give it, and how to add the duration output to a model
source: phoonnx/voice.py:TTSVoice.synthesize, phoonnx/voice.py:TTSVoice.render, phoonnx/voice.py:TTSVoice._infer, phoonnx/alignment.py, phoonnx/onnx_surgery.py:add_phoneme_alignment_output, phoonnx/cli.py:add_alignment, phoonnx/engines/base.py:_find_duration_output
generated: no
verify: .venv/bin/python -m doctest -o NORMALIZE_WHITESPACE docs/alignment.md
see: streaming.md, cli.md, training/export.md
-->

This page shows how to get the timing of each phoneme in synthesized audio, for developers
who build lip-sync, karaoke highlights or subtitles.

Phoneme alignment tells you how many audio samples each phoneme takes in the output. The
model must give a duration for each phoneme. A standard VITS export does not give it, but
phoonnx adds that output to a VITS-family model when it loads, if the `onnx` package is
installed. See [Runtime alignment](#runtime-alignment-for-models-exported-without-the-flag).

## Get alignments from `synthesize()`

Pass `include_alignments=True` to `TTSVoice.synthesize()`:

```python
from phoonnx.voice import TTSVoice
from phoonnx.config import SynthesisConfig

voice = TTSVoice.load("model.onnx", "model.json")
config = SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)

for chunk in voice.synthesize("Azul fellawen.", config, include_alignments=True):
    print(f"phonemes : {chunk.phonemes}")
    print(f"phoneme ids: {chunk.phoneme_ids}")

    if chunk.phoneme_alignments:
        for align in chunk.phoneme_alignments:
            duration_ms = align.num_samples / chunk.sample_rate * 1000
            print(f"  {align.phoneme!r:6s}  {duration_ms:6.1f} ms")
    else:
        print("  (no alignment available)")
```

With the files of the Kabyle voice `Phoonnx/phoonnx_kab-DZ_miro_o2i` and the `streaming`
extra (which installs `onnx`), the output is:

```text
phonemes : ['æ', 'z', 'u', 'l', ' ', 'f', 'ə', 'l', 'ː', 'æ', 'w', 'ə', 'n']
phoneme ids: [1, 0, 53, 0, 52, 0, 47, 0, 38, 0, 3, 0, 32, 0, 73, 0, 38, 0, 140, 0, 53, 0, 49, 0, 73, 0, 40, 0, 2]
  'æ'      174.1 ms
  'z'       69.7 ms
  'u'       58.0 ms
  'l'       46.4 ms
  ' '       58.0 ms
  'f'       34.8 ms
  'ə'       46.4 ms
  'l'       46.4 ms
  'ː'       69.7 ms
  'æ'       58.0 ms
  'w'       58.0 ms
  'ə'       81.3 ms
  'n'      197.4 ms
```

The zero noise settings make the durations the same on each run. The first call also
writes `model.alignment.onnx` next to `model.onnx`.

The ids hold a blank (`0`) between phonemes, and a start (`1`) and end (`2`) id. phoonnx
adds the samples of a blank or start id to the next phoneme, and the samples of the end id
to the last phoneme. Each alignment reads its symbol back from the vocabulary of the voice,
so a symbol that the tokenizer maps to another id shows as the symbol of that id.

### `PhonemeAlignment`

Each alignment is a `PhonemeAlignment` with two fields:

```python
>>> from phoonnx.voice import PhonemeAlignment
>>> align = PhonemeAlignment(phoneme="z", num_samples=1536)
>>> round(align.num_samples / 22050 * 1000, 1)
69.7

```

Divide `num_samples` by `chunk.sample_rate` and multiply by 1000 to get milliseconds.

### `AudioChunk` alignment fields

| Field | Type | Description |
|---|---|---|
| `phonemes` | `list[str]` | the phoneme tokens of this sentence |
| `phoneme_ids` | `list[int]` | the integer ids that went into the ONNX model |
| `phoneme_id_samples` | `np.ndarray \| None` | the sample count of each phoneme id |
| `phoneme_alignments` | `list[PhonemeAlignment] \| None` | the sample count of each phoneme |

`phonemes` and `phoneme_ids` are always set. `phoneme_alignments` is `None` in these cases:

* `include_alignments=False`, which is the default.
* The model gives no durations, and phoonnx cannot add them.
* The sample counts do not match the ids one to one.

The audio is the same with and without `include_alignments`, because the same inference
runs.

## Get alignments from `render()`

`TTSVoice.prepare()` turns text into one `Utterance` for each engine call.
`TTSVoice.render()` turns one `Utterance` into audio. Pass `alignments=True` to get the
durations:

```python
for utterance in voice.prepare("Azul fellawen."):
    chunk = voice.render(utterance, alignments=True)
    samples = chunk.phoneme_id_samples  # samples for each id, or None
```

To render your own token ids, build the `Utterance` yourself:

```python
from phoonnx.voice import Utterance

chunk = voice.render(Utterance(text="", symbols=None, ids=tuple(phoneme_ids)), alignments=True)
```

`render()` always gives an `AudioChunk`. When the model gives durations,
`chunk.phoneme_alignments` holds the samples of each phoneme, as with `synthesize()`.

## `hop_length`

The model gives each duration in frames. phoonnx multiplies each frame count by
`hop_length` to get samples. The default is 256, the hop size of the standard VITS
decoder:

```python
>>> from phoonnx.config import DEFAULT_HOP_LENGTH
>>> DEFAULT_HOP_LENGTH
256

```

Set another value in the voice config JSON:

```json
{
  "hop_length": 256
}
```

Or set it on the loaded voice:

```python
voice.config.hop_length = 256
```

## Engine support

Each adapter with a duration predictor looks for a duration tensor among the ONNX outputs
of the model, by name. `DURATION_OUTPUT_NAMES` on the adapter class lists the names:

```python
>>> from phoonnx.engines.vits import VitsAdapter
>>> VitsAdapter.DURATION_OUTPUT_NAMES
['phoneme_id_samples', 'durations', 'w_ceil', 'dur']

```

`TTSVoice` converts the tensor that the adapter finds to samples in the same way for each
engine: `samples = frames * hop_length`. For a single-stage engine, the frames are the
frames of the decoder. For a two-stage engine, the frames are mel frames, and `hop_length`
is the hop size of the vocoder.

| Engine family | Durations in a standard export? | Units |
|---|---|---|
| VITS (Piper, Mimic 3, Coqui, phoonnx) | only with `--add-phoneme-alignment`, or added when phoonnx loads the model | decoder frames |
| YourTTS | the same as VITS | decoder frames |
| x-vits | yes: the adapter computes the durations itself | decoder frames |
| OptiSpeech | yes: `durations` is the third output (`[wav, wav_lengths, durations]`) | decoder frames |
| tiny-voca | yes: `durations` is the second output | frames of 256 samples |
| FastPitch, Mixer-TTS | no: a standard export gives only `mel_spec` | mel frames |
| GlowTTS (Larynx, Coqui) | no: a standard export gives only the mel tensors | mel frames |
| Matcha-TTS | no: a two-stage export gives `[mel, mel_lengths]`, and a fused export gives no durations | mel frames |
| StyleTTS2, Kokoro | no: a standard export gives only the waveform | model frames |
| ZipVoice | not supported: the model has no duration predictor | none |

For a "no" row, the adapter still looks for the names in `DURATION_OUTPUT_NAMES`, so a
model that you export again with the tensor as an output gives alignments with no change
to the adapter. For a model without that tensor, `include_alignments=True` gives `None` for
`phoneme_id_samples` and `phoneme_alignments`, and synthesis does not change.

ZipVoice and the other in-context or autoregressive engines have no duration predictor,
so they cannot give alignments in this way. phoonnx has no forced aligner for their audio.

## Exporting a model with alignment support

A standard VITS model gives only the audio tensor. To give the duration tensor as a second
output, add `--add-phoneme-alignment` when you export:

```bash
python -m phoonnx_train.export_onnx checkpoint.ckpt -c config.json --add-phoneme-alignment
```

The flag makes the output of the `Ceil` node, which holds the phoneme durations, a named
output of the graph. `add_phoneme_alignment_output()` in `phoonnx/onnx_surgery.py` makes
this change.

You can also change a model that you have, without the training package:

```bash
phoonnx-voices add-alignment model.onnx
```

```text
Wrote alignment-capable model to: model.alignment.onnx
```

The command writes `model.alignment.onnx` next to the original. phoonnx looks for that
name when it loads the voice, so no request pays for the change. Use `-o` to write the file
to another path. When the model has no single `Ceil` tensor, the command stops with exit
status 1 and gives the reason:

```text
Error: no unique duration (Ceil-op) tensor found in 'model.decoder.onnx'; this model cannot produce phoneme alignments
```

When `onnx` is not installed, the error names the `phoonnx[streaming]` extra.

Do the same from Python:

```python
from phoonnx.alignment import export_alignment_model

export_alignment_model("model.onnx", "model-aligned.onnx")
```

A runtime that expects one output tensor can fail on a model with the second output. The
command writes a copy and keeps the original model, so use the original for such a runtime.

## Runtime alignment for models exported without the flag

You do not have to export a model again to get alignments from it. The first time you ask
for alignments with `synthesize(include_alignments=True)` or `render(utterance,
alignments=True)`, phoonnx adds the duration output if the model does not have it:

1. phoonnx finds the duration tensor in the graph. It uses the `Ceil` node search of
   `--add-phoneme-alignment`, through `phoonnx.onnx_surgery.add_phoneme_alignment_output`.
2. It writes a changed copy of the model to `<model>.alignment.onnx`.
3. It opens an ONNX Runtime session on the copy, with the execution providers and thread
   count of the original session, and runs the request again on it.

The copy goes next to the model. When `PHOONNX_ORT_CACHE_DIR` is set, the copy goes in
that folder, which helps when the folder of the model is read-only:

```python
>>> import os
>>> from phoonnx.alignment import alignment_model_path
>>> _ = os.environ.pop("PHOONNX_ORT_CACHE_DIR", None)
>>> alignment_model_path("voices/model.onnx").as_posix()
'voices/model.alignment.onnx'
>>> os.environ["PHOONNX_ORT_CACHE_DIR"] = "cache"
>>> alignment_model_path("voices/model.onnx").as_posix()
'cache/model.alignment.onnx'
>>> del os.environ["PHOONNX_ORT_CACHE_DIR"]

```

A lock guards the change for each copy, and phoonnx moves the finished copy into place in
one step. Requests on a shared voice at the same time wait for the copy or use the copy
that is already there.

phoonnx tries this at most once for each `TTSVoice`. It keeps the result on the voice,
also a failure: no duration tensor, no `onnx` package, or a copy that it cannot write.
Later calls use the open session of the copy, or give `None` with no new attempt. Between
processes, the `<model>.alignment.onnx` file is the cache. phoonnx uses it when it is newer
than the model.

`include_alignments=False` runs none of this and costs nothing.

If the model has no duration predictor, as with ZipVoice, or `onnx` is not installed,
`phoneme_id_samples` and `phoneme_alignments` stay `None`. phoonnx writes one log line at
debug or info level that gives the reason.

## Use cases

### Visemes and lip-sync

Map each phoneme to a viseme index. Then schedule the face-rig changes at the sample
offset that you add up from `num_samples`:

```python
VISEME = {"p": 0, "b": 0, "m": 0, "f": 1, "v": 1}  # your mapping

offset = 0
for align in chunk.phoneme_alignments or []:
    t = offset / chunk.sample_rate
    viseme = VISEME.get(align.phoneme, -1)
    schedule_viseme(t, viseme)
    offset += align.num_samples
```

### Karaoke and subtitle highlights

Add up the sample offsets to get the start time of each word. The space phoneme (`' '`)
marks each word boundary. Use these times to move the text highlight with the audio.

---

[← Streaming](streaming.md) · [Home](index.md) · [Configuration reference →](configuration.md)
