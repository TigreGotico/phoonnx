# Voice Conversion

<!-- agent-hint
covers: how to convert the output of any voice to a target speaker with audio.voice_conversion
source: phoonnx/config.py:AudioPost.voice_conversion, phoonnx/voice.py:TTSVoice._load_voice_conversion, scripts/vc_gate.py
generated: no
verify: python -I -c "from phoonnx.config import SynthesisConfig; print(SynthesisConfig().audio.voice_conversion_engine)" (prints None), plus the audio.voice_conversion example on the page
see: cloning.md, ovos_plugin.md
-->

This page shows how to convert the output of any voice to a target speaker, for developers who use phoonnx from Python.

Voice conversion changes **who** the audio sounds like, after synthesis. It runs on the
waveform, so it works with every phoonnx engine. This includes single-speaker models that
cannot clone. One model can thus speak in the voice of any reference clip.

```python
from phoonnx.voice import TTSVoice
from phoonnx.config import AudioPost, SpeakerReference, SynthesisConfig

voice = TTSVoice.load("model.onnx", "model.json")
for chunk in voice.synthesize(
    "This sentence comes out in the target speaker's voice.",
    SynthesisConfig(audio=AudioPost(voice_conversion=SpeakerReference(audio="target_speaker.wav"))),
):
    print(chunk.sample_rate, len(chunk.audio_float_array) > 0)
```

With a Portuguese unicode voice and a 6-second reference clip, the loop prints one line
for the sentence. The audio comes back at the 22 050 Hz rate of the OpenVoice engine:

```text
22050 True
```

[voiceclonnx](https://github.com/TigreGotico/voiceclonnx), a pure-ONNX library, does the
conversion. Install it with the extra:

```bash
pip install --pre "phoonnx[vc]"
```

## Voice conversion or native cloning?

phoonnx has two ways to make a voice sound like somebody else, and they are not
interchangeable.

| | `reference` (native cloning) | `audio.voice_conversion` (voice conversion) |
|---|---|---|
| When it happens | during generation | after generation, on the waveform |
| What it changes | timbre **and** prosody — the model is conditioned on the reference | timbre only — the source voice's rhythm and intonation stay |
| Which engines | cloning engines only (YourTTS, StyleTTS2, ZipVoice, F5-TTS, Chatterbox, …) | **all** of them |
| Quality | higher — nothing is re-synthesized from an already-lossy waveform | good, but it is a second pass over generated audio |
| Cost | none beyond the engine's own inference | one extra ONNX pass per sentence |

**Prefer `reference` when the voice supports it.** Reach for
`audio.voice_conversion` when it does not — a piper VITS voice, a kokoro voice, a kittentts
voice — or when you want one target speaker applied uniformly across several
different engines.

The two are independent and may be combined, though there is rarely a reason to:
native cloning already puts you at the target speaker, and a conversion pass on
top only adds loss.

## Configuration

| Field | Type | Default | Meaning |
|---|---|---|---|
| `audio.voice_conversion` | `SpeakerReference` | `None` | target speaker; its `audio` is a `str` path or URL, or `(audio, sample_rate)`; `None` disables the stage entirely |
| `audio.voice_conversion_engine` | `str` | `None` | which voiceclonnx engine to use; `None` uses `"openvoice"` |

`None` is a strict no-op: `voiceclonnx` is never imported and the audio is
bit-for-bit the engine's own output.

The reference clip should be 5–30 seconds of clean speech from one speaker. Paths,
URLs and in-memory `(audio, sample_rate)` tuples are all accepted — the same shapes
`reference.audio` takes, loaded by the same guarded loader, so remote clips are
subject to its address and size limits. Whatever you pass is decoded and written to
a temporary WAV, which is what voiceclonnx reads.

The loaded engine is cached against the reference's **samples**, not against the
path or URL you named, so rewriting a reference file in place gives you the changed
speaker rather than the previous one. The cost is that the reference is read —
and, for a URL, fetched — once per `synthesize()` call. Pass an
`(audio, sample_rate)` tuple if you want to do that reading yourself.

### Choosing an engine

`openvoice` is the default because it is the only engine tested here that transfers
the voice **and** keeps the words intact. `knnvc` transfers harder but costs about
9 percentage points of word error rate — pick it only if you care more about timbre
than about being understood, and measure it on your own material first. The full
engine list is in the voiceclonnx README.

```python
SynthesisConfig(audio=AudioPost(
    voice_conversion=SpeakerReference(audio="target.wav"),
    voice_conversion_engine="knnvc",
))
```

## What to expect

Measured on 5 utterances × 3 source voices × 2 target speakers, with
`wespeaker-resnet34` for similarity and parakeet for word error rate:

* Speaker similarity to the target rises **+0.24 cosine** on average over the
  unconverted floor, on every source/target pair.
* Word error rate rises **+2.5 pp** on average — conversion is close to
  intelligibility-neutral but not free.
* Prosody does not move. A flat source voice stays flat; converting it to an
  expressive speaker's timbre will not make it expressive.

Reproduce it all with:

```bash
python scripts/vc_gate.py --out vc-run.json
```

The script exits non-zero when a gate fails.

## How it fits in the pipeline

Conversion is applied **per chunk** — one sentence at a time — not to the whole
utterance. Every voiceclonnx engine is stateless across calls and conditions only
on the reference, so per-sentence conversion gives the same result as converting
the whole utterance while keeping synthesis streaming. Sentences under roughly
0.3 s carry too little content for the engines' content encoders and pass through
unconverted rather than becoming artefacts.

The converted audio comes back at the VC engine's own sample rate (22 050 Hz for
OpenVoice), and the `AudioChunk` reports that rate — not the voice's native one.
`synthesize_wav()` takes the header from the chunk, so files come out correct
without any extra work on your side.

When super-resolution (`SynthesisConfig(audio=AudioPost(super_resolution=True))`, see
[Audio super-resolution](ovos_plugin.md#audio-super-resolution)) is also on, conversion runs **first**, so
the upscaler works on the final timbre.

## Errors

Voice conversion fails loudly, unlike super-resolution. Returning the source
speaker after you asked for a different one is a wrong answer, not a degraded one,
so:

* `audio.voice_conversion` set without `voiceclonnx` installed raises `ImportError` naming
  the `phoonnx[vc]` extra.
* A reference path that is not a readable file raises `FileNotFoundError`:
  `audio.voice_conversion points at no readable file: 'missing.wav'`.
* A conversion failure mid-stream propagates instead of yielding the source voice.

## Models

voiceclonnx downloads its own weights from Hugging Face on first use
(`TigreGotico/voiceclonnx-openvoice-v2` for the default engine) and caches them.
phoonnx does not mirror or manage them. The voiceclonnx project owns the one mirror.

---

[Docs index](index.md) · Previous: [Voice cloning](cloning.md) · Next: [Streaming](streaming.md)
