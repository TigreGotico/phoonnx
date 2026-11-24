# Voice Conversion

<!-- agent-hint
covers: how to convert the output of any voice to a target speaker with audio.voice_conversion, and how scripts/vc_gate.py checks the conversion
source: phoonnx/config.py:AudioPost.voice_conversion, phoonnx/voice.py:TTSVoice._load_voice_conversion, phoonnx/voice.py:TTSVoice._maybe_convert, scripts/vc_gate.py
generated: no
verify: .venv/bin/python -m doctest -o NORMALIZE_WHITESPACE docs/voice_conversion.md && .venv/bin/python -c "import re, sys; sys.path.insert(0, 'scripts'); import vc_gate as g; t = open('docs/voice_conversion.md').read(); sys.exit(0 if re.search(r'margin of at least %.2f' % g.SIM_MARGIN_GATE, t) and 'at most %.2f' % g.LEAKAGE_GATE in t and 'at most %g percentage points' % g.WER_DELTA_GATE in t and '%d sentences' % len(g.UTTERANCES) in t else 1)"
see: cloning.md, ovos_plugin.md
-->

This page shows how to convert the output of any voice to a target speaker, for
developers who use phoonnx from Python.

Voice conversion changes who the audio sounds like, after synthesis. It runs on the
waveform, so it works with every phoonnx engine, also with single-speaker models that
cannot clone. One model can thus speak in the voice of any reference clip.

[voiceclonnx](https://github.com/TigreGotico/voiceclonnx), a pure-ONNX library, does the
conversion. Install it with the extra:

```bash
pip install --pre "phoonnx[vc]"
```

Then pass a clip of the target speaker as `audio.voice_conversion`:

```python
from phoonnx.voice import TTSVoice
from phoonnx.config import AudioPost, SpeakerReference, SynthesisConfig

voice = TTSVoice.load("model.onnx", "model.json")
for chunk in voice.synthesize(
    "This sentence comes out in the voice of the target speaker.",
    SynthesisConfig(audio=AudioPost(voice_conversion=SpeakerReference(audio="target_speaker.wav"))),
):
    print(chunk.sample_rate, len(chunk.audio_float_array) > 0)
```

The loop prints one line for the sentence. The default OpenVoice engine gives its audio
at 22050 Hz, whatever the rate of the voice:

```text
22050 True
```

## Voice conversion or native cloning?

phoonnx has two ways to make a voice sound like another person, and they do different
things:

| | `reference` (native cloning) | `audio.voice_conversion` (voice conversion) |
|---|---|---|
| When it happens | during generation | after generation, on the waveform |
| What it changes | timbre and prosody, because the model is conditioned on the reference | timbre only: the rhythm and intonation of the source voice stay |
| Which engines | cloning engines only (YourTTS, StyleTTS2, ZipVoice, F5-TTS, Chatterbox and others) | all engines |
| Quality | higher, because nothing is made again from generated audio | a second pass over generated audio |
| Cost | the inference of the engine only | one more ONNX pass per sentence |

Use `reference` when the voice supports it. Use `audio.voice_conversion` when it does not,
such as for a Piper VITS, Kokoro or KittenTTS voice, or when you want one target speaker
on several engines.

You can use the two together, but native cloning already gives the target speaker, and a
conversion pass after it only adds loss.

## Configuration

The two fields are in `SynthesisConfig.audio`, an `AudioPost`:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `voice_conversion` | `SpeakerReference` \| `None` | `None` | the clip of the target speaker, in its `audio` field: a `str` path or URL, or `(audio, sample_rate)`. `None` turns the stage off |
| `voice_conversion_engine` | `str` | `None` | the voiceclonnx engine. `None` uses `"openvoice"` |

```python
>>> from phoonnx.config import SynthesisConfig
>>> config = SynthesisConfig()
>>> config.audio.voice_conversion, config.audio.voice_conversion_engine
(None, None)

```

With `voice_conversion=None`, phoonnx does not import `voiceclonnx`, and the audio is the
output of the engine, bit for bit.

Use a reference clip of 5 to 30 seconds of clean speech from one speaker. A path, a URL
and an `(audio, sample_rate)` tuple all work. These are the forms that `reference.audio`
takes, and the same loader reads them, so a URL has the same address and size limits. See
[Voice cloning](cloning.md#reference-audio-formats). phoonnx decodes the reference and
writes it to a temporary WAV file, which voiceclonnx reads.

phoonnx caches the loaded engine by the samples of the reference, not by the path or URL.
If you write a new clip to the same file, you get the new speaker. In exchange, each
`synthesize()` call reads the reference again, and downloads it again for a URL. Pass an
`(audio, sample_rate)` tuple to read the clip yourself.

### Choose an engine

`openvoice` is the default engine. voiceclonnx has others, such as `knnvc`, `focalcodec`,
`facodec` and `cosyvoice`. Each has its own sample rate, and the voiceclonnx README lists
them. Check another engine on your own material with `scripts/vc_gate.py` (see below)
before you use it.

```python
SynthesisConfig(audio=AudioPost(voice_conversion=SpeakerReference(audio="target.wav"),
                                voice_conversion_engine="knnvc"))
```

## Check the conversion

`scripts/vc_gate.py` converts 5 sentences from 3 source voices to 2 target speakers. It
scores the speaker similarity with `wespeaker-resnet34` and the word error rate with
Parakeet. It writes a run JSON and a table, and it exits with a non-zero status when a
gate fails:

```bash
python scripts/vc_gate.py --out vc-run.json
```

The gates are:

1. Each pair of source voice and target speaker moves toward the target: its mean
   similarity to the target is above the similarity with no conversion. Over the whole
   run, the similarity to the target is above that floor by a margin of at least 0.10.
2. The similarity of the converted audio to the source voice is at most 0.85 for each
   pair. Audio that passes through with no conversion scores near 1.0.
3. The mean word error rate rises by at most 5 percentage points.

`--vc-engine` selects another voiceclonnx engine for the run.

Prosody does not change. A flat source voice stays flat: the timbre of an expressive
speaker does not make it expressive.

## How it fits in the pipeline

phoonnx converts each chunk, one sentence at a time, not the whole utterance. Each
voiceclonnx engine keeps no state between calls and conditions only on the reference, so
the result is the same, and synthesis can still stream. A sentence shorter than 0.3 s has
too little content for the content encoders of the engines, so it passes through with no
conversion.

The converted audio has the sample rate of the conversion engine, 22050 Hz for OpenVoice,
and the `AudioChunk` reports that rate, not the rate of the voice. `synthesize_wav()` takes
the header from the chunk, so the file is correct.

When super-resolution is also on (`SynthesisConfig(audio=AudioPost(super_resolution=True))`, see
[Audio super-resolution](ovos_plugin.md#audio-super-resolution)), the conversion runs first,
so the upscaler works on the final timbre.

## Errors

Voice conversion fails with an error, unlike super-resolution. Audio in the source voice,
after you asked for another speaker, is a wrong answer, not a degraded one:

* `audio.voice_conversion` without `voiceclonnx` installed raises `ImportError`, which names the
  `phoonnx[vc]` extra.
* A reference path that is not a file raises `FileNotFoundError`:
  `audio.voice_conversion points at no readable file: 'missing.wav'`.
* A conversion failure during synthesis raises the error, and no source voice audio comes
  out.

## Models

voiceclonnx downloads its own weights from Hugging Face on first use
(`TigreGotico/voiceclonnx-openvoice-v2` for the default engine) and caches them. phoonnx
does not mirror or manage them.

---

[Docs index](index.md) · Previous: [Voice cloning](cloning.md) · Next: [Streaming](streaming.md)
