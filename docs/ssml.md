# SSML

<!-- agent-hint
covers: what each SSML 1.1, Polly and Alexa element and attribute does in phoonnx, its status, the time rules, the bounds and the cost
source: phoonnx/ssml.py:parse_ssml, phoonnx/voice.py:TTSVoice.synthesize_ssml, phoonnx/audio_fx.py, tests/test_ssml_audit.py:MATRIX, tests/test_ssml_edges.py
generated: no
verify: .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_ssml_audit.py
see: usage.md, alignment.md
-->

This page lists what phoonnx does with each SSML element and attribute, with the time rules,
the limits and the cost. Read it when you write SSML for a phoonnx voice.

`TTSVoice.synthesize_ssml()` and `synthesize_ssml_wav()` read SSML 1.1 and the Amazon Polly
and Alexa extensions. `phoonnx.ssml.parse_ssml` reads the markup, and
`TTSVoice.synthesize_ssml` speaks the result. [Usage](usage.md#ssml) has a short example.

## Status

| Status | Meaning |
| --- | --- |
| supported | the markup changes what is heard, as the standard says |
| extension | a form that SSML 1.1 does not define, such as an SSML 1.0 or Polly form. The markup changes what is heard, as the row says |
| degraded | the element is read without its effect: its words are kept, and no tag is read aloud |
| unsupported | the attribute is ignored on purpose (see the end of this page) |

## Support table

Each row says what `parse_ssml` gives for the element: the text said, then the silence,
pace, pitch, effects, marks and sounds that go with it. "words" is the text inside the
element, such as `Hello world` or `Bonjour le monde`. The text frontend of the voice reads
`say-as`, so a `say-as` row says what `parse_ssml` gives to the frontend.
`tests/test_ssml_audit.py` checks each row in English and in French, and it fails when a row
or a status on this page differs from its table.

| Element or attribute | Status | What the reader does |
| --- | --- | --- |
| speak | supported | words kept |
| speak xml:lang | supported | words kept, language de-DE |
| speak with the SSML namespace declared | supported | says "words then x", pause 1000 ms |
| a prefix declared as the SSML namespace | supported | says "words then x", pause 1000 ms |
| an undeclared ssml prefix | supported | says "words then x", pause 1000 ms |
| lang xml:lang | supported | says "words then Haus", language de-DE |
| xml:lang on p, s and w | supported | says "hola then ciao", language es-ES, pause 750 ms, language it-IT |
| p | supported | says "words then words", pause 750 ms |
| s | supported | says "words then words", pause 150 ms |
| w, token | supported | says "words x" |
| w role | degraded | words kept |
| say-as interpret-as=cardinal | supported | the text frontend reads "1234" in the language in force |
| say-as interpret-as=number | supported | the text frontend reads "12" in the language in force |
| say-as interpret-as=ordinal | supported | the text frontend reads "3" in the language in force |
| say-as interpret-as=digits | supported | the text frontend reads "123" in the language in force |
| say-as interpret-as=fraction | supported | the text frontend reads "3/4" in the language in force |
| say-as interpret-as=unit | supported | the text frontend reads "5km" in the language in force |
| say-as interpret-as=date | supported | the text frontend reads "2026-11-09" in the language in force |
| say-as interpret-as=date format="dmy" | supported | the text frontend reads "1/2/2026" with format dmy, in the language in force |
| say-as interpret-as=time | supported | the text frontend reads "14:30" in the language in force |
| say-as interpret-as=time format="hms12" | supported | the text frontend reads "7:30 pm" with format hms12, in the language in force |
| say-as interpret-as=time detail="1" | degraded | the text frontend reads "9 pm" with detail 1, in the language in force |
| say-as interpret-as=date detail="written" | degraded | the text frontend reads "9 novembre 2026" with detail written, in the language in force |
| say-as interpret-as=currency | supported | the text frontend reads "$12.50" in the language in force |
| say-as interpret-as=address | supported | the text frontend reads "12 Rue de la Paix" in the language in force |
| say-as interpret-as=telephone | supported | the text frontend reads "555-0100" in the language in force |
| say-as interpret-as=characters | supported | the text frontend reads "XK7" in the language in force |
| say-as interpret-as=spell-out | supported | the text frontend reads "XK7" in the language in force |
| say-as interpret-as=verbatim | supported | the text frontend reads "XK7" in the language in force |
| say-as interpret-as=interjection | supported | the text frontend reads "wow" in the language in force |
| say-as interpret-as=an unknown type | degraded | the text frontend reads "abc" in the language in force |
| say-as detail on a type other than cardinal | degraded | the text frontend gets "12" with detail 2 |
| say-as expletive | supported | says "words then (a sound) then words", beep 350 ms |
| phoneme ipa | supported | says "[[ bɔ̃ʒuʁ ]]" |
| phoneme x-sampa | supported | says "[[ həˈləʊ ]]" |
| phoneme another alphabet | degraded | says "你" |
| phoneme without ph | degraded | words kept |
| sub alias | supported | says "World Wide Web" |
| sub with an empty alias | supported | says "a b" |
| lookup | degraded | words kept |
| lexicon | unsupported | words kept |
| break time | supported | says "a then b then c", pause 1500 ms, pause 200 ms |
| break strength | supported | says "a then b then c", pause 750 ms, pause 100 ms |
| break strength none | supported | says "a then b" |
| break without attributes | supported | says "a then b", pause 500 ms |
| break time without a unit | degraded | says "a then b", pause 500 ms |
| break over ten seconds | degraded | says "a then b", pause 10000 ms |
| emphasis level | supported | words kept, rate x0.9, volume x1.15 |
| emphasis level reduced | supported | words kept, rate x1.05, volume x0.85 |
| prosody rate keyword | supported | words kept, rate x0.8 |
| prosody rate percent | supported | words kept, rate x1.5 |
| prosody rate relative | extension | words kept, rate x0.8 |
| prosody rate number | extension | words kept, rate x1.5 |
| prosody pitch semitones | supported | words kept, pitch +3.0 st |
| prosody pitch percent | supported | words kept, pitch +12.0 st |
| prosody pitch keyword | supported | words kept, pitch +3.0 st |
| prosody pitch in Hz | unsupported | words kept |
| prosody contour | unsupported | words kept |
| prosody range keyword | supported | words kept, pitch range x2.0 on an [engine with a range control](#pitch-made-by-the-engine); no effect on the others |
| prosody range percent | supported | words kept, pitch range x0.5 on an [engine with a range control](#pitch-made-by-the-engine); no effect on the others |
| prosody range in semitones | unsupported | words kept |
| prosody volume keyword | supported | words kept, volume x0.65 |
| prosody volume dB | supported | words kept, volume x0.50 |
| prosody volume number | extension | words kept, volume x0.50 |
| prosody duration | supported | words kept, duration 3000 ms |
| prosody amazon:max-duration | extension | words kept, max duration 2000 ms |
| nested prosody | supported | words kept, rate x1.5625, pitch +2.0 st |
| voice languages | supported | words kept, voice language de-DE (the accent dropped), the language of the words stays as it was |
| voice name, gender, age, variant, required, ordering | degraded | words kept |
| audio src with fallback text | degraded | says "words after" |
| audio src without fallback | degraded | words kept |
| mark | supported | says "words then x", mark m |
| desc | degraded | words kept |
| metadata | degraded | words kept |
| meta | degraded | words kept |
| seq, par, media | degraded | says "words then one then two", pause 150 ms |
| amazon:effect name=whispered | extension | words kept, effects whisper |
| amazon:effect name=drc | extension | words kept, effects drc |
| amazon:effect phonation=soft | extension | words kept, effects soft |
| amazon:effect name=phonation value=soft | extension | words kept, effects soft |
| amazon:effect vocal-tract-length | extension | words kept, pitch +3.9 st |
| amazon:auto-breaths | extension | words kept, breath every 2 sentences (250 ms) |
| amazon:breath | extension | says "words then (a sound) then x", breath 500 ms |
| amazon:domain | degraded | words kept |
| amazon:emotion | degraded | words kept |
| alexa:name | degraded | words kept |
| amazon prefix declared by the document | extension | words kept, effects whisper |
| unknown element | degraded | says "words end" |

## Pitch made by the engine

`<prosody pitch>` is a shift of the finished audio (a resample and a time stretch), which moves
the formants with the pitch, as a smaller or larger speaker would sound. An engine with pitch
inputs of its own makes the shift itself instead, and keeps the formants where the voice has
them. `<prosody range>` (how far the pitch moves around its mean) has no audio form, so it
reaches only these engines:

| Engine | `pitch` | `range` |
| --- | --- | --- |
| tiny-voca | `pitch_shift`, ln(2)/12 for each semitone | `pitch_range` |
| Mixer-TTS, and FastPitch graphs with `pitch_mul` and `pitch_add` | `pitch_mul` and `pitch_add`, when the voice states the pitch mean and standard deviation in Hz it was trained with (`pitch_mean` and `pitch_std` in `engine_params`, or in the graph's metadata, which phoonnx's Mixer-TTS export writes; a voice z-scored per speaker gives a list of each, by speaker id). Without them, the audio is shifted | `pitch_mul`, on a single-speaker voice. A multi-speaker voice gets none: measured on the Arabic voices, the range moved each speaker's mean pitch and widened it unevenly |
| any other engine | the finished audio is shifted | no effect |

The engine's keys build on the voice's own and the call's `extra`: a voice that sets
`pitch_shift` is shifted from there. The part of the pitch that `vocal-tract-length` asks for
always stays on the audio, because it is meant to move the formants. Measured on real models:
a +4 st and a -4 st shift land within 0.4 st on tiny-voca and Mixer-TTS, and the spectral
envelope moves by 0 to 5 % against 19 to 24 % for the audio shift. A wider range works (x-high
widens the pitch movement 1.6 to 1.7 times); a narrower one works only in part (x-low narrows it
to 0.86 to 0.92 of the voice's), because part of the movement comes from the decoder, not the
predicted pitch. OptiSpeech's `p_factor` is not used: on the voice measured it moved the mean
pitch with the range (+3.1 st at x-high, against 0.04 st on Mixer-TTS).

## How a document is read

* phoonnx reads the input inside a `<speak>` of its own. Text after the closing tag, two
  roots, an `<?xml ...?>` line, comments and a `<!DOCTYPE>` all work. The doctype is dropped,
  and its entities are not expanded.
* Text that a person or a language model writes and XML refuses is repaired: `AT&T`,
  `1 < 2`, `&nbsp;` and the other HTML entities, and control characters. The markup around
  them still counts.
* The prefixes `amazon:`, `alexa:` and `ssml:` work when the document declares them and when
  it does not. `amazon:max-duration` is an attribute like any other. Element names are not
  case sensitive.
* A document that still does not parse is spoken as its text, with the tags taken out. A
  CDATA section keeps its content, and a `<` before a space is a word. The reader never
  raises.
* An unknown element is read as its words. `desc`, `metadata`, `meta` and `lexicon` describe
  the document, so they are not read.
* `xml:lang` on any element sets the language of its content, and of a `say-as` in it. The
  words are then spoken in that language by a multilingual voice that has it in its
  `lang_id_map`, and by an engine that is told the language tag itself: every SuperTonic voice
  speaks the 31 languages of its model, whatever the language in its name. Any other voice, and
  a language the model lacks, keeps the voice's own language.
* `say-as` goes to `TextFrontend.say_as` of the voice with its written value,
  `interpret-as`, `format`, `detail` and the language in force. phoonnx holds no reading
  rule. text2spoken reads the value.

## Engines without pace control

`<prosody rate>` and `<emphasis>` change the length scale of the call, and most engines read it
or a speed key of their own. The codec language models make speech token by token until the
model ends it, so a length scale cannot reach them: ArkTTS, Chatterbox, Indic Parler-TTS,
Llasa, Magpie, MOSS-TTS-Nano, NeuTTS, Orpheus, OuteTTS, Pocket TTS, Qwen3-TTS and Spark-TTS.
Shami has its speed fixed inside the exported graph. Their adapters set `PACE_CONTROL = False`.
Indic Parler-TTS can also be asked for a pace in its description text, such as "a moderate
speed"; that is separate from SSML.

On these engines a rate is dropped, unless `SynthesisConfig.audio.stretch_pace` is on. Then the
finished audio of each sentence is time-stretched (WSOLA, the pitch kept) by the call's length
scale over the voice's own, within 0.5x and 2x. A stretch changes the recording, not the
speech: the pauses and breaths get longer or shorter with the words. A speech recogniser reads
stretched Piper speech as well as Piper's own slower and faster speech, from 0.5x to 2x.
`TTSVoice.stream` yields a stretched sentence as one chunk. `<prosody duration>` fits the
stretched audio, as it fits any other.

## Time

* A `<break>` adds to the pause that is already pending. The edge of a `<p>` or `<s>` only
  makes sure that the pause is at least that long. The end of one paragraph and the start of
  the next therefore give one pause.
* A silence is exactly `sample_rate * ms // 1000` samples. It is yielded before the speech
  after it is synthesized, so a listener does not wait for the model to hear the pause.
* `on_mark(name, seconds)` is called in document order. `seconds` counts the samples from
  the start. A `<mark>` before a `<break>` is called before the silence is yielded, and a
  mark after it is called after the silence. A mark at the end of the document is called
  last, also after a trailing `<break>`.
* `<prosody duration>` and `amazon:max-duration` stretch or squeeze the speech with WSOLA
  (`audio_fx.time_stretch`) to the time asked for, within a factor of 0.5 to 2. The pauses
  and sounds in the element do not count in that time. The pieces share the rest of the time
  by their length. The innermost `duration` wins, and `max-duration` only shortens the
  speech. With `include_alignments=True`, the phoneme samples and alignments stretch with
  the speech, and their sum stays equal to its length.
* Speech that starts or stops on a loud sample is faded over 4 ms (`audio_fx.declick`). A
  quiet edge stays as it is.
* `seed=` on `synthesize_ssml` makes whispers and breaths repeat: the same markup and seed
  give the same samples.

## Bounds

| What | Limit |
| --- | --- |
| a `<break>` | 10 s |
| silence and sounds (breaks, breaths, bleeps) in one document | 120 s together. What does not fit is dropped, but a `<mark>` event is never dropped |
| `<prosody duration>` | 600 s. The stretch itself stays within 0.5x and 2x |
| rate | 0.5x to 2x |
| pitch | -12 to +12 semitones |
| range | 0 to 2x the voice's own |
| volume | 0 to 2x. A dB value counts up to 96 dB in either direction |
| nesting | 100 levels. Deeper elements are read as text |

The reading time grows in proportion to the size of the document. Hostile input, such as a
hundred thousand `<break>` elements, marks, unclosed tags or open CDATA sections, cannot make
the reader slow down more than that.

## Cost

The markup elements that change the text or the model settings cost nothing after the model:
`<prosody rate>`, `volume`, `<emphasis>`, `<break>`, `<mark>`, `say-as`, `sub` and `phoneme`.
With `audio.stretch_pace` on, `<prosody rate>` and `<emphasis>` on an engine without pace
control run a WSOLA time stretch (below).
The effects below run on the finished audio with numpy. Their cost grows in proportion to the
length of the audio, and it does not include the model:

| Feature | What it runs |
| --- | --- |
| `<prosody pitch>` (unless the [engine makes it](#pitch-made-by-the-engine)), `vocal-tract-length` | a pitch shift (resample and time stretch) |
| `<prosody duration>`, `amazon:max-duration` | a WSOLA time stretch |
| `<prosody rate>`, `<emphasis>` with `stretch_pace`, on an [engine without pace control](#engines-without-pace-control) | a WSOLA time stretch, 15 to 30 ms per 2 s of 24 kHz audio |
| `<amazon:effect name="whispered">`, `phonation="soft"` | shaped breath noise from a short-time Fourier transform |
| `<amazon:effect name="drc">` | a dynamic range compressor, under 1 ms per second of audio |
| `<amazon:auto-breaths>`, `<amazon:breath>` | a short generated breath |
| edge fade | under 0.1 ms per second of audio |

On one laptop CPU core, at 22.05 kHz, each effect costs less than 3 percent of the duration of
the audio it changes.

## Unsupported on purpose

* `<prosody pitch="150Hz">`, `contour`, and a `range` in semitones or Hz: a VITS voice has no
  pitch contour to steer, and an absolute frequency needs the pitch of the speaker, which
  phoonnx does not measure. The semitone and percent forms of `pitch` and the keyword and
  percent forms of `range` work, as [Pitch made by the engine](#pitch-made-by-the-engine) says.
* `<audio src>`: nothing is fetched or played. The fallback text inside it is read.
* `<lexicon>` and `<lookup>`: pronunciation lexicons are not loaded. Use `<phoneme>` or
  `<sub>`.
* `<voice name>`, `gender`, `age` and `variant`: a voice object is one voice. `languages`
  picks the language id of a multilingual voice from its first language, and drops an
  `:accent`. It does not change the language of the words: `say-as` reads in the `xml:lang`
  in force, as SSML 1.1 section 3.2.1 asks.
* `<par>` and `<media>` timing (`begin`, `end`, `repeatCount`, `soundLevel`, fades): the
  children are read one after the other.
* `<w role>`, `<amazon:domain>`, `<amazon:emotion>` and `<alexa:name>`: this engine has no
  styles and no parts of speech. The words are read.
* `<say-as detail>` on any type other than `cardinal`: the reader gives it to the text
  frontend, and text2spoken reads the text as if no detail were given (SSML 1.1 section
  3.1.9).
* Alphabets other than IPA and X-SAMPA for `<phoneme>`.
* `<prosody volume="silent">` skips its words. They are not synthesized, so they take no
  time.

---

[← Usage guide](usage.md) · [Home](index.md) · [Voice cloning →](cloning.md)
