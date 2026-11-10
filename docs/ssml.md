# SSML

`TTSVoice.synthesize_ssml()` and `synthesize_ssml_wav()` read SSML 1.1 and the Amazon Polly and
Alexa extensions (`phoonnx.ssml.parse_ssml`, then `phoonnx.voice.TTSVoice.synthesize_ssml`). This
page is what the reader was measured to do, in English and in French, by running it
(`tests/test_ssml_audit.py` is the table below, one row per line, and fails if the two disagree).
The short version, with examples, is in [usage](usage.md#ssml).

## Status

| Status | Meaning |
| --- | --- |
| supported | the markup changes what is heard, as the standard says |
| degraded | the element is read without its effect: its words are kept, nothing is read as a tag |
| unsupported | the attribute is ignored on purpose (see the end of this page) |

## Support table

"words" is the text inside the element (`Hello world`, `Bonjour le monde`). A row says what comes out
of `parse_ssml`: the text said, then the silence, pace, pitch, effects, marks and sounds that go with it.

| Element or attribute | Status | What the reader does |
| --- | --- | --- |
| speak | supported | words kept |
| speak xml:lang | supported | words kept; language de-DE |
| speak with the SSML namespace declared | supported | says "words then x"; pause 1000 ms |
| a prefix declared as the SSML namespace | supported | says "words then x"; pause 1000 ms |
| lang xml:lang | supported | says "words then Haus"; language de-DE |
| xml:lang on p, s and w | supported | says "hola then ciao"; language es-ES; pause 750 ms; language it-IT |
| p | supported | says "words then words"; pause 750 ms |
| s | supported | says "words then words"; pause 150 ms |
| w, token | supported | says "words x" |
| w role | degraded | words kept |
| say-as interpret-as=cardinal | supported | says "one thousand, two hundred and thirty four" (en), "mille deux cent trente-quatre" (fr) |
| say-as interpret-as=number | supported | says "twelve" (en), "douze" (fr) |
| say-as interpret-as=ordinal | supported | says "third" (en), "troisième" (fr) |
| say-as interpret-as=digits | supported | says "1 2 3" |
| say-as interpret-as=fraction | supported | says "three quarters" (en), "trois quarts" (fr) |
| say-as interpret-as=unit | supported | says "five kilometers" (en), "cinq kilomètres" (fr) |
| say-as interpret-as=date | supported | says "monday, november ninth, twenty twenty six" (en), "lundi neuf novembre deux-mille-vingt-six" (fr) |
| say-as interpret-as=date format="dmy" | supported | says "sunday, february first, twenty twenty six" (en), "dimanche premier février deux-mille-vingt-six" (fr) |
| say-as interpret-as=time | supported | says "half past two p.m." (en), "quatorze heures trente" (fr) |
| say-as interpret-as=currency | supported | says "twelve dollars and fifty cents" (en), "douze dollars et cinquante cents" (fr) |
| say-as interpret-as=address | supported | says "twelve Rue de la Paix" (en), "douze Rue de la Paix" (fr) |
| say-as interpret-as=telephone | supported | says "5 5 5 0 1 0 0" |
| say-as interpret-as=characters | supported | says "X K 7" |
| say-as interpret-as=spell-out | supported | says "X K 7" |
| say-as interpret-as=verbatim | supported | says "X K 7" |
| say-as interpret-as=interjection | supported | says "wow" |
| say-as interpret-as=an unknown type | degraded | says "abc" |
| say-as detail | degraded | says "1 2" |
| say-as expletive | supported | says "words then (a sound) then words"; beep 350 ms |
| phoneme ipa | supported | says "[[ bɔ̃ʒuʁ ]]" |
| phoneme x-sampa | supported | says "[[ həˈləʊ ]]" |
| phoneme another alphabet | degraded | says "你" |
| phoneme without ph | degraded | words kept |
| sub alias | supported | says "World Wide Web" |
| sub with an empty alias | supported | says "a b" |
| lookup | degraded | words kept |
| lexicon | unsupported | words kept |
| break time | supported | says "a then b then c"; pause 1500 ms; pause 200 ms |
| break strength | supported | says "a then b then c"; pause 750 ms; pause 100 ms |
| break strength none | supported | says "a then b" |
| break without attributes | supported | says "a then b"; pause 500 ms |
| break time without a unit | degraded | says "a then b"; pause 500 ms |
| break over ten seconds | degraded | says "a then b"; pause 10000 ms |
| emphasis level | supported | words kept; rate x0.9; volume x1.15 |
| emphasis level reduced | supported | words kept; rate x1.05; volume x0.85 |
| prosody rate keyword | supported | words kept; rate x0.8 |
| prosody rate percent | supported | words kept; rate x1.5 |
| prosody rate relative | supported | words kept; rate x0.8 |
| prosody rate number | supported | words kept; rate x1.5 |
| prosody pitch semitones | supported | words kept; pitch +3.0 st |
| prosody pitch percent | supported | words kept; pitch +12.0 st |
| prosody pitch keyword | supported | words kept; pitch +3.0 st |
| prosody pitch in Hz | unsupported | words kept |
| prosody contour | unsupported | words kept |
| prosody range | unsupported | words kept |
| prosody volume keyword | supported | words kept; volume x0.65 |
| prosody volume dB | supported | words kept; volume x0.50 |
| prosody volume number | supported | words kept; volume x0.50 |
| prosody duration | supported | words kept; duration 3000 ms |
| prosody amazon:max-duration | supported | words kept; max duration 2000 ms |
| nested prosody | supported | words kept; rate x1.5625; pitch +2.0 st |
| voice languages | supported | words kept; language de-DE |
| voice name, gender, age, variant, required, ordering | degraded | words kept |
| audio src with fallback text | degraded | says "words after" |
| audio src without fallback | degraded | words kept |
| mark | supported | says "words then x"; mark m |
| desc | degraded | words kept |
| metadata | degraded | words kept |
| meta | degraded | words kept |
| seq, par, media | degraded | says "words then one then two"; pause 150 ms |
| amazon:effect name=whispered | supported | words kept; effects whisper |
| amazon:effect name=drc | supported | words kept; effects drc |
| amazon:effect phonation=soft | supported | words kept; effects soft |
| amazon:effect name=phonation value=soft | supported | words kept; effects soft |
| amazon:effect vocal-tract-length | supported | words kept; pitch +3.9 st |
| amazon:auto-breaths | supported | words kept; breath every 2 sentences (250 ms) |
| amazon:breath | supported | says "words then (a sound) then x"; breath 500 ms |
| amazon:domain | degraded | words kept |
| amazon:emotion | degraded | words kept |
| alexa:name | degraded | words kept |
| amazon prefix declared by the document | supported | words kept; effects whisper |
| unknown element | degraded | says "words end" |

## How a document is read

* Whatever is given is read inside a `<speak>` of its own, so text after the closing tag, two roots, an
  `<?xml ...?>` line, comments and a `<!DOCTYPE>` (dropped, never expanded) all work.
* What a person or a language model writes and XML refuses is repaired, not punished: `AT&T`, `1 < 2`,
  `&nbsp;` and the other HTML entities, control characters. The markup around them still counts.
* Prefixes (`amazon:`, `alexa:`, `ssml:`) work whether the document declares them or not, and
  `amazon:max-duration` is an attribute like any other. Element names are not case sensitive.
* A document that still does not parse is spoken as its text with the tags taken out (a CDATA
  section keeps its content, a `<` before a space is a word); it never raises.
* An unknown element is its words. `desc`, `metadata`, `meta` and `lexicon` are about the document, not
  for the listener, and are not read.
* `xml:lang` on any element sets the language of what is inside it (and of `say-as` in it).

## Time

* A `<break>` adds to the pause already pending; the edge of a `<p>` or `<s>` only makes sure there is at
  least that much, so the end of one paragraph and the start of the next pause once.
* Silences are exactly `sample_rate * ms // 1000` samples and are yielded before the speech that follows
  is synthesized, so a listener does not wait for the model to hear the pause they asked for.
* `on_mark(name, seconds)` is called in document order, `seconds` counted in samples from the start. A
  `<mark>` written before a `<break>` is called before the silence is yielded, one written after it after.
  A mark with nothing after it (even after a trailing `<break>`) is still called, at the end.
* `<prosody duration>` and `amazon:max-duration` stretch or squeeze the speech with WSOLA
  (`audio_fx.time_stretch`) to the time asked for, within a factor of 0.5 to 2; the pauses and sounds inside
  the element are taken out of the time, and the rest is shared between the pieces by length. The innermost
  `duration` wins. `max-duration` only ever shortens.
* Speech that starts or stops on a loud sample (an effect cut it, or the model did) is faded over 4 ms
  (`audio_fx.declick`); a quiet edge, which is what a voice gives, is left as it is.
* `seed=` on `synthesize_ssml` makes whispers and breaths repeat: the same markup and seed give the same samples.

## Bounds

| What | Limit |
| --- | --- |
| a `<break>` | 10 s |
| silence and sounds (breaks, breaths, bleeps) in one document | 120 s together; what does not fit is dropped, `<mark>` events never are |
| `<prosody duration>` | 600 s; the stretch itself stays within 0.5x and 2x |
| rate | 0.5x to 2x; pitch +-12 semitones; volume 0 to 2x (`+-96 dB` is the most a dB value can ask) |
| nesting | 100 levels; deeper is read as text |
| `say-as` | 15 digits or 200 characters; beyond that it is read digit by digit or as written |

A hundred thousand `<break>`s, marks, `<amazon:breath/>`s, unclosed tags, open CDATA sections or
doctypes are each read in under a second.

## Cost

Measured on one core of a laptop CPU, 22.05 kHz, numpy only, per second of audio produced
(`FakeVoice` in `tests/test_ssml_edges.py`, so the model itself is not in it). Every feature is linear in
the length of the audio (5 s, 20 s and 80 s cost the same per second); at 48 kHz the filters cost about twice that.

| Feature | ms per second of audio |
| --- | --- |
| `<prosody rate>`, `volume`, `<emphasis>`, `<break>`, `<mark>`, `say-as`, `sub`, `phoneme` | none (done before or by the model) |
| `<prosody pitch>`, `vocal-tract-length` | 7 to 8 |
| `<prosody duration>` / `amazon:max-duration` | 5 to 10 |
| `<amazon:effect name="whispered">` | 4 |
| `<amazon:effect phonation="soft">` | 4 |
| `<amazon:effect name="drc">` | under 1 |
| `<amazon:auto-breaths>`, `<amazon:breath>` | about 1 |
| edge fade | under 0.1 |
| pitch, duration and whisper together | 23 |

Reading the markup takes about 20 ms for 40 kB (300 paragraphs with breaks, emphasis and `say-as`).

## Unsupported on purpose

* `<prosody pitch="150Hz">` and `contour`, `range`: a VITS voice has no pitch contour to steer; the
  semitone and percent forms are done on the finished audio, an absolute frequency would need the
  speaker's own pitch measured first.
* `<audio src>`: nothing is fetched or played; the fallback text inside it is read.
* `<lexicon>` / `<lookup>`: pronunciation lexicons are not loaded; use `<phoneme>` or `<sub>`.
* `<voice name>`, `gender`, `age`, `variant`: a voice object is one voice. `languages` picks the language
  id of a multilingual voice.
* `<par>` and `<media>` timing (`begin`, `end`, `repeatCount`, `soundLevel`, fades): read one after the other.
* `<w role>`, `<amazon:domain>`, `<amazon:emotion>`, `<alexa:name>`: styles and parts of speech are not
  something this engine does; the words are read.
* `<say-as detail>` and alphabets other than IPA and X-SAMPA for `<phoneme>`.
* `<prosody volume="silent">` skips its words (they are not synthesized silently, so they take no time).
