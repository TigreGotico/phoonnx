# Voice sweep

<!-- agent-hint
covers: how to load and synthesise every index voice, check the audio and the ASR transcript, and compare two runs
source: scripts/voice_sweep.py
generated: no
verify: ! comm -23 <(sed '/<!-- agent-hint/,/-->/d;/^uv /d;/^VIRTUAL_ENV/d' docs/voice_sweep.md | grep -oP -- '(?<![\w-])--[a-z][a-z-]*' | sort -u) <(for c in run floor diff; do .venv/bin/python -I scripts/voice_sweep.py $c --help; done | grep -oP -- '(?<![\w-])--[a-z][a-z-]*' | sort -u) | grep . && .venv/bin/python -c "import re,sys; sys.path.insert(0,'scripts'); import voice_sweep as v; p=open('docs/voice_sweep.md').read(); rows=re.findall('^[|] '+chr(96)+'([a-z_]+)'+chr(96)+' [|] (.*) [|]$', p, re.M)[:6]; nums=[float(x) for r in rows for x in re.findall('[0-9]+[.][0-9]+', r[1])]; pct=[float(x)/100 for r in rows for x in re.findall('([0-9]+)%', r[1])]; ok=[r[0] for r in rows]==list(v.PROBLEMS) and nums==[v.SILENCE_RMS, v.MIN_SECONDS_PER_CHAR, v.DC_LIMIT, v.FULL_SCALE, v.NOISE_FLATNESS] and pct==[v.CLIP_FRACTION]; print(rows); sys.exit(0 if ok else 1)"
see: benchmarking.md, choose_voice.md
-->

This page shows how to check every voice in the bundled index with `scripts/voice_sweep.py`. It is for maintainers who change the index or the synthesis code.

The script loads every voice, makes it speak, and checks the audio. Use it to find index entries that do not load or do not speak, and to compare two commits.

A voice passes when it loads through the index path (`TTSModelInfo.load`) and `TTSVoice.synthesize` returns clean audio for every sentence. The tool checks each sentence's audio for these faults:

| Fault | Rule |
| --- | --- |
| `nonfinite` | a NaN or infinite sample |
| `silent` | RMS below 0.001 |
| `too_short` | less than 0.02 s of audio per letter or digit of the text |
| `dc_offset` | a mean sample value above 0.1 |
| `clipped` | more than 1% of the samples at full scale (0.99) |
| `noise` | a median spectral flatness above 0.3 in the frames within 20 dB of the loudest |

Each bound sits far from the values of working voices.

Where an ASR model is verified for the language, the tool also transcribes the audio and records WER and CER. A wrong token id or a wrong phonemizer gives fluent audio that says the wrong thing, and only the transcript shows it. Coloured noise and speech under noise also pass the audio checks. For a language without a verified ASR model, `ok` means that the audio has none of the faults above. It does not mean that the speech is intelligible.

## Install

The sweep runs in an environment that has phoonnx with the language extras of the voices you test, plus the ASR stack:

```bash
uv venv --python 3.11 .venv-sweep
VIRTUAL_ENV=.venv-sweep uv pip install --prerelease=allow -e ".[espeak,ar,en,es,pt,ru,he,omnivoice,pockettts,cloning]" \
    onnx-asr soundfile scipy pyarrow
```

A voice whose phonemizer is not installed fails to load, and the sweep records the error.

## Verify the ASR models first

An ASR model that does not know a language still returns fluent text, so its WER means nothing. `floor` scores each model on 15 clips of the FLEURS test split. It keeps the languages where the WER is at most 0.30, or the CER is at most 0.20 for Chinese, Japanese, Thai and other scripts without spaces:

```bash
python scripts/voice_sweep.py floor --out floor.json --reference-dir refs/
```

The file also stores two FLEURS sentences per verified language. The sweep adds them to that language's sentences, so the transcript is checked on real text in the language. With `--reference-dir`, `floor` also saves one FLEURS clip per language, with its text, for the voices that clone a speaker.

## Run

```bash
HF_HUB_CACHE=/data/sweep-cache python scripts/voice_sweep.py run --sentences sentences.tsv \
    --asr-floor floor.json --reference-dir refs/ --out run.jsonl --jobs 3 --cpus 0-11 --threads 4 \
    --evict --max-cache-gb 30
```

A separate `HF_HUB_CACHE` keeps the downloads of the sweep away from the cache that your installed voices use.

- `--sentences` is a TSV with the columns `lang`, `id`, `category`, `text` and `source`. Each voice gets the rows of its language whose `id` is in `--sentence-ids` (default `plain digits`). A row whose source starts with `script:` is a sentence for the script, not the language, so it gets no WER.
- `--reference-dir` holds `<lang>.wav` and `<lang>.txt`, or `default.wav`, for voices with `requires_reference`.
- Voices that share model files run together in one worker process, so the files download once. A group of more than `--group-size` voices (default 40) is split between the slots. With `--evict` a repo that the run downloaded leaves the Hugging Face cache when no pending voice downloads from it. A repo that was in the cache when the run started stays, under its `Phoonnx/` id or its earlier `OpenVoiceOS/` id. `--max-cache-gb` starts no further group while the cache holds more than that. `--min-free-gb` waits while the cache disk has less free space than that.
- `--timeout` is the limit per voice in seconds. A voice that runs out of time is recorded as `timeout`, with the sentences it spoke before. A worker that crashes or stops answering is replaced, and the voice it was on is recorded as `crash` or `timeout`. When a worker ends before it starts a voice, the run records the failure on its next voice. So each replacement worker uses up one voice, and the run always ends.
- The run checks its options before it starts a worker. It stops with exit status 2 for a missing or unreadable sentence file, a sentence file with no row for `--sentence-ids`, an unreadable `--asr-floor` file, or a `--voices` id that is not in the index.
- `--max-wer` (default 0.5) and `--max-cer` (default 0.35, for scripts without spaces) are the limits for a voice's mean score. An ASR model passes `floor` at a WER of up to 0.30 on human speech, so the voice limit sits above that.
- `--threads` caps the onnxruntime threads of every session. The default is 4. With no cap, onnxruntime starts one pinned thread per core, and `--cpus` has no effect.

The run writes `run.jsonl.synth` while it synthesises, then adds the ASR results and writes `run.jsonl`. Both files resume: a stopped run continues where it stopped. The last line states the voices in, the results out and the number of failing voices. The command exits 1 when the counts differ or when any voice fails. Every status except `ok`, `no_sentences` and `needs_reference` is a failure.

Each line of `run.jsonl` is one voice:

| Field | Meaning |
| --- | --- |
| `status` | `ok`, or `partial` when some sentences pass and others do not, or `unintelligible` when the WER or CER is above the limit. A failure is `load_failed`, `synth_failed`, `timeout`, `crash`, or a fault from the table above when no sentence passes. `needs_reference` and `no_sentences` mean that the run could not test the voice |
| `error` | what failed: the load or synthesis error, or each failing sentence with its faults |
| `resolved` | adapter, language code, phoneme type, alphabet and sample rate that the loaded voice uses |
| `snapshot` | the Hugging Face revision of the model file |
| `sentences` | per sentence: duration, peak, RMS, DC offset, clipped fraction, flatness, the list of faults, RTF, and transcript, WER and CER where scored |
| `wer`, `cer` | the mean over the scored sentences |

## Compare two runs

```bash
python scripts/voice_sweep.py diff base.jsonl head.jsonl --wer-margin 0.15
```

`diff` counts these as regressions: a voice whose status is worse in the second run (`ok`, then `partial`, then any failure), a voice that is missing from the second run, a voice that has a WER in the first run and none in the second although it spoke, and a WER that rose by more than the margin. A voice that the second run did not test (`no_sentences` or `needs_reference`) is listed but is not a regression. `diff` exits with status 1 when it finds a regression.

---

[Docs index](index.md) · Previous: [Benchmarking](benchmarking.md) · Next: [Training quickstart](training/quickstart.md)
