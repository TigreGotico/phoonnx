# Datasets

This page is for anyone preparing training data. After reading it you will know every input
format phoonnx accepts, the exact LJSpeech `metadata.csv` layout, how audio files are located
and normalized, how speakers are assigned, and how to filter out bad clips before training. It
backs the [training quickstart](quickstart.md).

## Input formats

`preprocess.py` reads four on-disk shapes, selected with `--dataset-format` (default `auto`,
which detects per source):

| Format | Auto-detected from | Source is |
|---|---|---|
| `ljspeech` | a directory containing `metadata.csv` | an LJSpeech directory (`metadata.csv` + `wav(s)/`) |
| `jsonl` | a `.jsonl` file | one JSON object per line |
| `parquet` | a `.parquet` file, a shard glob, or a directory of shards | Parquet table(s) |
| `hf` | an `org/name` string that is not an existing path | a Hugging Face dataset repo id (loaded via `datasets`) |

LJSpeech remains the simplest path and is what the [quickstart](quickstart.md) uses. The
tabular formats (`jsonl`/`parquet`/`hf`) resolve columns by name; see
[Column mapping](#column-mapping-tabular-formats).

### Merging multiple sources

`-i/--input-dir` is repeatable (and accepts comma-separated values), so several datasets can be
merged in one run. Per-source speaker IDs are **namespaced** to avoid collisions between
datasets that happen to share speaker labels:

```bash
python -m phoonnx_train.preprocess \
  -i dataset_a -i dataset_b,my-org/hf-dataset \
  --output-dir train_out --language en-US
```

## LJSpeech directory layout

The default format uses the LJSpeech convention:

```
my_dataset/
├── metadata.csv
└── wavs/            # or wav/
    ├── 0001.wav
    └── ...
```

`metadata.csv` must sit directly under the `--input-dir`. If it is missing, preprocess logs
`Missing metadata file: .../metadata.csv` and produces nothing.

## metadata.csv format

The file is **pipe-delimited** (`|`), one utterance per line, and is parsed exactly as
follows:

- **Filename** is the first field (`row[0]`).
- **Text** is always the last field (`row[-1]`).
- **Speaker** is the second field (`row[1]`) **only when the row has more than two fields** and
  `--single-speaker` was not passed.

So both of these are valid:

```
0001|Hello and welcome.                    # single-speaker: filename|text
0002|alice|Hello and welcome.              # multi-speaker:  filename|speaker|text
```

A row with exactly two fields is always treated as single-speaker. A row with fewer than two
fields is skipped with `Skipping malformed row`.

## Audio file resolution

For each metadata filename, phoonnx searches `wav/` then `wavs/`, trying three names in order:

1. the filename exactly as written,
2. the filename with `.wav` appended,
3. the filename with leading zeros stripped, plus `.wav` (so `0001` also matches `1.wav`).

Behavior:

- A file that resolves nowhere is skipped with `Missing audio file for filename: ...`.
- A zero-byte file is skipped with `Empty audio file: ...`.
- With `--skip-audio`, audio is not required at all (text-only phonemization).

## Audio requirements and normalization

- Source sample rates need **not** match — every clip is resampled to `--sample-rate`
  (default 22050 Hz) and cached as normalized audio plus a spectrogram under
  `<output-dir>/cache/<sample-rate>/`.
- Clips should be clean single-speaker speech, trimmed of long silences (silence trimming is
  applied during caching).
- An utterance whose spectrogram has fewer frames than its phoneme-id count is dropped with
  `audio is too short for its text` — the monotonic aligner needs at least one frame per
  phoneme. A handful is normal; many indicates audio/transcript misalignment.

## Speakers

- `--single-speaker` forces the whole dataset to one speaker and ignores speaker columns.
- `--speaker-id N` assigns a fixed integer id to every utterance (single-speaker only; cannot
  be combined with `--single-speaker`).
- Otherwise, multi-speaker mode activates automatically when **more than one distinct** speaker
  label is present. Speaker IDs are then assigned by descending utterance count (the speaker
  with the most clips becomes id 0), and the mapping is written to `speaker_id_map` in
  `config.json`.

## Column mapping (tabular formats)

For `jsonl`/`parquet`/`hf` sources, columns are resolved by name with fallbacks. Override any
of them explicitly:

| Flag | Holds | Default fallback order |
|---|---|---|
| `--text-column` | Transcript text | `text`, `sentence`, `transcription`, `transcript` |
| `--audio-column` | Audio path or embedded bytes | `audio` |
| `--speaker-column` | Speaker label | `speaker`, `speaker_id` |
| `--phonemes-column` | Precomputed phonemes (opt-in) | unset |
| `--lang-column` | Per-row language code | unset |

Any unmapped columns are carried through into `dataset.jsonl` extras.

### Embedded audio bytes

Hugging Face audio columns (and phoonnx's own Parquet shards) frequently store the audio as
embedded **bytes** rather than a filesystem path — the `path` field is often `None`, and column
casting does not inline the bytes. The loaders read the bytes explicitly and materialize them
on demand, so no separate audio folder is required for `hf`/`parquet` sources.

### Precomputed phonemes

Pass `--phonemes-column COL` to use phonemes already present in the data instead of
phonemizing. Rows with a non-empty value in that column skip phonemization and are used
**verbatim**; their whitespace-separated symbols are validated against the final phoneme map
and a mismatch **fails loudly** rather than silently mistokenizing. Rows with an empty value
fall back to normal phonemization.

## Quality filtering

Bad clips (music, clipping, wrong transcript, off-speaker) degrade a voice. `preprocess.py`
can drop them on the way in with repeatable `--filter COLUMN:MIN:MAX` options. A clip must pass
**every** filter to be kept; `MIN` or `MAX` may be left empty for an unbounded side. Metrics
are computed fresh per clip (not read from precomputed columns). Quality filtering needs
`phoonnx[train-eval]`.

| Metric | What it measures | Notes |
|---|---|---|
| `wpm` | Words per minute | Catches too-fast / too-slow reads |
| `snr` | Energy-based signal-to-noise (dB) | Catches noisy clips |
| `clipping` | Fraction of near-full-scale samples | Catches distorted audio |
| `is_music_like` | 0/1 onset-rhythmicity heuristic | Coarse pre-filter only (~25–30% error at any threshold) |
| `vad_ratio` | Speech-activity fraction (via `vadonnx`) | Model set by `--vad-model` (default `silero`) |
| `speaker_consistency` | Min pairwise speaker-embedding similarity (via `speakeronnx`) | Model set by `--speaker-model` (default `wespeaker-resnet34`) |
| `utmos` | UTMOS naturalness (SpeechMOS) | Relative ranker; treat as coarse on non-English |
| `dnsmos_sig` / `dnsmos_bak` / `dnsmos_ovrl` | DNSMOS P.835 | Signal / background / overall |
| `plcmos` | Packet-loss-concealment quality | Catches VoIP / dropped-packet artifacts |
| `aecmos` | Echo-cancellation quality | Catches speakerphone / echo artifacts |
| `wer` | Word error rate of an ASR transcription vs the clip's own text | Most expensive; **always evaluated last**. ASR model set by `--asr-model` (default `whisper-base`) |

Referencing an unknown metric name warns and skips that one filter rather than failing.

**Recommended starting filters:**

```bash
python -m phoonnx_train.preprocess ... \
  --filter utmos:3.0: \
  --filter wpm:80:400 \
  --filter is_music_like:0:0 \
  --filter snr:15: \
  --filter clipping:0:0.01
```

Add `--filter wer:0:0.3` when transcript accuracy is in doubt (it re-transcribes every clip, so
it is the slowest). The full flag list is in the [preprocess reference](preprocess.md).

### Reusing metric values

Computing filters (especially `wer`) is expensive. You can cache and reuse the results:

- `--metrics-out sidecar.parquet` writes every computed metric value per row to a Parquet
  sidecar during filtering.
- `--metrics-in sidecar.parquet` reads a previously written sidecar back.
- `--filter-from-columns` makes each `--filter` prefer a per-row value from a dataset column of
  the same name (or from `--metrics-in`) before computing it on demand.

## Resuming an interrupted run

`--resume` skips rows already written to an existing `dataset.jsonl` (matched by row id or audio
path) and appends only new rows; writes are atomic (temp file + rename), so an interrupted run
never leaves a half-written manifest. `--resume` is **incompatible with `--corpus-only-map`**
(the corpus-only phoneme map cannot be reconstructed from already-written rows) and the two
together are rejected with an error.
