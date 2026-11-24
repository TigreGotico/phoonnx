# Datasets

<!-- agent-hint
covers: which training data preprocess reads, how it finds audio and speakers, and how to filter bad clips
source: phoonnx_train/dataset_loaders.py, phoonnx_train/preprocess.py:cli, phoonnx_train/quality_filter.py
generated: no
verify: .venv/bin/python -c "import re, sys; import phoonnx_train.dataset_loaders as L; from phoonnx_train import quality_filter as Q; p = open('docs/training/datasets.md').read(); cols = {'--text-column': L.DEFAULT_TEXT_COLUMNS, '--audio-column': L.DEFAULT_AUDIO_COLUMNS, '--speaker-column': L.DEFAULT_SPEAKER_COLUMNS}; bad = [f for f, c in cols.items() if not re.search(re.escape(f) + r'\x60 \|[^|]*\| ' + re.escape(', '.join(chr(96) + x + chr(96) for x in c)) + r' \|', p)]; table = p.split(chr(10) + '## Quality filtering')[1].split(chr(10) + '### ')[0]; names = set(re.findall(r'\x60([a-z_]+)\x60', ' '.join(r.split('|')[1] for r in table.splitlines() if r.startswith('| \x60')))); bad += sorted(names ^ set(Q._SCORER_REGISTRY)); print(bad); sys.exit(bool(bad))"
see: preprocess.md, quickstart.md
-->

This page describes the training data that preprocess accepts, how it finds audio and speakers, and how to filter out bad clips. Read it when you prepare a dataset for the [training quickstart](quickstart.md).

## Input formats

`phoonnx_train.preprocess` reads four kinds of source. The `--dataset-format` flag selects one. Its default, `auto`, detects the kind of each source:

| Format | Auto-detected from | Source is |
|---|---|---|
| `ljspeech` | a directory that contains `metadata.csv` | an LJSpeech directory (`metadata.csv` and `wav/` or `wavs/`) |
| `jsonl` | a `.jsonl` file | one JSON object per line |
| `parquet` | a `.parquet` file, a shard glob, or a directory of shards | one or more Parquet tables |
| `hf` | an `org/name` string that is not an existing path | a Hugging Face dataset repo id, loaded with `datasets` |

LJSpeech is the simplest format, and the [quickstart](quickstart.md) uses it. The tabular formats (`jsonl`, `parquet` and `hf`) find their columns by name. See [Column mapping](#column-mapping-tabular-formats).

### Merging multiple sources

You can give `-i/--input-dir` more than once, or give it a comma-separated list, to merge several datasets in one run. When there is more than one source, preprocess adds the source name to each speaker label, so two datasets that use the same label keep two speakers:

```bash
python -m phoonnx_train.preprocess \
  -i dataset_a -i dataset_b,my-org/hf-dataset \
  --output-dir train_out --language en-US
```

## LJSpeech directory layout

An LJSpeech directory holds `metadata.csv` and a folder of WAV files:

```
my_dataset/
├── metadata.csv
└── wavs/            # or wav/
    ├── 0001.wav
    └── ...
```

`metadata.csv` must be directly in the `--input-dir` directory. If it is missing, preprocess logs `Missing metadata file: .../metadata.csv` and reads no rows from that source.

## metadata.csv format

The file holds one utterance per line, with fields separated by a pipe (`|`). Preprocess reads the fields in this way:

- The first field is the file name.
- The last field is the text.
- The second field is the speaker, but only when the row has more than two fields and you did not give `--single-speaker`.

Both of these rows are valid:

```
0001|Hello and welcome.                    # single speaker: filename|text
0002|alice|Hello and welcome.              # multi-speaker:  filename|speaker|text
```

A row with two fields has no speaker. Preprocess skips a row with fewer than two fields and logs `Skipping malformed row`.

## Audio file resolution

For each file name in `metadata.csv`, preprocess looks in `wav/` and then in `wavs/`. In each folder it tries three names in this order:

1. the file name as written,
2. the file name with `.wav` added,
3. the file name without its leading zeros, with `.wav` added, so `0001` also finds `1.wav`.

A file that preprocess cannot find is skipped with `Missing audio file for filename: ...`. A file of zero bytes is skipped with `Empty audio file: ...`. With `--skip-audio`, preprocess needs no audio and phonemizes the text only.

## Audio requirements and normalization

The source clips can have different sample rates. Preprocess resamples every clip to `--sample-rate` (default 22050 Hz). It trims silence and caches the normalized audio and a spectrogram under `<output-dir>/cache/<sample-rate>/`.

Use clean speech from one speaker per clip. Preprocess drops an utterance whose spectrogram has fewer frames than its phoneme ids, and logs `audio is too short for its text`. The monotonic aligner needs at least one frame for each phoneme. A few dropped clips are normal. Many dropped clips show that the audio and the transcripts do not align.

## Speakers

- `--single-speaker` makes the dataset one speaker and ignores the speaker fields.
- `--speaker-id N` gives every utterance the fixed integer id `N`. You cannot use it together with `--single-speaker`.
- Otherwise, preprocess treats the dataset as multi-speaker when it holds more than one speaker label. It gives the ids in order of utterance count, so the speaker with the most clips gets id 0. It writes the map to `speaker_id_map` in `config.json`.

## Column mapping (tabular formats)

For `jsonl`, `parquet` and `hf` sources, preprocess finds each column by name, and tries a list of names in order. A flag sets the column name directly:

| Flag | Holds | Default names, in order |
|---|---|---|
| `--text-column` | Transcript text | `text`, `sentence`, `transcription`, `transcript` |
| `--audio-column` | Audio path or embedded bytes | `audio` |
| `--speaker-column` | Speaker label | `speaker`, `speaker_id` |
| `--phonemes-column` | Precomputed phonemes (opt-in) | unset |
| `--lang-column` | Language code of the row | unset |

Preprocess copies every other column into the extras of each row in `dataset.jsonl`.

### Embedded audio bytes

Hugging Face audio columns, and the Parquet shards that phoonnx writes, often hold the audio as bytes and not as a file path. The `path` field is then often `None`. The loaders read the bytes and write the audio when they need it, so `hf` and `parquet` sources need no separate audio folder.

### Precomputed phonemes

Give `--phonemes-column COL` to use phonemes that the data already holds. A row with a value in that column skips phonemization, and preprocess uses its symbols as written. The symbols are separated by spaces. Preprocess checks each symbol against the final phoneme map and stops with an error on a symbol that the map does not have. A row with an empty value gets normal phonemization.

## Quality filtering

Bad clips, such as music, clipped audio, a wrong transcript or a different speaker, make a voice worse. Preprocess can drop them with the `--filter COLUMN:MIN:MAX` option, which you can give more than once. A clip must pass every filter. Leave `MIN` or `MAX` empty for no limit on that side. Quality filtering needs `phoonnx[train-eval]`.

Start with these filters:

```bash
python -m phoonnx_train.preprocess ... \
  --filter utmos:3.0: \
  --filter wpm:80:400 \
  --filter is_music_like:0:0 \
  --filter snr:15: \
  --filter clipping:0:0.01
```

Add `--filter wer:0:0.3` when you do not trust the transcripts. That filter transcribes every clip again, so it is the slowest one.

Preprocess calculates each metric for each clip, unless `--filter-from-columns` gives it a stored value (see [Reusing metric values](#reusing-metric-values)). A filter on an unknown metric name logs a warning, and preprocess skips that filter. The metrics are:

| Metric | What it measures | Notes |
|---|---|---|
| `wpm` | Words per minute | Finds reads that are too fast or too slow |
| `snr` | Signal-to-noise ratio from the signal energy, in dB | Finds noisy clips |
| `clipping` | Fraction of samples near full scale | Finds distorted audio |
| `is_music_like` | 0 or 1, from an onset rhythm heuristic | A coarse pre-filter. It is wrong on about 25 to 30 percent of clips at any threshold |
| `vad_ratio` | Fraction of the clip that holds speech, from `vadonnx` | `--vad-model` sets the model (default `silero`) |
| `speaker_consistency` | Lowest similarity of speaker embeddings between windows of the clip, from `speakeronnx` | `--speaker-model` sets the model (default `wespeaker-resnet34`) |
| `utmos` | UTMOS naturalness score, from SpeechMOS | |
| `dnsmos_sig`, `dnsmos_bak`, `dnsmos_ovrl` | DNSMOS P.835 | Signal, background and overall |
| `plcmos` | Packet loss concealment quality | Finds VoIP and dropped-packet artifacts |
| `aecmos` | Echo cancellation quality | Finds speakerphone and echo artifacts |
| `wer` | Word error rate of an ASR transcript against the clip's own text | The most expensive metric. It always runs last. `--asr-model` sets the model (default `whisper-base`) |

The full list of flags is in the [preprocess reference](preprocess.md).

### Reusing metric values

The metrics take time to calculate, `wer` most of all. You can keep the values and use them again:

- `--metrics-out sidecar.parquet` writes each metric value of each row to a Parquet file.
- `--metrics-in sidecar.parquet` reads such a file.
- `--filter-from-columns` makes each `--filter` use the value from a dataset column of the same name, or from `--metrics-in`, before it calculates the metric.

## Resuming an interrupted run

`--resume` skips the rows that an existing `dataset.jsonl` already holds, by row id or audio path, and adds only the other rows. Preprocess writes to a temporary file and renames it, so an interrupted run does not leave a half-written manifest.

You cannot use `--resume` together with `--corpus-only-map`, because the rows already written cannot rebuild the corpus-only phoneme map. Preprocess stops with an error when you give both.

---

[Docs index](../index.md) · Previous: [Training quickstart](quickstart.md) · Next: [Preprocess reference](preprocess.md)
