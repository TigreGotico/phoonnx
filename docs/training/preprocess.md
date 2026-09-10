# Preprocess reference

This is the complete reference for `phoonnx_train.preprocess` — every flag, the files it
writes, and the fine-tuning behavior. It is for anyone building a training set; for the guided
path start with the [training quickstart](quickstart.md), and for dataset formatting see
[Datasets](datasets.md).

```bash
python -m phoonnx_train.preprocess --input-dir SRC --output-dir DST --language LANG [options]
```

## What it writes

- `<output-dir>/config.json` — model + dataset configuration (see the [config.json reference](training.md#configjson-reference))
- `<output-dir>/dataset.jsonl` — one normalized utterance per line (text, phonemes, phoneme IDs, audio references, engine extras)
- `<output-dir>/cache/<sample-rate>/` — normalized audio and cached spectrograms

## Required options

| Option | Description |
|---|---|
| `-i, --input-dir SOURCE` | Dataset source; **repeatable** (or comma-separated) to merge datasets with per-source speaker namespacing. A source may be an LJSpeech directory, a `.jsonl` file, a `.parquet` file / shard glob / directory of shards, or a Hugging Face `org/name` repo id |
| `-o, --output-dir DIR` | Where to write `config.json`, `dataset.jsonl`, and the cache |
| `-l, --language TEXT` | Phonemizer language code (e.g. `en`, `en-US`, `pt-PT`) |

## Input format and columns

| Option | Default | Description |
|---|---|---|
| `--dataset-format` | `auto` | `auto`, `ljspeech`, `jsonl`, `parquet`, or `hf`. `auto` detects per source (see [Datasets → Input formats](datasets.md#input-formats)) |
| `--text-column NAME` | `text`, `sentence`, `transcription`, `transcript` | Transcript column (jsonl/parquet/hf) |
| `--audio-column NAME` | `audio` | Audio path or embedded-bytes column (jsonl/parquet/hf) |
| `--speaker-column NAME` | `speaker`, `speaker_id` | Speaker-label column (jsonl/parquet/hf) |
| `--phonemes-column NAME` | unset | Opt-in precomputed phonemes, used verbatim and validated against the phoneme map (mismatch fails loudly) |
| `--lang-column NAME` | unset | Per-row language code, carried into `dataset.jsonl` extras |

Hugging Face and Parquet audio columns often hold embedded bytes rather than a path; the loaders
read the bytes explicitly and materialize audio on demand.

## Resuming

| Option | Description |
|---|---|
| `--resume` | Skip rows already present in an existing output `dataset.jsonl` (by row id / audio path) and append only new rows; writes are atomic. **Incompatible with `--corpus-only-map`** (rejected with an error) |

## Phonemization

| Option | Default | Description |
|---|---|---|
| `--phoneme-type` | `espeak` | Phonemizer backend; one of the [`PhonemeType`](../phonemizers.md) values |
| `--alphabet` | `ipa` | Output alphabet; one of the [`Alphabet`](../phonemizers.md#alphabet-reference) values |
| `--phonemizer-model TEXT` | `""` | Model path/name or variant selector for backends that take one (ByT5, AhoTTS, Cotovia, arbtok) |
| `--text-casing` | `ignore` | Casing applied before phonemization: `ignore`, `lower`, `upper`, `casefold` |
| `--add-diacritics` | off | Add diacritics before phonemization (Arabic/Hebrew, phonemizer-specific) |

## Audio

| Option | Default | Description |
|---|---|---|
| `-r, --sample-rate INT` | `22050` | Target sample rate; every clip is resampled to it |
| `--cache-dir DIR` | `<output-dir>/cache/<sample-rate>` | Where normalized audio and spectrograms are cached |
| `--skip-audio` | off | Phonemize text only; do not require or process audio |

## Speakers

| Option | Default | Description |
|---|---|---|
| `--single-speaker` | off | Force one speaker, ignoring metadata speaker columns |
| `--speaker-id INT` | none | Fixed speaker id for a single-speaker dataset (mutually exclusive with `--single-speaker`) |

## Fine-tuning (phoneme map)

| Option | Default | Description |
|---|---|---|
| `-c, --prev-config FILE` | none | Reuse the `phoneme_id_map` from a previous `config.json` so the vocabulary is preserved |
| `--drop-extra-phonemes BOOL` | `True` | When the new data has symbols absent from the previous map, discard them (so fine-tuning can proceed). Set to `False` to instead raise on any new symbol |
| `--corpus-only-map` | off | Build the phoneme map only from symbols the corpus actually contains, instead of seeding it with the full default IPA table. Symbols outside the map then fail at tokenization rather than mapping to untrained embeddings. Models built this way can only be fine-tuned from a compatible (subset) map |

> `--drop-extra-phonemes` defaults to **`True`**. When fine-tuning with `--prev-config`, a new
> symbol is dropped with a warning unless you pass `--drop-extra-phonemes False`, which turns
> the mismatch into an error.

## Engine-specific preprocessing

These run a **training engine's** extra per-utterance feature extraction at preprocess time and
record the produced fields in `dataset.jsonl`. This `--engine` is **distinct** from
`train.py --engine`; here it only controls sidecar feature extraction (d-vectors, F0), not the
model you later train.

| Option | Description |
|---|---|
| `--engine NAME` | Run this engine's `extra_preprocess` per utterance (e.g. `yourtts` d-vectors, `fastpitch` F0) |
| `--speaker-encoder-path PATH` | `[--engine yourtts]` Coqui ResNet ONNX speaker encoder used to compute d-vectors |
| `--language-id INT` | `[--engine yourtts]` language id recorded on every utterance (multilingual training) |

## Quality filters

Drop utterances outside `[MIN, MAX]` on an on-demand-computed metric; repeatable, and a clip
must pass every filter. Requires `phoonnx[train-eval]`. The metric catalog and recommended
thresholds are in [Datasets → Quality filtering](datasets.md#quality-filtering).

| Option | Default | Description |
|---|---|---|
| `--filter COLUMN:MIN:MAX` | none | Repeatable quality filter; `MIN`/`MAX` may be empty for unbounded |
| `--vad-model NAME` | `silero` | `vadonnx` model for the `vad_ratio` metric |
| `--speaker-model NAME` | `wespeaker-resnet34` | `speakeronnx` model for `speaker_consistency` |
| `--asr-model NAME` | `whisper-base` | `onnx_asr`-loadable model for `wer` |
| `--metrics-out FILE` | none | Write every computed filter metric per row to a Parquet sidecar |
| `--metrics-in FILE` | none | Read a previously written metrics sidecar (preferred over recomputation with `--filter-from-columns`) |
| `--filter-from-columns` | off | Make `--filter` prefer a per-row value from a dataset column of the same name (or `--metrics-in`) before computing it |

## Cross-machine paths

Preprocess on one machine and train on another by rewriting the paths stored in
`dataset.jsonl`:

| Option | Description |
|---|---|
| `--jsonl-audio-path BASE` | Override the audio base directory (everything before `/wav`) written into `dataset.jsonl` |
| `--jsonl-audio-spec-path BASE` | Override the cache base directory (everything before `/cache`) written into `dataset.jsonl` |

## Miscellaneous

| Option | Default | Description |
|---|---|---|
| `-w, --max-workers INT` | CPU count | Parallel worker processes (minimum 1) |
| `--dataset-name TEXT` | parent dir name | Dataset name recorded in `config.json` |
| `--audio-quality TEXT` | output dir name | Audio-quality label recorded in `config.json` |
| `--debug` | off | Print DEBUG logs |

## Text normalization

Before phonemization, every utterance is normalized: numbers, dates, times, fractions, units
and contractions are expanded to spoken words (via `ovos-number-parser` / `ovos-date-parser`,
with locale-aware separators keyed off `--language`). For example, English
`"I'm Dr. 3/4 of 10kg"` becomes `"I am Doctor three quarters of ten kilograms"`. This runs
automatically inside preprocess so training text matches what the phonemizer sees at inference.
