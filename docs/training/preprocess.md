# Preprocess reference

<!-- agent-hint
covers: every phoonnx_train.preprocess flag, the files it writes and the fine-tuning phoneme map
source: phoonnx_train/preprocess.py:cli, phoonnx_train/dataset_loaders.py, phoonnx_train/quality_filter.py, phoonnx/config.py:voice_frontend
generated: no
verify: ! comm -3 <(grep -oP -- '(?<![\w-])--[a-z][a-z0-9-]*' docs/training/preprocess.md | sort -u) <(.venv/bin/python -m phoonnx_train.preprocess --help | grep -oP -- '(?<![\w-])--[a-z][a-z0-9-]*' | sort -u) | grep . && .venv/bin/python -c "import re, sys; import phoonnx_train.dataset_loaders as L; from phoonnx_train.preprocess import cli; cols = {'--text-column': L.DEFAULT_TEXT_COLUMNS, '--audio-column': L.DEFAULT_AUDIO_COLUMNS, '--speaker-column': L.DEFAULT_SPEAKER_COLUMNS, '--phonemes-column': (), '--lang-column': ()}; show = lambda p, o: (', '.join(cols[o]) or 'unset') if o in cols else ('on' if p.default is True else 'off') if p.is_flag else '\"\"' if p.default == '' else str(p.default) if isinstance(p.default, (str, int, float)) else 'none'; want = {o: show(p, o) for p in cli.params for o in p.opts}; rows = [[c.strip() for c in r.split('|')[1:-1]] for r in open('docs/training/preprocess.md').read().splitlines() if r.startswith('| \x60-')]; got = [(re.search(r'--[a-z][a-z0-9-]*', r[0]).group(), r[1].replace('\x60', '')) for r in rows if len(r) == 3 and '<' not in r[1] and (r[1].startswith('\x60') or r[1] in ('on', 'off', 'none', 'unset'))]; bad = [(f, v, want.get(f)) for f, v in got if want.get(f) != v and not (v == 'unset' and want.get(f) == 'none')]; print(len(got), 'defaults checked', bad); sys.exit(bool(bad))"
see: datasets.md, training.md, ../phonemizers.md
-->

This page lists every `phoonnx_train.preprocess` flag and the files that preprocess writes. Read it when you build a training set.

For the guided path, start with the [training quickstart](quickstart.md). For the input formats, see [Datasets](datasets.md).

```bash
python -m phoonnx_train.preprocess --input-dir SRC --output-dir DST --language LANG [options]
```

## What it writes

- `<output-dir>/config.json`: the model and dataset configuration. See the [config.json reference](training.md#configjson-reference).
- `<output-dir>/dataset.jsonl`: one utterance per line, with its text, phonemes, phoneme ids, audio paths and engine extras.
- `<output-dir>/cache/<sample-rate>/`: the normalized audio and the cached spectrograms.

## Text normalization

Before phonemization, preprocess normalizes each utterance with `text2spoken.normalize` in the language of `--language`. Numbers, dates, times, fractions, units and titles become spoken words, and contractions stay as written. For example, the English text `"I'm Dr. Smith, 3/4 of 10kg"` becomes `"I'm Doctor Smith, three-quarters of ten kilograms"`.

Preprocess and synthesis build their text frontend with the same function, `phoonnx.config.voice_frontend`. A text thus gives the same phonemes at training and at inference.

## Required options

| Option | Description |
|---|---|
| `-i, --input-dir SOURCE` | The dataset source. Give it more than once, or as a comma-separated list, to merge datasets. Preprocess adds the source name to each speaker label. A source is an LJSpeech directory, a `.jsonl` file, a `.parquet` file, shard glob or directory of shards, or a Hugging Face `org/name` repo id |
| `-o, --output-dir DIR` | Where preprocess writes `config.json`, `dataset.jsonl` and the cache |
| `-l, --language TEXT` | The phonemizer language code, for example `en`, `en-US` or `pt-PT` |

## Input format and columns

| Option | Default | Description |
|---|---|---|
| `--dataset-format` | `auto` | `auto`, `ljspeech`, `jsonl`, `parquet` or `hf`. `auto` detects the format of each source (see [Datasets: Input formats](datasets.md#input-formats)) |
| `--text-column NAME` | `text`, `sentence`, `transcription`, `transcript` | The transcript column (jsonl, parquet, hf) |
| `--audio-column NAME` | `audio` | The column with an audio path or embedded bytes (jsonl, parquet, hf) |
| `--speaker-column NAME` | `speaker`, `speaker_id` | The speaker label column (jsonl, parquet, hf) |
| `--phonemes-column NAME` | unset | Opt-in precomputed phonemes. Preprocess uses them as written and stops with an error on a symbol that the phoneme map does not have |
| `--lang-column NAME` | unset | A language code for each row, copied into the extras of `dataset.jsonl` |

Hugging Face and Parquet audio columns often hold embedded bytes and not a path. The loaders read the bytes and write the audio when they need it.

## Resuming

| Option | Description |
|---|---|
| `--resume` | Skip the rows that an existing output `dataset.jsonl` holds, by row id or audio path, and add only the other rows. Writes are atomic. You cannot use it together with `--corpus-only-map` or `--segment-map` |

## Phonemization

| Option | Default | Description |
|---|---|---|
| `--phoneme-type` | `espeak` | The phonemizer backend, one of the [`PhonemeType`](../phonemizers.md) values |
| `--alphabet` | `ipa` | The output alphabet, one of the [`Alphabet`](../phonemizers.md#alphabet-reference) values |
| `--phonemizer-model TEXT` | `""` | A model path, model name or variant for the backends that take one, such as ByT5, AhoTTS, Cotovia and arbtok |
| `--text-casing` | `ignore` | The casing that preprocess applies before phonemization: `ignore`, `lower`, `upper` or `casefold`. `config.json` records it as `text_casing`, and synthesis applies the same casing. `--resume` keeps the value of the existing `config.json` |
| `--add-diacritics` | off | Add diacritics through scriptconv before phonemization. The run stops if the diacritizer for the language is not installed |
| `--keep-punctuation/--no-keep-punctuation` | on | Keep clause punctuation (`, . ? ! : ;`) in the phonemes, as Piper does. `config.json` records it as `keep_punctuation`, so synthesis does the same. `--resume` keeps the value of the existing `config.json` |

## Audio

| Option | Default | Description |
|---|---|---|
| `-r, --sample-rate INT` | `22050` | The target sample rate. Preprocess resamples every clip to it |
| `--cache-dir DIR` | `<output-dir>/cache/<sample-rate>` | Where preprocess caches the normalized audio and spectrograms |
| `--skip-audio` | off | Phonemize the text only, with no audio |

## Speakers

| Option | Default | Description |
|---|---|---|
| `--single-speaker` | off | Make the dataset one speaker and ignore the speaker columns |
| `--speaker-id INT` | none | A fixed speaker id for a single-speaker dataset. You cannot use it together with `--single-speaker` |

## Fine-tuning (phoneme map)

| Option | Default | Description |
|---|---|---|
| `-c, --prev-config FILE` | none | Use the `phoneme_id_map` of an earlier `config.json`, so the vocabulary stays the same |
| `--drop-extra-phonemes BOOL` | `True` | Drop the symbols that the earlier map does not have, so fine-tuning can continue. Set it to `False` to stop with an error on such a symbol |
| `--corpus-only-map` | off | Build the phoneme map only from the symbols in the corpus, not from the full default IPA table. A symbol outside the map then fails at tokenization, and does not map to an embedding that the model never trained. You can fine-tune such a model only from a config whose map is a subset of it |
| `--segment-map` | off | Give each phoneme segment one id. See [Segment map](#segment-map). It implies `--corpus-only-map`. You cannot use it together with `--resume` or `--prev-config` |

When you fine-tune with `--prev-config`, preprocess drops each symbol that the earlier map does not have and logs a warning. Give `--drop-extra-phonemes False` to stop with an error.

### Segment map

With `--segment-map`, one id holds a whole phoneme segment: a symbol, its combining marks, its attached modifier letters (`ʲ`, `ː`, `ʰ`) and the symbol that a tie bar joins to it. So `ɐ̃`, `tʲ`, `aː` and `d͡ʒ` get one id each.

A run of tone letters in one notation gets one id. The notations are Chao letters, left-stem letters and the numerals ¹ to ⁵, so `˥˩` and `⁵¹` get one id each. A change of notation starts a new id, and the neutral tone `⁰` keeps its own id. Stress marks, spaces and punctuation keep their own ids. A phonemizer item longer than one code point, such as a token from a dataset column, keeps its own id.

A `features.json` input table needs this map, because the table describes whole segments.

At inference, a segment that the training data did not hold loses its mark, so `ẽ` is read as `e`. A precomposed form does not match a decomposed form. Synthesize with the phonemizer that made the training data.

A fine-tune keeps the earlier map, and the tokenizer of that map reads the corpus. For this reason `--segment-map` and `--prev-config` cannot go together.

## Engine-specific preprocessing

These options run the per-utterance feature extraction of a training engine, and record the fields it makes in `dataset.jsonl`. This `--engine` is not the `--engine` of `phoonnx_train.train`. Here it only selects the extra features, such as d-vectors or F0, and not the model that you train later.

| Option | Description |
|---|---|
| `--engine NAME` | Run the `extra_preprocess` of this engine on each utterance, for example `yourtts` d-vectors or `fastpitch` F0 |
| `--speaker-encoder-path PATH` | With `--engine yourtts`: the Coqui ResNet ONNX speaker encoder that computes the d-vectors |
| `--language-id INT` | With `--engine yourtts`: the language id that preprocess records on every utterance, for multilingual training |
| `--f0-method {pyin,dio,harvest}` | The F0 extractor for the engines that use pitch. The default is `pyin`, which uses `librosa.pyin`. `dio` and `harvest` use WORLD through `pyworld`, from the `train-pyworld` extra, and are faster. The value must match the `f0_method` of the training engine config, because it selects the F0 cache file that training reads |

## Quality filters

These options drop the utterances whose metric is outside `[MIN, MAX]`. You can give `--filter` more than once, and a clip must pass every filter. Filters need `phoonnx[train-eval]`. The metrics and the suggested thresholds are in [Datasets: Quality filtering](datasets.md#quality-filtering).

| Option | Default | Description |
|---|---|---|
| `--filter COLUMN:MIN:MAX` | none | A quality filter. Leave `MIN` or `MAX` empty for no limit |
| `--vad-model NAME` | `silero` | The `vadonnx` model for the `vad_ratio` metric |
| `--speaker-model NAME` | `wespeaker-resnet34` | The `speakeronnx` model for `speaker_consistency` |
| `--asr-model NAME` | `whisper-base` | The model for `wer`, which `onnx_asr.load_model` must load |
| `--metrics-out FILE` | none | Write each metric value of each row to a Parquet file |
| `--metrics-in FILE` | none | Read a metrics file that `--metrics-out` wrote. With `--filter-from-columns`, its values come before a new calculation |
| `--filter-from-columns` | off | Make `--filter` use the value from a dataset column of the same name, or from `--metrics-in`, before it calculates the metric |

## Cross-machine paths

To preprocess on one machine and train on another, rewrite the paths in `dataset.jsonl`:

| Option | Description |
|---|---|
| `--jsonl-audio-path BASE` | Replace the audio base directory, the part before `/wav/`, in `dataset.jsonl` |
| `--jsonl-audio-spec-path BASE` | Replace the cache base directory, the part before `/cache/`, in `dataset.jsonl` |

## Miscellaneous

| Option | Default | Description |
|---|---|---|
| `-w, --max-workers INT` | CPU count | The number of worker processes, 1 or more |
| `--dataset-name TEXT` | name of the parent of `--output-dir` | The dataset name in `config.json` |
| `--audio-quality TEXT` | name of `--output-dir` | The audio quality label in `config.json` |
| `--debug` | off | Print DEBUG logs |

---

[Docs index](../index.md) · Previous: [Datasets](datasets.md) · Next: [Training reference](training.md)
