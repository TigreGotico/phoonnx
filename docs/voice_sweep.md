# Voice sweep

`scripts/voice_sweep.py` loads every voice in the bundled index, makes it speak, and checks the audio. Use it to find index entries that do not load or do not speak, and to compare two commits.

A voice passes when it loads through the index path (`TTSModelInfo.load`) and `TTSVoice.synthesize` returns audio that is not silent and holds no NaN. Where an ASR model is verified for the language, the tool also transcribes the audio and records WER and CER. A wrong token id or a wrong phonemizer gives fluent audio that says the wrong thing, and only the transcript shows it.

## Install

The sweep runs in an environment that has phoonnx with the language extras of the voices you test, plus the ASR stack:

```bash
uv venv --python 3.11 .venv-sweep
VIRTUAL_ENV=.venv-sweep uv pip install --prerelease=allow -e ".[espeak,ar,en,es,pt,ru,he,omnivoice,pockettts,cloning]" \
    onnx-asr "text2spoken[ru]" soundfile scipy pyarrow
```

A voice whose phonemizer is not installed fails to load, and the sweep records the error.

## Verify the ASR models first

An ASR model that does not know a language still returns fluent text, so its WER means nothing. `floor` scores each model on 15 clips of the FLEURS test split and keeps the languages where the WER is at most 0.30 (CER at most 0.20 for Chinese, Japanese, Thai and other scripts without spaces):

```bash
python scripts/voice_sweep.py floor --out floor.json --reference-dir refs/
```

The file also stores two FLEURS sentences per verified language. The sweep adds them to that language's sentences, so the transcript is checked on real text in the language. With `--reference-dir`, `floor` also saves one FLEURS clip per language, with its text, for the voices that clone a speaker.

## Run

```bash
python scripts/voice_sweep.py run --sentences sentences.tsv --asr-floor floor.json \
    --reference-dir refs/ --out run.jsonl --jobs 3 --cpus 0-11 --threads 4 --evict --max-cache-gb 30
```

- `--sentences` is a TSV with the columns `lang`, `id`, `category`, `text` and `source`. Each voice gets the rows of its language whose `id` is in `--sentence-ids` (default `plain digits`). A row whose source starts with `script:` is a sentence for the script, not the language, so it gets no WER.
- `--reference-dir` holds `<lang>.wav` and `<lang>.txt`, or `default.wav`, for voices with `requires_reference`.
- Voices that share model files run together in one worker process, so the files download once. A group of more than `--group-size` voices (default 40) is split between the slots. With `--evict` a repo leaves the Hugging Face cache, through huggingface_hub, when no pending voice downloads from it. `--max-cache-gb` starts no new group while the cache holds more than that, and `--min-free-gb` waits while the cache disk has less free space than that.
- `--timeout` is the limit per voice in seconds. A voice that runs out of time is recorded as `timeout`, with the sentences it spoke before. A worker that crashes or stops answering is replaced, and the voice it was on is recorded as `crash` or `timeout`.
- `--threads` caps the onnxruntime threads of every session. Without the cap onnxruntime starts one pinned thread per core, and `--cpus` has no effect.

The run writes `run.jsonl.synth` while it synthesises, then adds the ASR results and writes `run.jsonl`. Both files resume: a stopped run continues where it stopped. The last line states the voices in and the results out, and the command fails when they differ.

Each line of `run.jsonl` is one voice:

| Field | Meaning |
| --- | --- |
| `status` | `ok`, `partial`, `load_failed`, `synth_failed`, `silent`, `nan`, `timeout`, `crash`, `needs_reference` or `no_sentences` |
| `error` | the first error, for a failed voice |
| `resolved` | adapter, language code, phoneme type, alphabet and sample rate that the loaded voice uses |
| `snapshot` | the Hugging Face revision of the model file |
| `sentences` | per sentence: duration, peak, RMS, NaN, silent, RTF, and transcript, WER and CER where scored |
| `wer`, `cer` | the mean over the scored sentences |

## Compare two runs

```bash
python scripts/voice_sweep.py diff base.jsonl head.jsonl --wer-margin 0.15
```

`diff` lists each voice that spoke in the first run and fails in the second, and each voice whose WER rose by more than the margin. It exits with status 1 when it finds a regression.
