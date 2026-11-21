# Benchmarking TTS latency

> This page targets developers who need to reason about synthesis latency —
> where time goes inside a single `synthesize()` call, and how to compare
> voices or execution providers.

## Running the benchmark

```bash
python scripts/bench_latency.py MODEL [OPTIONS]
```

`MODEL` is either a path to a local `.onnx` voice or a voice id downloadable
through `TTSModelManager` (for example `piper/en_US-amy-low`). Useful options:

- `--config PATH` — voice config JSON, if it doesn't sit next to the model as `MODEL.json`.
- `--runs N` — number of warm runs to summarize (default 5).
- `--warmup / --no-warmup` — run one extra untimed pass before the reported cold run.
- `--text` / `--text-file` — override the built-in 1-sentence and 5-sentence benchmark texts.
- `--json-out PATH` — write machine-readable results alongside the printed table.

To force a specific ONNX Runtime execution provider, set `PHOONNX_ONNX_PROVIDERS`
before running, e.g. `PHOONNX_ONNX_PROVIDERS=CPUExecutionProvider` or
`PHOONNX_ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider`. The script
also reports which providers ONNX Runtime *claims* to have available — that is
not proof a GPU provider actually initializes; a missing CUDA/cuDNN shared
library makes ONNX Runtime fall back to CPU silently, so always check the
`session_create` time and the per-run numbers before trusting a GPU result.

## What the stages mean

Each stage mirrors a step inside `TTSVoice.synthesize()` (see
`phoonnx/voice.py`).

The script times its own staged path: it runs each step over the whole text before the next step starts, through `phonemize` and the deprecated `phonemes_to_ids` and `phoneme_ids_to_audio`. Synthesis itself runs `prepare` and `render`, which take the text one sentence at a time. Thus the stage times show where the time goes, but their sum can differ from the time of one `synthesize()` call. Nothing is patched or mocked in the timed path:

- **config_parse** / **session_create** — loading the voice: parsing its JSON
  config versus building the ONNX Runtime session. Reported once, cold, since
  a real deployment loads a voice exactly once and keeps it resident.
- **cold_warmup_run** — the first `session.run` after session creation, which
  pays for ONNX Runtime's graph optimization and kernel selection. This cost
  is paid once per process; every run after it is faster for that reason
  alone, not because anything about the text changed.
- **text_normalize** — phonetic-spelling substitution (user pronunciation
  overrides), when the voice ships an override table.
- **diacritics** — restoring diacritics (Arabic/Hebrew voices only); zero
  for voices that don't request it.
- **phonemize** — turning text into phoneme sequences, one per sentence.
- **tokenize** — mapping phonemes to the model's integer vocabulary.
- **session_run_total** — the sum of ONNX Runtime inference calls across all
  sentences in the text.
- **postprocess_total** — normalization, volume scaling, and clipping into
  the final `AudioChunk`.

## TTFA vs RTF

**Time-to-first-audio (TTFA)** is the wall-clock time from calling
`synthesize()` to receiving the *first* `AudioChunk`. `synthesize()` phonemizes,
tokenizes and synthesizes one sentence before it reads the next. Thus TTFA for a
multi-sentence paragraph carries the cost of the first sentence only, plus the
normalisation of the whole text, which runs before the text is cut into
sentences. This is the number that determines how long a user waits in a live
conversation before hearing anything.

**Real-time factor (RTF)** is total synthesis time divided by the seconds of
audio produced. An RTF below 1.0 means synthesis runs faster than playback —
the model can keep up with real-time streaming. RTF says nothing about how
long the *first* chunk takes to arrive; a voice can have a low RTF and still
feel slow to start if its first sentence is long.

Read them together: TTFA answers "how long until I hear anything," RTF
answers "can this keep up once it starts."
