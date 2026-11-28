# Benchmarking TTS latency

<!-- agent-hint
covers: how to measure where synthesis time goes and how to read time to first audio and the real-time factor
source: scripts/bench_latency.py, phoonnx/providers.py:PROVIDERS_ENV_VAR, phoonnx/voice.py:TTSVoice.synthesize
generated: no
verify: ! comm -3 <(sed '/<!-- agent-hint/,/-->/d' docs/benchmarking.md | grep -oP -- '(?<![\w-])--[a-z][a-z-]*' | sort -u) <(.venv/bin/python -I scripts/bench_latency.py --help | grep -oP -- '(?<![\w-])--[a-z][a-z-]*' | grep -vx -- --help | sort -u) | grep .
see: streaming.md, voice_sweep.md
-->

This page shows how to measure where the time of a `synthesize()` call goes. It is for developers who compare voices or execution providers.

## Run the benchmark

1. Download a voice and print the paths of its model and config files:

   ```bash
   python -c "
   from phoonnx.model_manager import TTSModelManager
   manager = TTSModelManager(); manager.load(); manager.merge_default_voices()
   info = manager.voices['OpenVoiceOS/phoonnx_kab-DZ_miro_o2i']
   print(info.download_model()); print(info.hub_path(info.config_url))"
   ```

   The two lines end in `miro_kab-DZ.onnx` and `miro_kab-DZ.json`.

2. Run the script from a checkout of this repository, with the model file and its config. In this example, the two files are copied to `model.onnx` and `config.json`:

   ```bash
   python scripts/bench_latency.py model.onnx --config config.json --runs 2 --text "Azul, amek telliḍ?"
   ```

3. Read the tables. A run on a busy 8-core laptop CPU prints this, followed by a second table for the five-sentence text:

   ```text
   === Load (cold, one-shot) ===
   stage                    seconds
   config_parse              0.0419
   session_create            1.7694
   cold_warmup_run           3.5121

   === 1sentence (warm, N=2) ===
   stage                       median       p95       min       max
   text_normalize              0.0000    0.0000    0.0000    0.0000
   diacritics                  0.0000    0.0000    0.0000    0.0000
   phonemize                   0.0276    0.0305    0.0247    0.0305
   tokenize                    0.0013    0.0025    0.0001    0.0025
   session_run_total           0.6187    0.6303    0.6070    0.6303
   postprocess_total           0.0013    0.0024    0.0002    0.0024
   ttfa                        0.5658    0.5803    0.5514    0.5803
   total                       0.5659    0.5803    0.5514    0.5803
   rtf                         0.2220    0.2262    0.2179    0.2262
   audio_seconds: 2.113
   ```

The numbers change from run to run on a busy machine, so compare medians over more runs.

## Options

The first argument is the path of a local `.onnx` voice. These options change the run:

- `--config PATH` gives the voice config JSON. Without it, the script reads `MODEL.json` next to the model.
- `--runs N` sets the number of warm runs to summarize. The default is 5.
- `--warmup` runs one more untimed pass before the cold run. The default is `--no-warmup`.
- `--text` replaces the built-in one-sentence and five-sentence texts. Both tables then use the same text.
- `--text-file PATH` reads the text from a file.
- `--json-out PATH` also writes the results as JSON. The JSON holds the execution providers that ONNX Runtime lists as available.

To select an ONNX Runtime execution provider, set `PHOONNX_ONNX_PROVIDERS` before the run, for example `PHOONNX_ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider`. The list of available providers does not prove that a GPU provider starts. When a CUDA or cuDNN shared library is missing, ONNX Runtime uses the CPU and shows no error. Read the `session_create` time and the run times before you trust a GPU result.

## What the stages mean

The script times each step of synthesis on its own. It runs each step over the whole text before the next step starts, through `phonemize` and the deprecated `phonemes_to_ids` and `phoneme_ids_to_audio` methods. `TTSVoice.synthesize()` runs `prepare` and `render`, which take the text one sentence at a time. The stage times show where the time goes, but their sum can differ from the time of one `synthesize()` call. The timed path patches and mocks nothing.

- `config_parse` and `session_create` load the voice: the first parses its JSON config, the second builds the ONNX Runtime session. The script reports them one time, because a deployment loads a voice one time.
- `cold_warmup_run` is the first synthesis after the session is made. ONNX Runtime optimizes the graph and selects kernels during this run, one time per process. The runs after it are faster for that reason alone.
- `text_normalize` applies the phonetic spellings of the voice, when the voice has a spelling table.
- `diacritics` adds diacritics, for voices that ask for them, such as Arabic voices. It is zero for other voices.
- `phonemize` turns the text into phonemes, one sequence per sentence.
- `tokenize` maps the phonemes to the integer ids of the model.
- `session_run_total` is the sum of the ONNX Runtime calls for all sentences.
- `postprocess_total` normalizes the audio, applies the volume and clips the samples into the final `AudioChunk`.

## Time to first audio and real-time factor

Time to first audio (`ttfa`) is the wall-clock time from the `synthesize()` call to the first `AudioChunk`. `synthesize()` speaks one sentence before it reads the next. For a paragraph, `ttfa` holds the cost of the first sentence, plus the normalization of the whole text, which runs before the text is split into sentences. This number tells how long a user waits before they hear speech.

The real-time factor (`rtf`) is the total synthesis time divided by the seconds of audio. An `rtf` below 1.0 means that synthesis runs faster than playback, so the voice can keep up with streaming. The `rtf` does not tell when the first chunk arrives: a voice with a low `rtf` can still start slowly when its first sentence is long.

Read the two numbers together. `ttfa` tells how long the user waits for the first audio. `rtf` tells if synthesis keeps up after it starts.

---

[← Deployment](deployment.md) · [Home](index.md) · [Voice sweep →](voice_sweep.md)
