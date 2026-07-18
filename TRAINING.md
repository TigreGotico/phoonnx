# Training Guide for **phoonnx**

This document explains how to prepare data, train models, and export them to ONNX for inference.

---

## 1. Preprocessing Data

Before training, you need to preprocess your dataset into a format compatible with `phoonnx`.


```
Usage: preprocess.py [OPTIONS]

  Preprocess a TTS dataset (e.g., LJSpeech format) for training a VITS-style
  model. This script handles text normalization, phonemization, and optional
  audio caching.

Options:
  -i, --input-dir DIRECTORY       Directory with audio dataset (e.g.,
                                  containing metadata.csv and wavs/)
                                  [required]
  -o, --output-dir DIRECTORY      Directory to write output files for training
                                  (config.json, dataset.jsonl)  [required]
  -l, --language TEXT             phonemizer language code (e.g., 'en', 'es',
                                  'fr')  [required]
  -c, --prev-config FILE          Optional path to a previous config.json from
                                  which to reuse phoneme_id_map. (for fine-tuning
                                  only)
  --drop-extra-phonemes BOOLEAN   If training data has more symbols than base
                                  model, discard new symbols. (for fine-tuning
                                  only)
  -r, --sample-rate INTEGER       Target sample rate for voice (hertz,
                                  Default: 22050)
  --cache-dir DIRECTORY           Directory to cache processed audio files.
                                  Defaults to <output-dir>/cache/<sample-
                                  rate>.
  -w, --max-workers INTEGER RANGE
                                  Maximum number of worker processes to use
                                  for parallel processing. Defaults to CPU
                                  count.  [x>=1]
  --single-speaker                Force treating the dataset as single
                                  speaker, ignoring metadata speaker columns.
  --speaker-id INTEGER            Specify a fixed speaker ID (0, 1, etc.) for
                                  a single speaker dataset.
  --phoneme-type [raw|unicode|graphemes|misaki|espeak|gruut|goruut|epitran|byt5|charsiu|transphone|mwl_phonemizer|deepphonemizer|openphonemizer|g2pen|g2pfa|openjtalk|cutlet|pykakasi|cotovia|phonikud|mantoq|viphoneme|g2pk|kog2p|g2pc|g2pm|pypinyin|xpinyin|jieba]
                                  Type of phonemes to use.
  --alphabet [unicode|ipa|arpa|sampa|x-sampa|hangul|kana|hira|hepburn|kunrei|nihon|pinyin|eraab|cotovia|hanzi|buckwalter]
                                  Phoneme alphabet to use (e.g., IPA).
  --phonemizer-model TEXT         Path or name of a custom phonemizer model,
                                  if applicable.
  --text-casing [ignore|lower|upper|casefold]
                                  Casing applied to utterance text before
                                  phonemization.
  --dataset-name TEXT             Name of dataset to put in config (default:
                                  name of <output_dir>/../).
  --audio-quality TEXT            Audio quality description to put in config
                                  (default: name of <output_dir>).
  --skip-audio                    Do not preprocess or cache audio files.
  --debug                         Print DEBUG messages to the console.
  --add-diacritics                Add diacritics to text (phonemizer specific,
                                  e.g., to denote stress).
  -h, --help                      Show this message and exit.
```

This step produces:

* `config.json`: Model + dataset configuration (see below).
* `dataset.jsonl`: Normalized utterances with phoneme IDs and audio references.
* Cached normalized audio + spectrograms (in `cache/`).


**Example Usage**

```bash
python preprocess.py  \
  --input-dir /path/to/dataset/  \
  --output-dir /tmp/tts_train  \
  --prev-config /path/to/previous.ckpt.json  \
  --language en  \
  --sample-rate 22050  \
  --phoneme-type espeak  \
  --alphabet ipa
```

---

## 2. Training a Model

Train a model using PyTorch Lightning.  The training pipeline is **engine-agnostic**: you select an architecture with `--engine` and the CLI delegates model creation, checkpoint loading, and ONNX export to the engine implementation.

```
Usage: train.py [OPTIONS]

Options:
  --dataset-dir DIRECTORY         Path to pre-processed dataset directory
                                  [required]
  --engine TEXT                   TTS architecture to train (default: vits)
  --checkpoint-epochs INTEGER     Save checkpoint every N epochs (default: 1)
  --quality TEXT                  Quality/size of model (default: medium)
  --resume-from-checkpoint TEXT   Load an existing checkpoint and resume
                                  training
  --resume-from-single-speaker-checkpoint TEXT
                                  For multi-speaker models only. Converts a
                                  single-speaker checkpoint to multi-speaker
                                  and resumes training
  --seed INTEGER                  Random seed (default: 1234)
  --max-epochs INTEGER            Stop training once this number of epochs is
                                  reached (default: 1000)
  --devices TEXT                  Number of devices or list of device IDs to
                                  train on (default: "1")
  --accelerator TEXT              Hardware accelerator to use (cpu, gpu, tpu,
                                  mps, etc.)  (default: "auto")
  --default-root-dir DIRECTORY    Default root directory for logs and
                                  checkpoints (default: None)
  --precision TEXT                Precision used in training (e.g. 16, 32,
                                  bf16, 16-mixed)  (default: "32")
  --learning-rate FLOAT           Learning rate for optimizer (default: 2e-4)
  --batch-size INTEGER            Training batch size (default: 16)
  --num-workers INTEGER           Number of data loader workers (default: 1)
  --validation-split FLOAT        Proportion of data used for validation
                                  (default: 0.05)
  --discard-encoder               Discard the encoder weights from base
                                  checkpoint (not yet supported by all engines)
  --help                          Show this message and exit.
```
Usage: train.py [OPTIONS]

Options:
  --dataset-dir DIRECTORY         Path to pre-processed dataset directory
                                  [required]
  --checkpoint-epochs INTEGER     Save checkpoint every N epochs (default: 1)
  --quality [x-low|medium|high]   Quality/size of model (default: medium)
  --resume-from-checkpoint TEXT   Load an existing checkpoint and resume
                                  training
  --resume-from-single-speaker-checkpoint TEXT
                                  For multi-speaker models only. Converts a
                                  single-speaker checkpoint to multi-speaker
                                  and resumes training
  --seed INTEGER                  Random seed (default: 1234)
  --max-epochs INTEGER            Stop training once this number of epochs is
                                  reached (default: 1000)
  --devices INTEGER               Number of devices or list of device IDs to
                                  train on (default: 1)
  --accelerator TEXT              Hardware accelerator to use (cpu, gpu, tpu,
                                  mps, etc.)  (default: "auto")
  --default-root-dir DIRECTORY    Default root directory for logs and
                                  checkpoints (default: None)
  --precision INTEGER             Precision used in training (e.g. 16, 32,
                                  bf16) (default: 32)
  --learning-rate FLOAT           Learning rate for optimizer (default: 2e-4)
  --batch-size INTEGER            Training batch size (default: 16)
  --num-workers INTEGER           Number of data loader workers (default: 1)
  --validation-split FLOAT        Proportion of data used for validation
                                  (default: 0.05)
  --help                          Show this message and exit.
```


**Example Usage**

```bash
python train.py \
  --dataset-dir /tmp/tts_train \
  --engine vits \
  --quality medium \
  --max_epochs 1000 \
  --batch-size 8 \
  --accelerator gpu \
  --resume_from_checkpoint /path/to/previous.ckpt
```


### ZipVoice engine

`--engine zipvoice` trains the [ZipVoice](https://github.com/k2-fsa/ZipVoice)
flow-matching TTS (Zipformer text encoder + flow-matching decoder,
arXiv:2506.13053), vendored in `phoonnx_train/zipvoice/`. It consumes the
same preprocessed dataset as every other engine; audio is resampled to
24 kHz internally and turned into the 100-bin Vocos log-mel features the
model is defined over (cached next to the audio cache).

- `--quality base` (default fallback) uses the upstream model size;
  `--quality low` is a tiny smoke-test tier.
- Training uses the upstream recipe: ScaledAdam + the Eden schedule,
  70–100% target-span infilling masks and text-condition dropout for
  classifier-free guidance.
- `--resume-from-checkpoint` accepts both Lightning checkpoints and the
  upstream `{"model": ...}` layout, so the released ZipVoice checkpoint can
  be used as a fine-tuning starting point.

```bash
python -m phoonnx_train.train \
  --dataset-dir /tmp/tts_train \
  --engine zipvoice \
  --quality base

python -m phoonnx_train.export_onnx last.ckpt \
  --config /tmp/tts_train/config.json \
  --engine zipvoice \
  --output-dir ./exported   # text_encoder.onnx + fm_decoder.onnx
```

Export produces the two-graph contract the phoonnx `ZipVoiceAdapter`
consumes: `text_encoder.onnx` and the guidance-folded `fm_decoder.onnx`
(see `docs/zipvoice.md`).

---

## 3. Exporting to ONNX

After training, export the model checkpoint (`.ckpt`) to the ONNX format for efficient, cross-platform inference.  Export is engine-aware, so passing `--engine optispeech` (or any registered engine) will use that engine’s export procedure and metadata format.

```
Usage: export_onnx.py [OPTIONS] CHECKPOINT

  Export a model checkpoint to ONNX format.

Options:
  -c, --config PATH      Path to the model configuration JSON file.
  -o, --output-dir PATH  Output directory for the ONNX model. (Default:
                         current directory)
  --engine TEXT          TTS architecture used for training (default: vits)
  -t, --generate-tokens  Generate tokens.txt alongside the ONNX model. Some
                         inference engines need this (eg. sherpa)
  -p, --piper            Generate a piper compatible .json file alongside the
                         ONNX model.
  --help                 Show this message and exit.
```


**Example Usage**


```bash
python export_onnx.py \
  checkpoints/epoch=500-step=100000.ckpt \
  --config /path/to/output/config.json \
  --engine vits \
  --output-dir ./exported \
  --generate-tokens \
  --piper
```

-----

## 3.5 Evaluating Checkpoints During Training

`phoonnx_train/eval_loop.py` watches the training directory for new
lightning checkpoints (`epoch=*.ckpt`) and scores each one on a fixed set of
held-out sentences, synthesized on CPU so training keeps the GPU. Results are
appended to `metrics.csv` (one summary row per epoch) plus per-utterance
scores under `perutt/`, and the wavs of the best epoch so far are kept under
`samples/`. The loop is idempotent: already-scored epochs are skipped on
restart, so it can run alongside training or after the fact.

Install the extra dependencies first:

```bash
pip install phoonnx[train-eval]
```

**Example Usage**

```bash
python eval_loop.py \
  --train-dir /tmp/tts_train \
  --config /tmp/tts_train/config.json \
  --sentences heldout.txt \
  --output-dir ./eval \
  --speaker-ref-dir /path/to/dataset/wavs
```

Useful flags: `--once` (single pass instead of polling), `--poll N` (seconds
between scans, default 60), `--dry-run` (verify config + sentence encoding
without synthesis), and `--noise-scale` / `--length-scale` / `--noise-w` to
override the inference scales from `config.json`.

Two metrics are reported:

- **UTMOS** (`utmos_mean` etc.): an automatic Mean-Opinion-Score predictor
  (SpeechMOS `utmos22_strong`, 1–5 scale) estimating perceived
  naturalness/quality of each clip. **Caveat:** UTMOS is trained on
  English/Japanese MOS data; on other languages treat it as a *relative
  ranker* between checkpoints of the same voice, not an absolute quality
  score.
- **Speaker similarity** (`spk_sim_mean` etc.): cosine similarity between
  each synthesized clip's speaker embedding and a centroid built from
  reference recordings of the target speaker (`--speaker-ref-dir`, using the
  `--num-ref-wavs` largest files, default 60). Values closer to 1.0 mean the
  model sounds more like the target voice. This metric is optional: without
  `speakeronnx` or a reference directory, only UTMOS is scored and the
  speaker-similarity columns are left empty.

-----

## 3.6 Training a Vocos Vocoder

Two-stage acoustic models (Matcha-TTS and other mel-emitting engines)
produce a mel spectrogram and rely on a separate vocoder to render the
waveform. `phoonnx_train/train_vocos.py` trains a
[Vocos](https://arxiv.org/abs/2306.00814)-style vocoder in-house, so a
voice can ship with a vocoder trained (or finetuned) on the same speaker's
audio instead of a generic pretrained one — usually worth doing when the
target voice or language sounds "buzzy" or dull through a stock vocoder.

**Data needs:** audio only — no transcripts. Any directory tree of
`.wav`/`.flac`/`.ogg`/`.mp3` files works, including the `cache/` folder of
a phoonnx preprocess output directory. A few hours of clean speech is
enough for finetuning; training from scratch wants much more (or a
`--warm-start`).

> **Warning — mel settings are locked.** The vocoder is trained on the
> exact mel configuration phoonnx acoustic models use (`n_fft=1024`,
> `hop=256`, `win=1024`, `80` mels, `fmax=8000`, log-mel as in
> `phoonnx_train/vits/mel_processing.py`). These are hard-coded on
> purpose; a vocoder trained with different settings will produce garbage
> for every phoonnx mel model.

```bash
python -m phoonnx_train.train_vocos \
  --audio-dir /path/to/voice1/wavs \
  --audio-dir /path/to/voice2/wavs \
  --sample-rate 22050 \
  --batch-size 16 \
  --crop-seconds 1.0 \
  --warm-start charactr/vocos-mel-24khz \
  --max-epochs 100 \
  --default-root-dir /tmp/vocos_train
```

`--warm-start` accepts a local `.ckpt`/`.bin` file or a HuggingFace repo
id holding reference-Vocos-layout weights (e.g. `charactr/vocos-mel-24khz`,
`projecte-aina/alvocat-vocos-22khz`); parameters are copied where names
and shapes match and the matched fraction is logged. Use
`--resume-from-checkpoint` to continue an interrupted run (restores
optimizers and epoch).

**Export + runtime wiring:**

```bash
python -m phoonnx_train.export_vocos \
  /tmp/vocos_train/lightning_logs/version_0/checkpoints/epoch=99-step=12345.ckpt \
  vocoder.onnx
```

This writes `vocoder.onnx` (plus `vocoder.onnx.json` with the STFT
parameters) in the layout the phoonnx vocoder registry auto-detects as
`vocos`, and prints a torch-vs-onnxruntime parity check. Point any
mel-emitting voice at it:

```python
tts = TTSModel(model_path, config_path,
               engine_params={"vocoder_path": "vocoder.onnx"})
```

-----

## 4. Workflow Summary

1. **Prepare dataset** in LJSpeech-style format.
2. **Preprocess**:

   ```bash
   python preprocess.py --input-dir ... --output-dir ... --language en-us --sample-rate 22050
   ```
3. **Train**:

   ```bash
   python train.py --dataset-dir ... --quality medium --max_epochs 500
   ```
4. **Export**:

   ```bash
   python export_onnx.py checkpoint.ckpt model.onnx
   ```


---

## 5. Text Normalization (Preprocessing Step)

During preprocessing, all input text is **normalized** before phonemization. This ensures consistent training data and makes the phonemizer’s job easier.

Normalization in `phoonnx` is powered by:

* **[ovos-number-parser](https://github.com/OpenVoiceOS/ovos-number-parser)** – Expands numbers and fractions into words.
* **[ovos-date-parser](https://github.com/OpenVoiceOS/ovos-date-parser)** – Converts dates and times into spoken forms.
* **[unicode-rbnf](https://github.com/Elvenson/unicode-rbnf)** – Fallback for language-specific number formatting rules.
* Custom mappings for contractions, titles, and units.

**What Happens in Normalization**

1. **Dates & Times**

   * Detects and expands dates (`08/03/2025` → `eighth of March twenty twenty five`).
   * Converts times to spoken forms (`19h30` → `nineteen thirty`).

2. **Numbers & Fractions**

   * Expands numbers (`123` → `one hundred twenty three`).
   * Handles locale-specific decimal/thousands separators:

     * English: `1,234.56` → `one thousand two hundred thirty four point five six`
     * Portuguese/Spanish/French/German: `1.234,56` → `mil duzentos e trinta e quatro vírgula cinquenta e seis`
   * Expands fractions (`3/4` → `three quarters`).

3. **Units & Symbols**

   * Converts units and symbols into words (`25ºC` → `twenty five degrees celsius`, `5kg` → `five kilograms`).

4. **Contractions & Titles**

   * Expands contractions (`I’m` → `I am`, `won’t` → `will not`).
   * Expands titles (`Dr.` → `Doctor`, `Sr.` → `Senhor`, `Mme` → `Madame`).

5. **Hyphenated Words with Digits**

   * Fixes cases like `sub-23` → `sub 23`.

6. **Language Awareness**

   * Uses the provided `--language` code to decide rules.
   * If the code isn’t exact, `phoonnx` uses the [`langcodes`](https://pypi.org/project/langcodes/) library to map it to a valid phonemizer language.

**Example**

Input:

```
"I'm Dr. Prof. 3/3 0.5% of 12345€, 5ft, and 10kg"
```

Normalized (English):

```
"I am Doctor Professor three thirds zero point five per cent of twelve thousand three hundred forty five euros five feet and ten kilograms"
```

👉 This normalization step runs automatically inside **`preprocess.py`** before phonemization, so you don’t need to do it manually.

---

## 6. Understanding `config.json`

The `config.json` file stores dataset and training parameters. A typical example looks like this:

```json
{
  "audio": {
    "sample_rate": 22050,
    "quality": "medium"
  },
  "lang_code": "en-us",
  "inference": {
    "noise_scale": 0.667,
    "length_scale": 1,
    "noise_w": 0.8,
    "add_diacritics": false
  },
  "alphabet": "ipa",
  "phoneme_type": "espeak",
  "phonemizer_model": "",
  "phoneme_id_map": { ... },
  "num_symbols": 133,
  "num_speakers": 1,
  "speaker_id_map": {},
  "phoonnx_version": "0.1.0"
}
```

### Key Fields

* **audio.sample_rate**: The training/inference sample rate.
* **audio.quality**: Arbitrary label (taken from `--audio-quality` or output folder name).
* **lang_code**: Language code used for phonemization.
  * Flexible format, normalized with `langcodes`.
  * Example: `en`, `en-US`, or `eng` will all resolve correctly.
* **inference**: Default inference-time parameters.
  * `noise_scale`: Controls variability in speech.
  * `length_scale`: Controls speech rate.
  * `noise_w`: Additional noise parameter.
  * `add_diacritics`: Whether to apply diacritics during inference.
    * Only meaningful for **Hebrew (phonikud)** and **Arabic (tashkeel)**.
* **alphabet**: The phoneme alphabet.
  * Depends on the phonemizer and phoneme type.
  * Typical values: `"ipa"`, `"unicode"`, `"arpa"`, `"pinyin"`.
* **phoneme_type**: Which phonemizer was used (`espeak`, `gruut`, `byt5`, etc.).
* **phonemizer_model**: Only applies to **ByT5-based phonemizers**.
* **phoneme_id_map**: Mapping from phoneme symbols to numeric IDs.
* **num_symbols**: Total number of symbols in the phoneme map.
* **num_speakers**: Number of speakers (1 for single-speaker datasets).
* **speaker_id_map**: Mapping of speaker labels to IDs (for multi-speaker datasets).
* **phoonnx_version**: Version of the preprocessing pipeline.

---

## 7. Tips

* Use `--debug` to troubleshoot preprocessing.
* Always match `--sample-rate` to your dataset’s audio files.
* For multi-speaker datasets, ensure `metadata.csv` includes speaker IDs.
* Consider `--quality high` for production voices, but train longer.
* Use GPUs for training; CPU training is not practical.
