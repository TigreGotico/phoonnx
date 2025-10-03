# Training Guide for **phoonnx**

This document explains how to prepare data, train models, and export them to ONNX for inference.

---

## 1. Preprocessing Data

Before training, you need to preprocess your dataset into a format compatible with `phoonnx`.

```bash
python preprocess.py \
  --input-dir /path/to/dataset \
  --output-dir /path/to/output \
  --language en-us \
  --sample-rate 22050
```

### Supported Options

* `--input-dir`: Path to your dataset (must contain `metadata.csv` in [LJSpeech format](https://keithito.com/LJ-Speech-Dataset/)).
* `--output-dir`: Directory for processed files (`config.json`, `dataset.jsonl`).
* `--language`: Language code (e.g., `en-us`).
    > The language code is passed to the phonemizer. `phoonnx` uses the [langcodes](https://pypi.org/project/langcodes/) library internally to normalize and “correct” the code if needed.
* `--sample-rate`: Target audio sample rate (e.g., `22050`).
* `--cache-dir`: Optional directory to store cached processed audio files (defaults to `<output-dir>/cache/<sample-rate>`).
* `--max-workers`: Maximum number of multiprocessing workers to use.
* `--single-speaker`: Treat the dataset as **single speaker** regardless of `metadata.csv` contents. **Cannot** be used with `--speaker-id`.
* `--speaker-id`: Manually assign a **numeric ID** for single-speaker training (only used if the dataset is single-speaker). **Cannot** be used with `--single-speaker`.
* `--phoneme-type`: Phoneme system (`espeak`, `gruut`, `byt5`, etc.). (Default: `espeak`).
* `--alphabet`: Phoneme alphabet (`ipa`, `unicode`, `arpa`, `pinyin`, etc.). The choices depend on the selected phonemizer. (Default: `ipa`).
* `--phonemizer-model`: Optional pretrained model (currently applies only to **ByT5-based phonemizers**).
* `--text-casing`: Adjust text casing **before normalization** (`ignore`, `lower`, `upper`, `casefold`). (Default: `ignore`).
* `--skip-audio`: Skip audio normalization and caching (for text-only runs).
* `--add-diacritics`: Add diacritics to text **after normalization** but **before phonemization**. (Only meaningful for **Hebrew (phonikud)** and **Arabic (tashkeel)**, depending on the phonemizer).
* `--dataset-name`: Name of dataset to put in `config.json`.
* `--audio-quality`: Audio quality label to put in `config.json`.
* `--debug`: Verbose logging.

This step produces:

* `config.json`: Model + dataset configuration (see below).
* `dataset.jsonl`: Normalized utterances with phoneme IDs and audio references.
* Cached normalized audio + spectrograms (in `cache/`).


Perfect — we should definitely explain the **normalization step** clearly in the training guide since it’s a key part of preprocessing. Based on the code you shared, here’s how I’d add it into the **TRAINING.md**:

---

## 2. Training a Model

Train a [VITS](https://arxiv.org/abs/2106.06103)-style model using PyTorch Lightning.

```bash
python -m phoonnx \
  --dataset-dir /path/to/output \
  --quality medium \
  --max_epochs 500 \
  --gpus 1
```

### Options

* `--dataset-dir`: Path containing `config.json` and `dataset.jsonl`.
* `--quality`: Model size (`x-low`, `medium`, `high`).
* `--checkpoint-epochs`: Save checkpoints every *N* epochs.
* `--resume_from_checkpoint`: Resume from a previous run.
* `--resume_from_single_speaker_checkpoint`: Convert single-speaker checkpoint to multi-speaker training.
* `--seed`: Random seed (default `1234`).

PyTorch Lightning arguments are also supported (e.g., `--max_epochs`, `--accelerator gpu`, etc.).

---

## 3. Exporting to ONNX

After training, export the model checkpoint (`.ckpt`) to the ONNX format for efficient, cross-platform inference.

```bash
python export_onnx.py \
  checkpoints/epoch=500-step=100000.ckpt \
  --config /path/to/output/config.json \
  --output-dir ./exported \
  --generate-tokens \
  --piper
```

### Options

* **Positional Argument: CHECKPOINT**
    * Path to the PyTorch checkpoint file (`.ckpt`).
* `-c`, `--config`: Path to the model configuration JSON file (`config.json`). **Required** for metadata and token map.
* `-o`, `--output-dir`: Output directory for the ONNX file and associated assets.
* `-t`, `--generate-tokens`: Generate a **`tokens.txt`** file alongside the ONNX model. Required by some inference engines (e.g., Sherpa).
* `-p`, `--piper`: Generate a Piper-compatible **`.json`** file alongside the ONNX model, setting appropriate metadata flags.

-----

## 4. Workflow Summary

1. **Prepare dataset** in LJSpeech-style format.
2. **Preprocess**:

   ```bash
   python preprocess.py --input-dir ... --output-dir ... --language en-us --sample-rate 22050
   ```
3. **Train**:

   ```bash
   python -m phoonnx --dataset-dir ... --quality medium --max_epochs 500
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
