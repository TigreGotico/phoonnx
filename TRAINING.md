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
* `--single-speaker`: Treat dataset as single speaker.
* `--speaker-id`: Manually assign ID for single-speaker training.
* `--phoneme-type`: Phoneme system (`espeak`, `gruut`, `byt5`, etc.).
* `--alphabet`: Alphabet (`ipa`, `unicode`, `arpa`, `pinyin`, depending on phonemizer).
* `--phonemizer-model`: Optional pretrained model (currently applies only to **ByT5-based phonemizers**).
* `--text-casing`: Adjust text casing (`lower`, `upper`, `casefold`).
* `--skip-audio`: Skip audio normalization (for text-only runs).
* `--add-diacritics`: Add diacritics (only meaningful for **Hebrew (phonikud)** and **Arabic (tashkeel)**).
* `--debug`: Verbose logging.

This step produces:

* `config.json`: Model + dataset configuration (see below).
* `dataset.jsonl`: Normalized utterances with phoneme IDs and audio references.
* Cached normalized audio + spectrograms (in `cache/`).

---

## 2. Understanding `config.json`

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

## 3. Training a Model

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

## 4. Exporting to ONNX

After training, export the model to ONNX for efficient inference.

```bash
python export_onnx.py \
  checkpoints/epoch=500-step=100000.ckpt \
  model.onnx
```

---

## 5. Workflow Summary

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

## 6. Tips

* Use `--debug` to troubleshoot preprocessing.
* Always match `--sample-rate` to your dataset’s audio files.
* For multi-speaker datasets, ensure `metadata.csv` includes speaker IDs.
* Consider `--quality high` for production voices, but train longer.
* Use GPUs for training; CPU training is not practical.
