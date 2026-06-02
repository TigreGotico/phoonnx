#!/usr/bin/env python3
"""
LoRA Voice Adaptation Demo: Argentine Spanish ("dii" voice)

Fine-tunes the phoonnx es-ES_dii_espeak voice on Argentine Spanish
from the Kukedlc/arg-spanish-tts dataset using LoRA.

Prerequisites:
  pip install datasets soundfile librosa
  pip install -e ".[train]"   # phoonnx with training deps
  pip install datasets==2.18.0  # avoid torchcodec issues

Steps:
  1. download   - download & preprocess dataset (resample 24kHz→22050Hz)
  2. preprocess  - phonemize with espeak (es-419)
  3. base-model  - download base model config
  4. extract-ckpt - create PyTorch checkpoint from config
  5. train       - LoRA training
  6. merge        - merge LoRA adapter + export ONNX
  7. demo         - synthesize demo audio

Usage:
  python scripts/lora_argentinian_demo.py --steps all
  python scripts/lora_argentinian_demo.py --steps download preprocess base-model extract-ckpt train merge demo
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK_DIR = Path(os.environ.get("LORA_WORK_DIR", "/mnt/homelab/Workspace/lora-arg-dii"))
DATASET_NAME = "Kukedlc/arg-spanish-tts"
BASE_VOICE_ID = "OpenVoiceOS/phoonnx_es-ES_dii_espeak"
SPEAKER_ID = "5223"

SAMPLE_RATE = 22050
LORA_SCOPE = "full-acoustic"
LORA_EPOCHS = 150
LORA_BATCH_SIZE = 8
HF_BASE_URL = "https://huggingface.co/OpenVoiceOS/phoonnx_es-ES_dii_espeak/resolve/main"


def download_dataset(output_dir: Path, speaker_id: str, max_samples: int = 0):
    import librosa
    import soundfile as sf
    import numpy as np
    from datasets import load_dataset

    print(f"[1/8] Downloading {DATASET_NAME} (speaker={speaker_id})...")
    ds = load_dataset(DATASET_NAME, split="train", keep_in_memory=True)

    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"

    count = 0
    rows = []
    for i in range(len(ds)):
        row = ds[i]
        sid = str(row["speaker_id"]).lstrip("0") if row["speaker_id"] else "0"
        if sid != speaker_id:
            continue
        if max_samples > 0 and count >= max_samples:
            break

        fname = f"arg_{count:05d}"
        wav_path = wavs_dir / f"{fname}.wav"

        try:
            audio = row["audio"]
            wav_array = np.array(audio["array"], dtype=np.float32)
            sr_orig = audio["sampling_rate"]
            if sr_orig != SAMPLE_RATE:
                wav_array = librosa.resample(wav_array, orig_sr=sr_orig, target_sr=SAMPLE_RATE)
            sf.write(str(wav_path), wav_array, SAMPLE_RATE)
        except Exception as e:
            short_msg = str(e).split("\n")[0][:80]
            print(f"  Skipping {i}: {short_msg}")
            continue

        text = row["text"].strip()
        rows.append(f"{fname}|{text}")
        count += 1
        if count % 50 == 0:
            print(f"  Extracted {count} samples...")

    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))

    print(f"  Extracted {count} samples for speaker {speaker_id}")
    return count


def phonemize_dataset(input_dir: Path, output_dir: Path, prev_config: Path):
    print(f"[2/8] Phonemizing dataset...")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "phoonnx_train.preprocess",
        "--language", "es-419",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--sample-rate", str(SAMPLE_RATE),
        "--phoneme-type", "espeak",
        "--alphabet", "ipa",
        "--single-speaker",
        "--prev-config", str(prev_config),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED:\n{result.stderr[-2000:]}")
        sys.exit(1)
    print(f"  Done: {output_dir}")


def download_base_model(output_dir: Path):
    print(f"[3/8] Downloading base model...")
    output_dir.mkdir(parents=True, exist_ok=True)
    import urllib.request

    for name in ["dii_es-ES.onnx", "dii_es-ES.json"]:
        url = f"{HF_BASE_URL}/{name}"
        dest = output_dir / name
        if dest.exists():
            print(f"  Already have {name}")
            continue
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(url, str(dest))

    return output_dir / "dii_es-ES.json"


def extract_base_checkpoint(config_path: Path, output_dir: Path):
    print(f"[4/8] Downloading original base checkpoint from HuggingFace...")
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / "base.ckpt"
    if ckpt_path.exists():
        print(f"  Already have {ckpt_path}")
        return ckpt_path

    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=BASE_VOICE_ID,
            filename="epoch=651-step=88672.ckpt",
        )
        import shutil
        shutil.copy2(downloaded, str(ckpt_path))
        print(f"  Downloaded checkpoint to {ckpt_path}")
    except Exception as e:
        print(f"  HF download failed ({e}), creating random base checkpoint instead (will sound worse)")
        import pytorch_lightning as pl
        import torch
        from phoonnx_train.vits.lightning import VitsModel
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        model = VitsModel(
            num_symbols=config.get("num_symbols", 256),
            num_speakers=config.get("num_speakers", 1),
            sample_rate=config.get("audio", {}).get("sample_rate", SAMPLE_RATE),
            dataset=None,
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": pl.__version__,
            },
            str(ckpt_path),
        )
        print(f"  Random checkpoint: {ckpt_path}")

    return ckpt_path


def train_lora(dataset_dir: Path, base_ckpt: Path, output_dir: Path,
               lora_scope: str, epochs: int, batch_size: int):
    print(f"[5-6/8] Training LoRA (scope={lora_scope}, epochs={epochs})...")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "phoonnx_train.train",
        "--dataset-dir", str(dataset_dir),
        "--resume-from-checkpoint", str(base_ckpt),
        "--lora-scope", lora_scope,
        "--max-epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--default-root-dir", str(output_dir),
        "--accelerator", "gpu",
        "--devices", "1",
        "--precision", "32",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  Training FAILED (exit code {result.returncode})")
        sys.exit(1)
    print(f"  Training complete.")


def find_lora_adapter(train_dir: Path) -> Path:
    adapter_dir = train_dir / "lora_adapter"
    adapter_path = adapter_dir / "lora_adapter.pt"
    if adapter_path.exists():
        return adapter_path
    raise FileNotFoundError(f"No LoRA adapter at {adapter_path}")


def merge_and_export(base_ckpt: Path, lora_adapter: Path, config_path: Path,
                     output_dir: Path, lora_scope: str):
    print(f"[7/8] Merging LoRA + ONNX export...")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "phoonnx_train.merge_lora",
        str(base_ckpt),
        "--lora-adapter", str(lora_adapter),
        "--config", str(config_path),
        "--lora-scope", lora_scope,
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  MERGE FAILED:\n{result.stderr[-2000:]}")
        sys.exit(1)
    print(f"  Done: {output_dir}")


def synthesize_demo(model_onnx: Path, config_json: Path, output_dir: Path):
    print(f"[8/8] Synthesizing demo audio...")
    output_dir.mkdir(parents=True, exist_ok=True)

    from phoonnx.voice import TTSVoice
    voice = TTSVoice.load(model_path=str(model_onnx), config_path=str(config_json))

    demo_texts = [
        "Hola, soy una voz argentina adaptada con LoRA.",
        "Buenos dias, como estas hoy?",
        "La economia argentina es compleja pero fascinante.",
        "Vamos a tomar un mate en la plaza.",
        "Este modelo fue entrenado con pocos minutos de audio.",
    ]

    import wave
    for i, text in enumerate(demo_texts):
        out_path = output_dir / f"demo_{i+1}.wav"
        with wave.open(str(out_path), "w") as wav_file:
            voice.synthesize_wav(text, wav_file)
        print(f"  {out_path}: \"{text}\"")

    print(f"\nDemo complete! Files in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Argentine Spanish voice adaptation demo")
    parser.add_argument("--steps", nargs="+", default=["all"],
                        choices=["all", "download", "preprocess", "base-model",
                                  "extract-ckpt", "train", "merge", "demo"])
    parser.add_argument("--work-dir", type=str, default=str(WORK_DIR))
    parser.add_argument("--speaker", type=str, default=SPEAKER_ID)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=LORA_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=LORA_BATCH_SIZE)
    parser.add_argument("--lora-scope", type=str, default=LORA_SCOPE,
                        choices=["generator-only", "full-acoustic", "aggressive"])
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    steps = args.steps
    if "all" in steps:
        steps = ["download", "preprocess", "base-model", "extract-ckpt",
                 "train", "merge", "demo"]

    raw_dir = work_dir / "raw_dataset"
    preprocessed_dir = work_dir / "preprocessed"
    base_model_dir = work_dir / "base_model"
    train_dir = work_dir / "lora_training"
    lora_output_dir = work_dir / "merged"
    demo_dir = work_dir / "demo_output"

    config_path = None
    base_ckpt = None

    if "download" in steps:
        download_dataset(raw_dir, args.speaker, args.max_samples)
    if "base-model" in steps:
        config_path = download_base_model(base_model_dir)
    else:
        config_path = base_model_dir / "dii_es-ES.json"

    if "preprocess" in steps:
        phonemize_dataset(raw_dir, preprocessed_dir, config_path)
    if "extract-ckpt" in steps:
        base_ckpt = extract_base_checkpoint(config_path, work_dir / "base_ckpt")
    else:
        base_ckpt = work_dir / "base_ckpt" / "base.ckpt"

    if "train" in steps:
        train_lora(preprocessed_dir, base_ckpt, train_dir,
                    args.lora_scope, args.epochs, args.batch_size)

    if "merge" in steps:
        lora_adapter = find_lora_adapter(train_dir)
        preprocessed_config = preprocessed_dir / "config.json"
        merge_and_export(base_ckpt, lora_adapter, preprocessed_config if preprocessed_config.exists() else config_path,
                         lora_output_dir, args.lora_scope)

    if "demo" in steps:
        model_onnx = lora_output_dir / "merged_model.onnx"
        model_config = lora_output_dir / "config.json"
        if not model_onnx.exists():
            print(f"ONNX not found at {model_onnx}. Run --steps merge first.")
            sys.exit(1)
        synthesize_demo(model_onnx, model_config, demo_dir)

    print("\nAll requested steps complete!")


if __name__ == "__main__":
    main()