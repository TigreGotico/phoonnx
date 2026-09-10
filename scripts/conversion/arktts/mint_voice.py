"""Turn a reference clip into an ArkTTS voice asset.

    python mint_voice.py --repo Audio8/Audio8-TTS-Preview-0.6b \
        --audio antton.wav --text "Inguru hura gerrillarien esku izan da denbora luzean." \
        --name antton --out voices/antton.json

    python mint_voice.py --from-npy codes.npy --text "..." --name maider \
        --out voices/maider.json

An ArkTTS voice is not a speaker id — the model has no speaker table and always emits
``<|speaker:0|>``. A voice is the codec codes of a short clip plus the transcription of
that clip, and the prompt embeds both. This script produces the small JSON the engine
reads, either by encoding a clip with the checkpoint's codec encoder or by repackaging
codes that upstream already published.

The encoder runs here and never ships: ``phoonnx`` only needs the codec *decoder* at
synthesis time, because voices arrive pre-encoded. Minting a new voice is therefore an
offline step, not something the runtime does.

Keep the clip short and prosodically flat. Upstream selected its own references by
pitch-range ratio precisely because an expressive reference bleeds its intonation into
every sentence the voice later speaks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NUM_CODEBOOKS = 10
CODEBOOK_SIZE = 4096


def encode_clip(repo: str, audio_path: Path) -> np.ndarray:
    """Run the checkpoint's codec encoder over one clip and return ``[10, frames]``."""
    import soundfile
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(repo, dtype=torch.float32, trust_remote_code=True).eval()
    samples, rate = soundfile.read(str(audio_path), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    target = int(model.config.codec_sample_rate)
    if rate != target:
        # scipy rather than torchaudio: the codec is the only thing that needs torch here,
        # and torchaudio pins its own build against a matching torch, which is a heavy
        # constraint to inherit for one polyphase resample.
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(rate), target)
        mono = resample_poly(mono, target // divisor, int(rate) // divisor).astype("float32")
    audio = torch.from_numpy(mono)
    with torch.inference_mode():
        codes, lengths = model.encode_audio(
            audio.reshape(1, 1, -1), torch.tensor([audio.numel()]))
    return codes[0, :, : int(lengths[0])].cpu().numpy().astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="checkpoint whose codec encodes the clip")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--from-npy", type=Path, help="repackage published codes instead")
    parser.add_argument("--text", required=True, help="what the reference clip says")
    parser.add_argument("--name", required=True)
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--provenance", default="", help="where the clip came from and its licence")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.from_npy:
        codes = np.load(args.from_npy).astype(np.int64)
    elif args.repo and args.audio:
        codes = encode_clip(args.repo, args.audio)
    else:
        raise SystemExit("give either --from-npy or both --repo and --audio")

    if codes.ndim != 2 or codes.shape[0] != NUM_CODEBOOKS or codes.shape[1] == 0:
        raise SystemExit(f"codes must have shape [{NUM_CODEBOOKS}, T>0], got {codes.shape}")
    if codes.min() < 0 or codes.max() >= CODEBOOK_SIZE:
        raise SystemExit(f"codes must be in [0, {CODEBOOK_SIZE - 1}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "name": args.name,
        "display_name": args.display_name or args.name.title(),
        "reference_text": " ".join(args.text.split()),
        "frames": int(codes.shape[1]),
        "provenance": args.provenance,
        "codes": codes.tolist(),
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out} — {codes.shape[1]} frames "
          f"({codes.shape[1] * 2048 / 44100:.2f} s of reference)")


if __name__ == "__main__":
    main()
