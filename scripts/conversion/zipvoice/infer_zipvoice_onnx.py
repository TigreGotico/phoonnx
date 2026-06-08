"""ZipVoice zero-shot voice cloning — standalone ONNX inference (validated reference).

ZipVoice (k2-fsa) is a 123M flow-matching TTS with **in-context** cloning: the
reference audio + its transcription are part of the model input, and the model
continues that voice. Unlike the d-vector engines (YourTTS, StyleTTS2) there is no
speaker encoder — the conditioning is the prompt mel + prompt tokens.

This is phoonnx's first **iterative** engine: a short ODE loop over a flow-matching
vector field, rather than a single static graph. Three ONNX (all on HF,
``k2-fsa/ZipVoice`` + ``jasonzhang76/zipvoice`` for the vocoder):

  text_encoder : tokens, prompt_tokens, prompt_features_len, speed -> text_condition[1,T,100]
  fm_decoder   : t, x[1,T,100], text_condition, speech_condition, guidance_scale -> v   (the vector field; CFG is internal)
  vocos_24khz  : mels[1,100,T] -> mag, x, y   (ISTFT components)

The two normalization constants are the gotcha — get either wrong and the output is a
quiet, range-compressed hiss:
  * the prompt wav is RMS-normalized to ``target_rms`` (0.1) before the mel;
  * the model works in feature space scaled by ``feat_scale`` (0.1): the prompt mel is
    multiplied by it, and the generated mel must be divided by it before the vocoder.

mel: torchaudio MelSpectrogram(24kHz, n_fft 1024, hop 256, n_mels 100, power=1) then
``log(clamp(x, 1e-7))``. Time steps: ``linspace(0, 1, num_step+1)``. Default num_step 16
(ZipVoice-Distill works at 4-8).

A phoonnx-native engine reimplements the mel without torch (the conv1d-DFT trick from
the speaker-encoder export) and runs the loop in a multi-ONNX adapter; this script is
the validated reference it is ported against.

Usage:
  python infer_zipvoice_onnx.py --prompt-wav ref.wav --prompt-text "..." \
      --text "text to speak in the cloned voice" --out clone.wav
"""
import argparse

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import wave
from huggingface_hub import hf_hub_download
import piper_phonemize

FEAT_SCALE = 0.1
TARGET_RMS = 0.1
REPO = "k2-fsa/ZipVoice"


def load_tokens():
    t2i = {}
    for line in open(hf_hub_download(REPO, "zipvoice/tokens.txt")):
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 2:
            t2i[parts[0]] = int(parts[1])
    return t2i


def tokenize(text, t2i, lang="en-us"):
    return [t2i[ph] for word in piper_phonemize.phonemize_espeak(text, lang)
            for ph in word if ph in t2i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-wav", required=True)
    ap.add_argument("--prompt-text", required=True, help="transcription of the prompt wav")
    ap.add_argument("--text", required=True, help="text to speak in the cloned voice")
    ap.add_argument("--out", default="clone.wav")
    ap.add_argument("--model", default="zipvoice", choices=["zipvoice", "zipvoice_distill"])
    ap.add_argument("--num-step", type=int, default=16)
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--lang", default="en-us")
    args = ap.parse_args()

    sess = lambda f, r=REPO: ort.InferenceSession(hf_hub_download(r, f), providers=["CPUExecutionProvider"])
    text_encoder = sess(f"{args.model}/text_encoder.onnx")
    fm_decoder = sess(f"{args.model}/fm_decoder.onnx")
    vocos = sess("vocos_24khz.onnx", "jasonzhang76/zipvoice")
    t2i = load_tokens()

    to_mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100, center=True, power=1)

    def mel_of(wav):
        return to_mel(torch.from_numpy(wav).float()).clamp(min=1e-7).log().T.unsqueeze(0).numpy()

    # prompt wav -> 24kHz -> rms-normalize -> mel * feat_scale
    pw, sr = torchaudio.load(args.prompt_wav)
    pw = pw.mean(0).numpy()
    if sr != 24000:
        pw = torchaudio.functional.resample(torch.from_numpy(pw), sr, 24000).numpy()
    prompt_rms = float(np.sqrt((pw ** 2).mean()))
    pw = pw * TARGET_RMS / prompt_rms
    prompt_mel = mel_of(pw) * FEAT_SCALE
    t_ref = prompt_mel.shape[1]

    tokens = np.array([tokenize(args.text, t2i, args.lang)], np.int64)
    prompt_tokens = np.array([tokenize(args.prompt_text, t2i, args.lang)], np.int64)

    text_condition = text_encoder.run(None, {
        "tokens": tokens, "prompt_tokens": prompt_tokens,
        "prompt_features_len": np.array(t_ref, np.int64),
        "speed": np.array(args.speed, np.float32)})[0]
    num_frames = text_condition.shape[1]

    # flow matching: x_0 ~ N(0,1) integrated to x_1 = data, conditioned on the prompt mel
    x = np.random.randn(1, num_frames, 100).astype(np.float32)
    speech_condition = np.zeros((1, num_frames, 100), np.float32)
    speech_condition[:, :t_ref] = prompt_mel
    steps = np.linspace(0, 1, args.num_step + 1).astype(np.float32)
    for i in range(args.num_step):
        v = fm_decoder.run(None, {
            "t": np.array(steps[i], np.float32), "x": x,
            "text_condition": text_condition, "speech_condition": speech_condition,
            "guidance_scale": np.array(args.guidance_scale, np.float32)})[0]
        x = x + v * (steps[i + 1] - steps[i])

    # target slice -> /feat_scale -> vocoder (mag * (x + i*y)) -> ISTFT
    target_mel = (x[:, t_ref:] / FEAT_SCALE).transpose(0, 2, 1).astype(np.float32)
    mag, vx, vy = vocos.run(None, {"mels": target_mel})
    spec = torch.from_numpy(mag) * (torch.from_numpy(vx) + 1j * torch.from_numpy(vy))
    audio = torch.istft(spec, n_fft=1024, hop_length=256, win_length=1024,
                        window=torch.hann_window(1024), center=True).numpy().reshape(-1)
    audio = audio * prompt_rms / TARGET_RMS                       # restore prompt loudness
    audio = np.clip(audio / max(np.abs(audio).max(), 1e-6) * 0.95, -1, 1)

    with wave.open(args.out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes((audio * 32767).astype("<i2").tobytes())
    print(f"cloned {len(audio) / 24000:.2f}s -> {args.out}")


if __name__ == "__main__":
    main()
