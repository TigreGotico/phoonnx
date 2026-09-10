# ZipVoice (k2-fsa) — flow-matching zero-shot cloning

ZipVoice is a 123M flow-matching TTS with **in-context** voice cloning. It is phoonnx's
first **iterative** engine: a short ODE loop over a flow-matching vector field, rather
than a single static `tokens → graph → audio`. Cloning is in-context (reference audio +
its transcription are model inputs) — there is **no speaker encoder / d-vector**, so it
does not use the `speaker_encoders` registry.

`infer_zipvoice_onnx.py` is the **validated standalone reference** — the phoonnx-native
engine is ported against it.

## Models (all on HF, standalone ONNX)
- `k2-fsa/ZipVoice` — `zipvoice/text_encoder.onnx`, `zipvoice/fm_decoder.onnx`,
  `tokens.txt` (360-token espeak+pinyin vocab); `zipvoice_distill/` for the 4–8 step variant.
- `jasonzhang76/zipvoice` — `vocos_24khz.onnx` (the Vocos vocoder head).

## Pipeline
```
text       → espeak (piper_phonemize / pypinyin) → tokens.txt ids
prompt wav → 24kHz → rms-norm to target_rms(0.1) → mel*feat_scale(0.1)
text_encoder(tokens, prompt_tokens, prompt_len, speed) → text_condition[1,T,100]
x = N(0,1)[1,T,100];  speech_condition = prompt mel in the prefix, 0 after
for i in range(num_step):                              # 16 (4–8 for distill)
    v = fm_decoder(t=linspace(0,1,n+1)[i], x, text_condition, speech_condition, guidance)
    x = x + v*(t[i+1]-t[i])                            # forward Euler; CFG is internal
target = x[:, prompt_len:] / feat_scale → vocos → ISTFT(mag*(x+iy)) → audio
```

## The two gotchas (get either wrong → quiet, range-compressed hiss)
- **`target_rms = 0.1`** — the prompt wav is RMS-normalized before the mel; the output
  is scaled back to the prompt's loudness.
- **`feat_scale = 0.1`** — the model works in scaled feature space: multiply the prompt
  mel by it, and **divide the generated mel by it before the vocoder**.

mel = `log(clamp(MelSpectrogram(24kHz, n_fft 1024, hop 256, n_mels 100, power=1), 1e-7))`.

## Validation
Clone of an English reference renders coherent speech: generated mel range
`[-13.4, 4.6]` (matches the reference's full dynamic range) and frame-energy
std/mean `0.66` (in the speech range). See `[[voice-cloning-tts]]` for the candidate
context.

## Native engine (next)
A `ZipVoiceAdapter` reimplements the mel without torch (the conv1d-DFT trick from
`scripts/conversion/yourtts/export_speaker_encoder.py`), runs the 3-ONNX + ODE loop, and
feeds the Vocos vocoder via the existing vocoder registry. Tokenizer = phoonnx's espeak +
pypinyin phonemizers against `tokens.txt`.
