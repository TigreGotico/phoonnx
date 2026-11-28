# YourTTS conversion

YourTTS is a multilingual VITS conditioned on an external 512-d speaker **d-vector**
(not a speaker-id `emb_g`) + a language id — the basis for zero-shot voice cloning.

## VITS model → ONNX
`export_vits_dvector.py` extends the coqui VITS exporter with the d-vector + langid
path: the ONNX takes `input, input_lengths, scales, d_vector(1,512), langid` and
returns a waveform. Runs on the `YourTTSAdapter` (`Engine.YOURTTS`).

## Speaker encoder → ONNX
`export_speaker_encoder.py` exports the coqui ResNet speaker encoder (`model_se.pth`)
to a single `waveform → 512-d d-vector` graph. torchaudio's complex STFT can't be
exported, so the mel front end is reimplemented as a conv1d against the windowed DFT
basis (`OnnxMel`) with the mel filterbank lifted verbatim from torchaudio, and the
`instance_norm` op is spelled out — both numerically identical (validated at cosine
**1.0000** vs the torchaudio reference). `resnet.py`/`base_encoder.py`/
`torch_transforms.py` are the upstream sources for reference.

## Runtime cloning
The encoder is registered in `phoonnx.engines.speaker_encoders` (mirrors the vocoder
registry; `coqui_resnet` type). A cloning voice carries `speaker_encoder_url` +
`speaker_encoder_type` in its index entry; the manager downloads it to
`engine_params["speaker_encoder_path"]`, and `YourTTSAdapter` clones from
`params["reference_audio"] = (audio, sample_rate)`:
`reference.wav → speaker_encoder.onnx → d_vector → YourTTS`. Validated end-to-end —
the encoder recovers the source speaker (cosine 0.56 vs 0.20 for a different speaker).
