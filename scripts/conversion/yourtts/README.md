# YourTTS conversion

YourTTS is a multilingual VITS conditioned on an external 512-d speaker **d-vector**
(not a speaker-id `emb_g`) + a language id — the basis for zero-shot voice cloning.

## VITS model → ONNX  (done)
`export_vits_dvector.py` extends the coqui VITS exporter with the d-vector + langid
path: the ONNX takes `input, input_lengths, scales, d_vector(1,512), langid` and
returns a waveform. Runs on the `YourTTSAdapter` (`Engine.YOURTTS`), which feeds the
d-vector either bundled (fixed voice, via `engine_params["d_vector"]`) or per-request
(cloning).

## Speaker encoder → ONNX  (TODO — unlocks runtime cloning)
The model ships `model_se.pth`, a coqui **ResNet** speaker encoder (`resnet.py` +
`base_encoder.py` + `torch_transforms.py` vendored here). It maps a 16 kHz reference
waveform → 64-mel (in-graph `torch_spec`, `use_torch_spec=True`) → ResNet + ASP → a
512-d L2-normalised d-vector. Exporting it (wav → d-vector, opset ≥17 for the STFT, or
mel → d-vector with the mel computed in phoonnx) gives runtime zero-shot cloning:
`reference.wav → encoder.onnx → d_vector → YourTTSAdapter`. Must be validated against
the model's `speakers.json` ground-truth embeddings before shipping.
