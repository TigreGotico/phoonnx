# ito (Lokutor): phoonnx export

Converts a Lokutor [ito](https://github.com/lokutor-ai/ito) voice checkpoint
into the single ONNX graph phoonnx's `ito` engine consumes. The model is a
4.4M-parameter distilled English front plus vocoder at 24 kHz.

The export wraps the `ito` package (`pip install ito-tts`). That package is an
export-time dependency only. Nothing is vendored, and no ito code enters
phoonnx or the exported artifacts.

The ONNX graph re-expresses only the ONNX-unfriendly pieces (length
regulation, the harmonic source, STFT/iSTFT) with standard ops. See
`ito_onnx_graph.py` for the numerics notes.

## The graph

```
input  int64  [1, L]   ito token ids, leading pad included
speed  float  [1]      duration scale (1.0 = the trained pace)
wav    float  [1, T*hop]
durations int64 [1, L] frames per token (pad included)
mel    float  [1, 100, T]
f0     float  [1, T]
```

The style vector and the excitation-noise buffer are baked into the graph. A
voice ships as one `model.onnx` plus its `config.json`, and no sidecar
artifacts exist.

The front end (token ids) is `phoneme_type: "styletts2"`. scriptconv's
StyleTTS2 phonemizer reproduces ito's espeak + punctuation + nltk pipeline,
and the ito symbol table rides in the config's `phoneme_id_map`.

## Weights and licensing

The voice weights are **CC BY-NC-SA 4.0 + Lokutor's terms** (non-commercial)
and live on a gated HuggingFace repository (`lokutor-ai/ito`).

Accept the terms there and run `hf auth login` before exporting. The
exported artifacts inherit the license. Do not mirror them without
Lokutor's terms.

## Usage

```bash
pip install ito-tts            # export-time dependency (GPL-3.0, tools only)
python export_ito_onnx.py --ckpt ito_female.pt --out-dir ./ito_female \
    --check-parity --wav parity.wav
```

`--check-parity` synthesizes sentences through the ito PyTorch reference
and through a float32 twin of the export graph. It then reports the ONNX
waveform's SNR against both.

Expect more than 65 dB against the float64 reference
(`ito.synth.Chain.infer_full`). The residual is float32 kernel arithmetic.
The chip engine itself is int8/int16-quantized against this same float
reference, so this is well inside the model's own tolerance. A wrong export
lands orders of magnitude lower and fails the check.
