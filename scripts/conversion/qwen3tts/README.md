# Qwen3-TTS conversion

Two scripts: one exports the graphs, one proves the export matches the source model.

```bash
pip install qwen-tts torch onnx onnxruntime onnxscript
python export_qwen3tts_onnx.py --out ./onnx
python verify_parity.py --onnx ./onnx
```

`export_qwen3tts_onnx.py` writes the seven graphs phoonnx loads plus `tokenizer.json`.
It checks three of its own assumptions before it exports anything: that mRoPE reduces to
plain RoPE at batch 1, that the hand-built causal masks match the stock model, and that
the hand-built sliding-window mask matches transformers.

`verify_parity.py` runs one greedy synthesis through both the PyTorch model and the
phoonnx adapter and reports prompt, logit, token and waveform differences. Greedy is the
only comparable setting: with sampling the two sides draw from different random streams.

Source model: [`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice), Apache-2.0.
Mirror: [`OpenVoiceOS/phoonnx-qwen3-tts`](https://huggingface.co/OpenVoiceOS/phoonnx-qwen3-tts).
