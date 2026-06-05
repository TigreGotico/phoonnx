# Model conversion scripts

Standalone toolchains that convert upstream PyTorch checkpoints into the
phoonnx ONNX format. They vendor the **minimal pure-torch model code** needed for
inference + export (no `coqui-tts`/`TTS` package install required — it does not
install cleanly and drags in heavy deps).

These are developer tools, not part of the installed `phoonnx` package.

## `coqui_fastpitch_export/`

Converts coqui `tts_models/*/fast_pitch` (FastPitch / `ForwardTTS`) to a faithful,
**truly dynamic-length** ONNX. coqui's `ForwardTTS` needs three changes to export
correctly (all applied in the vendored copy; pretrained weights load unchanged):

1. **Attention** (`tsmha.py`) — `FFTransformer` wraps `torch.nn.MultiheadAttention`,
   which bakes the sequence length under the legacy ONNX tracer. `TracerSafeMHA`
   is a drop-in with the same `in_proj_weight`/`in_proj_bias`/`out_proj` parameter
   names (weights load as-is) but `-1` + live-shape reshapes. Numerically identical
   (~2e-7).
2. **Dynamic mask** (`forward_tts.py::inference`) — the original does
   `x_lengths = torch.tensor(x.shape[1:2])`, freezing the input length as a constant
   and masking out every token past the export example → the mel length caps.
   Replaced with a live-shape all-ones mask (`torch.ones(B, 1, x.shape[1])`);
   inference is a single un-padded sequence. This is what makes length unbounded
   (validated L=600 → 3637 mel frames, linear; onnx == torch to 1e-4).
3. **Tokenization** — handled phoonnx-side by `voice_config_from_coqui`: honor the
   config `phonemizer` field (gruut vs espeak emit different IPA) and sort the
   symbol set (coqui `Graphemes`/`IPAPhonemes` default `is_sorted=True`).

```bash
python export_fp.py <coqui_config.json> <coqui_model.pth> out.onnx
```

The matched `hifigan_v2` vocoder exports from `hifigan_generator.py` (mel → audio).

## `coqui_glowtts_export/`

Converts coqui `tts_models/*/glow-tts` (GlowTTS / flow). GlowTTS already takes
`input_lengths` as an explicit ONNX input, so it is dynamic without the mask fix;
the same `voice_config_from_coqui` tokenization rules (sort + phonemizer) apply.

## Licensing

The vendored model code under each exporter is derived from
[idiap/coqui-ai-TTS](https://github.com/idiap/coqui-ai-TTS) (**MPL-2.0**) and from
[nipponjo](https://github.com/nipponjo)'s FastPitch/Mixer attention. Those files
retain their upstream license; see `NOTICE`.
