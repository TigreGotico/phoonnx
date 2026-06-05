# FastPitch Engine

[FastPitch](https://arxiv.org/abs/2006.06873) (NVIDIA) is a non-autoregressive,
FastSpeech2-style acoustic model: text → 80-channel mel with pace/pitch control.
Its exported ONNX inference contract is **identical to [Mixer-TTS](./mixertts.md)**
(`token_ids` + `pace`/`speaker`/`pitch_mul`/`pitch_add` → `mel_spec`), so
`FastPitchAdapter` subclasses `MixerTTSAdapter` and reuses its feed/parse logic;
the two are told apart by the native config `engine` field (`fastpitch`).

Two-stage — the mel is voiced by a separate vocoder ([vocoders.md](./vocoders.md)).

## Voice index

FastPitch voices ship in `phoonnx/voice_index/fastpitch.json`, mirrored under
`OpenVoiceOS/phoonnx-fastpitch`. The reference model is the multi-speaker Arabic
FastPitch from [nipponjo/tts_arabic](https://github.com/nipponjo/tts_arabic)
(`mantoq`/buckwalter tokenization, paired with the matched `tts-arabic-hifigan`).

```python
from phoonnx.model_manager import TTSModelManager
m = TTSModelManager(); m.merge_default_voices()
voice = m.voices["nipponjo/tts-arabic-fastpitch"].load()
for chunk in voice.synthesize("السَّلامُ عَلَيكُم.", ): ...
```


## Converting coqui FastPitch to ONNX

coqui's `ForwardTTS` needs three fixes to export a faithful, **truly
dynamic-length** ONNX (done in a vendored copy; pretrained weights load
unchanged):

1. **Attention** — its `FFTransformer` wraps `torch.nn.MultiheadAttention`, which
   bakes the sequence length under the legacy tracer. Replace it with a tracer-safe
   drop-in (same `in_proj_weight`/`in_proj_bias`/`out_proj` names, `-1` + live-shape
   reshapes). Numerically identical (~2e-7).
2. **Dynamic mask** — `inference()` does `x_lengths = torch.tensor(x.shape[1:2])`,
   which freezes the input length as a constant, masking out every token past the
   export example. Replace with a live-shape all-ones mask
   (`torch.ones(B, 1, x.shape[1])`) — inference is a single un-padded sequence. This
   is what makes the mel length unbounded (L=600 → 3637 frames, linear).
3. **Tokenization** — build the vocab coqui's way: `phonemizer` field (gruut vs
   espeak emit different IPA) and `is_sorted=True` (sort the symbol set before id
   assignment). Both handled by `voice_config_from_coqui`.

Then `torch.onnx.export(..., dynamo=False, opset=14)` gives a dynamic graph.
en/ljspeech is mirrored as `coqui/en-ljspeech-fast_pitch` (gruut, matched
`coqui-ljspeech-hifigan-v2`). Exporter: `ml/tts/_exporters/coqui_fastpitch`.

## References
- [FastPitch paper](https://arxiv.org/abs/2006.06873) · [tts_arabic](https://github.com/nipponjo/tts_arabic)
- [docs/mixertts.md](./mixertts.md) · [docs/vocoders.md](./vocoders.md)
