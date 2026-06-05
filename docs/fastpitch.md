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

coqui's `ForwardTTS` (`fast_pitch`) does **not** export to a dynamic-length ONNX
out of the box: its `FFTransformer` wraps `torch.nn.MultiheadAttention`, which
**bakes the example sequence length** under the legacy tracer (`dynamo=False`),
while `dynamo=True` chokes on the data-dependent duration→length expansion.

The fix (validated): in a vendored copy, replace the stock `nn.MultiheadAttention`
with a tracer-safe drop-in — same `in_proj_weight`/`in_proj_bias`/`out_proj`
parameter names (so pretrained weights load unchanged), but reshaping with `-1`
and live `tensor.shape` reads (`x.reshape(x.shape[0], -1, head_dim)`) the way
nipponjo's hand-written FastPitch attention does. With that single swap a plain
`torch.onnx.export(..., dynamo=False, opset=14)` yields a fully dynamic-length
graph. The en/ljspeech model is mirrored as `coqui/en-ljspeech-fast_pitch`.

## References
- [FastPitch paper](https://arxiv.org/abs/2006.06873) · [tts_arabic](https://github.com/nipponjo/tts_arabic)
- [docs/mixertts.md](./mixertts.md) · [docs/vocoders.md](./vocoders.md)
