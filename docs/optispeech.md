# OptiSpeech Engine

OptiSpeech is a non-autoregressive, **FastSpeech2-style** TTS: an acoustic model
with explicit duration / pitch / energy prediction and a GAN vocoder, exported
as a **single ONNX file** that outputs the waveform directly (no separate
vocoder stage).

## Inference

OptiSpeech uses a different ONNX I/O contract from VITS, and it embeds **all of
its config inside the ONNX metadata** (no external config file).

### ONNX Inputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| ``x`` | int64 | ``[B, T]`` | Phoneme IDs |
| ``x_lengths`` | int64 | ``[B]`` | Sequence lengths |
| ``scales`` | float32 | ``[3]`` | ``[d_factor, p_factor, e_factor]`` |
| ``sids`` | int64 | ``[B]`` | Speaker IDs (optional, multi-speaker) |
| ``lids`` | int64 | ``[B]`` | Language IDs (optional, multi-language) |

### ONNX Outputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| ``wav`` | float32 | ``[B, 1, T]`` | Waveform |
| ``wav_lengths`` | int64 | ``[B]`` | Sample counts |
| ``durations`` | int64 | ``[B, T_text]`` | Per-phoneme durations |

> Note the ``scales`` tensor differs from VITS, where it packs
> ``[noise_scale, length_scale, noise_w_scale]``. OptiSpeech shares the
> ``x``/``x_lengths``/``scales`` input names with Matcha-TTS, so the adapter is
> probed **before** Matcha — it is identified by its embedded metadata and
> ``wav``/``durations`` outputs.

### Parameters

| Param | Default | Description |
|-------|---------|-------------|
| ``d_factor`` | 1.0 | Duration scale — speaking rate (>1 slower, <1 faster) |
| ``p_factor`` | 1.0 | Pitch scale |
| ``e_factor`` | 1.0 | Energy scale |

Pass them per call via ``SynthesisConfig.extra_params``::

    voice.synthesize("hello", SynthesisConfig(extra_params={"d_factor": 0.9}))

## Config — embedded metadata → native config

OptiSpeech stores its config inside the ONNX model under the ``inference``
metadata key (a JSON blob): ``sample_rate``, ``input_symbols`` (symbol→id),
``special_symbols``, ``text_processor`` (``add_blank`` / ``add_bos_eos`` /
``tokenizer``), ``speakers`` and ``languages``.

``phoonnx.engines.optispeech_config.voice_config_from_optispeech_meta`` converts
that metadata into a native phoonnx ``VoiceConfig`` (folding ``input_symbols``
into the tokenizer ``phoneme_id_map`` and honouring the ``add_blank`` /
``add_bos_eos`` flags), e.g.::

    import onnxruntime as ort
    from phoonnx.engines.optispeech import OptiSpeechAdapter
    from phoonnx.engines.optispeech_config import voice_config_from_optispeech_meta

    sess = ort.InferenceSession("model.onnx")
    meta = OptiSpeechAdapter().parse_onnx_meta(sess)
    config = voice_config_from_optispeech_meta(meta)

The mirrored voices (below) ship this as a **native ``config.json``** alongside
the model, so they load through the standard path — ``engine: optispeech`` in
the config routes to ``OptiSpeechAdapter``, with no runtime metadata extraction.

## Voice index

OptiSpeech voices ship in ``phoonnx/voice_index/optispeech.json``, mirrored under
``OpenVoiceOS/phoonnx-optispeech`` (each subfolder = ``model.onnx`` + native
``config.json``). They are loaded by the voice manager like any other voice:

```python
from phoonnx.model_manager import TTSModelManager

m = TTSModelManager(); m.merge_default_voices()
voice = m.voices["hf_community/mush42/optispeech-lightspeech-en-us-emily"].load()
for chunk in voice.synthesize("Hello from OptiSpeech."):
    ...   # chunk.audio_float_array
```

Current entries (mirror of [`mush42/optispeech`](https://huggingface.co/mush42/optispeech), 24 kHz, en-US):

| voice_id | acoustic | speaker |
|----------|----------|---------|
| `…/optispeech-lightspeech-en-us-emily` | LightSpeech | emily |
| `…/optispeech-lightspeech-en-us-mike` | LightSpeech | mike |
| `…/optispeech-convnext-en-us-emily` | ConvNeXt-TTS | emily |

## Text processing

OptiSpeech's IPA tokenizer phonemizes with espeak (``phoneme_type: espeak``,
``alphabet: ipa``). The tokenizer flags come from the model's ``text_processor``
metadata (e.g. these LightSpeech/ConvNeXt voices use ``add_blank: false`` /
``add_bos_eos: false``), so phoonnx feeds the exact token sequence the model was
trained on.

## References

- [OptiSpeech](https://github.com/mush42/optispeech)
- [docs/engines.md](./engines.md) — the engine adapter framework
