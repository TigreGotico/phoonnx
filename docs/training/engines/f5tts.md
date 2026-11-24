# F5-TTS Engine

<!-- agent-hint
covers: how to load an F5-TTS, Habibi or SILMA voice, clone with a reference clip and its transcription, and set the dialect
source: phoonnx/engines/f5tts.py, phoonnx/voice_index/f5tts.json, phoonnx/opm.py
generated: no
verify: .venv/bin/python -c "import re,sys,json; import phoonnx.engines.f5tts as F; p=open('docs/training/engines/f5tts.md').read().replace(chr(96),''); d={k:float(v) for k,v in re.findall(r'^\| ([a-z_]+) \| (-?[\d.]+) \| [A-Z]',p,re.M)}; v=dict(re.findall(r'^\| ([a-z0-9]+/[\w-]+) \| (Phoonnx/[\w-]+) \|',p,re.M)); idx={k:re.search(r'co/([^/]+/[^/]+)',x['model_url']).group(1) for k,x in json.load(open('phoonnx/voice_index/f5tts.json')).items()}; m=dict(re.findall(r'\b([A-Z]{3}) ([⓪①-⑫])',p)); sys.exit(0 if d=={k:float(x) for k,x in F.F5TTSAdapter().default_params().items()} and v==idx and m==F.HABIBI_DIALECT_MAP else 1)"
see: ../../cloning.md, zipvoice.md, ../../ovos_plugin.md
-->

This page shows how to load an F5-TTS voice and clone a speaker from a reference clip and its transcription.

## What it is

F5-TTS is a flow-matching text-to-speech model with a Diffusion Transformer (DiT) backbone and ConvNeXt V2 blocks. [Habibi-TTS](https://github.com/SWivid/Habibi-TTS), an Arabic dialect TTS, and several community fine-tunes use this architecture. Like ZipVoice, F5-TTS runs an Euler ODE sampling loop, not a single-pass graph. It clones in context: the reference audio and its transcription go in front of the target text.

Pick F5-TTS for in-context cloning, and for Arabic dialects (Habibi, NAMAA, SILMA). Like [ZipVoice](zipvoice.md), it needs the transcription of the reference. YourTTS, StyleTTS2 and Chatterbox clone from the audio alone.

## Install

Reference loading and resampling need the `cloning` extra (`soundfile`, `scipy`):

```bash
pip install --pre "phoonnx[cloning]"
```

## Clone a voice

Every call needs the reference clip and its transcription. With an index voice, the model manager downloads the three ONNX graphs on first use:

```python
import wave

from phoonnx.config import SpeakerReference, SynthesisConfig
from phoonnx.model_manager import TTSModelManager

m = TTSModelManager()
m.merge_default_voices()
voice = m.voices["f5tts/v1-base"].load()
with wave.open("out.wav", "wb") as f:
    voice.synthesize_wav("A line the reference never spoke.", f,
        SynthesisConfig(reference=SpeakerReference(
            audio="reference.wav",
            text="transcription of the reference clip",
        )))
```

To load a downloaded voice folder yourself, name the auxiliary graphs in `engine_params`:

```python
from huggingface_hub import snapshot_download
from phoonnx.voice import TTSVoice

voice_dir = snapshot_download("Phoonnx/f5tts-multi-base-v1-graphemes")
voice = TTSVoice.load(
    f"{voice_dir}/model.onnx",
    config_path=f"{voice_dir}/config.json",
    engine_params={
        "preprocess_path": f"{voice_dir}/F5_Preprocess.onnx",
        "decode_path": f"{voice_dir}/F5_Decode.onnx",
    },
)
```

`model.onnx` is the transformer graph.

### OVOS plugin

The plugin reads the reference from its config:

```json
{
  "tts": {
    "module": "ovos-tts-plugin-phoonnx",
    "ovos-tts-plugin-phoonnx": {
      "voice": "f5tts/v1-base",
      "speaker_reference": "~/reference.wav",
      "speaker_reference_text": "transcription of the reference clip"
    }
  }
}
```

## Habibi-TTS dialect control

The Habibi-TTS Unified model was trained with dialect control tokens. Upstream wraps the text as `{dialect_char}〈text〉`, and each dialect maps to an enclosed-number character (`habibi_tts/model/utils.py`):

`UNK ⓪ · MSA ① · SAU ② · UAE ③ · ALG ④ · IRQ ⑤ · EGY ⑥ · MAR ⑦`

The vocabulary also holds `OMN ⑧ · TUN ⑨ · LEV ⑩ · SDN ⑪ · LBY ⑫`, but these dialects had no training data. Only the first eight are meaningful.

Set the tag with the `dialect` engine parameter. The `habibi/ar-unified` voice config sets `"dialect": "UNK"`. To change it for one call, use `SynthesisConfig(extra={"dialect": "EGY"})`. The adapter applies the tag at token level, through the `phoneme_id_map` of the voice. The per-dialect models take plain text with no tag.

An unknown dialect name raises `ValueError`. If the vocabulary of the voice lacks the control tokens, as in the per-dialect models and plain F5-TTS, the adapter logs a warning and skips the tag.

## How synthesis works

A voice has three ONNX graphs, in the layout of [DakeQQ/F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX):

```
text ─ tokenizer ─► text_ids ─┐
ref.wav ──────────────────────┤
                               ▼
        preprocess ─► noise, rope, cat_mel_text, ref_signal_len
                               │
        transformer × NFE ─────┘  (Euler ODE loop)
                               │
                               ▼
        decode ─► waveform      (Vocos + ISTFT)
```

- `preprocess` takes the reference audio, the text IDs (reference text and new text together) and the maximum duration. It gives the initial noise, the rotary position tables, `cat_mel_text`, `cat_mel_text_drop` and `ref_signal_len`.
- `transformer` runs one denoising step per call. It takes the noise, the four `rope_*` tables, `cat_mel_text`, `cat_mel_text_drop` and `time_step`, and gives `denoised` and the next `time_step`.
- `decode` turns the mel into a waveform, with Vocos and the ISTFT inside the graph.

The adapter estimates the maximum duration from the length of the reference audio and the ratio of the text lengths, divided by `speed`. Flow matching then integrates the noise from `t=0` to `t=1` on a sway-sampled time schedule. The export fixes the number of steps, so `nfe` is not a speed setting. A graph that stores `nfe_step` in its metadata always runs that many steps, and the adapter logs a warning and ignores another `nfe`. For a graph without that key, `nfe` must equal the step count of the export. The classifier-free guidance and the sway-sampled time schedule are inside the transformer graph, so `cfg_strength` and `sway_coefficient` do not change the output of the published graphs.

`torch.onnx.export` can rename an input that has the same name as an output, for example `time_step` to `time_step.1`. The adapter feeds the transformer inputs in the declared order of `session.get_inputs()`, not by name, so the renaming does not matter.

## Voice config

`F5TTSAdapter` (`phoonnx/engines/f5tts.py`) overrides `BaseOnnxAdapter.synthesize()`. The primary session of the voice is the `transformer`. `configure()` loads the `preprocess` and `decode` graphs, and an optional standalone Vocos vocoder, from `engine_params`:

```json
{
    "engine": "f5tts",
    "engine_params": {
        "preprocess_path": "F5_Preprocess.onnx",
        "decode_path": "F5_Decode.onnx",
        "nfe": 32,
        "cfg_strength": 2.0,
        "sway_coefficient": -1.0,
        "target_rms": 0.15,
        "sample_rate": 24000,
        "speed": 1.0
    }
}
```

If you give a standalone Vocos vocoder in `vocoder_path`, the adapter uses it in place of the `decode` graph:

```json
{
    "engine_params": {
        "preprocess_path": "F5_Preprocess.onnx",
        "vocoder_path": "vocos-mel-24khz.onnx",
        "vocoder_type": "vocos"
    }
}
```

If neither `decode_path` nor `vocoder_path` is set, synthesis raises `RuntimeError`.

## Notes

- The config `engine` field may be `f5tts`, `f5-tts` or `habibi`. A config with a `preprocess_path` and an `nfe` in `engine_params` also selects this adapter.
- Every index voice also has an `int8` variant. See [Precision variants](../../precision.md).
- Any model in the F5-TTS ONNX export layout (preprocess, transformer, decode) runs with this adapter. That includes F5-TTS, F5-TTS v1, Habibi-TTS, NAMAA-Saudi-TTS-V2 and SILMA TTS v1.

## Reference

### Parameters

`speed` is also settable per call in `SynthesisConfig.extra`.

| Param | Default | Description |
|---|---|---|
| `speed` | 1.0 | Speaking rate. It scales the estimated duration (above 1 is faster). |
| `nfe` | 32 | ODE steps. The export fixes this value. |
| `cfg_strength` | 2.0 | Classifier-free guidance strength. The export fixes it, and the adapter does not feed it. |
| `sway_coefficient` | -1.0 | Sway coefficient of the time schedule. The export fixes it, and the adapter does not feed it. |
| `target_rms` | 0.15 | Reference loudness. The adapter stores it but does not use it. |

`dialect` (default unset) is the Habibi Unified dialect tag described above.

### Indexed voices

The voices are in `phoonnx/voice_index/f5tts.json`, and each has its own Hub repository in the [F5-TTS collection](https://huggingface.co/collections/Phoonnx/f5-tts-6af8ea7f407234f6f1f08b30).

| voice id | Hub repo | Model | Language | Licence |
|---|---|---|---|---|
| `f5tts/v1-base` | `Phoonnx/f5tts-multi-base-v1-graphemes` | F5-TTS v1 base | multilingual | cc-by-nc-4.0 |
| `habibi/ar-unified` | `Phoonnx/f5tts-ar-habibi_unified-graphemes` | Habibi-TTS Unified, all dialects | Arabic | cc-by-nc-sa-4.0 |
| `habibi/ar-msa` | `Phoonnx/f5tts-ar-habibi_msa-graphemes` | Habibi-TTS MSA | Modern Standard Arabic | apache-2.0 |
| `habibi/ar-egy` | `Phoonnx/f5tts-ar_eg-habibi_egy-graphemes` | Habibi-TTS EGY | Egyptian Arabic | apache-2.0 |
| `habibi/ar-sau` | `Phoonnx/f5tts-ar_sa-habibi_sau-graphemes` | Habibi-TTS SAU | Saudi Arabic | cc-by-nc-sa-4.0 |
| `habibi/ar-uae` | `Phoonnx/f5tts-ar_ae-habibi_uae-graphemes` | Habibi-TTS UAE | Emirati Arabic | cc-by-nc-sa-4.0 |
| `habibi/ar-alg` | `Phoonnx/f5tts-ar_dz-habibi_alg-graphemes` | Habibi-TTS ALG | Algerian Arabic | apache-2.0 |
| `habibi/ar-irq` | `Phoonnx/f5tts-ar_iq-habibi_irq-graphemes` | Habibi-TTS IRQ | Iraqi Arabic | apache-2.0 |
| `habibi/ar-mar` | `Phoonnx/f5tts-ar_ma-habibi_mar-graphemes` | Habibi-TTS MAR | Moroccan Arabic | apache-2.0 |
| `namaa/ar-sa-v2` | `Phoonnx/f5tts-ar_sa-namaa_space-v2-graphemes` | NAMAA Saudi TTS V2, a Habibi fine-tune | Saudi Arabic | cc-by-nc-sa-4.0 |
| `silma/v1` | `Phoonnx/f5tts-multi-silma_ai-v1-graphemes` | SILMA TTS v1 | Arabic | apache-2.0 |
| `silma/v1-en` | `Phoonnx/f5tts-multi-silma_ai-v1-graphemes` | SILMA TTS v1, English entry | English | apache-2.0 |
| `openbible/tw-asante` | `Phoonnx/f5tts-tw-asante_openbible-graphemes` | F5-TTS OpenBible Twi (Asante) | Twi (Asante) | cc-by-sa-4.0 |
| `openbible/ckb` | `Phoonnx/f5tts-ckb-openbible-graphemes` | F5-TTS OpenBible Central Kurdish | Central Kurdish | cc-by-sa-4.0 |
| `f5tts/sema-afro` | `Phoonnx/f5tts-multi-sema_afro-unicode` | Sema TTS Afro (`travis-moore/sema-tts-afro`) | English and 22 African languages | apache-2.0 |
| `f5tts/spring` | `Phoonnx/f5tts-multi-spring-unicode` | SPRING F5 (`SPRINGLab/SPRING_F5`) | English and 23 Indian languages | apache-2.0 |

The Habibi-TTS licence is per model (see its [model card](https://huggingface.co/SWivid/Habibi-TTS)). Unified, SAU and UAE are non-commercial because of the SADA and Mixat datasets, and NAMAA inherits that licence from its Habibi base.

### Upstream

| | |
|---|---|
| Repo | <https://github.com/SWivid/F5-TTS> |
| Paper | [arXiv:2410.06885](https://arxiv.org/abs/2410.06885) |
| ONNX export | <https://github.com/DakeQQ/F5-TTS-ONNX> |
| ONNX voices | [F5-TTS collection](https://huggingface.co/collections/Phoonnx/f5-tts-6af8ea7f407234f6f1f08b30) |

---

[Docs index](../../index.md) · Previous: [ArkTTS](arktts.md) · Next: [MOSS-TTS-Nano](mosstts.md)
