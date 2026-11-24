# x-vits Engine (inference)

<!-- agent-hint
covers: how to load the x-vits voice, choose a style and mark a stressed word
source: phoonnx/engines/xvits.py, phoonnx/model_manager.py:TTSModelInfo, phoonnx/ssml.py:parse_ssml
generated: no
verify: .venv/bin/python -c "import re,sys; from phoonnx.engines.xvits import XVitsAdapter as A, EMPHASIS_MODES as E; p=open('docs/training/engines/xvits.md').read().replace(chr(96),''); d={k:float(v) for k,v in re.findall(r'^\| ([a-z_]+) \| [a-z_]+ \| ([\d.]+) \|$',p,re.M)}; m=tuple(re.search(r'^\| emphasis \| ([^|]+) \|',p,re.M).group(1).replace(' ','').split(',')); sys.exit(0 if d==A().default_params() and m==E else 1)"
see: ../../engines.md, ../../configuration.md
-->

This page shows how to run the x-vits voice, with its styles and its stressed words, for anyone who uses phoonnx from Python.

x-vits is a Piper/VITS voice with a prosody model added. It speaks in six styles and stresses the words you mark.
phoonnx runs it with the `xvits` adapter. There is no training engine for it in `phoonnx_train`.

> Related: [adapter architecture](../../engines.md) · [configuration](../../configuration.md)

## Speak with it

The voice is `fdemelo/x-vits` on the Hugging Face Hub. Download it, name its files in `engine_params`, and synthesize.

```python
from pathlib import Path

from huggingface_hub import snapshot_download

from phoonnx.config import SynthesisConfig
from phoonnx.voice import TTSVoice

voice_dir = Path(snapshot_download("fdemelo/x-vits"))
engine_params = {
    "durations_path": voice_dir / "durations.onnx",
    "acoustic_path": voice_dir / "acoustic.onnx",
    "predictor_path": voice_dir / "predictor.onnx",
    "prominence_path": voice_dir / "prominence.onnx",
    "dp_cond_path": voice_dir / "dp_cond.npz",
    "emphasis_template_path": voice_dir / "emphasis_template.json",
    "runtime_path": voice_dir / "runtime.json",
}
voice = TTSVoice.load(
    str(voice_dir / "encoder.onnx"),
    str(voice_dir / "config.json"),
    lang_code="en",
    engine_params={key: str(path) for key, path in engine_params.items()},
)

config = SynthesisConfig(extra={"style": "happy", "seed": 1})
for chunk in voice.synthesize("I *really* did not expect that.", config):
    print(chunk.sample_rate, len(chunk.audio_float_array))
```

The voice gives one chunk per sentence at 22050 Hz. The stressed word `really` is longer than with `"emphasis": "none"`, and its pitch moves.

Set `lang_code="en"`. The voice config names the espeak voice `en` and the language `en_US`. phoonnx reads the language, so without `lang_code` it phonemizes with `en-us`. The voice was trained on the IDs of the espeak `en` voice.

## Options

Pass the options in `SynthesisConfig.extra`. `length_scale` is a normal `SynthesisConfig` field.

| Option | Values | Effect |
| --- | --- | --- |
| `style` | `confused`, `default`, `enunciated`, `happy`, `laughing`, `sad`, or none | The speaking style. A name that is not in the list raises `ValueError`, and the message lists the valid names. With no style, the prosody model runs without a style and without guidance. |
| `emphasis` | `auto`, `marked`, `none` | `auto` stresses the marked words, or the words the prominence model picks when the text has no marks. `marked` stresses only marked words. `none` turns emphasis off. |
| `emphasis_scale` | a number | The size of the emphasis. The default is the `template_scale` of `runtime.json`. |
| `steps` | an integer | The steps of the prosody ODE. The default is the `pp_steps` of `runtime.json`, which is 16 for `fdemelo/x-vits`. Fewer steps run faster. |
| `seed` | an integer | The seed of the random generator. The same seed, text and options give the same audio. |

The noise scales and the length scale default to the values in `runtime.json`, not to the numbers in the Piper `config.json`. `SynthesisConfig.noise_scale`, `noise_w_scale` and `length_scale` change them for one call. When `runtime.json` lacks a key, the adapter uses the default in this table:

| Setting | `runtime.json` key | Default |
| --- | --- | --- |
| `noise_scale` | `noise_scale` | 0.667 |
| `noise_w_scale` | `noise_scale_w` | 0.4 |
| `length_scale` | `length_scale` | 1.08 |
| `steps` | `pp_steps` | 16 |
| `emphasis_scale` | `template_scale` | 1.5 |

## Mark a stressed word

Put asterisks around a word, or around several words, in the text: `You *want* the shark?`. Each marked word is stressed.
When a text has marks, only the marked words are stressed. A text with no marks lets the prominence model pick.

The marks reach the model as the word of each token. phoonnx reads the unmarked text once to make the ids. It finds the word of each phoneme by aligning that reading with the phonemes of each word alone, so a merged word such as `ɪnðə` ("in the") still gives its phonemes back to its two words. A mark follows its word only while the text normaliser keeps that word in place. When the normaliser moves a word, for example in an amount with a currency sign, the mark can be lost or land on another word. Put the mark on a word that the normaliser leaves alone, or write the amount in words.

SSML `<emphasis>` does not reach the x-vits model. `parse_ssml` turns an emphasised span into a separate piece with its own rate and volume, and phoonnx synthesizes each piece in a separate engine call.

## How the graphs fit

| Graph | Role |
| --- | --- |
| `encoder.onnx` | The voice's own graph. Ids to text states. |
| `predictor.onnx` | One step of an optimal-transport flow-matching model. It gives the pitch, voicing and energy of each token, for a style. |
| `durations.onnx` | The stochastic duration predictor. A per-style vector from `dp_cond.npz` is added to its input. |
| `acoustic.onnx` | The prior, the length regulation, the flow and the HiFi-GAN generator. It reads the prosody. |
| `prominence.onnx` | The words a speaker would stress. |

The adapter samples the prosody with a cosine time schedule and a midpoint step. For a style, it moves the velocity away from the unconditioned velocity by the guidance weight of each target. It then adds the emphasis offsets to the pitch and the duration multipliers to the durations. The durations are `ceil(exp(logw) * length_scale)` frames.
The random draws come in a fixed order: prosody noise, duration noise, latent noise.

## Index entry

The bundled index has no x-vits voice. An index entry names the encoder as the model and the other files as `aux_model_urls`:

```json
{
  "voice_id": "fdemelo/x-vits",
  "engine": "xvits",
  "model_url": "https://huggingface.co/<mirror>/resolve/<commit>/encoder.onnx",
  "config_url": "https://huggingface.co/<mirror>/resolve/<commit>/config.json",
  "phoneme_type": "espeak",
  "alphabet": "ipa",
  "lang": "en-us",
  "phonemizer_lang": "en",
  "aux_model_urls": {
    "durations_path": "https://huggingface.co/<mirror>/resolve/<commit>/durations.onnx",
    "acoustic_path": "https://huggingface.co/<mirror>/resolve/<commit>/acoustic.onnx",
    "predictor_path": "https://huggingface.co/<mirror>/resolve/<commit>/predictor.onnx",
    "prominence_path": "https://huggingface.co/<mirror>/resolve/<commit>/prominence.onnx",
    "dp_cond_path": "https://huggingface.co/<mirror>/resolve/<commit>/dp_cond.npz",
    "emphasis_template_path": "https://huggingface.co/<mirror>/resolve/<commit>/emphasis_template.json",
    "runtime_path": "https://huggingface.co/<mirror>/resolve/<commit>/runtime.json"
  }
}
```

`phonemizer_lang` gives the espeak voice `en`. The adapter is found from the encoder's inputs (`ids`, `lengths`) and outputs (`x`, `x_mask`).

## Licence

The adapter code is a port of the runtime of x-vits by Flavio De Melo (MIT, <https://github.com/femelo/x-vits>).
The voice weights `fdemelo/x-vits` are CC BY-NC 4.0, so you may not use them commercially. The voice was trained on speaker ex02 of the Expresso dataset.

---

[Docs index](../../index.md) · Previous: [FastPitch](fastpitch.md) · Next: [ZipVoice](zipvoice.md)
