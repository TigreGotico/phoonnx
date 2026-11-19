# x-vits Engine (inference)

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

config = SynthesisConfig(extra_params={"style": "happy", "seed": 1})
for chunk in voice.synthesize("I *really* did not expect that.", config):
    print(chunk.sample_rate, len(chunk.audio_float_array))
```

The output is one chunk per sentence at 22.05 kHz:

```
22050 45824
```

The same sentence with `"emphasis": "none"` gives `22050 44032`. The stressed word `really` is longer, and its pitch moves.

`lang_code="en"` matters. The voice config names the espeak voice `en` and the language `en_US`. phoonnx reads the language, so it would phonemize with `en-us`. The voice was trained on the ids of the `en` voice. With `lang_code="en"` the ids are the ones the training used.

## Options

Pass the options in `SynthesisConfig.extra_params`. `length_scale` is a normal `SynthesisConfig` field.

| Option | Values | Effect |
| --- | --- | --- |
| `style` | `confused`, `default`, `enunciated`, `happy`, `laughing`, `sad`, or none | The speaking style. A name that is not in the list raises `ValueError`, and the message lists the valid names. With no style, the prosody model runs without a style and without guidance. |
| `emphasis` | `auto`, `marked`, `none` | `auto` stresses the marked words, or the words the prominence model picks when the text has no marks. `marked` stresses only marked words. `none` turns emphasis off. |
| `emphasis_scale` | a number | The size of the emphasis. The default is the `template_scale` of `runtime.json`. |
| `steps` | an integer | The steps of the prosody ODE. The default is 16. Four steps is about 2.5 times faster on a CPU and moves the pitch by about 0.1 semitone on average. |
| `seed` | an integer | The seed of the random generator. The same seed, text and options give the same audio. |

The noise scales and the length scale default to the values in `runtime.json`, not to the numbers in the Piper `config.json`.
`SynthesisConfig.noise_scale`, `noise_w_scale` and `length_scale` change them for one call.

## Mark a stressed word

Put asterisks around a word, or around several words, in the text: `You *want* the shark?`. Each marked word is stressed.
When a text has marks, only the marked words are stressed. A text with no marks lets the prominence model pick.

The marks reach the model as the word of each token. phoonnx reads the unmarked text once to make the ids. It finds the word of each phoneme by aligning that reading with the phonemes of each word alone, so a merged word such as `ɪnðə` ("in the") still gives its phonemes back to its two words. A mark follows its word only while the normaliser keeps that word in place. When the normaliser moves a word, the mark can be lost or land on another word: `a £3 *billion* deal.` is read "a three billion pounds deal", and with espeak the mark then reaches no token. Put the mark on a word the normaliser leaves alone, or write the amount in words.

SSML `<emphasis>` does not reach the model. `parse_ssml` turns an emphasised span into a separate piece with its own rate and volume, and each piece is a separate engine call, so the span loses its sentence. SSML support needs the reader to keep the span inside its sentence as a mark when the voice reads marks, and to skip the piece-level rate and volume change for it.

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

An index entry names the encoder as the model and the other files as `aux_model_urls`:

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
The voice weights `fdemelo/x-vits` are CC BY-NC 4.0: you may not use them commercially. The voice was trained on speaker ex02 of the Expresso dataset.
