# BSC multispeaker StyleTTS2 (Spanish / Catalan)

This page is for anyone selecting a Spanish or Catalan StyleTTS2 voice. After reading it
you will know which named speakers exist, where their voices come from, and how good
each one is.

The Barcelona Supercomputing Center published two multispeaker StyleTTS2 checkpoints:

| Upstream | Language | Training corpus | Weights license |
|---|---|---|---|
| [`BSC-LT/styletts2-spanish-multispeaker`](https://huggingface.co/BSC-LT/styletts2-spanish-multispeaker) | Spanish | [CML-TTS](https://huggingface.co/datasets/ylacombe/cml-tts) Spanish, 7.9 h | Apache-2.0 |
| [`BSC-LT/styletts2-catalan-multispeaker`](https://huggingface.co/BSC-LT/styletts2-catalan-multispeaker) | Catalan | [Festcat](https://huggingface.co/datasets/projecte-aina/festcat_trimmed_denoised) + [OpenSLR-69](https://huggingface.co/datasets/projecte-aina/openslr-slr69-ca-trimmed-denoised), 14.8 h | GPL-3.0 |

Both are phonemized with **espeak** and use the yl4579 178-symbol IPA token set. Both use
the **hifigan** decoder, so no iSTFT graph is involved.

## Zero-shot parent and named speakers

Neither upstream repository ships reference audio, so `phoonnx` first wired both as
zero-shot cloning voices — `bsc/es-styletts2` and `bsc/ca-styletts2` — which need a
reference clip on every call (see [cloning](cloning.md)).

The **named speakers** remove that requirement. Each one is the *same* ONNX graph plus
its own 256-value style blob:

| | Voice IDs | Speakers |
|---|---|---|
| Catalan | `bsc/ca-bet`, `bsc/ca-eli`, `bsc/ca-eva`, `bsc/ca-jan`, `bsc/ca-mar`, `bsc/ca-ona`, `bsc/ca-pau`, `bsc/ca-pep`, `bsc/ca-pol`, `bsc/ca-teo`, `bsc/ca-uri` | 11 named Festcat speakers |
| Spanish | `bsc/es-cml3946`, `bsc/es-cml8882`, `bsc/es-cml9972`, `bsc/es-cml10246`, `bsc/es-cml11797`, `bsc/es-cml12367` | the 6 CML-TTS Spanish speakers that hold the corpus |

CML-TTS identifies speakers by number only, so the Spanish voice IDs carry the corpus id
rather than a name. The Catalan OpenSLR-69 half of the training data is anonymous and
contributes no named voices.

A named speaker still accepts a reference clip — cloning overrides the shipped style:

```python
from phoonnx.model_manager import TTSModelManager
from phoonnx.config import SynthesisConfig

manager = TTSModelManager(); manager.load(); manager.merge_default_voices()

voice = manager.voices["bsc/ca-ona"].load()
voice.synthesize("Bon dia, com estàs?")                       # shipped style
voice.synthesize("Bon dia.", SynthesisConfig(speaker_reference="ref.wav"))   # cloned
```

## How the style blobs are made

`scripts/conversion/styletts2/` holds the two-step recipe:

1. `fetch_bsc_speaker_refs.py` pulls four reference clips per speaker out of the
   HuggingFace parquet exports of the **training corpora**, with DuckDB, so the full
   corpora never have to be downloaded.
2. `export_bsc_speakers.py` runs the exported `style_encoder.onnx` over those clips and
   writes `<speaker>.bin` — the **mean** of the per-clip 256-value vectors. Averaging
   cancels the prosody of any single clip and leaves the speaker identity. The 256 values
   are `ref_p` ++ `ref_s`: acoustic style for the decoder, then prosodic style for the
   predictor. The adapter splits them again at 128.

The blobs are the same shape and role as the HiTZ Basque per-speaker styles, and they
reach the adapter the same way: `style_url` in the voice index, downloaded to
`engine_params["style_path"]`.

## Measured quality

Intelligibility is word error rate over five synthesized sentences per speaker. The
**ASR floor** column is the same recognizer over that speaker's real reference clips, so a
weak style vector is separable from recognition error on the language itself. Note the two
sets use different text: the reference clips are 19th-century prose from the training
corpora, the synthesized sentences are modern everyday ones.

`whisper-large-v3-turbo` is the recognizer for both languages. `canary-1b-v2`, phoonnx's
usual choice, was rejected: its vocabulary carries a `<|ca|>` tag but the model was never
trained on Catalan, so it silently translates Catalan into English and scores ~96 % WER on
*real human* Festcat audio. Never read a WER from an ASR you have not floor-checked on
real speech in that language.

**Style round-trip** re-encodes each synthesized clip with `style_encoder.onnx` and
compares it to the style it was asked to render. It answers a question WER cannot: whether
the voice is actually the requested speaker. Cosine similarity is to the target style;
nearest-style accuracy is how often the synthesized clip's own style is closest to the
correct speaker out of all speakers in that language.

<!--WER_TABLE-->

## Limitations

- The Spanish checkpoint saw only 7.9 hours of speech. It is noticeably weaker than the
  Catalan one, which had 14.8 hours.
- The Catalan weights are **GPL-3.0** upstream. That governs the weights, not `phoonnx`
  (Apache-2.0), which only loads them. Account for it in your own deployment.
- The model cache is keyed by voice ID, so each named speaker keeps its own copy of the
  same 545 MB graph. Use one speaker per language unless you need more. This applies to
  every shared-checkpoint voice family (the HiTZ Basque speakers behave the same way).
