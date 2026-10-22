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

The recognizer is the **per-language conformer-CTC** export for that language:
[`OpenVoiceOS/nvidia-ca-conformer-ctc-large-onnx`](https://huggingface.co/OpenVoiceOS/nvidia-ca-conformer-ctc-large-onnx)
and
[`OpenVoiceOS/nvidia-es-conformer-ctc-large-onnx`](https://huggingface.co/OpenVoiceOS/nvidia-es-conformer-ctc-large-onnx),
both run through `onnx-asr`. Prefer a per-language CTC model as the gate: it has one
language and no decoder prompt, so it cannot translate or switch language — it can only
mis-hear.

Multilingual sequence-to-sequence recognizers can fail in a way that looks like a bad
voice. `canary-1b-v2` carries a `<|ca|>` tag in its vocabulary but was never trained on
Catalan; it silently translates Catalan into English and scores ~96 % WER on *real human*
Festcat audio. The floor column is what catches this. Floor-check every gate on real
speech in the target language before you believe a WER.

**Style round-trip** re-encodes each synthesized clip with `style_encoder.onnx` and
compares it to the style it was asked to render. It answers a question WER cannot: whether
the voice is actually the requested speaker. Cosine similarity is to the target style;
nearest-style accuracy is how often the synthesized clip's own style is closest to the
correct speaker out of all speakers in that language.

### Catalan

Gate: [`OpenVoiceOS/nvidia-ca-conformer-ctc-large-onnx`](https://huggingface.co/OpenVoiceOS/nvidia-ca-conformer-ctc-large-onnx)

| Voice | WER | ASR floor on real clips | Style round-trip (cos) |
|---|---|---|---|
| `bsc/ca-bet` | 0.0% | 3.7% | 0.95 |
| `bsc/ca-eli` | 0.0% | 3.8% | 0.89 |
| `bsc/ca-eva` | 0.0% | 4.8% | 0.69 |
| `bsc/ca-jan` | 1.7% | 11.1% | 0.73 |
| `bsc/ca-mar` | 0.0% | 3.6% | 0.90 |
| `bsc/ca-ona` | 0.0% | 0.0% | 0.86 |
| `bsc/ca-pau` | 0.0% | 8.7% | 0.77 |
| `bsc/ca-pep` | 0.0% | 4.5% | 0.72 |
| `bsc/ca-pol` | 0.0% | 2.0% | 0.93 |
| `bsc/ca-teo` | 1.7% | 0.0% | 0.80 |
| `bsc/ca-uri` | 0.0% | 8.7% | 0.96 |
| **all speakers** | **0.3%** | **4.7%** | **0.84** |

Nearest-style accuracy: 42/55 clips (76 %).

### Spanish

Gate: [`OpenVoiceOS/nvidia-es-conformer-ctc-large-onnx`](https://huggingface.co/OpenVoiceOS/nvidia-es-conformer-ctc-large-onnx)

| Voice | WER | ASR floor on real clips | Style round-trip (cos) |
|---|---|---|---|
| `bsc/es-cml3946` | 3.6% | 1.8% | 0.89 |
| `bsc/es-cml8882` | 1.8% | 19.2% | 0.82 |
| `bsc/es-cml9972` | 7.3% | 7.4% | 0.94 |
| `bsc/es-cml10246` | 5.5% | 7.8% | 0.92 |
| `bsc/es-cml11797` | 9.1% | 1.2% | 0.92 |
| `bsc/es-cml12367` | 3.6% | 0.0% | 0.95 |
| **all speakers** | **5.2%** | **5.5%** | **0.91** |

Nearest-style accuracy: 27/30 clips (90 %).

## Limitations

- The Spanish checkpoint saw only 7.9 hours of speech against the Catalan one's 14.8, and
  it shows: 5.2 % WER against 0.3 %. Spanish sits at its own ASR floor rather than below
  it, so treat the Spanish speakers as usable but not clean.
- The Catalan weights are **GPL-3.0** upstream. That governs the weights, not `phoonnx`
  (Apache-2.0), which only loads them. Account for it in your own deployment.
- The model cache is keyed by voice ID, so each named speaker keeps its own copy of the
  same 545 MB graph. Use one speaker per language unless you need more. This applies to
  every shared-checkpoint voice family (the HiTZ Basque speakers behave the same way).
