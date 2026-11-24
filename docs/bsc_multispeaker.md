# BSC multispeaker StyleTTS2 (Spanish / Catalan)

<!-- agent-hint
covers: which BSC Spanish and Catalan StyleTTS2 voices exist, where they come from and how their speaker styles are made
source: phoonnx/voice_index/styletts2.json, phoonnx/engines/styletts2.py, scripts/conversion/styletts2/fetch_bsc_speaker_refs.py, scripts/conversion/styletts2/export_bsc_speakers.py
generated: no
verify: HF_HUB_OFFLINE=1 .venv/bin/python -m doctest -o NORMALIZE_WHITESPACE docs/bsc_multispeaker.md
see: cloning.md, galician.md
-->

This page lists the Spanish and Catalan StyleTTS2 voices of the Barcelona Supercomputing
Center (BSC), where they come from and how their speaker styles are made.

BSC published two multispeaker StyleTTS2 checkpoints:

| Upstream | Language | Training corpus | Weights license |
|---|---|---|---|
| [`BSC-LT/styletts2-spanish-multispeaker`](https://huggingface.co/BSC-LT/styletts2-spanish-multispeaker) | Spanish | [CML-TTS](https://huggingface.co/datasets/ylacombe/cml-tts) Spanish, 7.9 h | Apache-2.0 |
| [`BSC-LT/styletts2-catalan-multispeaker`](https://huggingface.co/BSC-LT/styletts2-catalan-multispeaker) | Catalan | [Festcat](https://huggingface.co/datasets/projecte-aina/festcat_trimmed_denoised) + [OpenSLR-69](https://huggingface.co/datasets/projecte-aina/openslr-slr69-ca-trimmed-denoised), 14.8 h | GPL-3.0 |

Both read espeak phonemes with the yl4579 178-symbol IPA token set. Both use the
hifigan decoder.

## Zero-shot voices and named speakers

Neither upstream repository ships reference audio. The voices `bsc/es-styletts2` and
`bsc/ca-styletts2` therefore clone from a reference clip, and they need one on each
call. Without one, synthesis raises a `ValueError`. See [cloning](cloning.md).

The named speakers need no clip. Each one is the same ONNX graph with its own style
file of 256 values. These are all the BSC voices in the index:

```python
>>> from phoonnx.model_manager import TTSModelManager
>>> manager = TTSModelManager()
>>> manager.merge_default_voices()
>>> for lang in ("ca", "es"):
...     ids = [v for v in manager.voices if v.startswith(f"bsc/{lang}-")]
...     print(lang, " ".join(ids))
ca bsc/ca-styletts2 bsc/ca-bet bsc/ca-eli bsc/ca-eva bsc/ca-jan bsc/ca-mar bsc/ca-ona
   bsc/ca-pau bsc/ca-pep bsc/ca-pol bsc/ca-teo bsc/ca-uri
es bsc/es-styletts2 bsc/es-cml3946 bsc/es-cml8882 bsc/es-cml9972 bsc/es-cml10246
   bsc/es-cml11797 bsc/es-cml12367
>>> [v for v in manager.voices if v.startswith("bsc/") and not manager.voices[v].style_url]
['bsc/es-styletts2', 'bsc/ca-styletts2']

```

The 11 Catalan names are the Festcat speakers. The 6 Spanish voices are the CML-TTS
speakers that hold most of that corpus. CML-TTS gives each speaker a number only, so
the Spanish voice ids carry that number. The OpenSLR-69 speakers are anonymous, so they
give no named voices.

A named speaker also accepts a reference clip. The clip replaces the shipped style:

```python
from phoonnx.model_manager import TTSModelManager
from phoonnx.config import SpeakerReference, SynthesisConfig

manager = TTSModelManager()
manager.load()
manager.merge_default_voices()

voice = manager.voices["bsc/ca-ona"].load()
shipped = list(voice.synthesize("Bon dia, com estàs?"))
cloned = list(voice.synthesize("Bon dia.", SynthesisConfig(reference=SpeakerReference(audio="ref.wav"))))
```

`synthesize` yields one `AudioChunk` per sentence, so the example collects them with
`list`.

## How the style files are made

`scripts/conversion/styletts2/` holds the recipe in two steps:

1. `fetch_bsc_speaker_refs.py` reads four reference clips per speaker from the Hugging
   Face parquet exports of the training corpora. It uses DuckDB, so it does not
   download the full corpora.
2. `export_bsc_speakers.py` runs the exported `style_encoder.onnx` on those clips. It
   writes `<speaker>.bin`, the mean of the style vectors of the clips. The mean cancels
   the prosody of a single clip and keeps the speaker identity.

The 256 values are `ref_p` followed by `ref_s`: the acoustic style for the decoder, then
the prosodic style for the predictor. The adapter splits them at 128. The style file
reaches the adapter through `style_url` in the voice index, which downloads to
`engine_params["style_path"]`. The HiTZ Basque speakers use the same mechanism.

## Check intelligibility

To measure a speaker, run a per-language ASR model on the synthesized audio. The
conformer-CTC exports
[`OpenVoiceOS/nvidia-ca-conformer-ctc-large-onnx`](https://huggingface.co/OpenVoiceOS/nvidia-ca-conformer-ctc-large-onnx)
and
[`OpenVoiceOS/nvidia-es-conformer-ctc-large-onnx`](https://huggingface.co/OpenVoiceOS/nvidia-es-conformer-ctc-large-onnx)
run through `onnx-asr`. A CTC model knows one language and has no decoder prompt, so it
cannot translate. It can only hear words wrong.

Run the same ASR on the speaker's real recordings and report that floor next to the
score of the synthesized audio. A multilingual sequence-to-sequence recognizer can
translate the audio into another language, and the floor shows that failure.

## Limitations

- The Spanish checkpoint learned from 7.9 hours of speech, and the Catalan checkpoint
  from 14.8 hours.
- The Catalan weights are GPL-3.0 upstream. That license applies to the weights, not to
  phoonnx (Apache-2.0), which only loads them. Account for it in your deployment.
- The Hugging Face hub cache keeps one copy of each file per URL and revision. The named
  speakers of one language therefore share one copy of the 545 MiB graph on disk, and
  each speaker adds only its 1 KiB style file.

---

[Docs index](index.md) · Previous: [Galician phonemizer](galician.md) · Next: [Usage guide](usage.md)
