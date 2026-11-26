# Orpheus

<!-- agent-hint
covers: what the Orpheus engine is, its voices, parameters and limits
source: phoonnx/engines/orpheus.py:OrpheusAdapter, phoonnx/voice_index/orpheus.json, scripts/conversion/orpheus/evidence/wer.json, scripts/conversion/orpheus/evidence/probe_prompt_output.txt
generated: no
verify: .venv/bin/python -c "import re, sys, json; from phoonnx.engines.orpheus import OrpheusAdapter, SAMPLES_PER_FRAME; q = chr(96); page = re.sub('<!-- agent-hint.*?\n-->', '', open('docs/orpheus.md').read(), flags=re.S); code = {k: float(v) for k, v in OrpheusAdapter().default_params().items()}; table = {k: float(v) for k, v in re.findall('^[|] ' + q + '([a-z_]+)' + q + ' [|] ([0-9.]+) [|]', page, re.M)}; idx = [v['engine_options']['voice'] for v in json.load(open('phoonnx/voice_index/orpheus.json')).values()]; names = re.findall(q + '([a-z]+)' + q, re.search('conversational realism:(.*?)Each index entry', page, re.S).group(1)); bad = [table != code, names != idx, str(SAMPLES_PER_FRAME) + ' samples' not in page]; print(bad, table, names); sys.exit(1 if any(bad) else 0)"
see: engines.md, benchmarking.md
-->

This page describes the Orpheus engine, its voices and its limits, for developers who consider a large codec language model voice.

[Orpheus](https://github.com/canopyai/Orpheus-TTS) (Canopy Labs, Apache-2.0) is a
Llama-3.2-3B causal language model whose vocabulary holds 28 672 audio tokens. It gives a
flat stream of tokens. Every seven tokens form one
[SNAC](https://huggingface.co/hubertsiuzdak/snac_24khz) frame, which the SNAC decoder turns
into 2048 samples at 24 kHz.

## Orpheus is a GPU engine

SNAC needs about 82 tokens for each second of audio, and the 3B backbone makes one token
per step. On a 12-core CPU in fp32, the backbone needs 0.43 to 0.56 s per token. The voice
thus runs 36 to 46 times slower than real time, 38.7 times on average over seven sentences
([measurements](../scripts/conversion/orpheus/evidence/wer.json), backbone time only).

Upstream serves Orpheus through vLLM on a GPU. Canopy Labs publishes only 3B checkpoints,
so no smaller Orpheus exists. The index holds the voices so that the catalog is complete.
Do not use one as a CPU default.

## Voices

The speaker is a name in the prompt text, not an embedding and not a speaker id. The English
checkpoint (`canopylabs/orpheus-3b-0.1-ft`) has these names, in the Canopy Labs order of
conversational realism:

`tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe`

Each index entry pins one name in its `engine_options`, for example `{"voice": "leo"}`.
To select another name for one call, pass it in `extra`:

```python
from phoonnx.config import SynthesisConfig

voice.synthesize("Hello there.", syn_config=SynthesisConfig(extra={"voice": "leo"}))
```

A name that the checkpoint does not have raises `ValueError`, and the message lists the
names it has. A string `speaker` is a name. An integer `speaker` selects a name only when the voice
config has a `speaker_id_map`. The index entries ship no voice config, so they have no map.

### Emotive tags

The BPE of the checkpoint encodes a tag as ordinary text, so no phonemizer removes it:

`<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`

```python
voice.synthesize("That is really funny <laugh> I cannot believe it.")
```

A tag is a hint, not a control. The model can render a tag, soften it, or ignore it.

## Cloning

Orpheus does not clone. A call with a `reference` raises `RuntimeError`, also
when `reference.text` is set. Use one of the named voices.

## Multilingual research releases

Canopy Labs also publishes six gated 3B research fine-tunes on the Hugging Face Hub. They
cover French, German, Spanish and Italian (one checkpoint), Mandarin, Korean and Hindi.
phoonnx does not mirror or index them, and each one has the CPU cost of the English
checkpoint.

## Prompt and sampler details

Two details of the upstream server decide what the model receives. A port that reads the
upstream source literally gets both of them wrong.

The prompt carries two BOS tokens. `OrpheusModel` builds its ids, decodes them back to a
string, and gives the string to vLLM. vLLM tokenizes it again with `add_special_tokens=True`
and puts a second `<|begin_of_text|>` in front. The adapter sends the same ids:

```
128000  <|begin_of_text|>   added by vLLM
128259  <custom_token_3>    start of human turn
128000  <|begin_of_text|>   added by the upstream tokenizer call
   ...  "{voice}: {text}"
128009  <|eot_id|>
128260  <custom_token_4>    end of human turn
128261  <custom_token_5>    start of AI turn
128257  <custom_token_1>    start of speech
```

The sampler order is the vLLM order, not the llama.cpp order. vLLM applies the penalties,
then the temperature, and cuts to the nucleus last, so `top_p` selects over the tempered
distribution. llama.cpp does the reverse. The [NeuTTS](engines.md) adapter uses the
llama.cpp order, because upstream serves NeuTTS through llama.cpp.

Upstream also passes `stop_token_ids=[49158]`. That id is the text token `Ġrez`, which never
appears in the audio part of a run, so it stops nothing. Generation ends on
`<custom_token_2>` (end of speech) or on the EOS of the checkpoint.

## Parameters

| Parameter | Default | What it does |
|---|---|---|
| `temperature` | 0.6 | Sampling temperature. 0 is greedy. |
| `top_p` | 0.8 | Nucleus, applied after the temperature. |
| `repetition_penalty` | 1.3 | Divides the scores of the tokens already seen. |
| `max_new_tokens` | 1200 | Limit on the generation loop, about 14.6 s of audio. |
| `max_chunk_chars` | 300 | Characters of text per model call. |

These are the values of the upstream server. The `generation_config.json` of the checkpoint
sets `top_p` to 0.9. The model was tuned against the served values, so the adapter uses them.

---

[← Pocket TTS](pockettts.md) · [Home](index.md) · [Galician phonemizer →](galician.md)
