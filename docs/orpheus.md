# Orpheus

[Orpheus](https://github.com/canopyai/Orpheus-TTS) (Canopy Labs, Apache-2.0) is a
Llama-3.2-3B causal language model whose vocabulary carries 28 672 audio tokens. It
emits a flat token stream; every seven tokens form one
[SNAC](https://huggingface.co/hubertsiuzdak/snac_24khz) frame, which SNAC's decoder
turns into 2048 samples at 24 kHz.

## Read this first: Orpheus is a GPU engine

A 3B backbone costs about **0.37 s per decode step** on 12 CPU cores, and SNAC needs
~82 tokens for every second of audio. Measured end to end, that is **37-46x slower
than real time**. Upstream serves it through vLLM on a GPU.

Canopy Labs announced 1B, 400M and 150M tiers, but **never released them**. Their own
loader still refuses those names with `"not supported ... will be released very soon"`,
and the `canopylabs` org on Hugging Face holds only 3B checkpoints. There is no smaller
Orpheus to fall back to.

The voices are indexed so the catalog is complete. Do not pick one as a CPU default.

## Voices

The speaker is a **name written into the prompt text** — not an embedding, not a
speaker id. English (`canopylabs/orpheus-3b-0.1-ft`), in Canopy Labs' own order of
conversational realism:

`tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe`

Select one per call, or pin it on the voice:

```python
voice.synthesize("Hello there.", extra_params={"voice": "leo"})
```

A plain `speaker_id` also works: the index ships a `speaker_id_map`, and the adapter
resolves the id back to the name before it reaches the model.

### Emotive tags

Tags are ordinary text that the checkpoint's own BPE encodes, so nothing phonemizes
them away:

`<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`

```python
voice.synthesize("That is really funny <laugh> I cannot believe it.")
```

They are a hint, not a control: the model may render a tag, soften it, or ignore it.

## Parameters

| Parameter | Default | What it does |
|---|---|---|
| `temperature` | 0.6 | sampling temperature; 0 is greedy |
| `top_p` | 0.8 | nucleus, applied **after** the temperature |
| `repetition_penalty` | 1.3 | divides the scores of tokens already seen |
| `max_new_tokens` | 1200 | ceiling on the AR loop (~14.6 s of audio) |
| `max_chunk_chars` | 300 | characters of text per model call |

These are the values upstream's **server** uses, which are not the checkpoint's
`generation_config.json` (that says `top_p` 0.9). The served values are the ones the
model was tuned against, so they are what the adapter defaults to.

## Two details a plausible reimplementation gets wrong

**The prompt carries a double BOS.** `OrpheusModel` builds its ids, decodes them back
to a *string*, and hands the string to vLLM, which re-tokenizes it with
`add_special_tokens=True` and prepends a second `<|begin_of_text|>`. Reading the
upstream source literally produces a prompt the model is never served:

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

**The sampler order is vLLM's, not llama.cpp's.** vLLM applies the penalties, then the
temperature, and truncates to the nucleus last — so `top_p` selects over the *tempered*
distribution. llama.cpp does the reverse. The
[NeuTTS](engines.md) adapter deliberately implements the other order, because that is
the stack *it* is served through.

Upstream also passes `stop_token_ids=[49158]`, which is the text token `Ġrez` and can
never appear mid-audio. It is dead code in their serving path; generation really ends on
`<custom_token_2>` (end of speech) or the checkpoint's EOS.

## Cloning

Orpheus clones **in context**, from a transcribed reference — not from a bare clip.
Passing `speaker_reference` alone raises. Use a named voice unless you also have the
reference transcript.

## Multilingual research releases

Canopy Labs also published seven 3B research checkpoints (2025-04-09). They are gated
on Hugging Face, and their voices are:

| Language | Repo | Voices |
|---|---|---|
| French | `canopylabs/3b-fr-ft-research_release` | pierre, amelie, marie |
| German | `canopylabs/3b-de-ft-research_release` | jana, thomas, max |
| Spanish | `canopylabs/3b-es_it-ft-research_release` | javi, sergio, maria |
| Italian | `canopylabs/3b-es_it-ft-research_release` | pietro, giulia, carlo |
| Mandarin | `canopylabs/3b-zh-ft-research_release` | 长乐, 白芷 |
| Korean | `canopylabs/3b-ko-ft-research_release` | 유나, 준서 |
| Hindi | `canopylabs/3b-hi-ft-research_release` | ऋतिका |

Canopy Labs rate their own Spanish, Italian and Hindi fine-tunes as poor, and the
others as fair. They are not mirrored or indexed yet: each is another 12.7 GB ONNX
export, and all of them share the English checkpoint's CPU cost.
