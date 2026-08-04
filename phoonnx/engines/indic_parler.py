"""Indic Parler-TTS inference adapter — AI4Bharat's description-controlled codec LM.

Indic Parler-TTS (``ai4bharat/indic-parler-tts``) is the Indic fine-tune of Parler-TTS
(Lacombe, Srivastav, Gandhi). It speaks 20 Indic languages plus English, and the voice is
chosen by a **natural-language description** ("Rohit's voice is clear and expressive...")
rather than by a speaker id or a reference clip.

This is phoonnx's first **encoder-decoder** engine. Every other autoregressive engine
here (Chatterbox, NeuTTS, Spark-TTS, OuteTTS, ArkTTS, Qwen3-TTS) is decoder-only: one
KV cache, one token stream. Parler splits the conditioning in two:

* a frozen Flan-T5 encoder turns the *description* into encoder states that every decoder
  layer **cross-attends** to. Those cross-attention keys/values do not change during
  generation, so they are computed once in the prefill graph and then held constant --
  the decode graph consumes them but never re-emits them;
* the *text to speak* is embedded by ``embed_prompts`` and **prepended to the decoder's
  input embeddings** (``prompt_cross_attention=False`` upstream). It is part of the
  self-attention stream, not of the cross-attention stream.

Four ONNX graphs (mirror: ``OpenVoiceOS/phoonnx-indic-parler``)::

    text_encoder     : input_ids, attention_mask -> encoder_hidden_states
    decoder_prefill  : codec_input_ids, prompt_input_ids, encoder_hidden_states,
                       encoder_attention_mask
                       -> logits, self-KV (24x2), cross-KV (24x2)      (= ``session``)
    decoder_decode   : codec_input_ids, encoder_attention_mask, self-KV, cross-KV
                       -> logits, self-KV (24x2)
    dac_decoder      : audio_codes (9 codebooks) -> waveform @ 44.1 kHz

The decoder writes 9 DAC codebooks under a **delay pattern**: codebook *k* is offset by
*k* steps, so a frame is only complete 9 steps after it starts. ``build_delay_pattern_mask``
and ``apply_delay_pattern_mask`` below are numpy ports of the upstream helpers, and
``_eos_delay_constraint`` ports ``ParlerTTSLogitsProcessor``: end-of-speech may only move
one codebook at a time, in order, so the tail of the delay pattern stays well-formed.

Upstream defaults to sampling (``do_sample=True``); greedy decoding tends to ramble past
the end of the sentence. Sampling is therefore the default here too, with ``temperature``
and ``top_k`` exposed as engine parameters. Set ``do_sample`` to 0 for the deterministic
path used by the parity tests.
"""
import json
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime
from quebra_frases import sentence_tokenize

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter
from phoonnx.providers import make_session

NUM_CODEBOOKS = 9
BOS_TOKEN_ID = 1025
EOS_TOKEN_ID = 1024
PAD_TOKEN_ID = 1024
AUDIO_VOCAB_SIZE = 1088
SAMPLE_RATE = 44100

# Languages AI4Bharat lists as officially supported (ISO 639 codes).
AVAILABLE_LANGS = ["as", "bn", "brx", "doi", "en", "gu", "hi", "kn", "kok", "mai",
                    "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta",
                    "te", "ur"]


def build_delay_pattern_mask(start_len: int, max_length: int,
                             num_codebooks: int = NUM_CODEBOOKS,
                             bos_token_id: int = BOS_TOKEN_ID,
                             pad_token_id: int = PAD_TOKEN_ID):
    """Numpy port of ``parler_tts.build_delay_pattern_mask`` for a bare BOS start.

    Returns ``(start_ids, pattern)``. ``pattern`` is ``(num_codebooks, max_length)``:
    ``-1`` marks a position the model must predict, anything else is a forced token
    (BOS in the lower triangle, PAD in the upper one). ``start_ids`` is the prefix that
    is already determined before the first prediction.
    """
    ids = np.full((1, num_codebooks, start_len), bos_token_id, dtype=np.int64)
    if max_length < 2 * num_codebooks - 1:
        # Too short for the staircase to close: upstream returns the ids untouched and a
        # mask of all -1 ("predict everything"). Matters at the tail, where a sequence
        # that stopped early is stripped against its own (short) length.
        return (ids.reshape(num_codebooks, -1),
                np.full((num_codebooks, max_length), -1, dtype=np.int64))
    shifted = np.full((1, num_codebooks, max_length), -1, dtype=np.int64)
    for codebook in range(num_codebooks):
        shifted[:, codebook, codebook:start_len + codebook] = ids[:, codebook]
    eos_delay = np.triu(np.ones((num_codebooks, max_length), bool),
                        k=max_length - num_codebooks + 1)
    bos_delay = np.tril(np.ones((num_codebooks, max_length), bool))
    keep = ~(bos_delay | eos_delay)
    pattern = keep * shifted + bos_delay * bos_token_id + eos_delay * pad_token_id
    pattern = pattern.reshape(num_codebooks, -1)
    predict_at = np.nonzero(pattern[0] == -1)[0]
    first = int(predict_at.min()) if len(predict_at) else start_len
    return pattern[:, :first].copy(), pattern


def apply_delay_pattern_mask(ids: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Keep predictions only where the pattern says ``-1``; force the rest."""
    p = pattern[..., : ids.shape[-1]]
    return np.where(p == -1, ids, p)


def strip_delay_pattern(raw_ids: np.ndarray, pattern: np.ndarray,
                        num_codebooks: int = NUM_CODEBOOKS) -> np.ndarray:
    """Undo the delay: drop the forced BOS/PAD cells and re-square the codebooks."""
    masked = apply_delay_pattern_mask(raw_ids, pattern)
    _, full = build_delay_pattern_mask(1, masked.shape[-1], num_codebooks)
    keep = (full != BOS_TOKEN_ID) & (full != PAD_TOKEN_ID)
    return masked[keep].reshape(num_codebooks, -1)


def sample_logits(logits: np.ndarray, temperature: float, top_k: int, rng) -> np.ndarray:
    """Temperature + top-k sampling, one draw per codebook row."""
    x = logits.astype(np.float64) / max(temperature, 1e-5)
    if top_k and top_k < x.shape[-1]:
        kth = np.partition(x, -top_k, axis=-1)[:, -top_k][:, None]
        x = np.where(x < kth, -np.inf, x)
    x = x - x.max(axis=-1, keepdims=True)
    probs = np.exp(x)
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.array([rng.choice(probs.shape[-1], p=probs[i]) for i in range(probs.shape[0])],
                    dtype=np.int64)


class IndicParlerAdapter(BaseOnnxAdapter):
    """Adapter for Indic Parler-TTS (T5 encoder -> AR decoder over DAC codes -> DAC decoder)."""

    #: hard ceiling from the upstream generation config (2610 total positions)
    MAX_LENGTH = 2610

    def __init__(self, temperature: float = 1.0, top_k: int = 0,
                 do_sample: bool = True, max_new_tokens: int = 1500):
        self.text_encoder: Optional[onnxruntime.InferenceSession] = None
        self.decode_session: Optional[onnxruntime.InferenceSession] = None
        self.dac_decoder: Optional[onnxruntime.InferenceSession] = None
        self.prompt_tokenizer = None
        self.description_tokenizer = None
        self.description: str = ""
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.do_sample = bool(do_sample)
        self.max_new_tokens = int(max_new_tokens)
        self.seed: Optional[int] = None
        self.sample_rate = SAMPLE_RATE
        self._decode_inputs: set = set()
        self._prefill_outputs: List[str] = []
        self._decode_outputs: List[str] = []

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def default_params(self) -> Dict[str, float]:
        return {"temperature": self.temperature, "top_k": float(self.top_k),
                "do_sample": float(self.do_sample), "max_new_tokens": float(self.max_new_tokens)}

    def param_labels(self) -> Dict[str, str]:
        return {"temperature": "Sampling temperature", "top_k": "Top-k cutoff (0 = off)",
                "do_sample": "Sample (1) or greedy (0)", "max_new_tokens": "Max codec frames"}

    def configure(self, voice_config: Any) -> None:
        """Load the three auxiliary graphs, both tokenizers and the voice description.

        The description is the *voice*: it is a plain string in ``engine_options`` of the
        index entry, taken verbatim from AI4Bharat's own recommended-speaker list, so a
        voice id resolves to exactly the wording the model was trained to answer to.
        """
        ep = getattr(voice_config, "engine_params", None) or {}
        providers = ep.get("providers")
        if self.text_encoder is None and ep.get("text_encoder_path"):
            self.text_encoder = make_session(ep["text_encoder_path"], providers=providers)
        if self.decode_session is None and ep.get("decoder_decode_path"):
            self.decode_session = make_session(ep["decoder_decode_path"], providers=providers)
            self._decode_inputs = {i.name for i in self.decode_session.get_inputs()}
            self._decode_outputs = [o.name for o in self.decode_session.get_outputs()]
        if self.dac_decoder is None and ep.get("dac_decoder_path"):
            self.dac_decoder = make_session(ep["dac_decoder_path"], providers=providers)
        if self.prompt_tokenizer is None and ep.get("prompt_tokenizer_path"):
            from tokenizers import Tokenizer
            self.prompt_tokenizer = Tokenizer.from_file(str(ep["prompt_tokenizer_path"]))
        if self.description_tokenizer is None and ep.get("description_tokenizer_path"):
            from tokenizers import Tokenizer
            self.description_tokenizer = Tokenizer.from_file(str(ep["description_tokenizer_path"]))
        if ep.get("description"):
            self.description = str(ep["description"])
        for key, cast in (("temperature", float), ("top_k", int), ("max_new_tokens", int)):
            if key in ep:
                setattr(self, key, cast(ep[key]))
        if "do_sample" in ep:
            self.do_sample = bool(int(ep["do_sample"]))
        if "seed" in ep and ep["seed"] is not None:
            self.seed = int(ep["seed"])
        if ep.get("model_config_path"):
            with open(ep["model_config_path"], "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.sample_rate = int(meta.get("sample_rate", self.sample_rate))

    # ------------------------------------------------------------------
    # text
    # ------------------------------------------------------------------

    def encode_text(self, text: str, voice: Any, syn_config: Any) -> List[List[int]]:
        """Tokenize raw text with the checkpoint's own prompt tokenizer, one call per
        sentence. Parler has no phoneme front end: the prompt tokenizer owns
        normalization, and the description tokenizer (a *different* vocabulary, the
        Flan-T5 one) is used only for the voice description."""
        if self.prompt_tokenizer is None:
            raise RuntimeError("Indic Parler voice missing prompt_tokenizer_path in engine_params")
        sentences = [s.strip() for s in sentence_tokenize(text) if s.strip()] or [text.strip()]
        return [list(self.prompt_tokenizer.encode(s).ids) for s in sentences if s]

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def encode_description(self, description: str):
        """Run the Flan-T5 encoder over the voice description.

        Returns ``(encoder_hidden_states, encoder_attention_mask)``. The mask is applied
        to the hidden states inside the exported graph, exactly as upstream does before
        handing them to the decoder.
        """
        if self.text_encoder is None or self.description_tokenizer is None:
            raise RuntimeError("Indic Parler voice missing text_encoder_path / "
                               "description_tokenizer_path in engine_params")
        enc = self.description_tokenizer.encode(description)
        input_ids = np.asarray([enc.ids], np.int64)
        attention_mask = np.ones_like(input_ids, np.int64)
        hidden = self.text_encoder.run(None, {"input_ids": input_ids,
                                              "attention_mask": attention_mask})[0]
        return np.asarray(hidden, np.float32), attention_mask

    @staticmethod
    def _eos_delay_constraint(logits: np.ndarray, raw_ids: np.ndarray,
                              first_unfinished: int) -> int:
        """Port of ``ParlerTTSLogitsProcessor``.

        Only one codebook at a time may emit end-of-speech, in codebook order. Every
        codebook above the lowest unfinished one has its EOS logit driven to ``-inf``,
        which is what keeps the delay pattern's tail consistent.
        """
        seen_eos = (raw_ids == EOS_TOKEN_ID).sum(axis=1)
        if seen_eos[first_unfinished] > 0 and first_unfinished < NUM_CODEBOOKS - 1:
            first_unfinished += 1
        logits[np.arange(NUM_CODEBOOKS) > first_unfinished, EOS_TOKEN_ID] = -np.inf
        return first_unfinished

    def generate_codes(self, prompt_ids: np.ndarray, encoder_hidden_states: np.ndarray,
                       encoder_attention_mask: np.ndarray,
                       session: onnxruntime.InferenceSession,
                       max_new_tokens: int, do_sample: bool, temperature: float,
                       top_k: int, rng) -> np.ndarray:
        """Autoregressive loop: prefill once, then step the decode graph with the cache.

        The cross-attention KV produced by the prefill graph is kept in the cache dict
        and fed back unchanged on every step; only the self-attention KV is refreshed
        from the decode graph's outputs.
        """
        if self.decode_session is None:
            raise RuntimeError("Indic Parler voice missing decoder_decode_path in engine_params")

        max_length = min(max_new_tokens + 1, self.MAX_LENGTH)
        start_ids, pattern = build_delay_pattern_mask(1, max_length)
        raw = apply_delay_pattern_mask(start_ids, pattern)

        outputs = session.run(None, {
            "codec_input_ids": raw[None, ...].astype(np.int64),
            "prompt_input_ids": prompt_ids.astype(np.int64),
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask.astype(np.int64),
        })
        if not self._prefill_outputs:
            self._prefill_outputs = [o.name for o in session.get_outputs()]
        named = dict(zip(self._prefill_outputs, outputs))
        logits = named["logits"]
        cache = {k.replace("present.", "past_key_values."): v
                 for k, v in named.items() if k.startswith("present.")}

        first_unfinished = 0
        unfinished = np.ones(NUM_CODEBOOKS, bool)
        for _ in range(max_new_tokens):
            step_logits = np.asarray(logits, np.float32).reshape(NUM_CODEBOOKS, -1).copy()
            first_unfinished = self._eos_delay_constraint(step_logits, raw, first_unfinished)
            if do_sample:
                nxt = sample_logits(step_logits, temperature, top_k, rng)
            else:
                nxt = step_logits.argmax(axis=-1).astype(np.int64)
            nxt = np.where(unfinished, nxt, PAD_TOKEN_ID)
            raw = np.concatenate([raw, nxt[:, None]], axis=-1)
            unfinished &= nxt != EOS_TOKEN_ID
            if not unfinished.any() or raw.shape[-1] >= max_length:
                break

            model_ids = apply_delay_pattern_mask(raw, pattern)
            feed = {"codec_input_ids": model_ids[None, :, -1:].astype(np.int64),
                    "encoder_attention_mask": encoder_attention_mask.astype(np.int64)}
            feed = {k: v for k, v in feed.items() if k in self._decode_inputs}
            for k, v in cache.items():
                if k in self._decode_inputs:
                    feed[k] = v
            step_out = dict(zip(self._decode_outputs, self.decode_session.run(None, feed)))
            logits = step_out["logits"]
            for k, v in step_out.items():
                if k.startswith("present."):
                    cache[k.replace("present.", "past_key_values.")] = v

        return strip_delay_pattern(raw, pattern)

    def decode_audio(self, codes: np.ndarray) -> np.ndarray:
        """Run the DAC decoder over ``(9, frames)`` codes and return mono float32."""
        if self.dac_decoder is None:
            raise RuntimeError("Indic Parler voice missing dac_decoder_path in engine_params")
        if codes.shape[-1] == 0:
            return np.zeros(0, np.float32)
        wav = self.dac_decoder.run(None, {"audio_codes": codes[None, ...].astype(np.int64)})[0]
        return np.asarray(wav, np.float32).reshape(-1)

    def synthesize(self, request: AdapterSynthesisRequest,
                   session: onnxruntime.InferenceSession) -> AdapterSynthesisResult:
        p = request.params
        description = str(p.get("description") or self.description)
        if not description:
            raise RuntimeError("Indic Parler voices need a speaker description "
                               "(engine_options['description'])")
        do_sample = bool(int(p.get("do_sample", self.do_sample)))
        temperature = float(p.get("temperature", self.temperature))
        top_k = int(p.get("top_k", self.top_k))
        max_new_tokens = int(p.get("max_new_tokens", self.max_new_tokens))
        seed = p.get("seed", self.seed)
        rng = np.random.default_rng(None if seed is None else int(seed))

        enc_hidden, enc_mask = self.encode_description(description)
        prompt_ids = np.asarray(request.phoneme_ids, np.int64).reshape(1, -1)
        codes = self.generate_codes(prompt_ids, enc_hidden, enc_mask, session,
                                    max_new_tokens, do_sample, temperature, top_k, rng)
        audio = self.decode_audio(codes)
        return AdapterSynthesisResult(audio=audio,
                                      extras={"codes": codes, "description": description,
                                              "sample_rate": self.sample_rate})

    # build_feed_dict / parse_outputs are required by the ABC but unused —
    # synthesize() drives the four-graph pipeline directly.
    def build_feed_dict(self, request: AdapterSynthesisRequest,
                        session: onnxruntime.InferenceSession) -> Dict[str, np.ndarray]:
        raise NotImplementedError("Indic Parler is multi-graph — use synthesize()")

    def parse_outputs(self, outputs: List[np.ndarray], request: AdapterSynthesisRequest,
                      output_names: Optional[List[str]] = None) -> AdapterSynthesisResult:
        raise NotImplementedError("Indic Parler is multi-graph — use synthesize()")

    @staticmethod
    def detect(config: Optional[Dict[str, Any]] = None,
               session: Optional[onnxruntime.InferenceSession] = None) -> bool:
        if config and config.get("engine") == "indic_parler":
            return True
        if session is not None:
            names = {i.name for i in session.get_inputs()}
            # the prefill graph's signature: a codec stream plus a *separate* prompt
            # stream plus cross-attention states — no other phoonnx engine has all three
            if {"codec_input_ids", "prompt_input_ids", "encoder_hidden_states"} <= names:
                return True
        return False
