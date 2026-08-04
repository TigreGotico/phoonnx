"""Orpheus inference adapter — Llama-backbone autoregressive codec-LM over SNAC.

Orpheus (Canopy Labs) is a Llama-3.2-3B causal LM whose vocabulary was extended with
28 672 ``<custom_token_N>`` audio tokens. It emits a flat stream of those tokens; every
seven of them form one **SNAC** frame, which SNAC's 24 kHz decoder turns into 2048 audio
samples. There is no duration model, no vocoder and no speaker encoder — the speaker is
a *name written into the prompt text*.

Two ONNX graphs::

    orpheus_lm.onnx    Llama backbone, KV-cached  (= the voice's ``session``)
    snac_decoder.onnx  three code streams -> waveform[1, 1, T] @ 24 kHz

``orpheus_lm.onnx`` serves prefill *and* decode — the same graph with a different past
length, exactly as :class:`phoonnx.engines.neutts.NeuTTSAdapter`'s does::

    inputs   input_ids                int64 [1, S]           prompt, or 1 token per step
             attention_mask           int64 [1, P + S]       ones over past and current
             past_key_values.<i>.key  fp32  [1, 8, P, 128]   for i in 0..27
             past_key_values.<i>.value
    outputs  logits                   fp32  [1, S, 156940]
             present.<i>.key / present.<i>.value   fp32 [1, 8, P + S, 128]

The KV geometry and the past/present input names are read off the graph's own signature,
so a differently-sized sibling checkpoint needs no code change.

Prompt format
~~~~~~~~~~~~~
Reproduced from what upstream's server *actually sends*, not from its source read
literally — the two differ, and the difference is a **double BOS**. ``OrpheusModel``
builds ``[128259] + tokenizer("{voice}: {text}") + [128009, 128260, 128261, 128257]``,
where the tokenizer has already prepended ``<|begin_of_text|>`` (128000). It then
*decodes that back to a string* and hands the string to vLLM, which re-tokenizes it with
``add_special_tokens=True`` and so prepends a **second** ``<|begin_of_text|>``. The
sequence the checkpoint is actually served is therefore::

    128000  <|begin_of_text|>      (added by vLLM)
    128259  <custom_token_3>       start of human turn
    128000  <|begin_of_text|>      (added by the upstream tokenizer call)
    ...     "{voice}: {text}"
    128009  <|eot_id|>
    128260  <custom_token_4>       end of human turn
    128261  <custom_token_5>       start of AI turn
    128257  <custom_token_1>       start of speech

Dropping the leading BOS — the reading a straightforward HuggingFace port produces — is a
different prompt from the one the model was served, so this adapter reproduces the served
form. See ``scripts/conversion/orpheus/probe_prompt.py`` for the probe that established
it (run it against ``tokenizer.json`` — no weights needed — and see
``scripts/conversion/orpheus/evidence/probe_prompt_output.txt`` for a reproduced run).

Audio tokens
~~~~~~~~~~~~
``<custom_token_10>`` .. ``<custom_token_28681>`` are ids 128266..156937, laid out as
seven interleaved SNAC positions of 4096 codes each. Position ``i`` in the generated run
maps back with::

    code = token_id - 128266 - (i % 7) * 4096

A frame's seven codes fill the three SNAC streams at rates 1 / 2 / 4::

    stream 0 <- [0]
    stream 1 <- [1], [4]
    stream 2 <- [2], [3], [5], [6]

CPU viability
~~~~~~~~~~~~~
SNAC runs at 2048 samples (85.3 ms) per 7-token frame, i.e. ~82 tokens per second of
audio. A 3B backbone needs ~0.37 s per decode step on 12 CPU cores in fp32, which is
around **30x slower than real time** — Orpheus is a GPU engine. Canopy Labs announced
1B / 400M / 150M tiers but never released them (their own loader still raises
``"not supported ... will be released very soon"``), so there is no smaller Orpheus to
fall back to. The voices are indexed for completeness; do not pick one as a CPU default.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime
from quebra_frases import sentence_tokenize

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter
from phoonnx.providers import make_session
from phoonnx.util import LOG

SAMPLE_RATE = 24000

#: samples SNAC emits per 7-token frame, hence the frame rate the LM must keep up with
SAMPLES_PER_FRAME = 2048
TOKENS_PER_FRAME = 7

#: control ids, fixed by the checkpoint's vocabulary (Llama-3 specials + custom_token_N)
BOS = 128000
EOT = 128009
START_OF_HUMAN = 128259     # <custom_token_3>
END_OF_HUMAN = 128260       # <custom_token_4>
START_OF_AI = 128261        # <custom_token_5>
START_OF_SPEECH = 128257    # <custom_token_1>
END_OF_SPEECH = 128258      # <custom_token_2>

#: first audio token id (``<custom_token_10>``) and the number of codes per SNAC position
AUDIO_TOKEN_BASE = 128266
CODEBOOK_SIZE = 4096
AUDIO_TOKEN_LAST = AUDIO_TOKEN_BASE + TOKENS_PER_FRAME * CODEBOOK_SIZE - 1   # 156937


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _apply_repetition_penalty(scores: np.ndarray, seen: np.ndarray, penalty: float) -> np.ndarray:
    """Divide the score of tokens in ``seen`` by ``penalty`` (multiply when negative).

    ``seen`` spans the **prompt as well as the generated run**, and is not windowed. That
    is vLLM's ``repetition_penalty``, which is the sampler upstream serves this checkpoint
    with — unlike llama.cpp's windowed penalty (see
    :data:`phoonnx.engines.neutts.REPETITION_WINDOW`) and unlike penalising only what has
    been generated. Orpheus needs a strong penalty (upstream default 1.3) or it stalls on
    runs of silence tokens.
    """
    if penalty == 1.0 or seen.size == 0:
        return scores
    out = scores.copy()
    idx = np.unique(seen[(seen >= 0) & (seen < out.shape[0])])
    if idx.size:
        s = out[idx]
        out[idx] = np.where(s < 0, s * penalty, s / penalty)
    return out


def _sample(scores: np.ndarray, temperature: float, top_p: float,
            rng: np.random.Generator) -> int:
    """Temperature, then nucleus — vLLM's sampler order (argmax if temperature<=0).

    vLLM applies penalties first, then divides by the temperature, and only then truncates
    to the nucleus, so ``top_p`` selects over the *tempered* distribution. llama.cpp does
    the opposite (truncate on the raw scores, temper the survivors), which is why this is
    spelled out rather than shared with :func:`phoonnx.engines.neutts._sample`: upstream
    serves Orpheus through vLLM, so vLLM's order is the one the defaults were tuned for.
    """
    s = np.asarray(scores, np.float32)
    if temperature <= 0:
        return int(np.argmax(s))
    probs = _softmax(s / float(temperature))
    if 0 < top_p < 1:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        # keep every token up to and including the one that crosses top_p
        keep = order[:max(1, int(np.searchsorted(cumulative, top_p)) + 1)]
        masked = np.zeros_like(probs)
        masked[keep] = probs[keep]
        probs = masked / masked.sum()
    return int(rng.choice(probs.shape[0], p=probs))


class OrpheusAdapter(BaseOnnxAdapter):
    """Adapter for the Orpheus family (Llama codec-LM + SNAC 24 kHz decoder)."""

    #: hard ceiling on the AR loop; upstream's ``max_tokens``. SNAC runs at ~82 tokens/s,
    #: so 1200 is ~14.6 s of audio.
    MAX_NEW_TOKENS = 1200
    #: characters of target text per model call
    MAX_CHUNK_CHARS = 300

    def __init__(self):
        self.snac: Optional[onnxruntime.InferenceSession] = None
        self.tokenizer = None
        self.voices: List[str] = []
        self.default_voice: Optional[str] = None
        self.past_names: List[str] = []
        self.present_names: List[str] = []
        self.num_kv_heads = 0
        self.head_dim = 0

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def default_params(self) -> Dict[str, Any]:
        """Upstream's serving defaults (``OrpheusModel.generate_tokens_sync``).

        Note these are *not* the checkpoint's ``generation_config.json`` values
        (``top_p`` 0.9 there, 0.8 here): the served stack overrides them, and the served
        values are what the model was tuned against.
        """
        return {
            "temperature": 0.6,
            "top_p": 0.8,
            "repetition_penalty": 1.3,
            "max_new_tokens": float(self.MAX_NEW_TOKENS),
            "max_chunk_chars": float(self.MAX_CHUNK_CHARS),
        }

    def param_labels(self) -> Dict[str, str]:
        return {
            "temperature": "sampling temperature",
            "top_p": "top-p (nucleus)",
            "repetition_penalty": "repetition penalty",
            "max_new_tokens": "max audio tokens (~82/s)",
            "max_chunk_chars": "target characters per model call",
            "voice": "voice name written into the prompt",
        }

    def configure(self, voice_config: Any) -> None:
        """Load the SNAC decoder, the checkpoint's BPE and the voice list from
        ``engine_params``."""
        ep = getattr(voice_config, "engine_params", None) or {}
        if self.snac is None and ep.get("snac_decoder_path"):
            self.snac = make_session(ep["snac_decoder_path"], providers=ep.get("providers"))
        if self.tokenizer is None and ep.get("tokenizer_path"):
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(str(ep["tokenizer_path"]))
        if not self.voices:
            self.voices = list(ep.get("voices") or [])
            self.default_voice = ep.get("default_voice") or (self.voices[0] if self.voices else None)

    def _read_kv_shape(self, session: onnxruntime.InferenceSession) -> None:
        """Read the KV-cache names and geometry off the graph's own signature."""
        self.past_names = [i.name for i in session.get_inputs()
                           if i.name.startswith("past_key_values")]
        self.present_names = [o.name for o in session.get_outputs()
                              if o.name.startswith("present")]
        for spec in session.get_inputs():
            if spec.name.startswith("past_key_values"):
                self.num_kv_heads = int(spec.shape[1])
                self.head_dim = int(spec.shape[3])
                break

    # ------------------------------------------------------------------
    # text
    # ------------------------------------------------------------------
    def resolve_voice(self, voice: Any, syn_config: Any) -> Optional[str]:
        """Pick the voice name: an explicit per-call ``voice``, else the speaker this
        phoonnx voice is pinned to, else the checkpoint's default.

        A multi-speaker Orpheus voice carries its names in ``speaker_id_map``, so a plain
        ``speaker_id`` selects one too — the name, not the index, is what reaches the
        model.
        """
        name = None
        if syn_config is not None:
            name = (syn_config.extra_params or {}).get("voice")
        cfg = getattr(voice, "config", None)
        if not name and syn_config is not None and syn_config.speaker_id is not None:
            id_map = getattr(cfg, "speaker_id_map", None) or {}
            for speaker, index in id_map.items():
                if int(index) == int(syn_config.speaker_id):
                    name = speaker
                    break
        if not name:
            name = (getattr(cfg, "engine_params", None) or {}).get("voice")
        if not name:
            name = self.default_voice
        if name and self.voices and name not in self.voices:
            raise ValueError(f"unknown Orpheus voice {name!r}; available: {sorted(self.voices)}")
        return name

    def chunk_text(self, text: str, max_chars: int) -> List[str]:
        """Pack whole sentences up to ``max_chars`` per model call.

        Each call is an independent prompt, so a long passage is split before synthesis
        rather than pushed through one 1200-token generation. Sentences come from the same
        ``quebra_frases`` splitter the rest of phoonnx uses; an over-long single sentence is
        split on word boundaries rather than dropped.
        """
        text = (text or "").strip()
        if not text:
            return []
        if len(text) <= max_chars:
            return [text]
        chunks: List[str] = []
        current = ""
        for sentence in sentence_tokenize(text) or [text]:
            sentence = sentence.strip()
            if not sentence:
                continue
            pieces = [sentence]
            if len(sentence) > max_chars:
                pieces, word_chunk = [], ""
                for word in sentence.split():
                    if word_chunk and len(word_chunk) + 1 + len(word) > max_chars:
                        pieces.append(word_chunk)
                        word_chunk = word
                    else:
                        word_chunk = f"{word_chunk} {word}".strip()
                if word_chunk:
                    pieces.append(word_chunk)
            for piece in pieces:
                if current and len(current) + 1 + len(piece) > max_chars:
                    chunks.append(current)
                    current = ""
                current = f"{current} {piece}".strip()
        if current:
            chunks.append(current)
        return chunks

    def build_prompt_ids(self, text: str, voice: Optional[str]) -> List[int]:
        """The exact id sequence upstream's vLLM server receives — including the
        double BOS documented in the module docstring."""
        body = f"{voice}: {text}" if voice else text
        # add_special_tokens=True reproduces the upstream tokenizer call, which prepends
        # the inner BOS; the outer BOS is the one vLLM adds when it re-tokenizes.
        core = list(self.tokenizer.encode(body, add_special_tokens=True).ids)
        return [BOS, START_OF_HUMAN] + core + [EOT, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]

    def encode_text(self, text: str, voice: Any, syn_config: Any) -> List[List[int]]:
        """Build one fully-formed prompt per model call and BPE-encode it.

        Orpheus consumes raw text, not phonemes: the voice name and the emotive tags
        (``<laugh>``, ``<sigh>``, ...) are ordinary text the checkpoint's own BPE encodes,
        so nothing here phonemizes.
        """
        if self.tokenizer is None:
            raise RuntimeError("Orpheus voice missing tokenizer_path in engine_params")
        budget = int(self.MAX_CHUNK_CHARS)
        if syn_config is not None:
            budget = int((syn_config.extra_params or {}).get("max_chunk_chars", budget))
        name = self.resolve_voice(voice, syn_config)
        return [self.build_prompt_ids(chunk, name)
                for chunk in self.chunk_text(text, max(1, budget))]

    # ------------------------------------------------------------------
    # codec
    # ------------------------------------------------------------------
    @staticmethod
    def token_ids_to_codes(token_ids: List[int]) -> List[List[int]]:
        """Flat audio-token run -> the three SNAC code streams.

        Position ``i`` carries a ``(i % 7)``-th offset, so the de-interleaving and the
        stream layout are one operation: a partial trailing frame is dropped, because SNAC
        needs whole frames.
        """
        codes = [int(t) - AUDIO_TOKEN_BASE - (i % TOKENS_PER_FRAME) * CODEBOOK_SIZE
                 for i, t in enumerate(token_ids)]
        frames = len(codes) // TOKENS_PER_FRAME
        s0: List[int] = []
        s1: List[int] = []
        s2: List[int] = []
        for f in range(frames):
            c = codes[f * TOKENS_PER_FRAME:(f + 1) * TOKENS_PER_FRAME]
            if any(v < 0 or v >= CODEBOOK_SIZE for v in c):
                # a frame whose codes fall outside their codebook is corrupt, not silent;
                # dropping it is better than handing SNAC an out-of-range index
                LOG.warning("Orpheus dropped an out-of-range SNAC frame at %d", f)
                continue
            s0.append(c[0])
            s1.extend((c[1], c[4]))
            s2.extend((c[2], c[3], c[5], c[6]))
        return [s0, s1, s2]

    def decode_codes(self, streams: List[List[int]]) -> np.ndarray:
        """Three SNAC code streams -> mono float32 waveform at 24 kHz."""
        if not streams[0]:
            return np.zeros(0, np.float32)
        names = [i.name for i in self.snac.get_inputs()]
        feed = {n: np.asarray(s, np.int64).reshape(1, -1) for n, s in zip(names, streams)}
        audio = self.snac.run(None, feed)[0]
        return np.asarray(audio, np.float32).reshape(-1)

    # ------------------------------------------------------------------
    # autoregressive loop
    # ------------------------------------------------------------------
    def generate(self, session: onnxruntime.InferenceSession, prompt_ids: List[int],
                 p: Dict[str, Any], rng: np.random.Generator) -> List[int]:
        """Prefill the prompt, then sample audio tokens until the model stops."""
        if not self.past_names:
            self._read_kv_shape(session)

        ids = np.asarray(prompt_ids, np.int64).reshape(1, -1)
        length = ids.shape[1]
        empty = np.zeros((1, self.num_kv_heads, 0, self.head_dim), np.float32)
        attention = np.ones((1, length), np.int64)
        out = session.run(None, {"input_ids": ids, "attention_mask": attention,
                                 **{n: empty for n in self.past_names}})
        logits, present = out[0], out[1:]

        temperature = float(p.get("temperature", 0.6))
        top_p = float(p.get("top_p", 0.8))
        penalty = float(p.get("repetition_penalty", 1.3))
        max_new = max(1, int(p.get("max_new_tokens", self.MAX_NEW_TOKENS)))

        generated: List[int] = []
        history = list(prompt_ids)
        for step in range(max_new):
            scores = np.asarray(logits, np.float32)[0, -1]
            scores = _apply_repetition_penalty(scores, np.asarray(history, np.int64), penalty)
            token = _sample(scores, temperature, top_p, rng)
            # upstream passes ``stop_token_ids=[49158]``, which is the *text* token
            # "Ġrez" and can never be emitted mid-audio — a no-op left in their serving
            # code. The run really ends on end-of-speech or the checkpoint's EOS.
            if token in (END_OF_SPEECH, EOT):
                break
            if not (AUDIO_TOKEN_BASE <= token <= AUDIO_TOKEN_LAST):
                LOG.warning("Orpheus emitted non-audio token %d at step %d — stopping",
                            token, step)
                break
            generated.append(token)
            history.append(token)
            if step == max_new - 1:
                LOG.warning("Orpheus hit the %d-token cap (~%.1fs) without an end token",
                            max_new, max_new / 82.0)
                break
            attention = np.concatenate([attention, np.ones((1, 1), np.int64)], axis=1)
            out = session.run(None, {
                "input_ids": np.asarray([[token]], np.int64),
                "attention_mask": attention,
                **dict(zip(self.past_names, present)),
            })
            logits, present = out[0], out[1:]
        return generated

    # ------------------------------------------------------------------
    # synthesis
    # ------------------------------------------------------------------
    def synthesize(self, request: AdapterSynthesisRequest,
                   session: onnxruntime.InferenceSession) -> AdapterSynthesisResult:
        if self.snac is None:
            raise RuntimeError("Orpheus voice missing snac_decoder_path in engine_params")
        if self.tokenizer is None:
            raise RuntimeError("Orpheus voice missing tokenizer_path in engine_params")
        if request.params.get("reference_audio") is not None:
            raise RuntimeError(
                "Orpheus clones in context from a transcribed reference, not from a bare "
                "clip: pass speaker_reference_text alongside the audio, or select a named "
                "voice with extra_params={'voice': ...}")

        p = {**self.default_params(), **request.params}
        seed = p.get("seed")
        rng = np.random.default_rng(None if seed is None else int(seed))
        prompt_ids = [int(i) for i in np.asarray(request.phoneme_ids).reshape(-1)]
        token_ids = self.generate(session, prompt_ids, p, rng)
        streams = self.token_ids_to_codes(token_ids)
        if not streams[0]:
            raise RuntimeError("Orpheus generated no complete SNAC frame — lower the "
                               "temperature or pick another voice")
        return AdapterSynthesisResult(audio=self.decode_codes(streams),
                                      extras={"frames": len(streams[0]),
                                              "tokens": len(token_ids)})

    # required by the ABC but unused — synthesize() drives the AR loop.
    def build_feed_dict(self, request, session):
        raise NotImplementedError("Orpheus is autoregressive — use synthesize()")

    def parse_outputs(self, outputs, request, output_names=None):
        raise NotImplementedError("Orpheus is autoregressive — use synthesize()")

    @staticmethod
    def detect(config: Optional[Dict[str, Any]] = None,
               session: Optional[onnxruntime.InferenceSession] = None) -> bool:
        return bool(config and config.get("engine") == "orpheus")
