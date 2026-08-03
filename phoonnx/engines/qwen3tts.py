"""Qwen3-TTS inference adapter — two-stage codec LM with 16 code groups.

Qwen3-TTS (Alibaba Qwen, Apache-2.0, ``QwenLM/Qwen3-TTS``) predicts the tokens of a
12.5 Hz neural codec and turns them into a waveform at 24 kHz. It has two stacked
autoregressive models, not one:

* the **talker** — 28 decoder layers that emit one token per audio frame. That token
  is code group 0, and its hidden state conditions the second model;
* the **code predictor** (upstream calls it the sub-talker) — 5 decoder layers that
  read the talker's hidden state and write the other 15 code groups of the *same*
  frame, one group per step.

So one 80 ms frame costs one talker step plus 15 code-predictor steps. The 16 groups
of a frame are then fed back to the talker as a single summed embedding, together
with the next slice of the text. This engine drives the overridable
``BaseOnnxAdapter.synthesize`` rather than the single-graph ``build_feed_dict`` path,
like the sibling :class:`phoonnx.engines.sparktts.SparkTTSAdapter`.

Seven ONNX graphs (HF: ``OpenVoiceOS/phoonnx-qwen3-tts``)::

    talker.onnx                 28-layer KV-cached talker step (= the voice's ``session``)
    text_embed.onnx             text ids -> projected text hidden (1024)
    codec_embed.onnx            codec ids -> codec hidden (1024)
    code_predictor_prefill.onnx [talker hidden, group-0 embed] -> group-1 logits + KV
    code_predictor_step.onnx    group-n embed -> group-(n+1) logits + KV
    sub_codec_embed.onnx        code-group token -> its group's embedding
    codec_decoder.onnx          (1, 16, T) codes -> waveform @ 24 kHz

plus ``tokenizer.json`` (the model's own Qwen2 subword BPE).

The talker reads *embeddings*, never ids: at every position a text hidden state and a
codec hidden state are summed. ``build_prompt`` reproduces the layout upstream builds
in ``Qwen3TTSForConditionalGeneration.generate`` for non-streaming input — the
assistant role tokens, a language "think" block, the speaker embedding, then the whole
text against codec padding, and finally the codec BOS that starts the audio.

Voices are the nine timbres the CustomVoice checkpoint was trained with; a voice is
one entry in ``spk_id``, so no reference clip and no speaker encoder are involved.
Two of them are Chinese dialect voices, and upstream overrides the language tag for
those, which :func:`resolve_language_id` reproduces.

The sampler order follows HuggingFace ``generate``, the stack upstream uses: a
repetition penalty over the tokens emitted so far, then the minimum-new-tokens floor
that blocks an immediate end of speech, then the suppressed control-token range, and
only then temperature, top-k and top-p. Reordering these changes which tokens survive.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime
from quebra_frases import sentence_tokenize

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter
from phoonnx.providers import make_session
from phoonnx.tokenizer import BPETokenizer
from phoonnx.util import LOG

SAMPLE_RATE = 24000
"""The codec decoder writes 24 kHz audio."""

FRAME_RATE = 12.5
"""Code frames per second; one frame is 1920 samples."""

UPSAMPLE = 1920
"""Waveform samples produced per code frame."""

NUM_CODE_GROUPS = 16
"""Codebooks per frame: group 0 from the talker, groups 1-15 from the code predictor."""

TALKER_LAYERS, TALKER_KV_HEADS, TALKER_HEAD_DIM = 28, 8, 128
PREDICTOR_LAYERS, PREDICTOR_KV_HEADS, PREDICTOR_HEAD_DIM = 5, 8, 128

TALKER_VOCAB = 3072
"""Talker codebook plus the control tokens above it."""

SUPPRESS_FROM = TALKER_VOCAB - 1024
"""Upstream suppresses every token from here up, except the end-of-speech token."""

MIN_NEW_TOKENS = 2
"""Upstream forbids ending speech before this many frames."""

MAX_NEW_TOKENS = 4096
"""Upstream generation budget — about 5.5 minutes of audio at 12.5 frames per second."""

MAX_CHUNK_CHARS = 300
"""Longest text handed to one autoregressive pass; longer input is split on sentences."""

DECODE_CHUNK, DECODE_LEFT_CONTEXT = 300, 25
"""Frames per codec-decoder call and the left context each call re-reads."""

# Text-side control tokens (ids in the Qwen2 text vocabulary).
TTS_PAD_TOKEN, TTS_BOS_TOKEN, TTS_EOS_TOKEN = 151671, 151672, 151673

# Codec-side control tokens (ids in the talker vocabulary).
CODEC_PAD, CODEC_BOS, CODEC_EOS = 2148, 2149, 2150
CODEC_THINK, CODEC_NOTHINK = 2154, 2155
CODEC_THINK_BOS, CODEC_THINK_EOS = 2156, 2157

ROLE_PREFIX_TOKENS = 3
"""``<|im_start|>assistant\\n`` — the part of the prompt that stays outside the sum."""

TAIL_TOKENS = 5
"""``<|im_end|>\\n<|im_start|>assistant\\n`` — the closing part, never spoken."""

LANGUAGE_IDS: Dict[str, int] = {
    "chinese": 2055, "english": 2050, "german": 2053, "italian": 2070,
    "portuguese": 2071, "spanish": 2054, "japanese": 2058, "korean": 2064,
    "french": 2061, "russian": 2069,
    "beijing_dialect": 2074, "sichuan_dialect": 2062,
}
"""Talker tokens that name the language of the utterance."""

LANG_CODE_TO_NAME: Dict[str, str] = {
    "zh": "chinese", "en": "english", "de": "german", "it": "italian",
    "pt": "portuguese", "es": "spanish", "ja": "japanese", "ko": "korean",
    "fr": "french", "ru": "russian",
}
"""BCP-47 primary subtag -> the language name the talker was trained with."""

SPEAKER_IDS: Dict[str, int] = {
    "vivian": 3065, "serena": 3066, "uncle_fu": 3010, "dylan": 2878, "eric": 2875,
    "ryan": 3061, "aiden": 2861, "ono_anna": 2873, "sohee": 2864,
}
"""The nine timbres of the CustomVoice checkpoint."""

SPEAKER_DIALECT: Dict[str, str] = {"eric": "sichuan_dialect", "dylan": "beijing_dialect"}
"""Voices that override the Chinese language tag with a dialect tag."""


def resolve_language_name(lang: Optional[str]) -> str:
    """Map a language tag or name to the name the talker knows.

    Accepts what a voice config carries (``"en-US"``), what upstream's API takes
    (``"English"``) and the ``"auto"`` sentinel, which drops the language block from
    the prompt and lets the model infer the language from the text.
    """
    if not lang:
        return "auto"
    name = str(lang).strip().lower().replace("-", "_")
    if name in LANGUAGE_IDS or name == "auto":
        return name
    primary = name.split("_")[0]
    return LANG_CODE_TO_NAME.get(primary, "auto")


def resolve_language_id(language: str, speaker: str) -> Optional[int]:
    """The language token for this call, or ``None`` for the no-language prompt.

    A dialect voice wins over the requested language whenever that language is
    Chinese or unset, which is what upstream does: Dylan always speaks Beijing
    Mandarin and Eric always speaks Sichuanese, whatever tag the caller passed.
    """
    speaker = speaker.lower()
    language = resolve_language_name(language)
    if language in ("chinese", "auto") and speaker in SPEAKER_DIALECT:
        return LANGUAGE_IDS[SPEAKER_DIALECT[speaker]]
    if language == "auto":
        return None
    return LANGUAGE_IDS[language]


def chunk_text(text: str, max_len: int = MAX_CHUNK_CHARS) -> List[str]:
    """Pack whole sentences into chunks of at most ``max_len`` characters.

    One chunk is one autoregressive pass. Sentence boundaries come from
    ``quebra_frases.sentence_tokenize``, the splitter the rest of phoonnx uses.
    """
    chunks: List[str] = []
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n+", text.strip())):
        if not paragraph:
            continue
        current = ""
        for sentence in sentence_tokenize(paragraph):
            if len(current) + len(sentence) + 1 <= max_len:
                current += (" " if current else "") + sentence
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence
        if current:
            chunks.append(current.strip())
    return chunks or ([text.strip()] if text.strip() else [])


def apply_logits_processors(logits: np.ndarray, generated: List[int],
                            repetition_penalty: float, step: int) -> np.ndarray:
    """Apply the processors HuggingFace ``generate`` builds for this model, in order.

    The order is the one upstream gets: repetition penalty, then the minimum-new-tokens
    floor, then the suppressed token range. Suppression comes last, so it cannot be
    undone by the penalty; the end-of-speech token is the one id inside the suppressed
    range that survives.
    """
    x = np.asarray(logits, np.float64).reshape(-1).copy()
    if repetition_penalty != 1.0 and generated:
        seen = np.unique(np.asarray(generated, np.int64))
        scores = x[seen]
        x[seen] = np.where(scores < 0, scores * repetition_penalty,
                           scores / repetition_penalty)
    eos = x[CODEC_EOS]
    x[SUPPRESS_FROM:] = -np.inf
    x[CODEC_EOS] = -np.inf if step < MIN_NEW_TOKENS else eos
    return x


def sample_token(logits: np.ndarray, temperature: float, top_k: int, top_p: float,
                 rng: np.random.Generator, do_sample: bool = True) -> int:
    """Draw one token the way HuggingFace ``generate`` does for this model.

    The warpers run in the order HuggingFace builds them — temperature, then top-k,
    then top-p. The repetition penalty is *not* here: it is a processor, and
    :func:`apply_logits_processors` has already run it on the talker's logits. The
    code predictor has no processors at all, so it calls this function directly.
    """
    x = np.asarray(logits, np.float64).reshape(-1)
    if not do_sample or temperature <= 0:
        return int(np.argmax(x))
    x = x / max(temperature, 1e-5)
    if top_k and 0 < top_k < x.size:
        x = np.where(x < np.partition(x, -top_k)[-top_k], -np.inf, x)
    x = x - x.max()
    probs = np.exp(x)
    probs /= probs.sum()
    if top_p >= 1.0:
        return int(rng.choice(probs.size, p=probs))
    order = np.argsort(probs)[::-1]
    cutoff = int(np.searchsorted(np.cumsum(probs[order]), top_p)) + 1
    keep = order[:max(1, cutoff)]
    p = probs[keep] / probs[keep].sum()
    return int(rng.choice(keep, p=p))


def empty_cache(layers: int, heads: int, head_dim: int) -> Dict[str, np.ndarray]:
    """A zero-length KV cache for a fresh autoregressive pass."""
    return {f"past_key_values.{i}.{kind}": np.zeros((1, heads, 0, head_dim), np.float32)
            for i in range(layers) for kind in ("key", "value")}


def roll_cache(outputs: Dict[str, np.ndarray], layers: int) -> Dict[str, np.ndarray]:
    """Map this step's ``present`` tensors onto the next step's ``past`` inputs."""
    return {f"past_key_values.{i}.{kind}": outputs[f"present.{i}.{kind}"]
            for i in range(layers) for kind in ("key", "value")}


class Qwen3TTSAdapter(BaseOnnxAdapter):
    """Adapter for Qwen3-TTS (talker + code predictor + 12.5 Hz codec decoder)."""

    def __init__(self):
        self.text_embed: Optional[onnxruntime.InferenceSession] = None
        self.codec_embed: Optional[onnxruntime.InferenceSession] = None
        self.predictor_prefill: Optional[onnxruntime.InferenceSession] = None
        self.predictor_step: Optional[onnxruntime.InferenceSession] = None
        self.sub_codec_embed: Optional[onnxruntime.InferenceSession] = None
        self.decoder: Optional[onnxruntime.InferenceSession] = None
        self.tokenizer: Optional[BPETokenizer] = None
        self.speaker: str = ""
        self.language: str = "auto"
        self._params: Dict[str, Any] = {}
        self._sub_tables = np.arange(NUM_CODE_GROUPS - 1, dtype=np.int64)

    def default_params(self) -> Dict[str, float]:
        # upstream generation_config.json for the CustomVoice checkpoints
        return {"temperature": 0.9, "top_k": 50.0, "top_p": 1.0,
                "repetition_penalty": 1.05, "subtalker_temperature": 0.9,
                "subtalker_top_k": 50.0, "subtalker_top_p": 1.0}

    def param_labels(self) -> Dict[str, str]:
        return {"temperature": "Sampling temperature", "top_k": "Top-k",
                "top_p": "Nucleus (top-p)", "repetition_penalty": "Repetition penalty",
                "subtalker_temperature": "Code-predictor temperature",
                "subtalker_top_k": "Code-predictor top-k",
                "subtalker_top_p": "Code-predictor nucleus (top-p)"}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def configure(self, voice_config: Any) -> None:
        """Open the six side graphs and the model's own subword BPE.

        The speaker comes from the voice, not from a reference clip: this checkpoint
        has nine trained timbres and no speaker encoder. The language comes from the
        voice's ``lang_code`` unless ``engine_params`` names one explicitly.
        """
        ep = getattr(voice_config, "engine_params", None) or {}
        self._params = dict(ep)
        providers = ep.get("providers")

        for attribute, key in (("text_embed", "text_embed_path"),
                               ("codec_embed", "codec_embed_path"),
                               ("predictor_prefill", "code_predictor_prefill_path"),
                               ("predictor_step", "code_predictor_step_path"),
                               ("sub_codec_embed", "sub_codec_embed_path"),
                               ("decoder", "codec_decoder_path")):
            if getattr(self, attribute) is None and ep.get(key):
                setattr(self, attribute, make_session(ep[key], providers=providers))

        if self.tokenizer is None and ep.get("bpe_tokenizer_path"):
            self.tokenizer = BPETokenizer(ep["bpe_tokenizer_path"])

        speaker = str(ep.get("speaker", "")).lower()
        if speaker and speaker not in SPEAKER_IDS:
            raise ValueError(f"Qwen3-TTS has no speaker '{speaker}'; "
                             f"known speakers: {sorted(SPEAKER_IDS)}")
        self.speaker = speaker
        self.language = resolve_language_name(
            ep.get("language") or getattr(voice_config, "lang_code", None))

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    def encode_text(self, text: str, voice: Any, syn_config: Any) -> List[List[int]]:
        """Turn text into one subword-id list per autoregressive pass.

        The chat wrapper is added here because the prompt builder needs to know where
        the role prefix ends and the spoken text begins, and it locates both by a fixed
        token count rather than by searching the sequence.
        """
        if self.tokenizer is None:
            raise RuntimeError("Qwen3-TTS voice missing bpe_tokenizer_path in engine_params")
        return [self.tokenizer.tokenize(
            f"<|im_start|>assistant\n{chunk}<|im_end|>\n<|im_start|>assistant\n")
            for chunk in chunk_text(text) if chunk.strip()]

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _text_hidden(self, ids) -> np.ndarray:
        return self.text_embed.run(
            None, {"input_ids": np.asarray(ids, np.int64).reshape(1, -1)})[0]

    def _codec_hidden(self, ids) -> np.ndarray:
        return self.codec_embed.run(
            None, {"input_ids": np.asarray(ids, np.int64).reshape(1, -1)})[0]

    def build_prompt(self, input_ids: np.ndarray, speaker: str,
                     language: str) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble the summed text/codec embedding sequence the talker was trained on.

        Returns ``(prompt, tts_pad)``. The second value is the text-side padding hidden
        state, which every generated frame adds once the text has run out.

        The layout, position by position:

        1. the three role tokens, text side only;
        2. the language "think" block, the speaker, and codec padding, each summed with
           text padding, closing with the text BOS;
        3. the whole text summed with codec padding, closing with the text EOS;
        4. text padding summed with the codec BOS, which is where audio starts.
        """
        speaker = speaker.lower()
        if speaker not in SPEAKER_IDS:
            raise ValueError(f"Qwen3-TTS has no speaker '{speaker}'; "
                             f"known speakers: {sorted(SPEAKER_IDS)}")
        input_ids = np.asarray(input_ids, np.int64).reshape(1, -1)
        if input_ids.shape[1] <= ROLE_PREFIX_TOKENS + TAIL_TOKENS:
            raise ValueError("Qwen3-TTS prompt has no text between the chat markers")

        text_hidden = self._text_hidden(input_ids)
        specials = self._text_hidden([TTS_BOS_TOKEN, TTS_EOS_TOKEN, TTS_PAD_TOKEN])
        tts_bos, tts_eos, tts_pad = specials[:, 0:1], specials[:, 1:2], specials[:, 2:3]

        language_id = resolve_language_id(language, speaker)
        if language_id is None:
            think = [CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS]
        else:
            think = [CODEC_THINK, CODEC_THINK_BOS, language_id, CODEC_THINK_EOS]
        codec_prefix = np.concatenate([
            self._codec_hidden(think),
            self._codec_hidden([SPEAKER_IDS[speaker]]),
            self._codec_hidden([CODEC_PAD, CODEC_BOS]),
        ], axis=1)

        role = text_hidden[:, :ROLE_PREFIX_TOKENS]
        head = np.concatenate(
            [np.repeat(tts_pad, codec_prefix.shape[1] - 2, axis=1), tts_bos],
            axis=1) + codec_prefix[:, :-1]

        body_text = text_hidden[:, ROLE_PREFIX_TOKENS:-TAIL_TOKENS]
        body = (np.concatenate([body_text, tts_eos], axis=1)
                + self._codec_hidden([CODEC_PAD] * (body_text.shape[1] + 1)))
        tail = tts_pad + self._codec_hidden([CODEC_BOS])

        prompt = np.concatenate([role, head, body, tail], axis=1)
        return prompt.astype(np.float32), tts_pad.astype(np.float32)

    # ------------------------------------------------------------------
    # Code predictor
    # ------------------------------------------------------------------

    def _sub_embed(self, tokens, tables) -> np.ndarray:
        return self.sub_codec_embed.run(None, {
            "input_ids": np.asarray(tokens, np.int64).reshape(-1),
            "tables": np.asarray(tables, np.int64).reshape(-1)})[0]

    def predict_code_groups(self, talker_hidden: np.ndarray, group0_hidden: np.ndarray,
                            rng: np.random.Generator, temperature: float, top_k: int,
                            top_p: float, do_sample: bool) -> List[int]:
        """Run the code predictor over code groups 1..15 of one frame.

        The prefill reads two positions — the talker's hidden state and the embedding
        of the group-0 token — and every later step reads the previous group's token
        through that group's own embedding table and its own output head, which is why
        the step graph takes the group index as an input.
        """
        feed = {"inputs_embeds": np.concatenate(
            [talker_hidden, group0_hidden], axis=1).astype(np.float32)}
        names = [o.name for o in self.predictor_prefill.get_outputs()]
        out = dict(zip(names, self.predictor_prefill.run(None, feed)))
        past = roll_cache(out, PREDICTOR_LAYERS)
        tokens = [sample_token(out["logits"][0], temperature, top_k, top_p, rng, do_sample)]

        step_names = [o.name for o in self.predictor_step.get_outputs()]
        for group in range(1, NUM_CODE_GROUPS - 1):
            feed = {"inputs_embeds": self._sub_embed(tokens[-1], group - 1).astype(np.float32),
                    "step": np.asarray(group, np.int64),
                    "position_ids": np.asarray([[group + 1]], np.int64), **past}
            out = dict(zip(step_names, self.predictor_step.run(None, feed)))
            past = roll_cache(out, PREDICTOR_LAYERS)
            tokens.append(sample_token(out["logits"][0], temperature, top_k, top_p,
                                       rng, do_sample))
        return tokens

    # ------------------------------------------------------------------
    # Talker
    # ------------------------------------------------------------------

    def generate_codes(self, session: onnxruntime.InferenceSession, prompt: np.ndarray,
                       tts_pad: np.ndarray, params: Dict[str, Any],
                       rng: np.random.Generator) -> np.ndarray:
        """Run both autoregressive loops and return the ``(frames, 16)`` code matrix."""
        do_sample = bool(params.get("do_sample", True))
        temperature = float(params.get("temperature", 0.9))
        top_k = int(params.get("top_k", 50))
        top_p = float(params.get("top_p", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.05))
        sub_temperature = float(params.get("subtalker_temperature", 0.9))
        sub_top_k = int(params.get("subtalker_top_k", 50))
        sub_top_p = float(params.get("subtalker_top_p", 1.0))
        max_new = int(params.get("max_new_tokens", MAX_NEW_TOKENS))

        names = [o.name for o in session.get_outputs()]
        past = empty_cache(TALKER_LAYERS, TALKER_KV_HEADS, TALKER_HEAD_DIM)
        step_embeds, position, codes, generated = prompt, 0, [], []

        for step in range(max_new):
            width = step_embeds.shape[1]
            feed = {"inputs_embeds": step_embeds,
                    "position_ids": np.arange(position, position + width,
                                              dtype=np.int64)[None], **past}
            out = dict(zip(names, session.run(None, feed)))
            position += width
            past = roll_cache(out, TALKER_LAYERS)

            scores = apply_logits_processors(
                out["logits"][0, -1], generated, repetition_penalty, step)
            token = sample_token(scores, temperature, top_k, top_p, rng, do_sample)
            generated.append(token)
            if token == CODEC_EOS:
                break

            group0_hidden = self._codec_hidden([token])
            groups = self.predict_code_groups(
                out["last_hidden"], group0_hidden, rng, sub_temperature,
                sub_top_k, sub_top_p, do_sample)
            codes.append([token] + groups)

            sub_hidden = self._sub_embed(groups, self._sub_tables)
            step_embeds = (group0_hidden + sub_hidden.sum(axis=1, keepdims=True)
                           + tts_pad).astype(np.float32)

        return np.asarray(codes, np.int64).reshape(-1, NUM_CODE_GROUPS)

    # ------------------------------------------------------------------
    # Codec decoder
    # ------------------------------------------------------------------

    def decode_codes(self, codes: np.ndarray) -> np.ndarray:
        """Turn a code matrix into a waveform, one bounded chunk at a time.

        The decoder's attention window is 72 frames, so upstream decodes in chunks with
        a left context that is thrown away afterwards. Decoding a long utterance in one
        call would allocate an attention mask of the whole length for no gain.
        """
        codes = np.asarray(codes, np.int64).reshape(-1, NUM_CODE_GROUPS).T[None]
        total = codes.shape[-1]
        parts, start = [], 0
        while start < total:
            end = min(start + DECODE_CHUNK, total)
            context = DECODE_LEFT_CONTEXT if start - DECODE_LEFT_CONTEXT > 0 else start
            chunk = np.ascontiguousarray(codes[..., start - context:end])
            wav = self.decoder.run(None, {"codes": chunk})[0]
            parts.append(wav[..., context * UPSAMPLE:])
            start = end
        return np.concatenate(parts, axis=-1).reshape(-1).astype(np.float32)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(self, request: AdapterSynthesisRequest,
                   session: onnxruntime.InferenceSession) -> AdapterSynthesisResult:
        for attribute, key in (("text_embed", "text_embed_path"),
                               ("codec_embed", "codec_embed_path"),
                               ("predictor_prefill", "code_predictor_prefill_path"),
                               ("predictor_step", "code_predictor_step_path"),
                               ("sub_codec_embed", "sub_codec_embed_path"),
                               ("decoder", "codec_decoder_path")):
            if getattr(self, attribute) is None:
                raise RuntimeError(f"Qwen3-TTS voice missing {key} in engine_params")

        params = request.params
        speaker = str(params.get("speaker") or self.speaker)
        language = str(params.get("language") or self.language)
        prompt, tts_pad = self.build_prompt(request.phoneme_ids, speaker, language)

        rng = np.random.default_rng(params.get("seed"))
        codes = self.generate_codes(session, prompt, tts_pad, params, rng)
        if codes.size == 0:
            LOG.warning("Qwen3-TTS produced no code frames for this chunk")
            return AdapterSynthesisResult(audio=np.zeros(0, np.float32))

        audio = self.decode_codes(codes)
        return AdapterSynthesisResult(
            audio=audio, extras={"frame_count": int(codes.shape[0]),
                                 "speaker": speaker, "language": language})

    # build_feed_dict / parse_outputs are required by the ABC but unused — synthesize()
    # drives the two autoregressive loops directly.
    def build_feed_dict(self, request: AdapterSynthesisRequest,
                        session: onnxruntime.InferenceSession) -> Dict[str, np.ndarray]:
        raise NotImplementedError("Qwen3-TTS is autoregressive — use synthesize()")

    def parse_outputs(self, outputs: List[np.ndarray],
                      request: AdapterSynthesisRequest,
                      output_names: Optional[List[str]] = None) -> AdapterSynthesisResult:
        raise NotImplementedError("Qwen3-TTS is autoregressive — use synthesize()")

    @staticmethod
    def detect(config: Optional[Dict[str, Any]] = None,
               session: Optional[onnxruntime.InferenceSession] = None) -> bool:
        return bool(config and config.get("engine") == "qwen3tts")
