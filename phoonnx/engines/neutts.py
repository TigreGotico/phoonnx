"""NeuTTS inference adapter — autoregressive NeuCodec LM (NeuTTS Air / VieNeu / Akiti).

The NeuTTS Air family (Neuphonic) pairs a Qwen3 causal LM with **NeuCodec**, a
single-codebook 24 kHz neural codec. The LM is an ordinary text model whose vocabulary
was extended with 65 536 ``<|speech_N|>`` tokens: give it phonemes, and it emits a flat
stream of codec tokens that NeuCodec's decoder turns back into a waveform. There is no
duration model, no vocoder and no speaker encoder.

phoonnx's third autoregressive engine, after
:class:`phoonnx.engines.chatterbox.ChatterboxAdapter` and
:class:`phoonnx.engines.mosstts.MossTTSNanoAdapter`. Compared to MOSS-TTS-Nano it is the
*simple* shape of the same idea: one token per step instead of a 16-token RVQ frame, so
the decode loop is a plain KV-cached LM loop.

Two ONNX graphs::

    neutts_lm.onnx        Qwen3 backbone, KV-cached  (= the voice's ``session``)
    neucodec_decoder.onnx codes[1, 1, N] -> waveform[..., T] @ 24 kHz

``neutts_lm.onnx`` serves prefill *and* decode — the same graph with a different past
length, exactly as Chatterbox's ``language_model`` does::

    inputs   input_ids       int64 [1, S]        prompt tokens, or 1 token per step
             attention_mask  int64 [1, P + S]    ones over past and current tokens
             position_ids    int64 [1, S]        absolute positions, P .. P+S-1
             past_key_<i>    fp32  [1, 4, P, 64] for i in 0..27
             past_value_<i>  fp32  [1, 4, P, 64]
    outputs  logits          fp32  [1, V]        last position only
             present_key_<i> / present_value_<i>  fp32 [1, 4, P + S, 64]

The KV geometry is read off the graph's own input signature, so a differently-sized
sibling checkpoint needs no code change. See
``scripts/conversion/neutts/export_neutts_onnx.py`` for the export.

Prompt format
~~~~~~~~~~~~~
Reproduced from upstream's inference code, which is what the checkpoint was trained on::

    <|TEXT_PROMPT_START|>{reference phones} {target phones}<|TEXT_PROMPT_END|>
    <|SPEECH_GENERATION_START|>{<|speech_c|> for c in reference codes}

Generation continues that speech-token run until ``<|SPEECH_GENERATION_END|>`` or EOS.
Both phone strings come from espeak-ng, and the tokenizer is the checkpoint's own BPE, so
text -> ids is byte-identical to training rather than re-derived here.

Cloning is **in-context and pre-encoded**: a voice preset is a reference transcript plus
the NeuCodec codes of its recording (a ``voices.json``, spec ``vieneu.voice.presets``),
shipped as a per-voice asset. Neuphonic publishes NeuCodec's *decoder* as ONNX but not
its encoder, so this adapter cannot encode a fresh reference clip — it clones from the
presets only, and ``speaker_reference`` is not supported.
"""
import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime
from quebra_frases import sentence_tokenize

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter
from phoonnx.providers import make_session
from phoonnx.util import LOG

SAMPLE_RATE = 24000

#: espeak-ng voice the family is phonemized with. Akiti-TTS phonemizes Asante Twi
#: through Lingua Franca Nova: espeak-ng has no Twi voice, and ``lfn``'s five-vowel,
#: near-one-to-one orthography is the closest reader for Twi spelling. This is what the
#: checkpoint was trained on, so it is a property of the model, not a fallback.
ESPEAK_VOICE = "lfn"

# Special ids are read from the tokenizer, but the token *strings* are fixed by the
# family's prompt format.
TEXT_PROMPT_START = "<|TEXT_PROMPT_START|>"
TEXT_PROMPT_END = "<|TEXT_PROMPT_END|>"
SPEECH_GENERATION_START = "<|SPEECH_GENERATION_START|>"
SPEECH_GENERATION_END = "<|SPEECH_GENERATION_END|>"
SPEECH_TOKEN_TEMPLATE = "<|speech_{}|>"

_SPEECH_TOKEN_RE = re.compile(r"^<\|speech_(\d+)\|>$")

#: Reference codes fed as the generation prefix. Upstream truncates at 200 (~4 s): the
#: prompt has to leave room for the target inside the LM's context.
MAX_REFERENCE_CODES = 200

#: Tokens the repetition penalty looks back over. Upstream runs the LM through
#: llama.cpp, whose penalty is windowed (``last_n_tokens_size``, default 64) and spans
#: the *prompt* as well as what has been generated. Penalising the whole generated run
#: instead — the HuggingFace convention the other phoonnx codec-LMs use — is a different
#: model, and it makes this one cut off early or babble.
REPETITION_WINDOW = 64

#: Punctuation ``phonemizer``'s ``preserve_punctuation=True`` keeps verbatim. espeak-ng
#: drops punctuation from its phoneme output, so the marks are split out before the
#: espeak call and spliced back after — the behaviour the training pipeline had.
_PUNCTUATION = ';:,.!?¡¿—…"«»“”'
_PUNCT_SPLIT = re.compile(f"([{re.escape(_PUNCTUATION)}]+)")


def preserve_punctuation_phonemize(text: str, phonemize) -> str:
    """espeak-phonemize *text* while keeping its punctuation in place.

    ``phonemize`` is a ``str -> str`` callable (the voice's espeak phonemizer). Words and
    punctuation runs are separated first, only the words go to espeak, and the marks are
    re-inserted between the phonemized pieces — which is what ``phonemizer``'s
    ``preserve_punctuation=True`` does, and therefore what the checkpoint saw at training
    time. espeak-ng on its own silently deletes the marks.
    """
    out: List[str] = []
    for part in _PUNCT_SPLIT.split(text):
        if not part.strip():
            continue
        if _PUNCT_SPLIT.fullmatch(part):
            out.append(part)
        else:
            phones = phonemize(part.strip())
            if phones:
                out.append(phones)
    # marks attach to the preceding phones ("pˈaa." not "pˈaa ."), everything else is
    # space-separated
    joined = ""
    for piece in out:
        if joined and not _PUNCT_SPLIT.fullmatch(piece):
            joined += " "
        joined += piece
    return joined


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _apply_repetition_penalty(scores: np.ndarray, seen: np.ndarray, penalty: float) -> np.ndarray:
    """Divide the score of tokens in ``seen`` by ``penalty`` (multiply when negative).

    >1 discourages repeats. ``seen`` is the recent-token *window* the caller chose, not
    the whole history — see :data:`REPETITION_WINDOW`. NeuTTS needs a fairly strong
    penalty (upstream default 1.3) or it stalls on runs of silence tokens.
    """
    if penalty == 1.0 or seen.size == 0:
        return scores
    out = scores.copy()
    idx = np.unique(seen[(seen >= 0) & (seen < out.shape[0])])
    if idx.size:
        s = out[idx]
        out[idx] = np.where(s < 0, s * penalty, s / penalty)
    return out


def _sample(scores: np.ndarray, temperature: float, top_k: int, top_p: float,
            rng: np.random.Generator) -> int:
    """Top-k, then nucleus, then temperature — llama.cpp's sampler order (argmax if temp<=0).

    The order matters and is not the HuggingFace one: llama.cpp truncates on the *raw*
    distribution and only then applies temperature, so a low temperature sharpens the
    surviving candidates rather than shrinking the candidate set. Upstream drives this
    checkpoint through llama.cpp, so this reproduces the sampler it was tuned against.
    """
    s = np.asarray(scores, np.float32)
    if temperature <= 0:
        return int(np.argmax(s))
    if 0 < top_k < s.shape[0]:
        threshold = float(np.partition(s, -top_k)[-top_k])
        s = np.where(s < threshold, -np.inf, s)
    if 0 < top_p < 1:
        order = np.argsort(s)[::-1]
        cumulative = np.cumsum(_softmax(s[order]))
        keep = np.ones(order.shape[0], bool)
        # keep every token up to and including the one that crosses top_p
        keep[1:] = cumulative[:-1] < top_p
        masked = np.full_like(s, -np.inf)
        masked[order[keep]] = s[order[keep]]
        s = masked
    probs = _softmax(s / float(temperature))
    return int(rng.choice(probs.shape[0], p=probs))


class NeuTTSAdapter(BaseOnnxAdapter):
    """Adapter for the NeuTTS Air family (Qwen3 codec-LM + NeuCodec decoder)."""

    MEMOIZED_WRITES = {
        # KV-cache geometry, read off the model's own input shapes.
        "_read_kv_shape": frozenset({"num_layers", "num_kv_heads", "head_dim"}),
        # The espeak backend the checkpoint was trained with, opened once.
        "_espeak": frozenset({"_phonemizer"}),
        # Each preset's reference transcript phonemized once; presets are
        # part of the voice, not of a request.
        "reference_phones": frozenset({"_ref_phones"}),
    }

    #: hard ceiling on the AR loop; NeuCodec runs at 50 tokens/s, so 600 is ~12 s
    MAX_NEW_TOKENS = 600
    #: characters of target text per model call (upstream's chunk budget)
    MAX_CHUNK_CHARS = 200

    def __init__(self):
        self.codec: Optional[onnxruntime.InferenceSession] = None
        self.tokenizer = None
        self.presets: Dict[str, Dict[str, Any]] = {}
        self.default_preset: Optional[str] = None
        self.speech_end_id: Optional[int] = None
        self.eos_id: Optional[int] = None
        self.speech_token_base: Optional[int] = None
        self.num_layers = 0
        self.num_kv_heads = 0
        self.head_dim = 0
        self._phonemizer = None
        self._ref_phones: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def default_params(self) -> Dict[str, Any]:
        """Upstream's CLI defaults."""
        return {
            "temperature": 0.4,
            "top_p": 0.8,
            "top_k": 50.0,
            "repetition_penalty": 1.3,
            "max_new_tokens": float(self.MAX_NEW_TOKENS),
            "max_chunk_chars": float(self.MAX_CHUNK_CHARS),
        }

    def param_labels(self) -> Dict[str, str]:
        return {
            "temperature": "sampling temperature",
            "top_p": "top-p (nucleus)",
            "top_k": "top-k",
            "repetition_penalty": "repetition penalty",
            "max_new_tokens": "max codec tokens (50/s)",
            "max_chunk_chars": "target characters per model call",
            "voice": "voice preset name",
        }

    def configure(self, voice_config: Any) -> None:
        """Load the NeuCodec decoder, the checkpoint's BPE and the voice presets from
        ``engine_params``."""
        ep = getattr(voice_config, "engine_params", None) or {}
        if self.codec is None and ep.get("codec_decoder_path"):
            self.codec = make_session(ep["codec_decoder_path"], providers=ep.get("providers"))
        if self.tokenizer is None and ep.get("tokenizer_path"):
            from phoonnx.tokenizer import load_hf_tokenizer
            self.tokenizer = load_hf_tokenizer(ep["tokenizer_path"])
            self.speech_end_id = self.tokenizer.token_to_id(SPEECH_GENERATION_END)
            self.speech_token_base = self.tokenizer.token_to_id(
                SPEECH_TOKEN_TEMPLATE.format(0))
            self.eos_id = self.tokenizer.token_to_id("<|endoftext|>")
        if not self.presets and ep.get("voices_path"):
            with open(ep["voices_path"], encoding="utf-8") as f:
                voices = json.load(f)
            self.presets = voices.get("presets") or {}
            self.default_preset = voices.get("default_voice")

    def _read_kv_shape(self, session: onnxruntime.InferenceSession) -> None:
        for spec in session.get_inputs():
            if spec.name.startswith("past_key_"):
                self.num_layers += 1
                if self.num_kv_heads == 0:
                    self.num_kv_heads = int(spec.shape[1])
                    self.head_dim = int(spec.shape[3])

    # ------------------------------------------------------------------
    # text
    # ------------------------------------------------------------------
    def _espeak(self, text: str) -> str:
        """Phonemize with espeak-ng at the voice the checkpoint was trained with.

        Deliberately *not* the phoonnx voice's own phonemizer: a NeuTTS voice is
        ``Alphabet.GRAPHEMES``, whose phonemizer passes text through unchanged. Which
        phonemes this model consumes is a property of the checkpoint, so espeak-ng is
        addressed directly.
        """
        if self._phonemizer is None:
            from scriptconv.phonemizers.mul import EspeakPhonemizer
            self._phonemizer = EspeakPhonemizer()
        return preserve_punctuation_phonemize(
            text, lambda part: self._phonemizer.phonemize_string(part, ESPEAK_VOICE).strip())

    def resolve_preset(self, voice: Any, syn_config: Any) -> Optional[Dict[str, Any]]:
        """Pick the voice preset: an explicit per-call ``voice``, else the one this
        phoonnx voice is pinned to, else ``voices.json``'s ``default_voice``."""
        name = None
        if syn_config is not None:
            name = (syn_config.extra_params or {}).get("voice")
        cfg = getattr(voice, "config", None)
        if not name:
            name = (getattr(cfg, "engine_params", None) or {}).get("voice")
        if not name:
            name = self.default_preset
        if not name:
            return None
        if name not in self.presets:
            raise ValueError(f"unknown NeuTTS voice preset {name!r}; "
                             f"available: {sorted(self.presets)}")
        return self.presets[name]

    def reference_phones(self, name: str, preset: Dict[str, Any]) -> str:
        """Phonemized reference transcript, cached per preset (it never changes)."""
        if name not in self._ref_phones:
            self._ref_phones[name] = self._espeak(preset.get("text", "")).strip()
        return self._ref_phones[name]

    def chunk_text(self, text: str, max_chars: int) -> List[str]:
        """Pack whole sentences up to ``max_chars`` per model call.

        The LM degrades on long prompts, so upstream splits before synthesizing.
        Sentences come from the same ``quebra_frases`` splitter the rest of phoonnx uses;
        an over-long single sentence is split on word boundaries rather than dropped.
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

    def build_prompt(self, target_phones: str, reference_phones: str = "",
                     reference_codes: Optional[List[int]] = None) -> str:
        """The exact prompt string the checkpoint was fine-tuned on."""
        text = f"{reference_phones} {target_phones}".strip() if reference_phones \
            else target_phones.strip()
        codes = "".join(SPEECH_TOKEN_TEMPLATE.format(int(c))
                        for c in (reference_codes or [])[:MAX_REFERENCE_CODES])
        return (f"{TEXT_PROMPT_START}{text}{TEXT_PROMPT_END}"
                f"{SPEECH_GENERATION_START}{codes}")

    def encode_text(self, text: str, voice: Any, syn_config: Any) -> List[List[int]]:
        """Build one fully-formed prompt per model call and BPE-encode it.

        The reference block is part of the prompt, so the whole string — reference phones,
        target phones, special tokens and the pre-encoded reference codes — is tokenized
        in one pass by the checkpoint's own BPE. Splicing separately-tokenized pieces
        would not reproduce training-time ids at the joins.
        """
        if self.tokenizer is None:
            raise RuntimeError("NeuTTS voice missing tokenizer_path in engine_params")
        budget = int(self.MAX_CHUNK_CHARS)
        if syn_config is not None:
            budget = int((syn_config.extra_params or {}).get("max_chunk_chars", budget))
        preset = self.resolve_preset(voice, syn_config)
        ref_phones, ref_codes = "", None
        if preset is not None:
            name = next(k for k, v in self.presets.items() if v is preset)
            ref_phones = self.reference_phones(name, preset)
            ref_codes = preset.get("codes")
        prompts = [self.build_prompt(self._espeak(chunk).strip(), ref_phones, ref_codes)
                   for chunk in self.chunk_text(text, max(1, budget))]
        return [list(self.tokenizer.encode(p, add_special_tokens=False).ids) for p in prompts]

    # ------------------------------------------------------------------
    # codec
    # ------------------------------------------------------------------
    def token_ids_to_codes(self, token_ids: List[int]) -> List[int]:
        """Keep only ``<|speech_N|>`` tokens and map them back to NeuCodec indices."""
        if self.speech_token_base is None:
            raise RuntimeError("NeuTTS tokenizer has no <|speech_0|> token")
        codes: List[int] = []
        for tid in token_ids:
            token = self.tokenizer.id_to_token(int(tid))
            match = _SPEECH_TOKEN_RE.match(token or "")
            if match:
                codes.append(int(match.group(1)))
        return codes

    def decode_codes(self, codes: List[int]) -> np.ndarray:
        """NeuCodec indices -> mono float32 waveform at 24 kHz."""
        if not codes:
            return np.zeros(0, np.float32)
        arr = np.asarray(codes, np.int32).reshape(1, 1, -1)
        name = self.codec.get_inputs()[0].name
        audio = self.codec.run(None, {name: arr})[0]
        return np.asarray(audio, np.float32).reshape(-1)

    # ------------------------------------------------------------------
    # autoregressive loop
    # ------------------------------------------------------------------
    def generate(self, session: onnxruntime.InferenceSession, prompt_ids: List[int],
                 p: Dict[str, Any], rng: np.random.Generator) -> List[int]:
        """Prefill the prompt, then sample codec tokens until the model stops."""
        if not self.num_layers:
            self._read_kv_shape(session)
        out_names = [o.name for o in session.get_outputs()]
        past_names = [f"past_{k}_{i}" for i in range(self.num_layers) for k in ("key", "value")]
        present_names = [f"present_{k}_{i}" for i in range(self.num_layers) for k in ("key", "value")]

        ids = np.asarray(prompt_ids, np.int64).reshape(1, -1)
        length = ids.shape[1]
        feed = {
            "input_ids": ids,
            "attention_mask": np.ones((1, length), np.int64),
            "position_ids": np.arange(length, dtype=np.int64)[None, :],
        }
        empty = np.zeros((1, self.num_kv_heads, 0, self.head_dim), np.float32)
        feed.update({n: empty for n in past_names})
        named = dict(zip(out_names, session.run(None, feed)))

        temperature = float(p.get("temperature", 0.4))
        top_p = float(p.get("top_p", 0.8))
        top_k = int(p.get("top_k", 50))
        penalty = float(p.get("repetition_penalty", 1.3))
        max_new = max(1, int(p.get("max_new_tokens", self.MAX_NEW_TOKENS)))

        generated: List[int] = []
        history = list(prompt_ids)
        for step in range(max_new):
            logits = np.asarray(named["logits"], np.float32).reshape(-1)
            logits = _apply_repetition_penalty(
                logits, np.asarray(history[-REPETITION_WINDOW:], np.int64), penalty)
            token = _sample(logits, temperature, top_k, top_p, rng)
            if token == self.speech_end_id or token == self.eos_id:
                break
            generated.append(token)
            history.append(token)
            if step == max_new - 1:
                LOG.warning("NeuTTS hit the %d-token cap (%.1fs) without an end token",
                            max_new, max_new / 50.0)
                break
            position = length + step
            feed = {
                "input_ids": np.asarray([[token]], np.int64),
                "attention_mask": np.ones((1, position + 1), np.int64),
                "position_ids": np.asarray([[position]], np.int64),
                **{p_name: named[q] for p_name, q in zip(past_names, present_names)},
            }
            named = dict(zip(out_names, session.run(None, feed)))
        return generated

    # ------------------------------------------------------------------
    # synthesis
    # ------------------------------------------------------------------
    def synthesize(self, request: AdapterSynthesisRequest,
                   session: onnxruntime.InferenceSession) -> AdapterSynthesisResult:
        if self.codec is None:
            raise RuntimeError("NeuTTS voice missing codec_decoder_path in engine_params")
        if self.tokenizer is None:
            raise RuntimeError("NeuTTS voice missing tokenizer_path in engine_params")
        if request.params.get("reference_audio") is not None:
            raise RuntimeError(
                "NeuTTS clones from pre-encoded voice presets, not from a reference clip: "
                "NeuCodec's encoder is not published as ONNX. Select a preset with "
                "extra_params={'voice': ...}")

        p = {**self.default_params(), **request.params}
        seed = p.get("seed")
        rng = np.random.default_rng(None if seed is None else int(seed))
        prompt_ids = [int(i) for i in np.asarray(request.phoneme_ids).reshape(-1)]
        token_ids = self.generate(session, prompt_ids, p, rng)
        codes = self.token_ids_to_codes(token_ids)
        if not codes:
            raise RuntimeError("NeuTTS generated no speech tokens — lower the temperature "
                               "or pick another voice preset")
        return AdapterSynthesisResult(audio=self.decode_codes(codes),
                                      extras={"codes": len(codes)})

    # required by the ABC but unused — synthesize() drives the AR loop.
    def build_feed_dict(self, request, session):
        raise NotImplementedError("NeuTTS is autoregressive — use synthesize()")

    def parse_outputs(self, outputs, request, output_names=None):
        raise NotImplementedError("NeuTTS is autoregressive — use synthesize()")

    @staticmethod
    def detect(config: Optional[Dict[str, Any]] = None,
               session: Optional[onnxruntime.InferenceSession] = None) -> bool:
        return bool(config and config.get("engine") == "neutts")
