"""SuperTonic inference adapter — Supertone's flow-matching TTS.

SuperTonic (Supertone Inc., ``Supertone/supertonic-3`` on HuggingFace) is a raw-text,
multilingual (31 langs) flow-matching TTS. It is phoonnx's second flow-matching engine
(after ZipVoice) but, unlike ZipVoice, it has no in-context cloning: pronunciation comes
from a fixed per-speaker *style* (style_ttl / style_dp vectors), not a reference clip.

Four ONNX graphs per model dir::

    duration_predictor : text_ids, style_dp, text_mask -> duration (seconds, per item)
    text_encoder        : text_ids, style_ttl, text_mask -> text_emb
    vector_estimator     : noisy_latent, text_emb, style_ttl, text_mask, latent_mask,
                            current_step, total_step -> denoised_latent   (= ``session``)
    vocoder              : latent -> wav_tts

plus ``tts.json`` (model config: sample rate, latent dims, chunk-compression factor) and
``unicode_indexer.json`` (a 65536-entry list mapping BMP codepoint -> token id, ``-1`` for
unmapped codepoints — turned into a standard :class:`phoonnx.tokenizer.Vocabulary` via
``Vocabulary.from_supertonic_indexer``). Voices are per-speaker *style* JSONs
(``style_ttl``/``style_dp`` arrays with a ``dims``/``data`` layout).

This is a **grapheme/text-token engine** in the same family as
:class:`phoonnx.engines.chatterbox.ChatterboxAdapter`: it owns text -> id conversion via
``encode_text`` (``BaseOnnxAdapter``'s ``Alphabet.GRAPHEMES`` bypass), rather than going
through the shared phonemizer dispatch (``Alphabet.UNICODE`` +
``UnicodeCodepointPhonemizer``). That dispatch does sentence-level chunking generically
(``BasePhonemizer.chunk_text``/``phonemize_lazy``), which would be a good fit on its own,
but SuperTonic's ``<lang>...</lang>`` wrap and its 120/300-char length cap must apply
*per model call* (each ONNX call gets its own wrap + is capped to the length the model
was trained on) — the generic sentence dispatch has no hook to enforce a length cap or to
splice per-call markup around already-tokenized sentences, so this stays adapter-owned.
What it *does* reuse rather than reinvent:

* character -> id mapping goes through the standard ``phoonnx.tokenizer.Vocabulary`` /
  ``TTSTokenizer`` (via ``Vocabulary.from_supertonic_indexer``), not a bespoke list
  lookup — same out-of-vocabulary handling (drop + warn) as every other engine;
* the NFKD step reuses ``scriptconv``'s ``UnicodeCodepointPhonemizer`` (the same class
  that backs ``Alphabet.UNICODE`` voices) instead of a local ``unicodedata.normalize``
  call;
* sentence boundaries reuse ``quebra_frases.sentence_tokenize`` (the same sentence
  splitter ``BasePhonemizer.chunk_text`` and ``TTSVoice`` itself use), packed up to
  SuperTonic's per-call length cap — pure text cleanup (emoji stripping, dash/quote
  normalization, terminal punctuation) has no existing equivalent in
  ``phoonnx.lang_preprocess`` (that module holds *script* transforms — Hangul/Kanji/
  Cangjie — not generic text cleanup) and stays local, adapted from upstream.

The text preprocessing / chunking / noisy-latent sampling below is adapted (MIT,
Copyright Supertone Inc.) from Supertone's official ``py/helper.py`` inference reference
in the ``Supertone/supertonic-3`` model card.
"""
import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime
from quebra_frases import sentence_tokenize

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter
from phoonnx.phonemizers.base import UnicodeCodepointPhonemizer
from phoonnx.providers import make_session
from phoonnx.tokenizer import TTSTokenizer, Vocabulary

AVAILABLE_LANGS = ["en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et",
                    "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl",
                    "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na"]

_NFKD = UnicodeCodepointPhonemizer(form="NFKD")

_EMOJI_PATTERN = re.compile(
    "[\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "☀-⛿"
    "✀-➿"
    "\U0001f1e6-\U0001f1ff]+",
    flags=re.UNICODE,
)

_REPLACEMENTS = {
    "–": "-", "‑": "-", "—": "-", "_": " ",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "´": "'", "`": "'",
    "[": " ", "]": " ", "|": " ", "/": " ", "#": " ", "→": " ", "←": " ",
}

_EXPR_REPLACEMENTS = {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}

_TERMINAL_PUNCT = re.compile(r"[.!?;:,'\"')\]}…。』】〉》›»]$")


def preprocess_text(text: str, lang: str) -> str:
    """NFKD-normalize (via the shared ``UnicodeCodepointPhonemizer``), strip emoji/stray
    symbols, fix punctuation spacing, terminate the sentence, and wrap the result in
    ``<lang>...</lang>``. Raises ``ValueError`` for a language SuperTonic was not trained
    on — a genuinely SuperTonic-specific constraint, not something the shared lang-match
    helpers (which pick the *closest* supported language) are meant to enforce here."""
    if lang not in AVAILABLE_LANGS:
        raise ValueError(f"Invalid language for supertonic: {lang}")
    text = _NFKD.phonemize_string(text, lang)
    text = _EMOJI_PATTERN.sub("", text)
    for k, v in _REPLACEMENTS.items():
        text = text.replace(k, v)
    text = re.sub(r"[♥☆♡©\\]", "", text)
    for k, v in _EXPR_REPLACEMENTS.items():
        text = text.replace(k, v)
    text = re.sub(r" ,", ",", text)
    text = re.sub(r" \.", ".", text)
    text = re.sub(r" !", "!", text)
    text = re.sub(r" \?", "?", text)
    text = re.sub(r" ;", ";", text)
    text = re.sub(r" :", ":", text)
    text = re.sub(r" '", "'", text)
    while '""' in text:
        text = text.replace('""', '"')
    while "''" in text:
        text = text.replace("''", "'")
    while "``" in text:
        text = text.replace("``", "`")
    text = re.sub(r"\s+", " ", text).strip()
    if text and not _TERMINAL_PUNCT.search(text):
        text += "."
    return f"<{lang}>{text}</{lang}>"


def chunk_text(text: str, max_len: int = 300) -> List[str]:
    """Split text into <= ``max_len``-char chunks, packing whole sentences (from the
    shared ``quebra_frases.sentence_tokenize`` splitter — the same one
    ``BasePhonemizer.chunk_text``/``TTSVoice`` use elsewhere) up to the per-model-call
    length SuperTonic was trained on. The generic phonemizer dispatch only splits on
    sentence boundaries and has no length cap, so that part stays adapter-owned."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    chunks: List[str] = []
    for paragraph in paragraphs:
        sentences = sentence_tokenize(paragraph)
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= max_len:
                current += (" " if current else "") + sentence
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence
        if current:
            chunks.append(current.strip())
    return chunks


def load_unicode_indexer(path: str) -> List[int]:
    """Load the raw unicode codepoint -> token id table from ``unicode_indexer.json``.

    The published format is a flat JSON array of length 65536 (one entry per BMP
    codepoint, ``-1`` for unmapped codepoints); a dict keyed by codepoint (int or numeric
    string) or by the literal character is also accepted for forward/backward tolerance.
    Feed the result to ``Vocabulary.from_supertonic_indexer`` to build a tokenizer.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    table = [-1] * 65536
    for k, v in raw.items():
        cp = ord(k) if len(k) == 1 else int(k)
        if 0 <= cp < len(table):
            table[cp] = v
    return table


def length_to_mask(length: int) -> np.ndarray:
    """Binary mask of shape (1, 1, length), all-ones (phoonnx synthesizes one
    unpadded chunk at a time, so there is never any padding to mask out)."""
    return np.ones((1, 1, max(length, 0)), dtype=np.float32)


def sample_noisy_latent(duration_s: float, sample_rate: int, base_chunk_size: int,
                         chunk_compress_factor: int, latent_dim: int,
                         rng: Optional[np.random.Generator] = None):
    """Gaussian noise seed for the Euler ODE loop, shaped from the predicted duration."""
    rng = rng or np.random.default_rng()
    wav_len = max(int(duration_s * sample_rate), 1)
    chunk_size = base_chunk_size * chunk_compress_factor
    latent_len = (wav_len + chunk_size - 1) // chunk_size
    latent_len = max(latent_len, 1)
    dim = latent_dim * chunk_compress_factor
    noisy_latent = rng.standard_normal((1, dim, latent_len)).astype(np.float32)
    latent_mask = length_to_mask(latent_len)
    return noisy_latent * latent_mask, latent_mask


class SuperTonicAdapter(BaseOnnxAdapter):
    """Adapter for SuperTonic (4-graph flow-matching, fixed per-speaker style)."""

    def __init__(self, total_step: int = 8, speed: float = 1.05, silence_duration: float = 0.3):
        self.duration_predictor: Optional[onnxruntime.InferenceSession] = None
        self.text_encoder: Optional[onnxruntime.InferenceSession] = None
        self.vocoder: Optional[onnxruntime.InferenceSession] = None
        self.tokenizer: Optional[TTSTokenizer] = None
        self.style_ttl: Optional[np.ndarray] = None
        self.style_dp: Optional[np.ndarray] = None
        self.sample_rate = 44100
        self.base_chunk_size = 512
        self.chunk_compress_factor = 6
        self.latent_dim = 24
        self.total_step = int(total_step)
        self.speed = float(speed)
        self.silence_duration = float(silence_duration)

    def default_params(self) -> Dict[str, float]:
        return {"total_step": float(self.total_step), "speed": self.speed,
                "silence_duration": self.silence_duration}

    def configure(self, voice_config: Any) -> None:
        """Load the three auxiliary graphs, the tts.json config, the unicode indexer
        (converted into a standard ``Vocabulary``/``TTSTokenizer``) and the voice's style
        from ``engine_params``; the vector_estimator is the voice's own primary session."""
        ep = getattr(voice_config, "engine_params", None) or {}
        sess = lambda p: make_session(p, providers=ep.get("providers"))
        if self.duration_predictor is None and ep.get("duration_predictor_path"):
            self.duration_predictor = sess(ep["duration_predictor_path"])
        if self.text_encoder is None and ep.get("text_encoder_path"):
            self.text_encoder = sess(ep["text_encoder_path"])
        if self.vocoder is None and ep.get("vocoder_path"):
            self.vocoder = sess(ep["vocoder_path"])
        if self.tokenizer is None and ep.get("unicode_indexer_path"):
            indexer = load_unicode_indexer(ep["unicode_indexer_path"])
            self.tokenizer = TTSTokenizer(
                Vocabulary.from_supertonic_indexer(indexer),
                add_blank_char=False, add_blank_word=False,
                use_eos_bos=False, blank_at_end=False, blank_at_start=False,
            )
        if ep.get("tts_config_path"):
            with open(ep["tts_config_path"], "r", encoding="utf-8") as f:
                cfgs = json.load(f)
            self.sample_rate = cfgs.get("ae", {}).get("sample_rate", self.sample_rate)
            self.base_chunk_size = cfgs.get("ae", {}).get("base_chunk_size", self.base_chunk_size)
            self.chunk_compress_factor = cfgs.get("ttl", {}).get(
                "chunk_compress_factor", self.chunk_compress_factor)
            self.latent_dim = cfgs.get("ttl", {}).get("latent_dim", self.latent_dim)
        if self.style_ttl is None and ep.get("style_path"):
            with open(ep["style_path"], "r", encoding="utf-8") as f:
                style = json.load(f)
            ttl_dims = style["style_ttl"]["dims"]
            dp_dims = style["style_dp"]["dims"]
            self.style_ttl = np.array(style["style_ttl"]["data"], np.float32).reshape(
                1, ttl_dims[1], ttl_dims[2])
            self.style_dp = np.array(style["style_dp"]["data"], np.float32).reshape(
                1, dp_dims[1], dp_dims[2])
        if "total_step" in ep:
            self.total_step = int(ep["total_step"])
        if "speed" in ep:
            self.speed = float(ep["speed"])
        if "silence_duration" in ep:
            self.silence_duration = float(ep["silence_duration"])

    def encode_text(self, text: str, voice: Any, syn_config: Any) -> List[List[int]]:
        """Preprocess + chunk raw text, then map each chunk through the standard
        tokenizer built from SuperTonic's own unicode indexer. Chunk length follows
        upstream (120 chars for ko/ja, 300 otherwise); out-of-vocabulary characters are
        dropped with a warning, the same as every other tokenizer-backed engine."""
        if self.tokenizer is None:
            raise RuntimeError("SuperTonic voice missing unicode_indexer_path in engine_params")
        cfg = getattr(voice, "config", None)
        lang = (getattr(cfg, "lang_code", None) or "en").split("-")[0].lower()
        if lang not in AVAILABLE_LANGS:
            raise ValueError(f"Invalid language for supertonic: {lang}")
        max_len = 120 if lang in ("ko", "ja") else 300
        chunks = chunk_text(text, max_len=max_len)
        return [self.tokenizer.tokenize(preprocess_text(chunk, lang)) for chunk in chunks if chunk]

    def synthesize(self, request: AdapterSynthesisRequest,
                   session: onnxruntime.InferenceSession) -> AdapterSynthesisResult:
        if self.duration_predictor is None or self.text_encoder is None or self.vocoder is None:
            raise RuntimeError("SuperTonic voice missing duration_predictor_path / "
                               "text_encoder_path / vocoder_path in engine_params")
        if self.style_ttl is None or self.style_dp is None:
            raise RuntimeError("SuperTonic voice missing style_path (per-speaker style) "
                               "in engine_params")

        text_ids = np.asarray(request.phoneme_ids, np.int64).reshape(1, -1)
        text_mask = length_to_mask(text_ids.shape[1])

        p = request.params
        total_step = int(p.get("total_step", self.total_step))
        speed = float(p.get("speed", self.speed))
        if total_step < 1:
            raise ValueError("supertonic total_step must be >= 1")

        duration, *_ = self.duration_predictor.run(
            None, {"text_ids": text_ids, "style_dp": self.style_dp, "text_mask": text_mask})
        duration = duration / speed

        text_emb, *_ = self.text_encoder.run(
            None, {"text_ids": text_ids, "style_ttl": self.style_ttl, "text_mask": text_mask})

        xt, latent_mask = sample_noisy_latent(
            float(np.asarray(duration).reshape(-1)[0]), self.sample_rate,
            self.base_chunk_size, self.chunk_compress_factor, self.latent_dim)

        total_step_np = np.array([total_step], np.float32)
        for step in range(total_step):
            current_step = np.array([step], np.float32)
            xt, *_ = session.run(None, {
                "noisy_latent": xt, "text_emb": text_emb, "style_ttl": self.style_ttl,
                "text_mask": text_mask, "latent_mask": latent_mask,
                "current_step": current_step, "total_step": total_step_np,
            })

        wav, *_ = self.vocoder.run(None, {"latent": xt})
        audio = np.asarray(wav, np.float32).reshape(-1)
        return AdapterSynthesisResult(audio=audio, extras={"duration": float(np.asarray(duration).reshape(-1)[0])})

    # build_feed_dict / parse_outputs are required by the ABC but unused — synthesize()
    # drives the multi-ONNX pipeline directly.
    def build_feed_dict(self, request: AdapterSynthesisRequest,
                        session: onnxruntime.InferenceSession) -> Dict[str, np.ndarray]:
        raise NotImplementedError("SuperTonic is multi-graph — use synthesize()")

    def parse_outputs(self, outputs: List[np.ndarray],
                      request: AdapterSynthesisRequest) -> AdapterSynthesisResult:
        raise NotImplementedError("SuperTonic is multi-graph — use synthesize()")

    @staticmethod
    def detect(config: Optional[Dict[str, Any]] = None,
               session: Optional[onnxruntime.InferenceSession] = None) -> bool:
        if config and config.get("engine") == "supertonic":
            return True
        if session is not None:
            names = {i.name for i in session.get_inputs()}
            # vector_estimator's distinctive input signature
            if {"noisy_latent", "text_emb", "style_ttl", "current_step", "total_step"} <= names:
                return True
        return False
