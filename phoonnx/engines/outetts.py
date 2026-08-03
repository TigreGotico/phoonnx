"""OuteTTS 1.0 inference adapter — Llama/Qwen codec-LM + DAC.speech decoder.

OuteTTS 1.0 (OuteAI) is a causal language model whose vocabulary was extended with
the two codebooks of **DAC.speech.v1.0**, a 24 kHz residual-vector-quantized codec at
1.5 kbps. The model reads a text prompt and emits interleaved ``<|c1_N|><|c2_N|>``
pairs; the codec decoder turns that pair stream back into a waveform. There is no
duration model, no vocoder and no phonemizer — the LM consumes raw text.

Two checkpoints share one interface (upstream's "interface v3")::

    OuteTTS-1.0-0.6B        Qwen3-0.6B backbone, Apache-2.0, 14 languages
    Llama-OuteTTS-1.0-1B    Llama-3.2-1B backbone, CC-BY-NC-SA-4.0, 23+ languages

phoonnx defaults to the 0.6B: it is the permissively licensed one and the one that
runs comfortably on CPU. The 1B is indexed as well because it is the only member of
the family that covers the low-resource languages (Bengali, Persian, Swahili, Tamil,
Lithuanian, Ukrainian, ...) — see ``VOICES.md`` for the license note.

Two ONNX graphs::

    model.onnx          the codec-LM, KV-cached  (= the voice's ``session``)
    decoder_model.onnx  audio_codes[1, 2, N] -> audio_values[1, 1, T] @ 24 kHz

``model.onnx`` serves prefill *and* decode — the same graph with a different past
length, exactly as :class:`phoonnx.engines.neutts.NeuTTSAdapter` does::

    inputs   input_ids                 int64 [1, S]        prompt, or 1 token per step
             attention_mask            int64 [1, P + S]    ones over past and current
             position_ids              int64 [1, S]        absolute positions P .. P+S-1
             past_key_values.<i>.key   fp32  [1, H, P, D]  for i in 0 .. L-1
             past_key_values.<i>.value fp32  [1, H, P, D]
    outputs  logits                    fp32  [1, S, V]     **all** positions
             present.<i>.key / present.<i>.value  fp32 [1, H, P + S, D]

Unlike NeuTTS's export, ``logits`` carries the whole sequence, so the decode step reads
``logits[0, -1]``. The KV geometry (L, H, D) is read off the graph's own input
signature, so the 0.6B and the 1B need no code change.

Prompt format
~~~~~~~~~~~~~
Reproduced from ``outetts.version.v3.prompt_processor.PromptProcessor``::

    <|im_start|>
    <|text_start|>{speaker text}{separator}{target text}<|text_end|>
    <|audio_start|>
    <|word_start|>{word}<|features|><|t_0.20|><|energy_N|><|spectral_centroid_N|>
        <|pitch_N|><|code|><|c1_a|><|c2_a|>...<|word_end|>
    ... one line per speaker word ...
    <|word_start|>

Generation continues that stream until ``<|audio_end|>`` or ``<|im_end|>``.

Cloning is **in-context and pre-encoded**: a speaker profile is a JSON transcript whose
words each carry a duration, three prosody buckets and their DAC codes (upstream's
``speaker`` dict, ``interface_version: 3``). Building a profile from a fresh clip needs
Whisper word-alignment plus the DAC *encoder*; this adapter clones from published
profiles only, so ``speaker_reference`` is not supported.

Sampling follows upstream's **HuggingFace** stack — ``outetts``'s default backend is
``Backend.HF`` — which is a different order from the llama.cpp one NeuTTS uses:
windowed repetition penalty, then temperature, then top-k, top-p and min-p over the
*tempered* distribution. See :func:`_sample`.
"""
import copy
import json
import math
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter
from phoonnx.providers import make_session
from phoonnx.util import LOG

SAMPLE_RATE = 24000

#: DAC.speech.v1.0 hop length; 24000 / 512 = 46.875 code frames per second.
CODEC_HOP_LENGTH = 512
FRAMES_PER_SECOND = SAMPLE_RATE / CODEC_HOP_LENGTH

# Special token strings — fixed by the family's prompt format
# (``outetts.version.v3.tokens.SpecialTokens``). The *ids* are read from the
# checkpoint's own tokenizer, so a sibling export needs no table here.
BOS = "<|im_start|>"
EOS = "<|im_end|>"
TEXT_START = "<|text_start|>"
TEXT_END = "<|text_end|>"
AUDIO_START = "<|audio_start|>"
AUDIO_END = "<|audio_end|>"
WORD_START = "<|word_start|>"
WORD_END = "<|word_end|>"
FEATURES = "<|features|>"
CODE = "<|code|>"
TIME = "<|t_{:.2f}|>"
C1 = "<|c1_{}|>"
C2 = "<|c2_{}|>"

#: Both DAC codebooks hold 1024 entries; upstream maps 1025 ids per codebook (the extra
#: slot exists in the vocabulary but is never emitted by the encoder).
CODEBOOK_TOKENS = 1025

#: Tokens the repetition penalty looks back over. Upstream ships a *patched*
#: ``RepetitionPenaltyLogitsProcessor`` that windows the penalty to the last 64 tokens
#: instead of HuggingFace's whole-context default, because the unwindowed penalty
#: destroys long generations for this model. The window spans the prompt too.
REPETITION_WINDOW = 64

#: Word-count bounds of upstream's ``chunk_text``.
CHUNK_MIN_WORDS = 10
CHUNK_MAX_WORDS = 30

_SENTENCE_END = re.compile(r"([.!?。！？︕︖]+\s*)")
_C1_RE = re.compile(r"^<\|c1_(\d+)\|>$")
_C2_RE = re.compile(r"^<\|c2_(\d+)\|>$")


# ----------------------------------------------------------------------------------
# text
# ----------------------------------------------------------------------------------
def _is_cjk(text: str) -> bool:
    """True when *text* contains Hiragana, Katakana or Han — upstream's language probe."""
    return any("぀" <= c <= "ゟ" or "゠" <= c <= "ヿ"
               or "一" <= c <= "鿿" for c in text)


def text_normalizations(text: str) -> str:
    """Upstream's ``PromptProcessor.text_normalizations``.

    Upstream runs ``ftfy.fix_text`` first to repair mojibake. phoonnx does not carry
    ``ftfy``, and NFKC — which upstream applies straight after — covers the
    compatibility folding that actually changes tokenization here. Text that reaches
    phoonnx has not been through a broken decode, so the mojibake repair has nothing
    to do; the rest of the pipeline is reproduced exactly.
    """
    if not isinstance(text, str):
        return "" if text is None else str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("…", "...")
    text = re.sub(r"\.{2,}", "...", text)
    text = re.sub(r"[“”„‟«»]", '"', text)
    text = re.sub(r"[‘’‛‹›`´]", "'", text)
    text = re.sub(r"[–—―−‐]", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    text = re.sub(r"[\x00-\x1F\x7F-\x9F­​-‍﻿]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.?!:;])", r"\1", text)
    text = re.sub(r"([,.?!:;])(?=[^\s,.?!:;])", r"\1 ", text)
    text = re.sub(r"(\w)\s+'\s*(\w)", r"\1'\2", text)
    text = re.sub(r"(\w)\s+'\s*([,.?!:;\s]|$)", r"\1'\2", text)
    text = re.sub(r"(\w)\s*'\s*([tsdmreSNTDMRE])\b", r"\1'\2", text, flags=re.IGNORECASE)
    text = re.sub(r"([?!])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.replace('"', "").strip()


def _tokenize_words(text: str) -> List[str]:
    """Split *text* into the units ``chunk_text`` counts.

    Upstream sends CJK text through MeCab (``-Owakati``) and everything else through
    ``str.split``. MeCab is a Japanese-only morphological analyser and a heavyweight
    optional dependency, so CJK is split per character here instead. That changes only
    *where* a long passage is cut into chunks, never the tokens inside a chunk, and a
    character is a tighter bound than a MeCab word — so a chunk stays under the model's
    budget either way.
    """
    if not text.strip():
        return []
    if _is_cjk(text):
        return [c for c in text if not c.isspace()]
    return text.split()


def split_into_sentences(text: str) -> List[str]:
    """Split on sentence-final punctuation, keeping the marks attached."""
    parts = _SENTENCE_END.split(text)
    sentences = []
    for i in range(0, len(parts), 2):
        current = parts[i]
        if i + 1 < len(parts):
            current += parts[i + 1]
        if current.strip():
            sentences.append(current.strip())
    return sentences


def chunk_text(text: str, min_words: int = CHUNK_MIN_WORDS,
               max_words: int = CHUNK_MAX_WORDS) -> List[str]:
    """Pack sentences into chunks of ``min_words``..``max_words`` — upstream's default
    ``GenerationType.CHUNKED`` behaviour.

    The LM degrades on long prompts, so upstream synthesizes chunk by chunk and
    concatenates the code streams. An over-long sentence is cut at ``max_words``.
    """
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    if not text:
        return []

    def join(tokens: List[str]) -> str:
        return "".join(tokens) if _is_cjk("".join(tokens)) else " ".join(tokens)

    chunks: List[str] = []
    current = ""
    count = 0
    for sentence in split_into_sentences(text) or [text]:
        sentence = sentence.strip()
        if not sentence:
            continue
        tokens = _tokenize_words(sentence)
        n = len(tokens)

        if n > max_words:
            if current:
                chunks.append(current)
                current, count = "", 0
            part: List[str] = []
            for token in tokens:
                part.append(token)
                if len(part) >= max_words:
                    chunks.append(join(part))
                    part = []
            if part:
                chunks.append(join(part))
            continue

        if count + n <= max_words:
            current = f"{current} {sentence}" if current else sentence
            count += n
        elif count >= min_words:
            chunks.append(current)
            current, count = sentence, n
        else:
            space_left = max_words - count
            head, tail = tokens[:space_left], tokens[space_left:]
            if current:
                joined = join(head)
                chunks.append(current + ("" if _is_cjk(joined) else " ") + joined)
            else:
                chunks.append(join(head))
            current = join(tail)
            count = len(tail)
    if current:
        chunks.append(current)
    return chunks


# ----------------------------------------------------------------------------------
# sampling — HuggingFace order (upstream's default backend)
# ----------------------------------------------------------------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _apply_repetition_penalty(scores: np.ndarray, seen: np.ndarray,
                              penalty: float) -> np.ndarray:
    """Divide positive scores of tokens in *seen* by *penalty*, multiply negative ones.

    ``seen`` is the recent-token window the caller chose, not the whole history —
    see :data:`REPETITION_WINDOW`.
    """
    if penalty == 1.0 or seen.size == 0:
        return scores
    out = scores.copy()
    idx = np.unique(seen[(seen >= 0) & (seen < out.shape[0])])
    if idx.size:
        s = out[idx]
        out[idx] = np.where(s <= 0, s * penalty, s / penalty)
    return out


def _sample(scores: np.ndarray, temperature: float, top_k: int, top_p: float,
            min_p: float, rng: np.random.Generator) -> int:
    """Temperature, then top-k, then top-p, then min-p — HuggingFace's warper order.

    ``outetts``'s default backend is ``Backend.HF``, so this reproduces the sampler the
    published defaults (``generation_config.json``) were tuned against. The order is
    *not* llama.cpp's: temperature is applied **first**, so the truncations below all
    operate on the already-tempered distribution. ``min_p`` has no llama.cpp-order
    equivalent in phoonnx's other codec-LMs — it drops every token whose probability is
    below ``min_p`` times the top token's.
    """
    s = np.asarray(scores, np.float32)
    if temperature <= 0:
        return int(np.argmax(s))
    s = s / float(temperature)
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
    if 0 < min_p < 1:
        probs = _softmax(s)
        s = np.where(probs < min_p * probs.max(), -np.inf, s)
    return int(rng.choice(s.shape[0], p=_softmax(s)))


# ----------------------------------------------------------------------------------
# loudness — ITU-R BS.1770-4, as pyloudnorm implements it
# ----------------------------------------------------------------------------------
def _biquad(kind: str, rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """K-weighting stage coefficients, ``pyloudnorm.IIRfilter`` designs them per rate."""
    if kind == "high_shelf":
        gain, q, fc = 4.0, 0.7071752369554196, 1681.9744509555319
    else:
        gain, q, fc = 0.0, 0.5003270373238773, 38.13547087602444
    a = 10 ** (gain / 40.0)
    w0 = 2.0 * math.pi * (fc / rate)
    alpha = math.sin(w0) / (2.0 * q)
    cos = math.cos(w0)
    if kind == "high_shelf":
        root = math.sqrt(a)
        b = np.array([a * ((a + 1) + (a - 1) * cos + 2 * root * alpha),
                      -2 * a * ((a - 1) + (a + 1) * cos),
                      a * ((a + 1) + (a - 1) * cos - 2 * root * alpha)])
        den = np.array([(a + 1) - (a - 1) * cos + 2 * root * alpha,
                        2 * ((a - 1) - (a + 1) * cos),
                        (a + 1) - (a - 1) * cos - 2 * root * alpha])
    else:
        b = np.array([(1 + cos) / 2, -(1 + cos), (1 + cos) / 2])
        den = np.array([1 + alpha, -2 * cos, 1 - alpha])
    return b / den[0], den / den[0]


def integrated_loudness(audio: np.ndarray, rate: int = SAMPLE_RATE,
                        block_size: float = 0.400) -> float:
    """Gated integrated loudness of mono *audio* in LUFS (ITU-R BS.1770-4).

    Reimplemented in numpy/scipy rather than pulled in as a dependency: it is two
    biquads and a two-stage gate, and ``pyloudnorm`` would be a new hard requirement
    for every phoonnx install. Returns ``-inf`` for silence, which
    :func:`normalize_loudness` passes through unchanged.
    """
    from scipy.signal import lfilter
    x = np.asarray(audio, np.float64).reshape(-1)
    for kind in ("high_shelf", "high_pass"):
        b, a = _biquad(kind, rate)
        x = lfilter(b, a, x)

    step = 0.25  # 75 % overlap
    duration = x.shape[0] / rate
    num_blocks = int(round((duration - block_size) / (block_size * step)) + 1)
    if num_blocks < 1:
        return -np.inf
    z = np.empty(num_blocks)
    for j in range(num_blocks):
        lo = int(block_size * (j * step) * rate)
        hi = int(block_size * (j * step + 1) * rate)
        z[j] = np.sum(np.square(x[lo:hi])) / (block_size * rate)
    with np.errstate(divide="ignore"):
        loudness = -0.691 + 10.0 * np.log10(z)

    gated = z[loudness > -70.0]
    if gated.size == 0:
        return -np.inf
    with np.errstate(divide="ignore"):
        relative = -0.691 + 10.0 * np.log10(gated.mean()) - 10.0
    gated = z[(loudness > -70.0) & (loudness > relative)]
    if gated.size == 0:
        return -np.inf
    return float(-0.691 + 10.0 * np.log10(gated.mean()))


def normalize_loudness(audio: np.ndarray, target: float = -18.0,
                       peak_limit: float = -1.0, rate: int = SAMPLE_RATE,
                       block_size: float = 0.400) -> np.ndarray:
    """Upstream's ``process_audio_tensor``: loudness-normalize, then peak-limit.

    Every OuteTTS waveform goes through this on the way out of the codec, so it is part
    of the model's output level, not a phoonnx-side taste decision. Clips shorter than
    one gating block are zero-padded for the measurement and trimmed back after, as
    upstream does.
    """
    x = np.asarray(audio, np.float64).reshape(-1)
    original = x.shape[0]
    if original == 0:
        return np.zeros(0, np.float32)
    minimum = int(block_size * rate)
    measured = integrated_loudness(
        np.pad(x, (0, max(0, minimum - original))), rate, block_size)
    if np.isfinite(measured):
        x = x * (10 ** ((target - measured) / 20.0))
    peak = float(np.max(np.abs(x)))
    threshold = 10 ** (peak_limit / 20.0)
    if peak > threshold:
        x = x * threshold / peak
    return np.asarray(x[:original], np.float32)


class OuteTTSAdapter(BaseOnnxAdapter):
    """Adapter for OuteTTS 1.0 (codec-LM + DAC.speech decoder)."""

    #: hard ceiling on the AR loop; DAC runs at ~46.9 frames/s and each frame is two
    #: tokens, so 4096 tokens is ~44 s of speech.
    MAX_NEW_TOKENS = 4096
    #: code frames per DAC decoder call (upstream's ``dac_decoding_chunk``)
    DECODE_CHUNK = 2048
    #: crossfade applied to every decoder chunk, in seconds (upstream's ``apply_fade``)
    FADE_SECONDS = 0.015

    def __init__(self):
        self.codec: Optional[onnxruntime.InferenceSession] = None
        self.tokenizer = None
        self.speakers: Dict[str, Dict[str, Any]] = {}
        self.default_speaker: Optional[str] = None
        self.audio_end_id: Optional[int] = None
        self.eos_id: Optional[int] = None
        self.word_start_id: Optional[int] = None
        self.c1: Dict[int, int] = {}
        self.c2: Dict[int, int] = {}
        self.num_layers = 0
        self.num_kv_heads = 0
        self.head_dim = 0

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def default_params(self) -> Dict[str, Any]:
        """Upstream's ``SamplerConfig`` / ``generation_config.json`` defaults."""
        return {
            "temperature": 0.4,
            "top_p": 0.9,
            "top_k": 40.0,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "max_new_tokens": float(self.MAX_NEW_TOKENS),
            "max_chunk_words": float(CHUNK_MAX_WORDS),
        }

    def param_labels(self) -> Dict[str, str]:
        return {
            "temperature": "sampling temperature",
            "top_p": "top-p (nucleus)",
            "top_k": "top-k",
            "min_p": "min-p (relative probability floor)",
            "repetition_penalty": "repetition penalty (64-token window)",
            "max_new_tokens": "max codec tokens (~94/s)",
            "max_chunk_words": "target words per model call",
            "voice": "speaker profile name",
        }

    def configure(self, voice_config: Any) -> None:
        """Load the DAC decoder, the checkpoint's tokenizer and the speaker profiles."""
        ep = getattr(voice_config, "engine_params", None) or {}
        if self.codec is None and ep.get("codec_decoder_path"):
            self.codec = make_session(ep["codec_decoder_path"], providers=ep.get("providers"))
        if self.tokenizer is None and ep.get("tokenizer_path"):
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(str(ep["tokenizer_path"]))
            self._build_token_maps()
        if not self.speakers and ep.get("speakers_path"):
            with open(ep["speakers_path"], encoding="utf-8") as f:
                data = json.load(f)
            # a bare upstream profile is a single speaker; phoonnx also accepts a
            # ``{"default_voice": ..., "speakers": {...}}`` bundle
            if "words" in data:
                name = data.get("name") or "default"
                self.speakers = {name: data}
                self.default_speaker = name
            else:
                self.speakers = data.get("speakers") or {}
                self.default_speaker = data.get("default_voice")

    def _build_token_maps(self) -> None:
        """Map ``<|c1_N|>`` / ``<|c2_N|>`` token ids back to codebook indices.

        Built from the tokenizer rather than from an offset table: the two checkpoints
        have different vocabularies, and nothing guarantees the codec tokens are
        contiguous in either.
        """
        for template, table in ((C1, self.c1), (C2, self.c2)):
            for i in range(CODEBOOK_TOKENS):
                tid = self.tokenizer.token_to_id(template.format(i))
                if tid is not None:
                    table[int(tid)] = i
        self.audio_end_id = self.tokenizer.token_to_id(AUDIO_END)
        self.eos_id = self.tokenizer.token_to_id(EOS)
        self.word_start_id = self.tokenizer.token_to_id(WORD_START)

    def _read_kv_shape(self, session: onnxruntime.InferenceSession) -> None:
        for spec in session.get_inputs():
            if spec.name.startswith("past_key_values.") and spec.name.endswith(".key"):
                self.num_layers += 1
                if self.num_kv_heads == 0:
                    self.num_kv_heads = int(spec.shape[1])
                    self.head_dim = int(spec.shape[3])

    # ------------------------------------------------------------------
    # prompt
    # ------------------------------------------------------------------
    def resolve_speaker(self, voice: Any, syn_config: Any) -> Optional[Dict[str, Any]]:
        """Pick the speaker profile: an explicit per-call ``voice``, else the one this
        phoonnx voice is pinned to, else the bundle's ``default_voice``."""
        name = None
        if syn_config is not None:
            name = (syn_config.extra_params or {}).get("voice")
        if not name:
            cfg = getattr(voice, "config", None)
            name = (getattr(cfg, "engine_params", None) or {}).get("voice")
        if not name:
            name = self.default_speaker
        if not name:
            return None
        if name not in self.speakers:
            raise ValueError(f"unknown OuteTTS speaker profile {name!r}; "
                             f"available: {sorted(self.speakers)}")
        return self.speakers[name]

    @staticmethod
    def _separator(text: str) -> str:
        """Sentence separator for the speaker transcript's script."""
        if any("぀" <= c <= "ゟ" or "゠" <= c <= "ヿ"
               or "一" <= c <= "鿿" for c in text):
            return "。"
        return ". "

    def merge_speaker_text(self, target: str, speaker_text: str) -> Tuple[str, str]:
        """Glue the speaker's transcript in front of the target text.

        The profile's audio *is* the start of the utterance, so its transcript has to
        lead the prompt; the separator that joins them is also appended to the profile's
        last word, so the codes and the text stay aligned.
        """
        speaker_text = speaker_text.strip()
        separator = self._separator(speaker_text)
        ends = ["。", "？", "！", "?", "!"] if separator == "。" \
            else [".", "?", "!"]
        joiner = ""
        if speaker_text:
            if speaker_text[-1] not in ends:
                joiner = separator
            elif separator != "。":
                joiner = " "
        return speaker_text + joiner + target.strip(), joiner.strip()

    @staticmethod
    def _feature_tokens(features: Dict[str, Any]) -> str:
        return "".join(f"<|{key}_{features.get(key, 0)}|>"
                       for key in ("energy", "spectral_centroid", "pitch"))

    def create_codes(self, words: List[Dict[str, Any]]) -> str:
        """Render the speaker profile's words as the model's in-context audio prompt."""
        lines = []
        for word in words:
            body = word["word"] + FEATURES + TIME.format(float(word["duration"]))
            body += self._feature_tokens(word.get("features") or {})
            pairs = "".join(C1.format(a) + C2.format(b)
                            for a, b in zip(word["c1"], word["c2"]))
            lines.append(WORD_START + body + CODE + pairs + WORD_END)
        return "\n".join(lines)

    def build_prompt(self, text: str, speaker: Optional[Dict[str, Any]] = None) -> str:
        """The exact completion prompt upstream's ``get_completion_prompt`` produces.

        The speaker dict is deep-copied first: upstream appends the separator to the
        profile's last word *in place*, which corrupts the profile for every later call
        in the same process.
        """
        text = text_normalizations(text)
        codes = ""
        if speaker is not None:
            speaker = copy.deepcopy(speaker)
            text, separator = self.merge_speaker_text(text, speaker["text"])
            speaker["words"][-1]["word"] += separator
            codes = self.create_codes(speaker["words"])
        prompt = f"{BOS}\n{TEXT_START}{text}{TEXT_END}\n{AUDIO_START}\n"
        if speaker is not None:
            prompt += codes + "\n" + WORD_START
        return prompt

    def encode_text(self, text: str, voice: Any, syn_config: Any) -> List[List[int]]:
        """Build one fully-formed prompt per model call and tokenize it.

        The speaker block is part of the prompt, so the whole string is tokenized in one
        pass by the checkpoint's own tokenizer: splicing separately-tokenized pieces
        would not reproduce training-time ids at the joins.
        """
        if self.tokenizer is None:
            raise RuntimeError("OuteTTS voice missing tokenizer_path in engine_params")
        budget = CHUNK_MAX_WORDS
        if syn_config is not None:
            budget = int((syn_config.extra_params or {}).get("max_chunk_words", budget))
        speaker = self.resolve_speaker(voice, syn_config)
        prompts = [self.build_prompt(chunk, speaker)
                   for chunk in chunk_text(text, max_words=max(1, budget))]
        return [list(self.tokenizer.encode(p, add_special_tokens=False).ids)
                for p in prompts]

    # ------------------------------------------------------------------
    # codec
    # ------------------------------------------------------------------
    def token_ids_to_codes(self, token_ids: List[int]) -> List[List[int]]:
        """Split the generated stream into the two DAC codebooks.

        The two runs are truncated to a common length: the model can stop mid-frame,
        having emitted a ``c1`` without its ``c2``.
        """
        first = [self.c1[t] for t in token_ids if t in self.c1]
        second = [self.c2[t] for t in token_ids if t in self.c2]
        n = min(len(first), len(second))
        return [first[:n], second[:n]]

    def _fade(self, audio: np.ndarray) -> np.ndarray:
        """Fade a decoder chunk in and out, as upstream does before concatenating."""
        total = audio.shape[0]
        length = min(int(SAMPLE_RATE * self.FADE_SECONDS), total // 2)
        if length <= 0:
            return audio
        audio = audio.copy()
        audio[:length] *= np.linspace(0.0, 1.0, length, dtype=audio.dtype)
        audio[total - length:] *= np.linspace(1.0, 0.0, length, dtype=audio.dtype)
        return audio

    def decode_codes(self, codes: List[List[int]]) -> np.ndarray:
        """Two DAC codebooks -> mono float32 waveform at 24 kHz."""
        if not codes or not codes[0]:
            return np.zeros(0, np.float32)
        arr = np.asarray(codes, np.int64).reshape(1, 2, -1)
        name = self.codec.get_inputs()[0].name
        pieces = []
        for start in range(0, arr.shape[-1], self.DECODE_CHUNK):
            chunk = arr[..., start:start + self.DECODE_CHUNK]
            audio = self.codec.run(None, {name: chunk})[0]
            pieces.append(self._fade(np.asarray(audio, np.float32).reshape(-1)))
        return normalize_loudness(np.concatenate(pieces))

    # ------------------------------------------------------------------
    # autoregressive loop
    # ------------------------------------------------------------------
    def generate(self, session: onnxruntime.InferenceSession, prompt_ids: List[int],
                 p: Dict[str, Any], rng: np.random.Generator) -> List[int]:
        """Prefill the prompt, then sample codec tokens until the model stops."""
        if not self.num_layers:
            self._read_kv_shape(session)
        out_names = [o.name for o in session.get_outputs()]
        past_names = [f"past_key_values.{i}.{k}"
                      for i in range(self.num_layers) for k in ("key", "value")]
        present_names = [f"present.{i}.{k}"
                         for i in range(self.num_layers) for k in ("key", "value")]

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
        top_p = float(p.get("top_p", 0.9))
        top_k = int(p.get("top_k", 40))
        min_p = float(p.get("min_p", 0.05))
        penalty = float(p.get("repetition_penalty", 1.1))
        max_new = max(1, int(p.get("max_new_tokens", self.MAX_NEW_TOKENS)))

        generated: List[int] = []
        history = list(prompt_ids)
        for step in range(max_new):
            # the export returns logits for every position, so the next-token
            # distribution is the last row — not the whole tensor as in NeuTTS
            logits = np.asarray(named["logits"], np.float32)[0, -1]
            logits = _apply_repetition_penalty(
                logits, np.asarray(history[-REPETITION_WINDOW:], np.int64), penalty)
            token = _sample(logits, temperature, top_k, top_p, min_p, rng)
            if token == self.audio_end_id or token == self.eos_id:
                break
            generated.append(token)
            history.append(token)
            if step == max_new - 1:
                LOG.warning("OuteTTS hit the %d-token cap (~%.1fs) without an end token",
                            max_new, max_new / (2 * FRAMES_PER_SECOND))
                break
            position = length + step
            feed = {
                "input_ids": np.asarray([[token]], np.int64),
                "attention_mask": np.ones((1, position + 1), np.int64),
                "position_ids": np.asarray([[position]], np.int64),
                **{name: named[q] for name, q in zip(past_names, present_names)},
            }
            named = dict(zip(out_names, session.run(None, feed)))
        return generated

    # ------------------------------------------------------------------
    # synthesis
    # ------------------------------------------------------------------
    def synthesize(self, request: AdapterSynthesisRequest,
                   session: onnxruntime.InferenceSession) -> AdapterSynthesisResult:
        if self.codec is None:
            raise RuntimeError("OuteTTS voice missing codec_decoder_path in engine_params")
        if self.tokenizer is None:
            raise RuntimeError("OuteTTS voice missing tokenizer_path in engine_params")
        if request.params.get("reference_audio") is not None:
            raise RuntimeError(
                "OuteTTS clones from pre-encoded speaker profiles, not from a reference "
                "clip: building one needs Whisper word alignment and the DAC encoder. "
                "Select a profile with extra_params={'voice': ...}")

        p = {**self.default_params(), **request.params}
        seed = p.get("seed")
        rng = np.random.default_rng(None if seed is None else int(seed))
        prompt_ids = [int(i) for i in np.asarray(request.phoneme_ids).reshape(-1)]
        token_ids = self.generate(session, prompt_ids, p, rng)
        codes = self.token_ids_to_codes(token_ids)
        if not codes[0]:
            raise RuntimeError("OuteTTS generated no codec tokens — lower the temperature "
                               "or pick another speaker profile")
        return AdapterSynthesisResult(audio=self.decode_codes(codes),
                                      extras={"frames": len(codes[0])})

    # required by the ABC but unused — synthesize() drives the AR loop.
    def build_feed_dict(self, request, session):
        raise NotImplementedError("OuteTTS is autoregressive — use synthesize()")

    def parse_outputs(self, outputs, request, output_names=None):
        raise NotImplementedError("OuteTTS is autoregressive — use synthesize()")

    @staticmethod
    def detect(config: Optional[Dict[str, Any]] = None,
               session: Optional[onnxruntime.InferenceSession] = None) -> bool:
        return bool(config and config.get("engine") == "outetts")
