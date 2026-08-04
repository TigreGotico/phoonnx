"""
Adapter for NVIDIA Magpie-TTS Multilingual.

Magpie is a transformer encoder-decoder that predicts discrete audio codec tokens
autoregressively. A 6-layer causal encoder reads the text. A 12-layer causal decoder
cross-attends to those states and emits 8 codebooks for 2 stacked frames per step, so
16 tokens at a time. A 2-layer local transformer then refines those 16 tokens one by
one. NanoCodec turns the finished codes into a 22.05 kHz waveform.

Speaker identity is a baked context embedding of shape (5, 217, 768) that is prepended
to the decoder input. This checkpoint dropped voice cloning, so the five voices are the
only voices.

Two things make this engine unusual.

**Classifier-free guidance.** The conditional and unconditional branches run as one
batch of 2. The unconditional branch gets a zeroed encoder output, a conditioning mask
that keeps only the first text position, and a zeroed context prefix.

**The attention prior.** Magpie does not learn a monotonic alignment on its own. Every
step, the loop reads the decoder's cross-attention, decides which text token is being
spoken, and builds a prior over the text for the *next* step that keeps the model moving
forward. The same signal decides when the text has run out and end-of-speech may be
predicted. That is why ``decoder_step`` returns cross-attention probabilities.

Because the prior is rebuilt every step, NeMo's default
(``use_kv_cache_for_inference: false``) re-applies the newest prior to every past query
position, which changes the past. A KV cache therefore produces a *different* sample
path, not a rounding difference. Both modes are supported and both match the matching
NeMo setting exactly; see ``exact_decode`` in the engine params.
"""
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime
from quebra_frases import sentence_tokenize

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter
from phoonnx.providers import make_session

# Byte-level tokenizers are byt5: <pad>, </s>, <unk>, then the 256 raw byte values.
BYTE_TOKENIZER_PREFIX = 3

# Languages whose tokenizer is a plain character or byte table, so phoonnx can reproduce
# it exactly from the shipped symbol table. The rest need NeMo's IPA G2P, which belongs
# in scriptconv rather than here.
BYTE_LANGUAGES = {"fr", "it", "vi", "ko"}
CHAR_LANGUAGES = {"ar"}

# Special tokens live at the end of each codebook, after the codec tokens.
AUDIO_BOS, AUDIO_EOS = 0, 1
N_SPECIAL_TOKENS = 8

MAX_CHUNK_CHARS = 200


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def chunk_text(text: str, max_len: int = MAX_CHUNK_CHARS) -> List[str]:
    """Split text into pieces short enough for one autoregressive pass.

    Magpie drifts on long inputs: the attention prior only tracks one text position at a
    time, and ``max_decoder_steps`` caps a pass at a few seconds of audio. Sentences are
    the natural unit, and oversized sentences are cut on whitespace.
    """
    chunks: List[str] = []
    for sentence in sentence_tokenize(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_len:
            chunks.append(sentence)
            continue
        current = ""
        for word in sentence.split():
            if current and len(current) + len(word) + 1 > max_len:
                chunks.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            chunks.append(current)
    return chunks or ([text.strip()] if text.strip() else [])


class MagpieTokenizer:
    """Reproduces Magpie's aggregated tokenizer from its exported symbol table.

    The checkpoint aggregates one sub-tokenizer per language into a single id space, each
    at a fixed offset. Character and byte sub-tokenizers are reproduced exactly here. The
    IPA sub-tokenizers (en, de, es, pt, hi) and the two that need external engines
    (zh via jieba, ja via pyopenjtalk) need a grapheme-to-phoneme step, which phoonnx
    keeps in scriptconv, so they are refused with a clear message rather than
    approximated.
    """

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.tokens: List[str] = data["tokens"]
        self.offsets: Dict[str, int] = data["tokenizer_offsets"]
        self.sizes: Dict[str, int] = data["num_tokens_per_tokenizer"]
        self.eos_id: int = int(data["eos_id"])
        self.vocab_size: int = int(data["vocab_size"])
        self._symbol_maps: Dict[str, Dict[str, int]] = {}

    def symbol_map(self, tokenizer_name: str) -> Dict[str, int]:
        """Symbol to global id for one sub-tokenizer, built once and kept."""
        if tokenizer_name not in self._symbol_maps:
            if tokenizer_name not in self.offsets:
                raise ValueError(f"Magpie tokenizer has no sub-tokenizer {tokenizer_name!r}")
            start = self.offsets[tokenizer_name]
            end = start + self.sizes[tokenizer_name]
            self._symbol_maps[tokenizer_name] = {
                symbol: start + i for i, symbol in enumerate(self.tokens[start:end])
            }
        return self._symbol_maps[tokenizer_name]

    def encode(self, text: str, lang: str, tokenizer_name: str) -> List[int]:
        """Turn text into global token ids, with the end-of-sentence id appended."""
        base = lang.split("-")[0].split("_")[0].lower()
        if base not in BYTE_LANGUAGES and base not in CHAR_LANGUAGES:
            raise NotImplementedError(
                f"Magpie needs grapheme-to-phoneme conversion for {base!r} "
                f"(sub-tokenizer {tokenizer_name!r}). phoonnx keeps phonemizers in "
                f"scriptconv; only the character and byte tokenizers "
                f"({sorted(BYTE_LANGUAGES | CHAR_LANGUAGES)}) are supported so far."
            )
        offset = self.offsets.get(tokenizer_name)
        if offset is None:
            raise ValueError(f"Magpie tokenizer has no sub-tokenizer {tokenizer_name!r}")

        if base in BYTE_LANGUAGES:
            ids = [offset + BYTE_TOKENIZER_PREFIX + b for b in text.encode("utf-8")]
        elif base in CHAR_LANGUAGES:
            symbols = self.symbol_map(tokenizer_name)
            ids = [symbols[c] for c in text if c in symbols]
        return ids + [self.eos_id]


class MagpieAdapter(BaseOnnxAdapter):
    """Adapter for NVIDIA Magpie-TTS Multilingual (multi-codebook autoregressive)."""

    def __init__(self):
        self.encoder: Optional[onnxruntime.InferenceSession] = None
        self.cross_kv: Optional[onnxruntime.InferenceSession] = None
        self.decoder: Optional[onnxruntime.InferenceSession] = None
        self.local: Optional[onnxruntime.InferenceSession] = None
        self.audio_embed: Optional[onnxruntime.InferenceSession] = None
        self.lt_embed: Optional[onnxruntime.InferenceSession] = None
        self.codec: Optional[onnxruntime.InferenceSession] = None
        self.tokenizer: Optional[MagpieTokenizer] = None
        self.context_embeddings: Optional[np.ndarray] = None
        self.speakers: Dict[str, int] = {}
        self.config: Dict[str, Any] = {}
        self._params: Dict[str, Any] = {}
        self._lang: str = "en"

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def default_params(self) -> Dict[str, float]:
        return {"temperature": 0.6, "top_k": 80.0, "cfg_scale": 2.5}

    def param_labels(self) -> Dict[str, str]:
        return {"temperature": "Sampling temperature", "top_k": "Top-k",
                "cfg_scale": "Classifier-free guidance scale"}

    def configure(self, voice_config: Any) -> None:
        """Open the six auxiliary graphs and read the static assets.

        The main session held by the voice layer is the decoder step, the graph that runs
        once per frame. Everything else is opened here.
        """
        ep = getattr(voice_config, "engine_params", None) or {}
        self._params = dict(ep)
        providers = ep.get("providers")

        for attribute, key in (("encoder", "text_encoder_path"),
                               ("cross_kv", "cross_kv_path"),
                               ("local", "local_step_path"),
                               ("audio_embed", "audio_embed_path"),
                               ("lt_embed", "lt_embed_path"),
                               ("codec", "codec_decoder_path")):
            if getattr(self, attribute) is None and ep.get(key):
                setattr(self, attribute, make_session(ep[key], providers=providers))

        if self.tokenizer is None and ep.get("tokenizer_path"):
            self.tokenizer = MagpieTokenizer(ep["tokenizer_path"])
        if self.context_embeddings is None and ep.get("context_embeddings_path"):
            self.context_embeddings = np.load(ep["context_embeddings_path"]).astype(np.float32)
        if not self.config and ep.get("magpie_config_path"):
            with open(ep["magpie_config_path"], "r", encoding="utf-8") as f:
                self.config = json.load(f)
        if not self.speakers and ep.get("speakers_path"):
            with open(ep["speakers_path"], "r", encoding="utf-8") as f:
                self.speakers = json.load(f)

        self._lang = (ep.get("lang")
                      or getattr(voice_config, "lang_code", None)
                      or getattr(voice_config, "lang", None)
                      or "en")

    # ------------------------------------------------------------------
    # Derived model constants
    # ------------------------------------------------------------------

    @property
    def num_codebooks(self) -> int:
        return int(self.config.get("num_audio_codebooks", 8))

    @property
    def frame_stacking(self) -> int:
        return int(self.config.get("frame_stacking_factor", 2))

    @property
    def stacked_codebooks(self) -> int:
        return self.num_codebooks * self.frame_stacking

    @property
    def codebook_size(self) -> int:
        return int(self.config.get("codebook_size", 2016))

    @property
    def tokens_per_codebook(self) -> int:
        return int(self.config.get("num_all_tokens_per_codebook", 2024))

    @property
    def audio_bos_id(self) -> int:
        return self.codebook_size + AUDIO_BOS

    @property
    def audio_eos_id(self) -> int:
        return self.codebook_size + AUDIO_EOS

    def _inference_param(self, key: str, default: Any) -> Any:
        return self.config.get("inference", {}).get(key, default)

    def forbidden_token_ids(self, forbid_eos: bool = False) -> List[int]:
        """Special-token ids that must never be sampled.

        Every special token is forbidden except end-of-speech, which is how generation
        stops. Early in a pass even that is forbidden, so a stray end token cannot cut a
        sentence off before it starts.
        """
        ids = [self.codebook_size + i for i in range(N_SPECIAL_TOKENS)]
        if not forbid_eos:
            ids.remove(self.audio_eos_id)
        return ids

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    def tokenizer_name_for(self, lang: str) -> str:
        """Sub-tokenizer that the checkpoint assigns to a language."""
        mapping = self.config.get("language_to_tokenizer", {})
        base = lang.split("-")[0].split("_")[0].lower()
        for key, names in mapping.items():
            if key.lower() == lang.lower() or key.split("-")[0].lower() == base:
                return names[0]
        raise ValueError(f"Magpie has no tokenizer for language {lang!r}. "
                         f"Known: {sorted(mapping)}")

    def encode_text(self, text: str, voice: Any, syn_config: Any) -> List[List[int]]:
        """Turn text into one token-id list per autoregressive pass."""
        if self.tokenizer is None:
            raise RuntimeError("Magpie voice missing tokenizer_path in engine_params")
        lang = (self._params.get("lang")
                or getattr(voice, "lang", None)
                or getattr(getattr(voice, "config", None), "lang_code", None)
                or self._lang)
        name = self.tokenizer_name_for(lang)
        return [self.tokenizer.encode(chunk, lang, name) for chunk in chunk_text(text)]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_codebook(self, logits: np.ndarray, temperature: float, top_k: int,
                        forbid_eos: bool, force_eos: bool, rng: np.random.Generator) -> int:
        """Pick one token from one codebook's logits.

        ``forbid_eos`` masks the end-of-speech token out while the attention prior's
        reading of the alignment says text remains, so a stray end token cannot cut a
        sentence off before it starts. ``force_eos`` is a hard override that always
        returns end-of-speech regardless of the sampled logits; the decode loop never
        sets it (real termination is sampled once ``forbid_eos`` lifts, detected by
        ``detect_eos``, see ``generate_codes``), but it is kept as a primitive that
        tests exercise directly and that a future explicit cutoff (e.g. a hard step
        budget) can call into without new plumbing.
        """
        if force_eos:
            return self.audio_eos_id
        logits = logits.astype(np.float64).copy()
        logits[self.forbidden_token_ids(forbid_eos)] = -np.inf
        k = max(1, min(int(top_k), logits.shape[-1]))
        threshold = np.partition(logits, -k)[-k]
        logits[logits < threshold] = -np.inf
        if temperature <= 0:
            return int(np.argmax(logits))
        probs = _softmax(logits / temperature)
        return int(rng.choice(probs.shape[-1], p=probs))

    def sample_frame_from_logits(self, logits: np.ndarray, temperature: float, top_k: int,
                                 forbid_eos: bool, force_eos: bool,
                                 rng: np.random.Generator) -> np.ndarray:
        """Sample all 16 stacked codebooks straight from the decoder logits.

        Used for the end-of-speech check, which compares a greedy read of the decoder
        against what the local transformer actually produced.
        """
        vocab = self.tokens_per_codebook
        out = np.zeros((self.num_codebooks, self.frame_stacking), np.int64)
        for fs in range(self.frame_stacking):
            for cb in range(self.num_codebooks):
                start = (cb + self.num_codebooks * fs) * vocab
                out[cb, fs] = self.sample_codebook(
                    logits[start:start + vocab], temperature, top_k, forbid_eos, force_eos, rng)
        return out

    def refine_frame(self, dec_out: np.ndarray, temperature: float, top_k: int,
                     cfg_scale: float, forbid_eos: bool, force_eos: bool,
                     rng: np.random.Generator) -> np.ndarray:
        """Run the local transformer over the 16 stacked codebooks.

        The decoder predicts all 16 tokens at once and independently. The local
        transformer walks them in order, each conditioned on the ones already chosen, so
        the codebooks of a frame agree with each other.
        """
        batch = dec_out.shape[0]
        hidden = dec_out[:, None, :].astype(np.float32)
        n_layers = int(self.config.get("local_transformer_n_layers", 2))
        heads = int(self.config.get("sa_n_heads", 12))
        head_dim = int(self.config.get("sa_d_head", 64))
        cache_k = np.zeros((n_layers, batch, 0, heads, head_dim), np.float32)
        cache_v = np.zeros((n_layers, batch, 0, heads, head_dim), np.float32)

        picks: List[int] = []
        for cb in range(self.stacked_codebooks):
            logits, cache_k, cache_v = self.local.run(None, {
                "h": hidden,
                "pos": np.array([cb], np.int64),
                "cache_k": cache_k,
                "cache_v": cache_v,
                "cb": np.array(cb, np.int64),
            })
            merged = logits[0]
            if batch == 2:
                merged = cfg_scale * logits[0] + (1.0 - cfg_scale) * logits[1]
            token = self.sample_codebook(merged, temperature, top_k, forbid_eos, force_eos, rng)
            picks.append(token)
            tokens = np.full((batch,), token, np.int64)
            hidden = self.lt_embed.run(None, {"tok": tokens, "cb": np.array(cb, np.int64)})[0]

        # picks is ordered frame-major (stack index outer, codebook inner)
        return np.asarray(picks, np.int64).reshape(self.frame_stacking, self.num_codebooks).T

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def mean_cross_attention(self, cross_attn_probs: np.ndarray,
                             layers: Optional[List[int]] = None) -> np.ndarray:
        """Average the cross-attention of the newest step over heads and layers.

        Shape in is (L, B, H, T, T_text); shape out is (B, T_text). Restricting to a
        subset of layers is how the model's own config picks the layers whose attention
        tracks the alignment most cleanly.
        """
        transcript_layers = self.config.get("transcript_decoder_layers")
        selected = []
        for layer_idx in range(cross_attn_probs.shape[0]):
            if layers is not None and layer_idx not in layers:
                continue
            if transcript_layers is not None and layer_idx not in transcript_layers:
                continue
            selected.append(cross_attn_probs[layer_idx].mean(axis=1))
        stacked = np.stack(selected, axis=1).mean(axis=1)
        return stacked[:, -1, :]

    def most_attended_position(self, alignment_scores: np.ndarray, last_attended: int,
                               text_len: int, counter: Dict[int, int]) -> int:
        """Decide which text position the model is speaking right now.

        The search is a small forward window from the last position, so the alignment can
        only move forward. A position attended too many times in a row is an attention
        sink, and the window steps past it. The last 3 positions are excluded so the
        search cannot park on the end of the sentence.
        """
        sink_threshold = int(self._inference_param("attention_sink_threshold", 4))
        window = int(self._inference_param("attention_prior_lookahead_window", 6))
        if counter.get(last_attended, 0) >= sink_threshold:
            last_attended += 1
        window_end = min(last_attended + window, text_len - 3)
        scores = alignment_scores[last_attended:window_end]
        if scores.size == 0:
            attended = text_len - 1
        else:
            attended = int(np.argmax(scores)) + last_attended
        counter[attended] = counter.get(attended, 0) + 1
        return attended

    def build_prior(self, text_len: int, attended: int, counter: Dict[int, int],
                    batch: int, text_positions: int) -> np.ndarray:
        """Build the attention prior for the next step.

        Everything is damped to ``epsilon`` except a small window around the position
        being spoken: one position back for pronunciation context, the current one, and
        the lookahead window. Positions that turned into attention sinks, and everything
        before them, are damped too, so the model cannot fall back into them.
        """
        epsilon = float(self._inference_param("attention_prior_epsilon", 0.1))
        window = int(self._inference_param("attention_prior_lookahead_window", 6))
        sink_threshold = int(self._inference_param("attention_sink_threshold", 4))

        prior = np.full((batch, 1, text_positions), epsilon, np.float32)
        if text_len <= 5:
            prior[:] = 1.0
            return prior
        prior[0, 0, max(1, attended - 1)] = 1.0
        prior[0, 0, attended] = 1.0
        for step in range(1, window + 1):
            prior[0, 0, min(attended + step, text_len - 1)] = 1.0
        for position, count in counter.items():
            if count >= sink_threshold:
                prior[0, 0, :position + 1] = epsilon
        if batch > 1:
            prior[1:] = prior[0]
        return prior

    # ------------------------------------------------------------------
    # End of speech
    # ------------------------------------------------------------------

    def find_eos_frame(self, codes: np.ndarray) -> float:
        """Index of the first frame in the stack that carries an end token."""
        method = str(self._inference_param("eos_detection_method", "argmax_or_multinomial_any"))
        mask = codes == self.audio_eos_id
        if method.endswith("_all"):
            per_frame = mask.all(axis=0)
        elif method.endswith("_zero_cb"):
            per_frame = mask[:1, :].any(axis=0)
        else:
            per_frame = mask.any(axis=0)
        hits = np.flatnonzero(per_frame)
        return float(hits[0]) if hits.size else float("inf")

    def detect_eos(self, sampled: np.ndarray, greedy: np.ndarray) -> float:
        """Whether this frame ends the utterance, and where inside the stack.

        The default method trusts either read: a greedy decode of the decoder logits or
        what the local transformer actually sampled. Whichever ends first wins.
        """
        method = str(self._inference_param("eos_detection_method", "argmax_or_multinomial_any"))
        if method.startswith("argmax_or_multinomial"):
            return min(self.find_eos_frame(greedy), self.find_eos_frame(sampled))
        return self.find_eos_frame(greedy)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _resolve_speaker(self, request: AdapterSynthesisRequest) -> int:
        speaker = request.speaker_id
        if speaker is None:
            speaker = self._params.get("speaker_id", 0)
        if isinstance(speaker, str):
            if speaker not in self.speakers:
                raise ValueError(f"Magpie has no voice {speaker!r}. Known: {sorted(self.speakers)}")
            speaker = self.speakers[speaker]
        speaker = int(speaker)
        if self.context_embeddings is not None and not 0 <= speaker < len(self.context_embeddings):
            raise ValueError(f"Magpie speaker index {speaker} is outside "
                             f"[0, {len(self.context_embeddings)})")
        return speaker

    def encode_conditioning(self, token_ids: np.ndarray, use_cfg: bool
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the encoder and pre-compute the cross-attention K/V for every layer.

        With guidance on, the unconditional branch is a zeroed encoder output whose mask
        keeps only the first position, which is what the checkpoint was trained against.
        """
        text_len = token_ids.shape[0]
        text = token_ids.reshape(1, -1).astype(np.int64)
        mask = np.ones((1, text_len), np.float32)
        cond = self.encoder.run(None, {"text": text, "text_mask": mask})[0]
        if use_cfg:
            cond = np.concatenate([cond, np.zeros_like(cond)], axis=0)
            cond_mask = np.zeros((2, text_len), np.float32)
            cond_mask[0, :] = 1.0
            cond_mask[1, 0] = 1.0
        else:
            cond_mask = mask
        cross_k, cross_v = self.cross_kv.run(None, {"cond": cond})
        return cond, cond_mask, cross_k, cross_v

    def generate_codes(self, token_ids: np.ndarray, speaker: int, params: Dict[str, Any]
                       ) -> np.ndarray:
        """Run the autoregressive loop and return codec codes of shape (8, T)."""
        temperature = float(params.get("temperature", 0.6))
        top_k = int(params.get("top_k", 80))
        cfg_scale = float(params.get("cfg_scale", self._inference_param("cfg_scale", 2.5)))
        use_cfg = bool(params.get("use_cfg", True))
        exact = bool(self._params.get("exact_decode", True))
        rng = np.random.default_rng(params.get("seed"))

        text_len = int(token_ids.shape[0])
        cond, cond_mask, cross_k, cross_v = self.encode_conditioning(token_ids, use_cfg)
        batch = cond.shape[0]
        text_positions = cond.shape[1]

        context = self.context_embeddings[speaker][None].astype(np.float32)
        if use_cfg:
            context = np.concatenate([context, np.zeros_like(context)], axis=0)

        bos = np.full((batch, self.stacked_codebooks), self.audio_bos_id, np.int64)
        x = np.concatenate([context, self.audio_embed.run(None, {"codes": bos})[0]], axis=1)
        positions = np.arange(x.shape[1], dtype=np.int64)

        heads = int(self.config.get("sa_n_heads", 12))
        head_dim = int(self.config.get("sa_d_head", 64))
        n_layers = int(self.config.get("decoder_n_layers", 12))
        empty_k = np.zeros((n_layers, batch, 0, heads, head_dim), np.float32)
        self_k, self_v = empty_k, empty_k

        prior = np.ones((batch, 1, text_positions), np.float32)
        alignment_layers = self.config.get("estimate_alignment_from_layers")
        max_steps = int(self._inference_param("max_decoder_steps", 500)) // self.frame_stacking
        min_frames = int(self._inference_param("min_generated_frames", 0))

        frames: List[np.ndarray] = []
        counter: Dict[int, int] = {}
        last_attended = 1
        end_frame: Optional[int] = None
        keep_open = False
        finished_steps = 0

        for step in range(max_steps):
            logits, dec_out, self_k, self_v, cross_attn = self.decoder.run(None, {
                "x": x, "pos": positions, "self_k": self_k, "self_v": self_v,
                "cross_k": cross_k, "cross_v": cross_v,
                "cond_mask": cond_mask, "attn_prior": prior,
            })

            step_logits = logits[:, -1, :]
            if use_cfg:
                step_logits = cfg_scale * step_logits[0] + (1.0 - cfg_scale) * step_logits[1]
            else:
                step_logits = step_logits[0]

            alignment = self.mean_cross_attention(cross_attn, alignment_layers)
            last_attended = self.most_attended_position(alignment[0], last_attended,
                                                        text_len, counter)
            prior = self.build_prior(text_len, last_attended, counter, batch, text_positions)

            # The text has run out once the alignment reaches its final positions. Until
            # then end-of-speech is forbidden, so the model cannot stop mid-sentence. Once
            # it is allowed, sampling decides when to actually emit it (see detect_eos
            # below) — generation is never force-terminated mid-loop.
            if last_attended >= text_len - 2 or end_frame is not None:
                finished_steps += 1
            keep_open = last_attended < text_len - 3 and end_frame is None
            if finished_steps > 5:
                keep_open = False

            forbid_eos = step * self.frame_stacking < min_frames or keep_open
            frame = self.refine_frame(dec_out[:, -1, :], temperature, top_k, cfg_scale,
                                      forbid_eos, False, rng)
            greedy = self.sample_frame_from_logits(step_logits, 0.0, 1, forbid_eos,
                                                   False, rng)

            if end_frame is None:
                found = self.detect_eos(frame, greedy)
                if found != float("inf"):
                    end_frame = step * self.frame_stacking + int(found)

            frames.append(frame)
            if end_frame is not None and len(frames) >= 4:
                break

            codes = frame.T.reshape(1, -1).astype(np.int64)
            next_embedding = self.audio_embed.run(
                None, {"codes": np.repeat(codes, batch, axis=0)})[0]
            if exact:
                # NeMo's default re-applies the newest prior over the whole history, so
                # the past changes every step and must be recomputed.
                x = np.concatenate([x, next_embedding], axis=1)
                self_k, self_v = empty_k, empty_k
                positions = np.arange(x.shape[1], dtype=np.int64)
            else:
                x = next_embedding
                positions = np.array([self_k.shape[2]], np.int64)

        stacked = np.concatenate(frames, axis=-1)
        length = end_frame if end_frame is not None else stacked.shape[-1]
        return stacked[:, :max(4, int(length))]

    def synthesize(self, request: AdapterSynthesisRequest,
                   session: onnxruntime.InferenceSession) -> AdapterSynthesisResult:
        """Generate one chunk of audio.

        ``session`` is the decoder step graph held by the voice layer; it is used when
        ``configure`` was not given a separate ``decoder_step_path``.
        """
        if self.decoder is None:
            self.decoder = session
        missing = [name for name in ("encoder", "cross_kv", "decoder", "local",
                                     "audio_embed", "lt_embed", "codec")
                   if getattr(self, name) is None]
        if missing:
            raise RuntimeError(f"Magpie voice is missing graphs: {missing}")
        if self.context_embeddings is None:
            raise RuntimeError("Magpie voice missing context_embeddings_path in engine_params")

        params = dict(self.default_params())
        params.update(request.params or {})
        token_ids = np.asarray(request.phoneme_ids, np.int64).reshape(-1)
        speaker = self._resolve_speaker(request)

        codes = self.generate_codes(token_ids, speaker, params)
        audio = self.codec.run(None, {"codes": codes[None].astype(np.int64)})[0]
        return AdapterSynthesisResult(audio=np.asarray(audio, np.float32).reshape(-1),
                                      extras={"codes": codes})

    # ------------------------------------------------------------------
    # Unused single-graph hooks
    # ------------------------------------------------------------------

    def build_feed_dict(self, request: AdapterSynthesisRequest,
                        session: onnxruntime.InferenceSession) -> Dict[str, np.ndarray]:
        raise NotImplementedError("Magpie drives seven graphs; see synthesize()")

    def parse_outputs(self, outputs: List[np.ndarray], request: AdapterSynthesisRequest,
                      output_names: Optional[List[str]] = None) -> AdapterSynthesisResult:
        raise NotImplementedError("Magpie drives seven graphs; see synthesize()")

    @staticmethod
    def detect(config: Optional[Dict[str, Any]] = None,
               session: Optional[onnxruntime.InferenceSession] = None) -> bool:
        """Match on the decoder step's distinctive signature.

        No other engine here takes a cross-attention prior alongside a split self- and
        cross-attention cache, so the input names alone are decisive.
        """
        if config and str(config.get("engine", "")).lower() == "magpie":
            return True
        if config and str(config.get("model_type", "")).lower() == "magpie-tts":
            return True
        if session is not None:
            names = {inp.name for inp in session.get_inputs()}
            return {"attn_prior", "cross_k", "self_k", "cond_mask"}.issubset(names)
        return False
