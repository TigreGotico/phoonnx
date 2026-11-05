"""ArkTTS inference adapter — DualAR codec LM with a 10-codebook 44.1 kHz codec.

ArkTTS is the architecture behind ``Audio8/Audio8-TTS-Preview-0.6b`` (Apache-2.0,
11 languages, zero-shot cloning) and its Basque fine-tune ``itzune/zortzi-tts``
(Apache-2.0, two voices). The two checkpoints ship byte-identical model code, tokenizer
and codec weights, so one adapter drives both and the differences live in the voice index.

Like Qwen3-TTS it has two stacked autoregressive models, not one:

* the **slow AR** — 24 decoder layers that emit one token per audio frame. That token is
  codebook 0, and its hidden state conditions the second model;
* the **fast AR** — 4 decoder layers that read the slow hidden state and write codebooks
  1..9 of the *same* frame, one codebook per step.

One 46 ms frame therefore costs one slow step plus nine fast steps. The ten codebooks are
fed back to the slow AR as a single summed embedding on the next position. This engine
drives the overridable ``BaseOnnxAdapter.synthesize`` rather than the single-graph
``build_feed_dict`` path, like its siblings :mod:`phoonnx.engines.qwen3tts` and
:mod:`phoonnx.engines.sparktts`.

Three ONNX graphs (HF: ``OpenVoiceOS/phoonnx-zortzi-tts``, ``OpenVoiceOS/phoonnx-audio8-tts``)::

    slow_ar.onnx        24-layer KV-cached backbone step (= the voice's ``session``)
    fast_ar.onnx        4-layer depth transformer over the ten codebooks
    codec_decoder.onnx  (1, 10, T) codes -> waveform @ 44.1 kHz

plus ``tokenizer.json`` (the model's own Qwen2 subword BPE) and one small JSON per voice
holding that voice's pre-encoded reference codes and the transcription of the clip they
came from.

**The voice is a reference clip, not a speaker id.** ArkTTS has no speaker table; the
speaker token is always ``<|speaker:0|>``. A voice is carried by the codec codes of a
short clip, which the prompt embeds directly, so the mirrors ship those codes rather than
the audio — the adapter never needs the codec *encoder*, only the decoder.

Two details of the ONNX contract are load-bearing:

* **The KV cache is a fixed 2048-wide window**, not a growing tensor. Each graph writes
  its new keys and values at ``input_pos`` and attends over the whole window, masking by
  ``key <= input_pos[query]``. The graph returns only the delta, so this module keeps the
  window and scatters into it. Writing a delta at the wrong offset gives wrong attention
  and no error.
* **The slow logits are sliced to 4097**, not the full 155776-entry vocabulary: the 4096
  semantic logits followed by the EOS logit. Upstream masks everything else away before it
  samples, so the export drops it. An index below 4096 is therefore already codebook 0's
  value, with no offset to subtract.

The sampler reproduces upstream's ``ArkttsModel._sample_semantic`` rather than the
HuggingFace ``generate`` stack, and the order is *not* the usual one: the semantic mask
runs first, then top-k and top-p together against the softmax of the **unscaled** logits,
and temperature divides last. Upstream calls this the "legacy" order; applying temperature
first, as HuggingFace does, changes which tokens survive the nucleus cut. On top of that
sits repetition-aware sampling — see :func:`sample_semantic`.

Greedy decoding is not offered. Both model cards state it runs into repetition loops that
never reach EOS, so ``do_sample`` defaults to true and the parity harness in
``scripts/conversion/arktts/`` is the only place greedy is used.
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime
from quebra_frases import sentence_tokenize

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter
from phoonnx.providers import make_session
from phoonnx.tokenizer import BPETokenizer
from phoonnx.util import LOG

SAMPLE_RATE = 44100
"""The codec decoder writes 44.1 kHz audio."""

FRAME_SIZE = 2048
"""Waveform samples per code frame — about 21.5 frames per second."""

NUM_CODEBOOKS = 10
"""Codebooks per frame: codebook 0 from the slow AR, 1..9 from the fast AR."""

CODEBOOK_SIZE = 4096
"""Entries per codebook, and the width of the slow AR's semantic logit block."""

EOS_INDEX = CODEBOOK_SIZE
"""Where EOS sits in the sliced slow logits, right after the 4096 semantic entries."""

SLOW_LAYERS, SLOW_KV_HEADS, SLOW_HEAD_DIM = 24, 2, 64
FAST_LAYERS, FAST_KV_HEADS, FAST_HEAD_DIM = 4, 2, 64

MAX_SEQ_LEN = 2048
"""The model's hard ceiling and the width of the slow cache."""

MAX_NEW_TOKENS = 512
"""Upstream's generation budget — about 24 seconds of audio at 21.5 frames per second."""

RAS_WINDOW_SIZE = 10
RAS_TOP_P = 0.9
RAS_TEMPERATURE = 1.0
"""Repetition-aware sampling: window length and the settings of the fallback draw."""

SEMANTIC_BEGIN_ID = 151678
"""First semantic token in the text vocabulary; sliced index ``i`` is this plus ``i``."""

MAX_CHUNK_CHARS = 200
"""Longest text handed to one autoregressive pass; longer input is split on sentences."""

SYSTEM_PROMPT = "convert the provided text to speech reference to the following:\n\nText:\n"
"""Opens the reference block. Upstream ``ArkttsProcessor._prompt_segments`` builds this."""


def chunk_text(text: str, max_len: int = MAX_CHUNK_CHARS) -> List[str]:
    """Pack whole sentences into chunks of at most ``max_len`` characters.

    One chunk is one autoregressive pass. The budget is smaller than Qwen3-TTS's because
    ArkTTS emits 21.5 frames per second against a 512-frame default, so a long chunk runs
    out of frames before it runs out of text.
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


def filter_top_k_top_p(logits: np.ndarray, top_k: int, top_p: float) -> np.ndarray:
    """Upstream's ``ArkttsLegacyTopKTopPLogitsProcessor``, kept in its own order.

    Two things differ from the usual HuggingFace warpers and both change the result:

    * the nucleus is measured on ``softmax(logits)`` *before* temperature is applied, so
      temperature cannot widen or narrow it;
    * the token that crosses the ``top_p`` threshold is removed rather than kept, because
      the test is ``cumulative > top_p`` on the inclusive cumulative sum.

    The top-ranked token always survives, which is what keeps the distribution non-empty
    when one token already carries more than ``top_p`` of the mass.
    """
    scores = np.asarray(logits, np.float64).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    shifted = np.exp(ordered - ordered.max())
    cumulative = np.cumsum(shifted / shifted.sum())
    remove = (cumulative > top_p) | (np.arange(ordered.size) >= top_k)
    remove[0] = False
    out = scores.copy()
    out[order[remove]] = -np.inf
    return out


def draw(scores: np.ndarray, rng: np.random.Generator, do_sample: bool) -> int:
    """Draw one index from already-filtered, already-tempered scores."""
    scores = np.asarray(scores, np.float64).reshape(-1)
    if not do_sample:
        return int(np.argmax(scores))
    finite = np.isfinite(scores)
    shifted = np.where(finite, scores - scores[finite].max(), -np.inf)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return int(rng.choice(probabilities.size, p=probabilities))


def sample_semantic(logits: np.ndarray, window: List[int], rng: np.random.Generator,
                    temperature: float, top_k: int, top_p: float, do_sample: bool) -> int:
    """Draw codebook 0 for one frame, with upstream's repetition-aware fallback.

    The regular draw uses the caller's settings. If it lands on a semantic token that is
    already in the last :data:`RAS_WINDOW_SIZE` tokens, upstream discards it and takes a
    second draw made under tighter settings (``top_p`` 0.9 at temperature 1.0) instead.
    That is what stops the model looping on one frame, and it is why the model cards say
    greedy decoding never terminates: with ``do_sample`` off both draws are the same
    argmax, so the fallback cannot break the loop.

    ``window`` holds *text-vocabulary* ids, not sliced indices. That is not cosmetic —
    upstream seeds the window with zeros, and zero is not a semantic token in the text
    vocabulary but *is* a valid sliced index, so a window kept in sliced space would treat
    an unfilled slot as a repeat of codebook value 0.
    """
    normal = draw(filter_top_k_top_p(logits, top_k, top_p) / max(temperature, 1e-5),
                  rng, do_sample)
    if not do_sample or normal == EOS_INDEX:
        return normal
    if SEMANTIC_BEGIN_ID + normal not in window:
        return normal
    return draw(filter_top_k_top_p(logits, top_k, RAS_TOP_P) / max(RAS_TEMPERATURE, 1e-5),
                rng, do_sample)


class ArkTTSAdapter(BaseOnnxAdapter):
    """Adapter for ArkTTS (slow AR + fast AR + 44.1 kHz codec decoder)."""

    MEMOIZED_WRITES = {
        # The reference system block never changes for a loaded voice.
        "prefix_ids": frozenset({"_prefix_ids"}),
    }

    def __init__(self):
        self.fast_ar: Optional[onnxruntime.InferenceSession] = None
        self.decoder: Optional[onnxruntime.InferenceSession] = None
        self.tokenizer: Optional[BPETokenizer] = None
        self.reference_codes: Optional[np.ndarray] = None
        self.reference_text: str = ""
        self._params: Dict[str, Any] = {}
        self._prefix_ids: List[int] = []

    def default_params(self) -> Dict[str, float]:
        # Both model cards pin these and warn against greedy decoding; the checkpoints'
        # generation_config.json is looser (0.7 / 0.9) and the cards override it.
        return {"temperature": 0.8, "top_k": 50.0, "top_p": 0.95,
                "max_new_tokens": float(MAX_NEW_TOKENS)}

    def param_labels(self) -> Dict[str, str]:
        return {"temperature": "Sampling temperature", "top_k": "Top-k",
                "top_p": "Nucleus (top-p)", "max_new_tokens": "Frame budget per chunk"}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def configure(self, voice_config: Any) -> None:
        """Open the two side graphs, the model's subword BPE, and the voice's codes."""
        ep = getattr(voice_config, "engine_params", None) or {}
        self._params = dict(ep)
        providers = ep.get("providers")

        for attribute, key in (("fast_ar", "fast_ar_path"), ("decoder", "codec_decoder_path")):
            if getattr(self, attribute) is None and ep.get(key):
                setattr(self, attribute, make_session(ep[key], providers=providers))

        if self.tokenizer is None and ep.get("bpe_tokenizer_path"):
            self.tokenizer = BPETokenizer(ep["bpe_tokenizer_path"])

        if self.reference_codes is None and ep.get("voice_codes_path"):
            self.load_voice(ep["voice_codes_path"])

    def load_voice(self, path: str) -> None:
        """Read a voice asset: the reference clip's codes and its transcription.

        The transcription is not decoration. It goes into the system prompt next to the
        codes, so the model sees what the reference clip says; a wrong transcription
        degrades the voice rather than raising anything.
        """
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        codes = np.asarray(data["codes"], np.int64)
        if codes.ndim != 2 or codes.shape[0] != NUM_CODEBOOKS or codes.shape[1] == 0:
            raise ValueError(f"ArkTTS voice codes must have shape [{NUM_CODEBOOKS}, T>0], "
                             f"got {codes.shape} from '{path}'")
        if codes.min() < 0 or codes.max() >= CODEBOOK_SIZE:
            raise ValueError(f"ArkTTS voice codes must be in [0, {CODEBOOK_SIZE - 1}]")
        reference_text = str(data.get("reference_text") or "").strip()
        if not reference_text:
            raise ValueError(f"ArkTTS voice '{path}' has no reference_text")
        self.reference_codes = codes
        self.reference_text = reference_text
        self._prefix_ids = []

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    def prefix_ids(self) -> List[int]:
        """The system block that opens the reference, tokenized once per voice.

        Upstream builds it by encoding each literal separately rather than encoding the
        joined string, and the two are not the same under BPE — a merge across a boundary
        would produce different ids. This reproduces the split.
        """
        if not self._prefix_ids:
            speaker = self.reference_text
            if not re.search(r"<\|speaker:\d+\|>", speaker):
                speaker = f"<|speaker:0|>{speaker}"
            self._prefix_ids = [
                token
                for part in ("<|im_start|>system\n", SYSTEM_PROMPT, speaker, "\n\nSpeech:\n")
                for token in self.tokenizer.tokenize(part)
            ]
        return self._prefix_ids

    def encode_text(self, text: str, voice: Any, syn_config: Any) -> List[List[int]]:
        """Turn text into one *suffix* id list per autoregressive pass.

        Only the suffix is returned. The prefix and the reference codes have to sit between
        the two halves of the prompt, so :meth:`build_prompt` reassembles them; the prefix
        depends on the voice alone and is cached by :meth:`prefix_ids`.
        """
        if self.tokenizer is None:
            raise RuntimeError("ArkTTS voice missing bpe_tokenizer_path in engine_params")
        return [
            [token
             for part in ("<|im_end|>\n", "<|im_start|>user\n", " ".join(chunk.split()),
                          "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>")
             for token in self.tokenizer.tokenize(part)]
            for chunk in chunk_text(text) if chunk.strip()
        ]

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def build_prompt(self, suffix_ids: np.ndarray) -> np.ndarray:
        """Assemble the ``[1, 11, T]`` prompt the slow AR was trained on.

        Row 0 is the token stream: the system prefix, then the reference clip's codebook-0
        codes shifted into the semantic range, then the user text. Rows 1..10 hold the
        reference clip's ten codebooks, aligned under those semantic tokens and zero
        everywhere else — the model reads them only where row 0 names a semantic token.
        """
        if self.reference_codes is None:
            raise RuntimeError("ArkTTS voice missing voice_codes_path in engine_params")
        prefix = np.asarray(self.prefix_ids(), np.int64)
        suffix = np.asarray(suffix_ids, np.int64).reshape(-1)
        codes = self.reference_codes
        frames = codes.shape[1]

        row0 = np.concatenate([prefix, codes[0] + SEMANTIC_BEGIN_ID, suffix])
        prompt = np.zeros((1, NUM_CODEBOOKS + 1, row0.size), np.int64)
        prompt[0, 0] = row0
        prompt[0, 1:, prefix.size:prefix.size + frames] = codes
        if prompt.shape[2] >= MAX_SEQ_LEN:
            raise ValueError(f"ArkTTS prompt is {prompt.shape[2]} tokens; the model holds "
                             f"fewer than {MAX_SEQ_LEN}")
        return prompt

    # ------------------------------------------------------------------
    # Cache plumbing
    # ------------------------------------------------------------------

    @staticmethod
    def cache_dtype(session: onnxruntime.InferenceSession) -> np.dtype:
        """Half or single precision, whichever this graph was exported in."""
        types = {i.name: i.type for i in session.get_inputs()}
        return np.float16 if types.get("cache_key_0") == "tensor(float16)" else np.float32

    def empty_cache(self, session, layers: int, width: int, heads: int,
                    head_dim: int) -> Dict[str, np.ndarray]:
        """The fixed-width window a fresh pass starts from.

        Slots are zero and stay unread until the loop writes them: the graph masks by
        ``key <= input_pos[query]``, so an unwritten slot can never enter a softmax.
        """
        dtype = self.cache_dtype(session)
        return {f"cache_{kind}_{i}": np.zeros((1, heads, width, head_dim), dtype)
                for i in range(layers) for kind in ("key", "value")}

    @staticmethod
    def scatter_cache(cache: Dict[str, np.ndarray], outputs: Dict[str, np.ndarray],
                      layers: int, input_pos: np.ndarray) -> None:
        """Write this step's deltas into the window at the positions they belong to."""
        for index in range(layers):
            for kind in ("key", "value"):
                cache[f"cache_{kind}_{index}"][:, :, input_pos] = outputs[f"{kind}_delta_{index}"]

    # ------------------------------------------------------------------
    # Fast AR
    # ------------------------------------------------------------------

    def generate_codebooks(self, slow_hidden: np.ndarray, first: int, rng, temperature: float,
                           top_k: int, top_p: float, do_sample: bool) -> List[int]:
        """Run the fast AR over codebooks 1..9 of one frame.

        Codebook 0 is the slow AR's own token and is passed in, not predicted here.
        Position 0 of the fast AR reads the slow hidden state and its logits are thrown
        away — it exists to seed the depth cache, which is what upstream's
        ``_generate_codebooks`` does before its loop starts.
        """
        names = [o.name for o in self.fast_ar.get_outputs()]
        dtype = np.float16 if self.fast_ar.get_inputs()[0].type == "tensor(float16)" else np.float32
        cache = self.empty_cache(self.fast_ar, FAST_LAYERS, NUM_CODEBOOKS,
                                 FAST_KV_HEADS, FAST_HEAD_DIM)

        def step(position: int, hidden: np.ndarray, token: int, use_hidden: bool):
            input_pos = np.asarray([position], np.int64)
            outputs = dict(zip(names, self.fast_ar.run(None, {
                "slow_hidden": hidden.astype(dtype),
                "token_id": np.asarray([[token]], np.int64),
                "use_slow_hidden": np.asarray([use_hidden], bool),
                "input_pos": input_pos, **cache})))
            self.scatter_cache(cache, outputs, FAST_LAYERS, input_pos)
            return outputs["logits"]

        zeros = np.zeros((1, 1, slow_hidden.shape[-1]), dtype)
        step(0, slow_hidden, 0, True)
        codebooks = [first]
        for position in range(1, NUM_CODEBOOKS):
            logits = step(position, zeros, codebooks[-1], False)
            scores = filter_top_k_top_p(logits[0, -1], top_k, top_p) / max(temperature, 1e-5)
            codebooks.append(draw(scores, rng, do_sample))
        return codebooks

    # ------------------------------------------------------------------
    # Slow AR
    # ------------------------------------------------------------------

    def generate_codes(self, session: onnxruntime.InferenceSession, prompt: np.ndarray,
                       params: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
        """Run both autoregressive loops and return the ``(10, frames)`` code matrix."""
        do_sample = bool(params.get("do_sample", True))
        temperature = float(params.get("temperature", 0.8))
        top_k = int(params.get("top_k", 50))
        top_p = float(params.get("top_p", 0.95))
        width = prompt.shape[2]
        budget = min(int(params.get("max_new_tokens", MAX_NEW_TOKENS)), MAX_SEQ_LEN - width)

        names = [o.name for o in session.get_outputs()]
        cache = self.empty_cache(session, SLOW_LAYERS, MAX_SEQ_LEN, SLOW_KV_HEADS, SLOW_HEAD_DIM)
        input_pos = np.arange(width, dtype=np.int64)
        step_codes = prompt
        window: List[int] = []
        frames: List[List[int]] = []

        for _ in range(budget):
            outputs = dict(zip(names, session.run(
                None, {"codes": step_codes, "input_pos": input_pos, **cache})))
            self.scatter_cache(cache, outputs, SLOW_LAYERS, input_pos)

            semantic = sample_semantic(outputs["logits"][0, -1], window, rng,
                                       temperature, top_k, top_p, do_sample)
            if semantic == EOS_INDEX:
                break
            codebooks = self.generate_codebooks(
                outputs["slow_hidden"][:, -1:], semantic, rng, temperature, top_k, top_p, do_sample)
            frames.append(codebooks)

            # Upstream fills the window only from the *second* frame on: it creates the
            # buffer after the first draw and writes into it from the next one. Reproduced
            # here, so the first frame is never treated as a repeat of itself.
            if window:
                window.append(SEMANTIC_BEGIN_ID + semantic)
                del window[:-RAS_WINDOW_SIZE]
            else:
                window = [0] * RAS_WINDOW_SIZE

            input_pos = np.asarray([input_pos[-1] + 1], np.int64)
            step_codes = np.asarray(codebooks, np.int64).reshape(1, NUM_CODEBOOKS, 1)
            step_codes = np.concatenate(
                [np.full((1, 1, 1), SEMANTIC_BEGIN_ID + semantic, np.int64), step_codes], axis=1)

        if not frames:
            return np.zeros((NUM_CODEBOOKS, 0), np.int64)
        return np.asarray(frames, np.int64).T

    # ------------------------------------------------------------------
    # Codec decoder
    # ------------------------------------------------------------------

    def decode_codes(self, codes: np.ndarray) -> np.ndarray:
        """Turn a ``(10, frames)`` code matrix into a waveform at 44.1 kHz."""
        feed = np.asarray(codes, np.int64).reshape(1, NUM_CODEBOOKS, -1)
        audio = self.decoder.run(None, {"codes": feed})[0]
        return np.asarray(audio, np.float32).reshape(-1)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(self, request: AdapterSynthesisRequest,
                   session: onnxruntime.InferenceSession) -> AdapterSynthesisResult:
        for attribute, key in (("fast_ar", "fast_ar_path"), ("decoder", "codec_decoder_path")):
            if getattr(self, attribute) is None:
                raise RuntimeError(f"ArkTTS voice missing {key} in engine_params")

        params = request.params
        prompt = self.build_prompt(request.phoneme_ids)
        rng = np.random.default_rng(params.get("seed"))
        codes = self.generate_codes(session, prompt, params, rng)
        if codes.shape[1] == 0:
            LOG.warning("ArkTTS produced no code frames for this chunk")
            return AdapterSynthesisResult(audio=np.zeros(0, np.float32))

        audio = self.decode_codes(codes)
        return AdapterSynthesisResult(
            audio=audio, extras={"frame_count": int(codes.shape[1]),
                                 "reference_text": self.reference_text})

    # build_feed_dict / parse_outputs are required by the ABC but unused — synthesize()
    # drives the two autoregressive loops directly.
    def build_feed_dict(self, request: AdapterSynthesisRequest,
                        session: onnxruntime.InferenceSession) -> Dict[str, np.ndarray]:
        raise NotImplementedError("ArkTTS is autoregressive — use synthesize()")

    def parse_outputs(self, outputs: List[np.ndarray], request: AdapterSynthesisRequest,
                      output_names: Optional[List[str]] = None) -> AdapterSynthesisResult:
        raise NotImplementedError("ArkTTS is autoregressive — use synthesize()")

    @staticmethod
    def detect(config: Optional[Dict[str, Any]] = None,
               session: Optional[onnxruntime.InferenceSession] = None) -> bool:
        return bool(config and config.get("engine") == "arktts")
