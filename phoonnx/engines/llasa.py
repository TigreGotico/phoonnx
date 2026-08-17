"""Llasa inference adapter — autoregressive XCodec2 codec-LM (HKUST).

Llasa (`HKUSTAudio <https://huggingface.co/HKUSTAudio/Llasa-1B>`_, arXiv 2502.04128)
is a LLaMA-3.2 backbone whose vocabulary was extended with the 65,536
``<|s_N|>`` speech tokens of **XCodec2**, a single-codebook 16 kHz codec running at
50 tokens/s. Give it text, and it emits a flat run of speech tokens that XCodec2's
decoder turns back into a waveform. There is no duration model, no vocoder and no
speaker encoder — the same shape as
:class:`phoonnx.engines.neutts.NeuTTSAdapter`, with a different backbone and codec.

Two ONNX graphs::

    model.onnx           LLaMA backbone, KV-cached  (= the voice's ``session``)
    xcodec2_decoder.onnx codes[1, 1, N] -> audio[1, T] @ 16 kHz

``model.onnx`` serves prefill *and* decode — one graph, different past length::

    inputs   input_ids                  int64 [1, S]        prompt, or 1 token per step
             attention_mask             int64 [1, P + S]    ones over past and current
             position_ids               int64 [1, S]        absolute positions P .. P+S-1
             past_key_values.<i>.key    fp32  [1, 8, P, 64] for i in 0..15
             past_key_values.<i>.value  fp32  [1, 8, P, 64]
    outputs  logits                     fp32  [1, 1, V]     last position only
             present.<i>.key / present.<i>.value

The KV geometry is read off the graph's own input signature, so the 3B and 8B
siblings need no code change. See ``scripts/conversion/llasa/`` for the export.

Prompt format
~~~~~~~~~~~~~
Llasa was trained on the stock LLaMA-3.2-Instruct chat template, so the prompt is
that template verbatim::

    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Cutting Knowledge Date: December 2023
    Today Date: 26 Jul 2024

    <|eot_id|><|start_header_id|>user<|end_header_id|>

    Convert the text to speech:<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|><|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>

    <|SPEECH_GENERATION_START|>

The system block is what ``apply_chat_template`` injects, and the date inside it is
whatever day inference runs on. phoonnx pins it to :data:`TEMPLATE_DATE` instead, so
the same text always produces the same prompt ids; the model saw a moving date
during training and does not key on it.

Voices
~~~~~~
Llasa needs no reference at all: with the plain prompt above it invents a speaker
per call. To give the engine stable, repeatable voices, a preset supplies a
reference transcript plus the XCodec2 codes of its recording, appended after
``<|SPEECH_GENERATION_START|>`` so generation continues the same speaker
in-context. That is upstream's own cloning recipe with the encode step done ahead
of time.

Cloning is therefore **pre-encoded**, like NeuTTS: ``speaker_reference`` (a fresh
clip) is not supported, because tokenising one needs XCodec2's encoder together
with the w2v-BERT filterbank front end, neither of which this bundle ships.
"""
import json
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime
from quebra_frases import sentence_tokenize

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter
from phoonnx.providers import make_session
from phoonnx.util import LOG

SAMPLE_RATE = 16000

#: XCodec2 emits 50 tokens per second of audio.
TOKENS_PER_SECOND = 50

#: Number of entries in the XCodec2 codebook; ``<|s_0|>`` .. ``<|s_65535|>``.
NUM_SPEECH_TOKENS = 65536

#: Frozen "Today Date" for the chat template. Any date reproduces training-time
#: conditions; freezing one makes phoonnx's prompt ids deterministic.
TEMPLATE_DATE = "26 Jul 2024"

PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "Cutting Knowledge Date: December 2023\nToday Date: {date}\n\n"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Convert the text to speech:"
    "<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "<|SPEECH_GENERATION_START|>"
)

SPEECH_GENERATION_END = "<|SPEECH_GENERATION_END|>"
SPEECH_TOKEN_ZERO = "<|s_0|>"

#: Reference codes fed as the generation prefix. Llasa was trained at a 2048-token
#: context, so a long reference leaves no room for the target; 500 codes is 10 s.
MAX_REFERENCE_CODES = 500


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _sample(scores: np.ndarray, temperature: float, top_p: float,
            rng: np.random.Generator) -> int:
    """Temperature, then nucleus — HuggingFace's warper order (argmax if temp<=0).

    Upstream drives this checkpoint through ``transformers.generate``, whose default
    processor list scales the logits by ``temperature`` *before* ``top_p`` truncates
    the softmax. Truncating first (llama.cpp's order, which
    :mod:`phoonnx.engines.neutts` needs) would select from a different candidate set.
    """
    s = np.asarray(scores, np.float64)
    if temperature <= 0:
        return int(np.argmax(s))
    probs = _softmax(s / float(temperature))
    if 0 < top_p < 1:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        keep = np.ones(order.shape[0], bool)
        # keep every token up to and including the one that crosses top_p
        keep[1:] = cumulative[:-1] < top_p
        kept = order[keep]
        p = probs[kept] / probs[kept].sum()
        return int(rng.choice(kept, p=p))
    return int(rng.choice(probs.shape[0], p=probs))


class LlasaAdapter(BaseOnnxAdapter):
    """Adapter for the Llasa family (LLaMA codec-LM + XCodec2 decoder).

    ``synthesize()`` retries the AR loop up to ``MAX_ATTEMPTS`` times if it comes
    back with zero speech tokens (see ``generate()``/``_mask_logits``). That
    pathology — the model ending a run before it ever emits a speech token — is
    already foreclosed for a normal call: step 0 of ``generate()`` masks out
    ``<|SPEECH_GENERATION_END|>``, so the first sampled token can only come from
    the ``<|s_N|>`` speech-token block. The retry loop is belt-and-braces for
    anything that reaches ``token_ids_to_codes`` with an empty id list despite
    that mask (a zero ``max_new_tokens``, or a caller driving ``generate()``
    directly), and it still raises ``RuntimeError`` if every attempt comes back
    empty — see the regression test in ``tests/test_llasa.py`` that forces this
    path by monkeypatching ``token_ids_to_codes``.
    """

    MEMOIZED_WRITES = {
        # KV-cache geometry, read off the model's own input shapes.
        "_read_kv_shape": frozenset({"num_layers", "num_kv_heads", "head_dim"}),
    }

    #: hard ceiling on the AR loop; 50 tokens/s, so 1000 is ~20 s
    MAX_NEW_TOKENS = 1000
    #: characters of target text per model call — the LM degrades on long prompts
    MAX_CHUNK_CHARS = 300
    #: resample attempts when the LM ends the run before emitting any speech token
    MAX_ATTEMPTS = 3

    def __init__(self):
        self.codec: Optional[onnxruntime.InferenceSession] = None
        self.tokenizer = None
        self.presets: Dict[str, Dict[str, Any]] = {}
        self.default_preset: Optional[str] = None
        self.speech_end_id: Optional[int] = None
        self.speech_token_base: Optional[int] = None
        self.num_layers = 0
        self.num_kv_heads = 0
        self.head_dim = 0

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def default_params(self) -> Dict[str, Any]:
        """Upstream's model-card settings (the 2025-05-10 "more stable" pair)."""
        return {
            "temperature": 0.9,
            "top_p": 0.95,
            "max_new_tokens": float(self.MAX_NEW_TOKENS),
            "max_chunk_chars": float(self.MAX_CHUNK_CHARS),
        }

    def param_labels(self) -> Dict[str, str]:
        return {
            "temperature": "sampling temperature",
            "top_p": "top-p (nucleus)",
            "max_new_tokens": "max codec tokens (50/s)",
            "max_chunk_chars": "target characters per model call",
            "voice": "voice preset name",
        }

    def configure(self, voice_config: Any) -> None:
        """Load the XCodec2 decoder, the checkpoint's BPE and the presets."""
        ep = getattr(voice_config, "engine_params", None) or {}
        if self.codec is None and ep.get("codec_decoder_path"):
            self.codec = make_session(ep["codec_decoder_path"], providers=ep.get("providers"))
        if self.tokenizer is None and ep.get("tokenizer_path"):
            from phoonnx.tokenizer import load_hf_tokenizer
            self.tokenizer = load_hf_tokenizer(ep["tokenizer_path"])
            self.speech_end_id = self.tokenizer.token_to_id(SPEECH_GENERATION_END)
            self.speech_token_base = self.tokenizer.token_to_id(SPEECH_TOKEN_ZERO)
        if not self.presets and ep.get("voices_path"):
            with open(ep["voices_path"], encoding="utf-8") as f:
                voices = json.load(f)
            self.presets = voices.get("presets") or {}
            self.default_preset = voices.get("default_voice")

    def _read_kv_shape(self, session: onnxruntime.InferenceSession) -> None:
        for spec in session.get_inputs():
            if spec.name.startswith("past_key_values.") and spec.name.endswith(".key"):
                self.num_layers += 1
                if self.num_kv_heads == 0:
                    self.num_kv_heads = int(spec.shape[1])
                    self.head_dim = int(spec.shape[3])

    # ------------------------------------------------------------------
    # text
    # ------------------------------------------------------------------
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
            raise ValueError(f"unknown Llasa voice preset {name!r}; "
                             f"available: {sorted(self.presets)}")
        return self.presets[name]

    def chunk_text(self, text: str, max_chars: int) -> List[str]:
        """Pack whole sentences up to ``max_chars`` per model call.

        Llasa was trained at a 2048-token context and drifts on long prompts, so the
        target is split before synthesis. Sentences come from the ``quebra_frases``
        splitter the rest of phoonnx uses; an over-long single sentence is split on
        word boundaries rather than dropped.
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

    def build_prompt(self, target_text: str, reference_text: str = "") -> str:
        """The chat-template string the checkpoint was trained on.

        For a cloning call upstream prepends the reference transcript to the target
        inside the *same* ``<|TEXT_UNDERSTANDING_...|>`` block — the model reads one
        continuous utterance whose first half it has already been given the audio for.
        """
        text = f"{reference_text} {target_text}".strip() if reference_text else target_text.strip()
        return PROMPT_TEMPLATE.format(date=TEMPLATE_DATE, text=text)

    def codes_to_token_ids(self, codes: List[int]) -> List[int]:
        """XCodec2 indices -> ``<|s_N|>`` vocabulary ids (a contiguous block)."""
        if self.speech_token_base is None:
            raise RuntimeError("Llasa tokenizer has no <|s_0|> token")
        return [self.speech_token_base + int(c) for c in codes]

    def encode_text(self, text: str, voice: Any, syn_config: Any) -> List[List[int]]:
        """Build one fully-formed prompt per model call and BPE-encode it.

        The reference *codes* are appended as raw ids rather than as ``<|s_N|>`` text:
        they are single, unmergeable special tokens, so the two are identical, and a
        500-token reference would otherwise build a 5 kB string per call.
        """
        if self.tokenizer is None:
            raise RuntimeError("Llasa voice missing tokenizer_path in engine_params")
        budget = int(self.MAX_CHUNK_CHARS)
        if syn_config is not None:
            budget = int((syn_config.extra_params or {}).get("max_chunk_chars", budget))
        preset = self.resolve_preset(voice, syn_config)
        ref_text, ref_ids = "", []
        if preset is not None:
            ref_text = (preset.get("text") or "").strip()
            ref_ids = self.codes_to_token_ids(
                list(preset.get("codes") or [])[:MAX_REFERENCE_CODES])
        out: List[List[int]] = []
        for chunk in self.chunk_text(text, max(1, budget)):
            prompt = self.build_prompt(chunk, ref_text)
            out.append(list(self.tokenizer.encode(prompt, add_special_tokens=False).ids) + ref_ids)
        return out

    # ------------------------------------------------------------------
    # codec
    # ------------------------------------------------------------------
    def token_ids_to_codes(self, token_ids: List[int]) -> List[int]:
        """Keep only ids inside the ``<|s_N|>`` block and map them back to indices."""
        if self.speech_token_base is None:
            raise RuntimeError("Llasa tokenizer has no <|s_0|> token")
        base = self.speech_token_base
        return [int(t) - base for t in token_ids
                if base <= int(t) < base + NUM_SPEECH_TOKENS]

    def decode_codes(self, codes: List[int]) -> np.ndarray:
        """XCodec2 indices -> mono float32 waveform at 16 kHz."""
        if not codes:
            return np.zeros(0, np.float32)
        arr = np.asarray(codes, np.int64).reshape(1, 1, -1)
        name = self.codec.get_inputs()[0].name
        audio = self.codec.run(None, {name: arr})[0]
        return np.asarray(audio, np.float32).reshape(-1)

    # ------------------------------------------------------------------
    # autoregressive loop
    # ------------------------------------------------------------------
    def _mask_logits(self, logits: np.ndarray, allow_end: bool) -> np.ndarray:
        """Keep only the ``<|s_N|>`` block, plus the end marker once a run has started."""
        base = self.speech_token_base
        masked = np.full_like(logits, -np.inf)
        stop = min(base + NUM_SPEECH_TOKENS, logits.shape[0])
        masked[base:stop] = logits[base:stop]
        if allow_end and self.speech_end_id is not None:
            masked[self.speech_end_id] = logits[self.speech_end_id]
        return masked

    def generate(self, session: onnxruntime.InferenceSession, prompt_ids: List[int],
                 p: Dict[str, Any], rng: np.random.Generator) -> List[int]:
        """Prefill the prompt, then sample speech tokens until the model stops."""
        if not self.num_layers:
            self._read_kv_shape(session)
        out_names = [o.name for o in session.get_outputs()]
        past_names = [f"past_key_values.{i}.{k}"
                      for i in range(self.num_layers) for k in ("key", "value")]
        present_names = [f"present.{i}.{k}"
                         for i in range(self.num_layers) for k in ("key", "value")]

        ids = np.asarray(prompt_ids, np.int64).reshape(1, -1)
        length = ids.shape[1]
        empty = np.zeros((1, self.num_kv_heads, 0, self.head_dim), np.float32)
        feed = {
            "input_ids": ids,
            "attention_mask": np.ones((1, length), np.int64),
            "position_ids": np.arange(length, dtype=np.int64)[None, :],
            **{n: empty for n in past_names},
        }
        named = dict(zip(out_names, session.run(None, feed)))

        temperature = float(p.get("temperature", 0.9))
        top_p = float(p.get("top_p", 0.95))
        max_new = max(1, int(p.get("max_new_tokens", self.MAX_NEW_TOKENS)))

        generated: List[int] = []
        for step in range(max_new):
            logits = np.asarray(named["logits"], np.float32).reshape(-1)
            # Only a speech token or the end marker can be a valid continuation. The
            # rest of the 193,800-entry vocabulary is text, and a stray text token
            # would be dropped later, silently cutting a hole in the audio. At step 0
            # the end marker is masked too: an empty run is never an answer, and this
            # prompt reaches p(end) ~ 0.25 on some sentences.
            logits = self._mask_logits(logits, allow_end=step > 0)
            token = _sample(logits, temperature, top_p, rng)
            if token == self.speech_end_id:
                break
            generated.append(token)
            if step == max_new - 1:
                LOG.warning("Llasa hit the %d-token cap (%.1fs) without an end token",
                            max_new, max_new / TOKENS_PER_SECOND)
                break
            position = length + step
            feed = {
                "input_ids": np.asarray([[token]], np.int64),
                "attention_mask": np.ones((1, position + 1), np.int64),
                "position_ids": np.asarray([[position]], np.int64),
                **{past: named[present] for past, present in zip(past_names, present_names)},
            }
            named = dict(zip(out_names, session.run(None, feed)))
        return generated

    # ------------------------------------------------------------------
    # synthesis
    # ------------------------------------------------------------------
    def synthesize(self, request: AdapterSynthesisRequest,
                   session: onnxruntime.InferenceSession) -> AdapterSynthesisResult:
        if self.codec is None:
            raise RuntimeError("Llasa voice missing codec_decoder_path in engine_params")
        if self.tokenizer is None:
            raise RuntimeError("Llasa voice missing tokenizer_path in engine_params")
        if request.params.get("reference_audio") is not None:
            raise RuntimeError(
                "Llasa clones from pre-encoded voice presets, not from a reference clip: "
                "tokenising one needs XCodec2's encoder and its w2v-BERT front end, which "
                "this bundle does not ship. Select a preset with extra_params={'voice': ...}")

        p = {**self.default_params(), **request.params}
        seed = p.get("seed")
        rng = np.random.default_rng(None if seed is None else int(seed))
        prompt_ids = [int(i) for i in np.asarray(request.phoneme_ids).reshape(-1)]
        codes: List[int] = []
        for attempt in range(self.MAX_ATTEMPTS):
            codes = self.token_ids_to_codes(self.generate(session, prompt_ids, p, rng))
            if codes:
                break
            # generate() masks out <|SPEECH_GENERATION_END|> at step 0 (see
            # _mask_logits), so an empty-first-token accident is already foreclosed
            # for any prompt that reaches this code — every token in a normal run is
            # drawn from the speech-token block. This retry is belt-and-braces for
            # paths that bypass that mask (e.g. ``max_new_tokens=0``, or a future
            # caller that samples outside ``generate()``), not a routine event.
            LOG.warning("Llasa returned no speech tokens (attempt %d/%d) — resampling",
                        attempt + 1, self.MAX_ATTEMPTS)
        if not codes:
            raise RuntimeError(f"Llasa generated no speech tokens in {self.MAX_ATTEMPTS} "
                               f"attempts — lower the temperature or pick another preset")
        return AdapterSynthesisResult(audio=self.decode_codes(codes),
                                      extras={"codes": len(codes)})

    # required by the ABC but unused — synthesize() drives the AR loop.
    def build_feed_dict(self, request, session):
        raise NotImplementedError("Llasa is autoregressive — use synthesize()")

    def parse_outputs(self, outputs, request, output_names=None):
        raise NotImplementedError("Llasa is autoregressive — use synthesize()")

    @staticmethod
    def detect(config: Optional[Dict[str, Any]] = None,
               session: Optional[onnxruntime.InferenceSession] = None) -> bool:
        return bool(config and config.get("engine") == "llasa")
