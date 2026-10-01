"""MOSS-TTS-Nano inference adapter — autoregressive codec-LM with zero-shot cloning.

MOSS-TTS-Nano (OpenMOSS, Apache-2.0) is a ~100M-parameter codec-LM TTS: a
12-layer LLM backbone predicts *frames* of RVQ-16 audio tokens, which a ~20M "Cat"
audio tokenizer (MOSS-Audio-Tokenizer-Nano) decodes to **48 kHz stereo** audio. It
covers 20 languages and clones a voice zero-shot from a reference clip alone — no
transcription, no d-vector, no speaker table.

It is phoonnx's second autoregressive engine after
:class:`phoonnx.engines.chatterbox.ChatterboxAdapter`, and shares that shape (raw text
-> subword ids, KV-cached decode loop, codec decode). What is different is that a single
decode step emits a whole **frame of 16 codebook tokens**, produced by a second, tiny
*local* transformer that runs once per frame on top of the global hidden state. So the
loop is two-level: global step -> local frame -> feed the frame back as the next global
input row.

Weights are mirrored in the OVOS HuggingFace org, under
``OpenVoiceOS/phoonnx-moss`` (``tts-nano-100m-onnx/`` and
``audio-tokenizer-nano-onnx/`` subfolders), crediting upstream OpenMOSS
(Apache-2.0).

Official ONNX exports are used as published (no re-export):

``OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX``
  ``moss_tts_prefill.onnx``          prompt rows -> global_hidden + 12 present KV pairs  (= ``session``)
  ``moss_tts_decode_step.onnx``      one row + past KV -> global_hidden + present KV
  ``moss_tts_local_fixed_sampled_frame.onnx``
                                     global_hidden -> should_continue + 16 sampled tokens
                                     (temperature/top-p/top-k/repetition penalty are
                                     **baked into the graph** at upstream defaults)
  ``moss_tts_local_cached_step.onnx`` per-channel local step with its own KV cache, so
                                     the host can sample with arbitrary parameters
  ``tokenizer.model``                SentencePiece text tokenizer
  ``*.data``                         external weights, resolved by filename next to the graphs

``OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX``
  ``moss_audio_tokenizer_encode.onnx``       reference wav[1,2,T]@48k -> codes[1,F,16]
  ``moss_audio_tokenizer_decode_full.onnx``  codes[1,F,16] -> audio[1,2,T]

Graph protocol (mirrors upstream ``ort_cpu_runtime.OrtCpuRuntime``)::

    rows          = text_rows(user_prefix + <audio_start>)
                  + audio_rows(reference codes, slot=<audio_user_slot>)
                  + text_rows(<audio_end> + instruction + text ids
                              + assistant_prefix + <audio_start>)
    global_hidden, past = prefill(rows, attention_mask)
    repeat up to max_new_frames:
        should_continue, frame[16] = local_frame(global_hidden)     # fixed | host-sampled
        if not should_continue: break
        row                 = [<audio_assistant_slot>, *frame]      # width 17
        global_hidden, past = decode_step(row, past_valid_lengths, past)
    audio[1,2,T] = codec.decode_full(frames)

Each input row is ``1 + n_vq = 17`` wide: column 0 is the text/slot token, columns 1..16
are the RVQ codebook tokens (``audio_pad_token_id`` on text rows).

Two deliberate divergences from upstream:

* **No WeTextProcessing.** Upstream normalizes text with WeTextProcessing (a heavy
  pynini/CJK-oriented dependency). phoonnx has its own preprocessing pipeline
  (``prepare_text``) and applies it before ``encode_text`` is ever called, so this
  adapter only adds the token-budget chunking upstream also does — sentence-split with
  the same ``quebra_frases`` splitter every other phoonnx engine uses, packed to
  ``max_text_tokens`` SentencePiece tokens per model call.
* **Whole-utterance codec decode.** The codec also ships a streaming
  ``decode_step`` graph with 4 transformer offsets + 12 attention caches; it exists to
  emit audio while the LM is still generating. phoonnx's ``synthesize`` returns a
  finished waveform, so ``decode_full`` is the faithful mapping and avoids carrying
  ~60 cache tensors for no benefit.

Output is downmixed stereo -> mono (channel mean), 1-D float32 in [-1, 1], at the
model's native 48 kHz.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime
from quebra_frases import sentence_tokenize

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter
from phoonnx.providers import make_session
from phoonnx.util import LOG

SAMPLE_RATE = 48000
CHANNELS = 2
N_VQ = 16
ROW_WIDTH = N_VQ + 1
CODEBOOK_SIZE = 1024
DOWNSAMPLE_RATE = 3840          # 48000 / 3840 = 12.5 frames per second
FRAMES_PER_SECOND = SAMPLE_RATE / DOWNSAMPLE_RATE

# special ids (tts_browser_onnx_meta.json -> model_config)
AUDIO_PAD_TOKEN_ID = 1024
AUDIO_START_TOKEN_ID = 6
AUDIO_END_TOKEN_ID = 7
AUDIO_USER_SLOT_TOKEN_ID = 8
AUDIO_ASSISTANT_SLOT_TOKEN_ID = 9

# The chat template MOSS-TTS-Nano was trained with, pre-tokenized (browser_poc_manifest
# .json -> prompt_templates). These are plain SentencePiece ids of the fixed system /
# instruction turns; keeping them as ids (rather than re-tokenizing prose here) is what
# makes the prompt reproducible without shipping the template text.
USER_PROMPT_PREFIX_TOKEN_IDS = (
    600, 289, 10356, 13, 10356, 10425, 1860, 4546, 12907, 10363, 13325, 11492,
)
USER_PROMPT_AFTER_REFERENCE_TOKEN_IDS = (
    10356, 10425, 3965, 7738, 11492, 505, 587, 10356, 10425, 352, 500, 856, 11492,
    505, 587, 10356, 10425, 2128, 1247, 594, 11492, 505, 587, 10356, 10425, 348,
    909, 561, 3648, 11492, 505, 587, 10356, 10425, 2818, 1305, 355, 348, 909,
    11492, 505, 587, 10356, 10425, 484, 339, 10367, 783, 11492, 505, 587, 10356,
    10425, 2427, 980, 11492,
)
ASSISTANT_PROMPT_PREFIX_TOKEN_IDS = (10356, 14, 5, 4, 8165, 430)

# Sampling constants compiled into moss_tts_local_fixed_sampled_frame.onnx. Asking for
# anything else means the host has to sample, via the local_cached_step graph.
FIXED_SAMPLING = {
    "text_temperature": 1.0,
    "text_top_p": 1.0,
    "text_top_k": 50,
    "audio_temperature": 0.8,
    "audio_top_p": 0.95,
    "audio_top_k": 25,
    "audio_repetition_penalty": 1.2,
}

SAMPLE_MODE_FIXED = "fixed"      # in-graph sampling (fast, upstream default)
SAMPLE_MODE_FULL = "full"        # host sampling via local_cached_step (tunable)
SAMPLE_MODE_GREEDY = "greedy"    # host argmax via local_cached_step


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target_sr)
        return resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
    except Exception:
        n = int(round(len(audio) * target_sr / sr))
        return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _apply_repetition_penalty(scores: np.ndarray, seen: set, penalty: float) -> np.ndarray:
    """Divide the score of already-emitted tokens by ``penalty`` (multiply when negative).

    Same convention as HuggingFace / :mod:`phoonnx.engines.chatterbox`: >1 discourages
    repeats. ``seen`` is the set of tokens this *channel* has already emitted — MOSS
    penalizes per RVQ channel, not across the whole frame.
    """
    if not seen or penalty == 1.0:
        return scores
    out = scores.copy()
    idx = np.fromiter((t for t in seen if 0 <= t < out.shape[0]), np.int64)
    if idx.size:
        s = out[idx]
        out[idx] = np.where(s < 0, s * penalty, s / penalty)
    return out


def _sample(scores: np.ndarray, temperature: float, top_k: int, top_p: float,
            rng: np.random.Generator, do_sample: bool = True) -> int:
    """Temperature + top-k + nucleus sampling over a 1-D score row (argmax if not sampling)."""
    if not do_sample:
        return int(np.argmax(scores))
    if temperature <= 0:
        raise ValueError("temperature must be positive when sampling")
    s = np.asarray(scores, np.float32) / float(temperature)
    if 0 < top_k < s.shape[0]:
        threshold = float(np.sort(s)[::-1][top_k - 1])
        s = np.where(s < threshold, -np.inf, s)
    if 0 < top_p < 1:
        order = np.argsort(s)[::-1]
        cumulative = np.cumsum(_softmax(s[order]))
        # keep every token up to and including the one that crosses top_p
        keep = np.ones(order.shape[0], bool)
        keep[1:] = cumulative[:-1] <= top_p
        s = np.full_like(s, -np.inf)
        s[order[keep]] = np.asarray(scores, np.float32)[order[keep]] / float(temperature)
    probs = _softmax(s)
    return int(rng.choice(probs.shape[0], p=probs))


class MossTTSNanoAdapter(BaseOnnxAdapter):
    """Adapter for MOSS-TTS-Nano (autoregressive RVQ codec-LM, zero-shot cloning)."""

    #: hard ceiling on the AR loop, ~30 s of audio at 12.5 frames/s
    MAX_NEW_FRAMES = 375
    #: SentencePiece tokens per model call (upstream's voice-clone chunk budget)
    MAX_TEXT_TOKENS = 75

    def __init__(self):
        self.decode_step: Optional[onnxruntime.InferenceSession] = None
        self.local_fixed_frame: Optional[onnxruntime.InferenceSession] = None
        self.local_cached_step: Optional[onnxruntime.InferenceSession] = None
        self.codec_encode: Optional[onnxruntime.InferenceSession] = None
        self.codec_decode: Optional[onnxruntime.InferenceSession] = None
        self.sp = None
        self.local_layers = 1
        self.local_heads = 12
        self.local_head_dim = 64

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def default_params(self) -> Dict[str, Any]:
        """Upstream's ``generation_defaults``, plus phoonnx's chunking budget."""
        return {
            "sample_mode": SAMPLE_MODE_FIXED,
            "max_new_frames": float(self.MAX_NEW_FRAMES),
            "max_text_tokens": float(self.MAX_TEXT_TOKENS),
            **FIXED_SAMPLING,
        }

    def param_labels(self) -> Dict[str, str]:
        return {
            "sample_mode": "sampling mode (fixed/full/greedy)",
            "max_new_frames": "max generated frames (12.5/s)",
            "max_text_tokens": "text tokens per model call",
            "text_temperature": "temperature (frame-continue decision)",
            "text_top_p": "top-p (frame-continue decision)",
            "text_top_k": "top-k (frame-continue decision)",
            "audio_temperature": "temperature (codebook tokens)",
            "audio_top_p": "top-p (codebook tokens)",
            "audio_top_k": "top-k (codebook tokens)",
            "audio_repetition_penalty": "repetition penalty (codebook tokens)",
        }

    def configure(self, voice_config: Any) -> None:
        """Load the five auxiliary graphs and the SentencePiece model from
        ``engine_params`` (the prefill graph is the voice's own session)."""
        ep = getattr(voice_config, "engine_params", None) or {}
        providers = ep.get("providers")

        def sess(key):
            path = ep.get(key)
            return make_session(path, providers=providers) if path else None

        if self.decode_step is None:
            self.decode_step = sess("decode_step_path")
        if self.local_fixed_frame is None:
            self.local_fixed_frame = sess("local_fixed_frame_path")
        if self.local_cached_step is None:
            self.local_cached_step = sess("local_cached_step_path")
        if self.codec_encode is None:
            self.codec_encode = sess("codec_encode_path")
        if self.codec_decode is None:
            self.codec_decode = sess("codec_decode_path")
        if self.sp is None and ep.get("sp_model_path"):
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor(model_file=str(ep["sp_model_path"]))
        if self.local_cached_step is not None:
            # local KV-cache geometry comes from the graph itself, not a hardcoded map
            for i in self.local_cached_step.get_inputs():
                if i.name == "local_past_key_0":
                    self.local_heads = int(i.shape[2])
                    self.local_head_dim = int(i.shape[3])
            self.local_layers = sum(
                1 for i in self.local_cached_step.get_inputs()
                if i.name.startswith("local_past_key_")
            )

    # ------------------------------------------------------------------
    # text
    # ------------------------------------------------------------------
    def encode_text(self, text: str, voice: Any, syn_config: Any) -> List[List[int]]:
        """SentencePiece-tokenize *text*, one id list per model call.

        MOSS-TTS-Nano degrades on long prompts, so upstream chunks by a token budget.
        Sentences come from the same ``quebra_frases`` splitter the rest of phoonnx
        uses; they are then packed greedily up to ``max_text_tokens`` tokens.
        """
        if self.sp is None:
            raise RuntimeError("MOSS-TTS-Nano voice missing sp_model_path in engine_params")
        budget = self.MAX_TEXT_TOKENS
        if syn_config is not None:
            budget = int(syn_config.extra_params.get("max_text_tokens", budget))
        return self._chunk(text, max(1, budget))

    def _encode(self, text: str) -> List[int]:
        return [int(i) for i in self.sp.encode(text, out_type=int)]

    def _chunk(self, text: str, budget: int) -> List[List[int]]:
        text = (text or "").strip()
        if not text:
            return []
        chunks: List[List[int]] = []
        current: List[int] = []
        for sentence in sentence_tokenize(text) or [text]:
            sentence = sentence.strip()
            if not sentence:
                continue
            ids = self._encode(sentence)
            # a sentence longer than the budget is hard-split rather than dropped
            pieces = ([ids[i:i + budget] for i in range(0, len(ids), budget)]
                      if len(ids) > budget else [ids])
            for piece in pieces:
                if current and len(current) + len(piece) > budget:
                    chunks.append(current)
                    current = []
                current.extend(piece)
        if current:
            chunks.append(current)
        return chunks

    # ------------------------------------------------------------------
    # prompt construction
    # ------------------------------------------------------------------
    @staticmethod
    def _text_rows(token_ids) -> List[List[int]]:
        """Text tokens occupy column 0; the 16 codebook columns are padded."""
        return [[int(t)] + [AUDIO_PAD_TOKEN_ID] * N_VQ for t in token_ids]

    @staticmethod
    def _audio_rows(codes, slot_token_id: int = AUDIO_USER_SLOT_TOKEN_ID) -> List[List[int]]:
        """Audio frames occupy columns 1..16; column 0 carries the speaker *slot* token."""
        rows = []
        for frame in codes:
            row = [int(slot_token_id)] + [AUDIO_PAD_TOKEN_ID] * N_VQ
            for i in range(min(len(frame), N_VQ)):
                row[i + 1] = int(frame[i])
            rows.append(row)
        return rows

    def build_prompt_rows(self, prompt_codes, text_token_ids) -> List[List[int]]:
        """The full ``[1, S, 17]`` prefill input: instruction, reference codes, text."""
        prefix = list(USER_PROMPT_PREFIX_TOKEN_IDS) + [AUDIO_START_TOKEN_ID]
        suffix = ([AUDIO_END_TOKEN_ID]
                  + list(USER_PROMPT_AFTER_REFERENCE_TOKEN_IDS)
                  + [int(t) for t in text_token_ids]
                  + list(ASSISTANT_PROMPT_PREFIX_TOKEN_IDS)
                  + [AUDIO_START_TOKEN_ID])
        return (self._text_rows(prefix)
                + self._audio_rows(prompt_codes)
                + self._text_rows(suffix))

    # ------------------------------------------------------------------
    # codec
    # ------------------------------------------------------------------
    def encode_reference(self, audio: np.ndarray, sr: int) -> List[List[int]]:
        """Reference clip -> RVQ-16 prompt codes. Mono is duplicated to stereo, since
        the tokenizer was trained on 2-channel 48 kHz audio."""
        wav = _resample(np.asarray(audio, np.float32).reshape(-1), int(sr), SAMPLE_RATE)
        stereo = np.repeat(wav[None, None, :], CHANNELS, axis=1).astype(np.float32)
        codes, lengths = self.codec_encode.run(None, {
            "waveform": stereo,
            "input_lengths": np.asarray([stereo.shape[-1]], np.int32)})
        n = int(np.asarray(lengths).reshape(-1)[0])
        codes = np.asarray(codes, np.int32)
        return [[int(codes[0, f, q]) for q in range(N_VQ)] for f in range(n)]

    def decode_frames(self, frames: List[List[int]]) -> np.ndarray:
        """RVQ frames -> mono float32 waveform (stereo output averaged down)."""
        if not frames:
            return np.zeros(0, np.float32)
        codes = np.asarray(frames, np.int32).reshape(1, len(frames), N_VQ)
        audio, lengths = self.codec_decode.run(None, {
            "audio_codes": codes,
            "audio_code_lengths": np.asarray([len(frames)], np.int32)})
        n = int(np.asarray(lengths).reshape(-1)[0])
        audio = np.asarray(audio, np.float32)[0, :, :n]     # [channels, T]
        return audio.mean(axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # local decoder (one frame of 16 codebook tokens)
    # ------------------------------------------------------------------
    @staticmethod
    def _seen_mask(seen_by_channel: List[set]) -> np.ndarray:
        mask = np.zeros((1, N_VQ, CODEBOOK_SIZE), np.int32)
        for ch, seen in enumerate(seen_by_channel):
            for token in seen:
                if 0 <= token < CODEBOOK_SIZE:
                    mask[0, ch, token] = 1
        return mask

    def _fixed_frame(self, global_hidden: np.ndarray, seen_by_channel: List[set],
                     rng: np.random.Generator) -> Tuple[bool, List[int]]:
        """In-graph sampled frame. The graph consumes uniform randoms so the host still
        owns the RNG (and therefore reproducibility) despite the baked-in parameters."""
        out = self.local_fixed_frame.run(None, {
            "global_hidden": global_hidden.astype(np.float32, copy=False),
            "repetition_seen_mask": self._seen_mask(seen_by_channel),
            "assistant_random_u": np.asarray([min(0.99999994, float(rng.random()))], np.float32),
            "audio_random_u": np.asarray(
                [[min(0.99999994, float(rng.random())) for _ in range(N_VQ)]], np.float32),
        })
        named = dict(zip([o.name for o in self.local_fixed_frame.get_outputs()], out))
        should_continue = bool(int(np.asarray(named["should_continue"]).reshape(-1)[0]))
        frame = [int(t) for t in np.asarray(named["frame_token_ids"]).reshape(-1)]
        return should_continue, frame

    def _empty_local_past(self) -> Dict[str, np.ndarray]:
        return {name: np.zeros((1, 0, self.local_heads, self.local_head_dim), np.float32)
                for i in range(self.local_layers)
                for name in (f"local_past_key_{i}", f"local_past_value_{i}")}

    def _local_step(self, global_hidden, text_token_id, audio_token_id, channel_index,
                    step_type, past_valid_lengths, past):
        out = self.local_cached_step.run(None, {
            "global_hidden": global_hidden.astype(np.float32, copy=False),
            "text_token_id": np.asarray([int(text_token_id)], np.int32),
            "audio_token_id": np.asarray([int(audio_token_id)], np.int32),
            "channel_index": np.asarray([int(channel_index)], np.int32),
            "step_type": np.asarray([int(step_type)], np.int32),
            "past_valid_lengths": np.asarray([int(past_valid_lengths)], np.int32),
            **past})
        named = dict(zip([o.name for o in self.local_cached_step.get_outputs()], out))
        nxt = {k.replace("local_present_", "local_past_"): v
               for k, v in named.items() if k.startswith("local_present_")}
        return np.asarray(named["text_logits"]).reshape(-1), np.asarray(named["audio_logits"]), nxt

    def _hosted_frame(self, global_hidden: np.ndarray, seen_by_channel: List[set],
                      p: Dict[str, Any], do_sample: bool,
                      rng: np.random.Generator) -> Tuple[bool, List[int]]:
        """Host-sampled frame via the cached local steps, honouring the caller's
        temperature / top-p / top-k / repetition penalty.

        Three ``step_type``s, per the export: 0 = ask whether another frame follows,
        1 = first codebook channel, 2 = each subsequent channel conditioned on the
        previously sampled one.
        """
        past = self._empty_local_past()
        valid = 0
        text_logits, _, past = self._local_step(global_hidden, 0, 0, 0, 0, valid, past)
        valid += 1
        # the LM only ever chooses between "another assistant frame" and "audio ends"
        candidates = np.asarray([AUDIO_ASSISTANT_SLOT_TOKEN_ID, AUDIO_END_TOKEN_ID], np.int64)
        choice = _sample(text_logits[candidates],
                         float(p["text_temperature"]),
                         min(int(p["text_top_k"]), candidates.shape[0]),
                         float(p["text_top_p"]), rng, do_sample)
        if int(candidates[choice]) != AUDIO_ASSISTANT_SLOT_TOKEN_ID:
            return False, []

        penalty = float(p["audio_repetition_penalty"])
        temperature = float(p["audio_temperature"])
        top_k, top_p = int(p["audio_top_k"]), float(p["audio_top_p"])

        def pick(audio_logits, channel):
            per_channel = int(audio_logits.shape[-1])
            flat = audio_logits.reshape(-1)
            scores = np.asarray(flat[channel * per_channel:(channel + 1) * per_channel], np.float32)
            scores = _apply_repetition_penalty(scores, seen_by_channel[channel], penalty)
            return _sample(scores, temperature, top_k, top_p, rng, do_sample)

        _, audio_logits, past = self._local_step(
            global_hidden, AUDIO_ASSISTANT_SLOT_TOKEN_ID, 0, 0, 1, valid, past)
        valid += 1
        frame = [pick(audio_logits, 0)]
        for channel in range(1, N_VQ):
            _, audio_logits, past = self._local_step(
                global_hidden, 0, frame[-1], channel - 1, 2, valid, past)
            valid += 1
            frame.append(pick(audio_logits, channel))
        return True, frame

    # ------------------------------------------------------------------
    # synthesis
    # ------------------------------------------------------------------
    def generate_frames(self, session: onnxruntime.InferenceSession,
                        rows: List[List[int]], p: Dict[str, Any],
                        rng: np.random.Generator) -> List[List[int]]:
        """Prefill, then run the global decode loop until the model stops or the
        ``max_new_frames`` safety cap is hit."""
        mode = str(p.get("sample_mode", SAMPLE_MODE_FIXED))
        if mode == SAMPLE_MODE_FIXED and any(
                float(p[k]) != float(v) for k, v in FIXED_SAMPLING.items()):
            LOG.debug("MOSS-TTS-Nano: sampling params differ from the graph's baked-in "
                      "constants — switching to host sampling")
            mode = SAMPLE_MODE_FULL
        if mode == SAMPLE_MODE_FIXED and self.local_fixed_frame is None:
            mode = SAMPLE_MODE_FULL
        if mode != SAMPLE_MODE_FIXED and self.local_cached_step is None:
            raise RuntimeError(f"MOSS-TTS-Nano sample_mode={mode!r} needs local_cached_step_path")

        ids = np.asarray(rows, np.int32).reshape(1, len(rows), ROW_WIDTH)
        mask = np.ones((1, len(rows)), np.int32)
        out = session.run(None, {"input_ids": ids, "attention_mask": mask})
        named = dict(zip([o.name for o in session.get_outputs()], out))
        global_hidden = self._last_hidden(named["global_hidden"])
        past = {k.replace("present_", "past_"): v
                for k, v in named.items() if k.startswith("present_")}
        valid_length = len(rows)

        frames: List[List[int]] = []
        seen_by_channel = [set() for _ in range(N_VQ)]
        max_frames = int(p.get("max_new_frames", self.MAX_NEW_FRAMES))
        for _ in range(max(1, max_frames)):
            if mode == SAMPLE_MODE_FIXED:
                keep_going, frame = self._fixed_frame(global_hidden, seen_by_channel, rng)
            else:
                keep_going, frame = self._hosted_frame(
                    global_hidden, seen_by_channel, p, mode != SAMPLE_MODE_GREEDY, rng)
            if not keep_going:
                break
            for channel, token in enumerate(frame):
                seen_by_channel[channel].add(int(token))
            frames.append(frame)

            row = np.full((1, 1, ROW_WIDTH), AUDIO_PAD_TOKEN_ID, np.int32)
            row[0, 0, 0] = AUDIO_ASSISTANT_SLOT_TOKEN_ID
            row[0, 0, 1:len(frame) + 1] = frame
            feed = {"input_ids": row,
                    "past_valid_lengths": np.asarray([valid_length], np.int32),
                    **past}
            out = self.decode_step.run(None, feed)
            named = dict(zip([o.name for o in self.decode_step.get_outputs()], out))
            global_hidden = self._last_hidden(named["global_hidden"])
            past = {k.replace("present_", "past_"): v
                    for k, v in named.items() if k.startswith("present_")}
            valid_length += 1
        else:
            LOG.warning("MOSS-TTS-Nano hit the %d-frame cap (%.1fs) without an end token",
                        max_frames, max_frames / FRAMES_PER_SECOND)
        return frames

    @staticmethod
    def _last_hidden(hidden: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden, np.float32)
        return hidden if hidden.ndim == 2 else hidden[:, -1, :]

    def synthesize(self, request: AdapterSynthesisRequest,
                   session: onnxruntime.InferenceSession) -> AdapterSynthesisResult:
        """Encode the reference clip (or reuse precomputed codes), build the
        prompt rows and run the autoregressive frame loop to produce audio.

        ``params['prompt_audio_codes']`` is an advanced/test-caller entry point
        for feeding already-encoded reference codes directly; production
        callers clone through ``params['reference_audio']``, which this method
        runs through :meth:`encode_reference` on every call.
        """
        missing = [k for k, v in (("decode_step_path", self.decode_step),
                                  ("codec_encode_path", self.codec_encode),
                                  ("codec_decode_path", self.codec_decode)) if v is None]
        if missing:
            raise RuntimeError(f"MOSS-TTS-Nano voice missing {', '.join(missing)} in engine_params")

        p = request.params
        codes = p.get("prompt_audio_codes")
        if codes is None:
            ref = p.get("reference_audio")
            if ref is None:
                raise RuntimeError("MOSS-TTS-Nano needs a reference clip "
                                   "(params['reference_audio']) — it clones zero-shot "
                                   "and has no built-in speakers")
            codes = self.encode_reference(np.asarray(ref[0], np.float32), int(ref[1]))
        if not codes:
            raise RuntimeError("reference clip is too short to produce any codec frames")

        rows = self.build_prompt_rows(codes, np.asarray(request.phoneme_ids).reshape(-1))
        seed = p.get("seed")
        rng = np.random.default_rng(None if seed is None else int(seed))
        frames = self.generate_frames(session, rows, {**self.default_params(), **p}, rng)
        audio = self.decode_frames(frames)
        return AdapterSynthesisResult(audio=audio, extras={"frames": len(frames)})

    # required by the ABC but unused — synthesize() drives the multi-graph loop.
    def build_feed_dict(self, request, session):
        raise NotImplementedError("MOSS-TTS-Nano is autoregressive — use synthesize()")

    def parse_outputs(self, outputs, request, output_names=None):
        raise NotImplementedError("MOSS-TTS-Nano is autoregressive — use synthesize()")

    @staticmethod
    def detect(config: Optional[Dict[str, Any]] = None,
               session: Optional[onnxruntime.InferenceSession] = None) -> bool:
        return bool(config and config.get("engine") == "mosstts")
