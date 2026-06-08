"""Chatterbox inference adapter — autoregressive codec-LM TTS with voice cloning.

Chatterbox (Resemble AI) is a Llama-based TTS that autoregressively generates speech
tokens conditioned on text + a reference speaker, then decodes them to audio. It is
phoonnx's first **autoregressive** engine (the second iterative subtype after the
flow-matching ZipVoice), driven through the overridable ``BaseOnnxAdapter.synthesize``.

Four ONNX graphs (HF: ``onnx-community/chatterbox-ONNX``):
  speech_encoder       : reference wav@24kHz -> (cond_emb, prompt_token, x_vector, feat)
  embed_tokens         : token ids -> embeddings, applying ``exaggeration``
  language_model[_q4]  : Llama, KV-cached AR step (= the voice's ``session``)
  conditional_decoder  : speech tokens (+ speaker x_vector/feat) -> waveform

Cloning is d-vector style (a reference wav, **no transcription**), so it uses the
``speaker_reference`` API and not ``speaker_reference_text``. Text is tokenized with the
model's own BPE (``phoonnx.tokenizer.BPETokenizer`` + the UNICODE phonemizer), so
``request.phoneme_ids`` already carries the subword ids.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter

S3GEN_SR = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
SILENCE_TOKEN = 4299


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


def _apply_repetition_penalty(prev_ids: np.ndarray, scores: np.ndarray, penalty: float) -> np.ndarray:
    """Divide the logits of already-emitted tokens by ``penalty`` (>1 discourages repeats)."""
    out = scores.copy()
    for b in range(prev_ids.shape[0]):
        ids = prev_ids[b]
        s = out[b, ids]
        out[b, ids] = np.where(s < 0, s * penalty, s / penalty)
    return out


def _sample_top_p(logits: np.ndarray, temperature: float, top_p: float, rng) -> np.ndarray:
    """Temperature + nucleus (top-p) sampling over a [1, vocab] logits row."""
    x = logits[0].astype(np.float64) / max(temperature, 1e-5)
    x -= x.max()
    probs = np.exp(x); probs /= probs.sum()
    order = np.argsort(probs)[::-1]
    cutoff = int(np.searchsorted(np.cumsum(probs[order]), top_p)) + 1
    keep = order[:max(1, cutoff)]
    p = probs[keep]; p /= p.sum()
    return np.array([[int(rng.choice(keep, p=p))]], np.int64)


class ChatterboxAdapter(BaseOnnxAdapter):
    """Adapter for Chatterbox (autoregressive codec-LM, d-vector cloning, exaggeration)."""

    REPETITION_PENALTY = 1.2
    MAX_NEW_TOKENS = 1000

    def __init__(self):
        self.embed_tokens = None
        self.speech_encoder = None
        self.cond_decoder = None
        self.past_names: List[str] = []
        self.num_kv_heads = 0
        self.head_dim = 0
        self.embed_inputs: set = set()   # which inputs embed_tokens accepts
        self.lm_needs_pos = False        # GPT-2-style LMs take position_ids (turbo)

    def default_params(self) -> Dict[str, float]:
        return {"exaggeration": 0.5, "temperature": 0.8, "top_p": 0.95}

    def encode_text(self, text: str, voice: Any, syn_config: Any) -> List[List[int]]:
        """BPE the raw text directly — Chatterbox's subword tokenizer owns normalization,
        so no phoneme front end (which would strip punctuation / expand numbers)."""
        return [voice.tokenizer.tokenize(text)]

    def configure(self, voice_config: Any) -> None:
        """Load the auxiliary graphs from ``engine_params`` and read the LM's KV-cache
        shape from its own input signature (no hardcoded layer counts)."""
        ep = getattr(voice_config, "engine_params", None) or {}
        sess = lambda p: onnxruntime.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        if self.embed_tokens is None and ep.get("embed_tokens_path"):
            self.embed_tokens = sess(ep["embed_tokens_path"])
        if self.embed_tokens is not None:
            self.embed_inputs = {i.name for i in self.embed_tokens.get_inputs()}
        if self.speech_encoder is None and ep.get("speech_encoder_path"):
            self.speech_encoder = sess(ep["speech_encoder_path"])
        if self.cond_decoder is None and ep.get("conditional_decoder_path"):
            self.cond_decoder = sess(ep["conditional_decoder_path"])

    def _read_kv_shape(self, session: onnxruntime.InferenceSession) -> None:
        names = [i.name for i in session.get_inputs()]
        self.past_names = [n for n in names if n.startswith("past_key_values")]
        self.lm_needs_pos = "position_ids" in names
        for i in session.get_inputs():
            if i.name.startswith("past_key_values"):
                shape = i.shape   # [B, num_kv_heads, seq, head_dim]
                self.num_kv_heads = int(shape[1])
                self.head_dim = int(shape[3])
                break

    def synthesize(self, request: AdapterSynthesisRequest,
                   session: onnxruntime.InferenceSession) -> AdapterSynthesisResult:
        if self.embed_tokens is None or self.speech_encoder is None or self.cond_decoder is None:
            raise RuntimeError("Chatterbox voice missing embed_tokens / speech_encoder / "
                               "conditional_decoder paths in engine_params")
        ref = request.params.get("reference_audio")
        if ref is None:
            raise RuntimeError("Chatterbox needs a reference clip (params['reference_audio'])")
        if not self.past_names:
            self._read_kv_shape(session)

        # speaker conditioning from the reference clip (no transcription needed)
        audio = _resample(np.asarray(ref[0], np.float32).reshape(-1), int(ref[1]), S3GEN_SR)
        cond_emb, prompt_token, x_vector, prompt_feat = self.speech_encoder.run(
            None, {"audio_values": audio[None, :].astype(np.float32)})

        input_ids = np.asarray(request.phoneme_ids, np.int64).reshape(1, -1)
        exaggeration = np.array([float(request.params.get("exaggeration", 0.5))], np.float32)
        temperature = float(request.params.get("temperature", 0.8))
        top_p = float(request.params.get("top_p", 0.95))
        rng = np.random.default_rng()
        embed_pos = np.where(input_ids >= START_SPEECH_TOKEN, 0,
                             np.arange(input_ids.shape[1])[None, :] - 1).astype(np.int64)

        def embed_feed(ids, pos):
            # feed only what this model's embed_tokens accepts (base/multilingual take
            # position_ids + exaggeration; turbo takes input_ids alone)
            feed = {"input_ids": ids}
            if "position_ids" in self.embed_inputs:
                feed["position_ids"] = pos
            if "exaggeration" in self.embed_inputs:
                feed["exaggeration"] = exaggeration
            return feed

        generated = np.array([[START_SPEECH_TOKEN]], np.int64)
        past = attention_mask = next_token = lm_pos = None
        for step in range(self.MAX_NEW_TOKENS):
            feed_in = embed_feed(input_ids, embed_pos) if step == 0 \
                else embed_feed(next_token, np.full((1, 1), step, np.int64))
            inputs_embeds = self.embed_tokens.run(None, feed_in)[0]
            if step == 0:
                inputs_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1)
                batch, seq_len, _ = inputs_embeds.shape
                past = {n: np.zeros([batch, self.num_kv_heads, 0, self.head_dim], np.float32)
                        for n in self.past_names}
                attention_mask = np.ones((batch, seq_len), np.int64)
                lm_pos = np.arange(seq_len)[None, :].astype(np.int64).repeat(batch, axis=0)
            feed = dict(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **past)
            if self.lm_needs_pos:                       # GPT-2-style LM (turbo) takes positions
                feed["position_ids"] = lm_pos
            logits, *present = session.run(None, feed)
            # repetition penalty over ALL emitted tokens (penalising only the last lets
            # the codec-LM babble), then sample (temperature/top_p) or greedy (temp<=0).
            logits = _apply_repetition_penalty(generated, logits[:, -1, :], self.REPETITION_PENALTY)
            if temperature > 0:
                next_token = _sample_top_p(logits, temperature, top_p, rng)
            else:
                next_token = np.argmax(logits, axis=-1, keepdims=True).astype(np.int64)
            generated = np.concatenate((generated, next_token), axis=-1)
            if (next_token.flatten() == STOP_SPEECH_TOKEN).all():
                break
            attention_mask = np.concatenate([attention_mask, np.ones((attention_mask.shape[0], 1), np.int64)], axis=1)
            lm_pos = lm_pos[:, -1:] + 1                  # positions stay cumulative
            for j, name in enumerate(self.past_names):
                past[name] = present[j]

        # strip start/stop, prepend the reference prompt token, append a little silence
        speech_tokens = generated[:, 1:-1]
        silence = np.full((speech_tokens.shape[0], 3), SILENCE_TOKEN, np.int64)
        speech_tokens = np.concatenate([prompt_token, speech_tokens, silence], axis=1)
        wav = self.cond_decoder.run(None, {"speech_tokens": speech_tokens,
                                           "speaker_embeddings": x_vector,
                                           "speaker_features": prompt_feat})[0]
        return AdapterSynthesisResult(audio=np.asarray(wav, np.float32).reshape(-1))

    def build_feed_dict(self, request, session):
        raise NotImplementedError("Chatterbox is autoregressive — use synthesize()")

    def parse_outputs(self, outputs, request):
        raise NotImplementedError("Chatterbox is autoregressive — use synthesize()")

    @staticmethod
    def detect(config: Optional[Dict[str, Any]] = None,
               session: Optional[onnxruntime.InferenceSession] = None) -> bool:
        return bool(config and config.get("engine") == "chatterbox")
