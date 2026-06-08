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

    def default_params(self) -> Dict[str, float]:
        return {"exaggeration": 0.5}

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
        if self.speech_encoder is None and ep.get("speech_encoder_path"):
            self.speech_encoder = sess(ep["speech_encoder_path"])
        if self.cond_decoder is None and ep.get("conditional_decoder_path"):
            self.cond_decoder = sess(ep["conditional_decoder_path"])

    def _read_kv_shape(self, session: onnxruntime.InferenceSession) -> None:
        self.past_names = [i.name for i in session.get_inputs() if i.name.startswith("past_key_values")]
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
        position_ids = np.where(input_ids >= START_SPEECH_TOKEN, 0,
                                np.arange(input_ids.shape[1])[None, :] - 1).astype(np.int64)
        embed_in = {"input_ids": input_ids, "position_ids": position_ids, "exaggeration": exaggeration}

        generated = np.array([[START_SPEECH_TOKEN]], np.int64)
        past = attention_mask = None
        for step in range(self.MAX_NEW_TOKENS):
            inputs_embeds = self.embed_tokens.run(None, embed_in)[0]
            if step == 0:
                inputs_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1)
                batch, seq_len, _ = inputs_embeds.shape
                past = {n: np.zeros([batch, self.num_kv_heads, 0, self.head_dim], np.float32)
                        for n in self.past_names}
                attention_mask = np.ones((batch, seq_len), np.int64)
            logits, *present = session.run(None, dict(inputs_embeds=inputs_embeds,
                                                      attention_mask=attention_mask, **past))
            logits = _apply_repetition_penalty(generated[:, -1:], logits[:, -1, :], self.REPETITION_PENALTY)
            next_token = np.argmax(logits, axis=-1, keepdims=True).astype(np.int64)
            generated = np.concatenate((generated, next_token), axis=-1)
            if (next_token.flatten() == STOP_SPEECH_TOKEN).all():
                break
            embed_in["input_ids"] = next_token
            embed_in["position_ids"] = np.full((1, 1), step + 1, np.int64)
            attention_mask = np.concatenate([attention_mask, np.ones((attention_mask.shape[0], 1), np.int64)], axis=1)
            for j, name in enumerate(self.past_names):
                past[name] = present[j]

        speech_tokens = np.concatenate([prompt_token, generated[:, 1:-1]], axis=1)
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
