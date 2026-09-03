"""ZipVoice inference adapter — flow-matching zero-shot cloning.

ZipVoice (k2-fsa) is a 123M flow-matching TTS with **in-context** cloning: the
reference audio + its transcription are part of the model input and the model
continues that voice. There is no speaker encoder / d-vector, so this does not use
the :mod:`phoonnx.engines.speaker_encoders` registry.

This is phoonnx's first **iterative** engine. It overrides :meth:`synthesize` to drive
a short ODE loop over a flow-matching vector field across three ONNX graphs:

  mel          : prompt wav@24kHz -> log-mel[1,100,T]   (VocosFbank, exported no-torch)
  text_encoder : tokens, prompt_tokens, prompt_len, speed -> text_condition[1,T,100]
  fm_decoder   : t, x, text_condition, speech_condition, guidance -> vector field   (= ``session``)

then the Vocos vocoder (the existing vocoder registry) inverts the mel. The voice's
primary model is the ``fm_decoder``; the mel + text_encoder graphs and the vocoder are
loaded in :meth:`configure` from ``engine_params``.

Two normalization constants matter (else the output is a quiet, range-compressed hiss):
the prompt wav is RMS-normalized to ``target_rms`` (0.1) before the mel, and the model
works in feature space scaled by ``feat_scale`` (0.1) — the prompt mel is multiplied by
it and the generated mel is divided by it before the vocoder.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter
from phoonnx.providers import make_session


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


class ZipVoiceAdapter(BaseOnnxAdapter):
    """Adapter for ZipVoice (flow-matching, in-context cloning)."""

    FEAT_SCALE = 0.1
    TARGET_RMS = 0.1
    SAMPLE_RATE = 24000

    def __init__(self, num_step: int = 16, guidance_scale: float = 1.0):
        self.text_encoder: Optional[onnxruntime.InferenceSession] = None
        self.mel: Optional[onnxruntime.InferenceSession] = None
        self.vocoder = None
        self.num_step = int(num_step)
        self.guidance_scale = float(guidance_scale)

    def default_params(self) -> Dict[str, float]:
        return {"speed": 1.0, "num_step": float(self.num_step), "guidance_scale": self.guidance_scale}

    def configure(self, voice_config: Any) -> None:
        """Load the auxiliary graphs (mel front end, text encoder) and the Vocos
        vocoder from ``engine_params``; the fm_decoder is the voice's own session."""
        ep = getattr(voice_config, "engine_params", None) or {}
        sess = lambda p: make_session(p, providers=ep.get("providers"))
        if self.text_encoder is None and ep.get("text_encoder_path"):
            self.text_encoder = sess(ep["text_encoder_path"])
        if self.mel is None and ep.get("mel_path"):
            self.mel = sess(ep["mel_path"])
        if self.vocoder is None and ep.get("vocoder_path"):
            from phoonnx.engines.vocoders import build_vocoder
            self.vocoder = build_vocoder(ep["vocoder_path"], ep.get("vocoder_type", "vocos"), ep)
        if "num_step" in ep:
            self.num_step = int(ep["num_step"])
        if "guidance_scale" in ep:
            self.guidance_scale = float(ep["guidance_scale"])

    def _prompt_mel(self, audio: np.ndarray, sr: int):
        """ref wav -> 24kHz -> rms-norm to target_rms -> log-mel * feat_scale ([1,T,100])."""
        audio = _resample(np.asarray(audio, np.float32).reshape(-1), int(sr), self.SAMPLE_RATE)
        rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-9)
        audio = audio * self.TARGET_RMS / rms
        mel = self.mel.run(None, {"waveform": audio[None, :].astype(np.float32)})[0]   # [1,100,T]
        return mel.transpose(0, 2, 1) * self.FEAT_SCALE, rms                            # [1,T,100]

    def synthesize(self, request: AdapterSynthesisRequest,
                   session: onnxruntime.InferenceSession) -> AdapterSynthesisResult:
        if self.text_encoder is None or self.mel is None or self.vocoder is None:
            raise RuntimeError("ZipVoice voice missing text_encoder_path / mel_path / vocoder_path")
        p = request.params
        ref = p.get("reference_audio")
        prompt_tokens = p.get("prompt_tokens")
        if ref is None or prompt_tokens is None:
            raise RuntimeError("ZipVoice needs a reference clip + transcription "
                               "(params 'reference_audio' and 'prompt_tokens')")

        tokens = np.asarray(request.phoneme_ids, np.int64).reshape(1, -1)
        prompt_tokens = np.asarray(prompt_tokens, np.int64).reshape(1, -1)
        prompt_mel, prompt_rms = self._prompt_mel(*ref)
        t_ref = prompt_mel.shape[1]

        text_condition = self.text_encoder.run(None, {
            "tokens": tokens, "prompt_tokens": prompt_tokens,
            "prompt_features_len": np.array(t_ref, np.int64),
            "speed": np.array(p.get("speed", 1.0), np.float32)})[0]
        num_frames = text_condition.shape[1]

        # flow matching: x_0 ~ N(0,1) integrated to x_1, conditioned on the prompt mel
        x = np.random.default_rng(0).standard_normal((1, num_frames, 100)).astype(np.float32)
        speech_condition = np.zeros((1, num_frames, 100), np.float32)
        speech_condition[:, :t_ref] = prompt_mel
        num_step = int(p.get("num_step", self.num_step))
        guidance = np.array(p.get("guidance_scale", self.guidance_scale), np.float32)
        # upstream default t_shift=0.5: warp the time grid so more ODE steps
        # land early in the trajectory (t = s*t / (1 + (s-1)*t))
        t_shift = float(p.get("t_shift", 0.5))
        steps = np.linspace(0, 1, num_step + 1).astype(np.float32)
        steps = (t_shift * steps / (1 + (t_shift - 1) * steps)).astype(np.float32)
        for i in range(num_step):
            v = session.run(None, {
                "t": np.array(steps[i], np.float32), "x": x,
                "text_condition": text_condition, "speech_condition": speech_condition,
                "guidance_scale": guidance})[0]
            x = x + v * (steps[i + 1] - steps[i])

        target_mel = (x[:, t_ref:] / self.FEAT_SCALE).transpose(0, 2, 1).astype(np.float32)   # [1,100,T]
        audio = np.asarray(self.vocoder.mel_to_audio(target_mel), np.float32).reshape(-1)
        # undo the prompt RMS normalization only when the prompt was quieter
        # than target_rms (as upstream): rescaling up from a loud prompt clips
        if prompt_rms < self.TARGET_RMS:
            audio = audio * prompt_rms / self.TARGET_RMS
        return AdapterSynthesisResult(audio=audio)

    # build_feed_dict / parse_outputs are required by the ABC but unused — synthesize()
    # drives the multi-ONNX loop directly.
    def build_feed_dict(self, request: AdapterSynthesisRequest,
                        session: onnxruntime.InferenceSession) -> Dict[str, np.ndarray]:
        raise NotImplementedError("ZipVoice is iterative — use synthesize()")

    def parse_outputs(self, outputs: List[np.ndarray],
                      request: AdapterSynthesisRequest) -> AdapterSynthesisResult:
        raise NotImplementedError("ZipVoice is iterative — use synthesize()")

    @staticmethod
    def detect(config: Optional[Dict[str, Any]] = None,
               session: Optional[onnxruntime.InferenceSession] = None) -> bool:
        return bool(config and config.get("engine") == "zipvoice")
