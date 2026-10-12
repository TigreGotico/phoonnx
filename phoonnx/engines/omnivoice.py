"""OmniVoice inference adapter — masked-diffusion LM, zero-shot cloning, 600+ languages.

OmniVoice (k2-fsa, ``k2-fsa/OmniVoice``) is a Qwen3-0.6B backbone that writes eight
streams of Higgs Audio V2 codec tokens. It is neither autoregressive like
:class:`phoonnx.engines.chatterbox.ChatterboxAdapter` nor a flow-matching ODE like
:class:`phoonnx.engines.zipvoice.ZipVoiceAdapter`: it is a **discrete diffusion** (masked)
language model. Every audio position starts as a MASK token (id 1024 of a 1025-entry
vocabulary) and a fixed number of steps progressively replace the most confident MASK
slots with real codes. Attention is **bidirectional** — a position must see the whole
sequence, past and future — and the model runs a full-sequence forward every step, so
there is no KV cache to thread.

Because it never emits an end-of-speech token, the output length is decided **before**
decoding from a rule-based estimate over the character weights of the text (see
:class:`phoonnx.thirdparty.omnivoice.RuleDurationEstimator`); ``length_scale`` rescales it.

Two graph families (HF: ``OpenVoiceOS/phoonnx-omnivoice``)::

    omnivoice_backbone.onnx    (input_ids[B,8,S], audio_mask[B,S]) -> logits[B,8,S,1025]
    acoustic_encoder.onnx      reference wav @24 kHz -> acoustic features   (cloning only)
    semantic_encoder.onnx      reference wav @16 kHz -> semantic features   (cloning only)
    quantizer_encoder.onnx     acoustic + semantic  -> reference codes      (cloning only)
    higgs_decoder.onnx         codes[8,1,T] -> waveform @24 kHz

plus ``tokenizer.json`` (the model's own Qwen3 subword BPE — OmniVoice reads raw text,
so no phonemizer is involved).

Prompt layout follows upstream ``_prepare_inference_inputs`` exactly::

    <|denoise|>                                   (cloning only)
    <|lang_start|>{lang or None}<|lang_end|>
    <|instruct_start|>{instruct or None}<|instruct_end|>
    <|text_start|>{ref_text + " " + text}<|text_end|>
    {reference codec codes}                       (cloning only)
    {MASK x target_len}                           <- what gets decoded

Classifier-free guidance runs the backbone twice per step: once over that full prompt and
once over the target span alone. Upstream batches the two rows behind a ``[2B,1,S,S]``
block mask; for one item that block mask says exactly "each row attends within its own
real length", which is what two separate forwards of different lengths compute.

.. warning::
   The community export ``onnx-community/OmniVoice-Onnx`` routes the backbone through
   ``com.microsoft::GroupQueryAttention``, which is unconditionally **causal**. Its
   hidden states match a causal PyTorch run (cos 0.9995) and not the bidirectional one
   the model needs (cos 0.954), which leaves only **18 %** greedy-token agreement with
   upstream. ``OpenVoiceOS/phoonnx-omnivoice`` therefore ships a fresh bidirectional
   export (cos 1.0000000, 100 % agreement). Its Higgs codec graphs *are* exact and are
   mirrored unchanged. Do not point this adapter at the community backbone.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime

from phoonnx.engines.base import (AdapterSynthesisRequest, AdapterSynthesisResult,
                                  BaseOnnxAdapter)
from phoonnx.providers import make_session
from phoonnx.thirdparty.omnivoice import (LANG_IDS, LANG_NAME_TO_ID, RuleDurationEstimator,
                                          add_punctuation, combine_text, fade_and_pad_audio,
                                          remove_silence, resample,
                                          tokenize_with_nonverbal_tags)
from phoonnx.util import LOG

SAMPLE_RATE = 24000
"""The Higgs Audio V2 codec decodes at 24 kHz."""

SEMANTIC_SAMPLE_RATE = 16000
"""Its semantic (HuBERT) branch wants 16 kHz."""

HOP_LENGTH = 960
"""24 000 / 960 = 25 codec frames per second."""

FRAME_RATE = SAMPLE_RATE // HOP_LENGTH

NUM_CODEBOOKS = 8
AUDIO_VOCAB_SIZE = 1025
AUDIO_MASK_ID = 1024
"""Codebook entries are 0..1023; 1024 is the MASK token the diffusion loop removes."""

TARGET_RMS = 0.1
"""Reference clips are normalised to this RMS before tokenization, and the generated
audio is scaled back by the clip's original RMS so a quiet prompt stays quiet."""

MAX_REF_SECONDS = 20.0


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=axis, keepdims=True))


def _gumbel(logits: np.ndarray, temperature: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gumbel noise to scaled logits — argmax over the result samples the softmax."""
    u = rng.random(logits.shape)
    return logits / temperature + (-np.log(-np.log(u + 1e-10) + 1e-10))


def _filter_top_k(log_probs: np.ndarray, ratio: float = 0.1) -> np.ndarray:
    """Keep the top ``ratio`` of the vocabulary, mask the rest to -inf."""
    k = math.ceil(ratio * log_probs.shape[-1])
    idx = np.argpartition(-log_probs, k - 1, axis=-1)[..., :k]
    out = np.full_like(log_probs, -np.inf)
    np.put_along_axis(out, idx, np.take_along_axis(log_probs, idx, -1), -1)
    return out


class OmniVoiceAdapter(BaseOnnxAdapter):
    """Adapter for OmniVoice (masked-diffusion codec LM, in-context cloning)."""

    MEMOIZED_WRITES = {
        # The cloning graphs, several hundred megabytes each, opened on the
        # first request that needs them and kept for the ones after.
        # (the attribute is the caller's; the call sites name the acoustic,
        # semantic and quantizer encoders)
        "_cloning_session": frozenset({"*attribute"}),
        # A one-entry cache keyed on the request's own clip: a request whose
        # clip differs recomputes rather than reading the previous caller's
        # codes, and one that repeats it skips the encoders.
        "encode_reference": frozenset({"_ref_cache", "_ref_cache_key"}),
    }

    NUM_STEP = 32
    GUIDANCE_SCALE = 2.0
    T_SHIFT = 0.1
    LAYER_PENALTY_FACTOR = 5.0
    POSITION_TEMPERATURE = 5.0
    CLASS_TEMPERATURE = 0.0

    def __init__(self):
        self.acoustic_encoder: Optional[onnxruntime.InferenceSession] = None
        self.semantic_encoder: Optional[onnxruntime.InferenceSession] = None
        self.quantizer_encoder: Optional[onnxruntime.InferenceSession] = None
        self.decoder: Optional[onnxruntime.InferenceSession] = None
        self.tokenizer = None
        self._params: Dict[str, Any] = {}
        self._estimator = RuleDurationEstimator()
        self._ref_cache_key: Optional[tuple] = None
        self._ref_cache: Tuple[Optional[np.ndarray], Optional[float]] = (None, None)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def default_params(self) -> Dict[str, float]:
        return {
            "num_step": float(self.NUM_STEP),
            "guidance_scale": self.GUIDANCE_SCALE,
            "t_shift": self.T_SHIFT,
            "layer_penalty_factor": self.LAYER_PENALTY_FACTOR,
            "position_temperature": self.POSITION_TEMPERATURE,
            "class_temperature": self.CLASS_TEMPERATURE,
            "length_scale": 1.0,
        }

    def param_labels(self) -> Dict[str, str]:
        return {
            "num_step": "Unmasking steps",
            "guidance_scale": "Classifier-free guidance",
            "t_shift": "Time-step shift",
            "layer_penalty_factor": "Codebook-order penalty",
            "position_temperature": "Position-choice temperature",
            "class_temperature": "Token-sampling temperature (0 = greedy)",
            "length_scale": "Speech duration (>1 slower)",
        }

    def configure(self, voice_config: Any) -> None:
        """Load the codec graphs and the model's own BPE from ``engine_params``.

        The three encoder graphs are only needed to clone, and the semantic encoder alone
        is several hundred megabytes, so they open on first use rather than at load.
        """
        ep = getattr(voice_config, "engine_params", None) or {}
        self._params = dict(ep)
        if self.decoder is None and ep.get("decoder_path"):
            self.decoder = make_session(ep["decoder_path"], providers=ep.get("providers"))
        if self.tokenizer is None and ep.get("bpe_tokenizer_path"):
            from phoonnx.tokenizer import BPETokenizer
            self.tokenizer = BPETokenizer(ep["bpe_tokenizer_path"])

    def _cloning_session(self, attribute: str, key: str) -> onnxruntime.InferenceSession:
        session = getattr(self, attribute)
        if session is None:
            path = self._params.get(key)
            if not path:
                raise RuntimeError(f"OmniVoice voice cannot clone: no {key} in engine_params")
            session = make_session(path, providers=self._params.get("providers"))
            setattr(self, attribute, session)
        return session

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.tokenize(text)

    def encode_text(self, text: str, voice: Any, syn_config: Any) -> List[List[int]]:
        """Tokenize raw text with the model's own Qwen3 BPE.

        OmniVoice reads text, not phonemes, so nothing goes through the shared phonemizer
        — its ``prompt_tokens`` are phoneme ids and mean nothing here. A cloning
        reference's transcription is kept as a *string*, because upstream joins it to the
        target text before tokenizing (:func:`combine_text`) rather than concatenating two
        token sequences. It travels to :meth:`synthesize` on ``request.params`` rather than
        on ``self`` — this adapter instance is shared across concurrent requests (one
        ``TTSVoice``/adapter per voice_id under the threaded server), and stashing it here
        would let one request's reference text bleed into another's audio.
        """
        if self.tokenizer is None:
            raise RuntimeError("OmniVoice voice missing bpe_tokenizer_path in engine_params")
        return [self._encode(text)] if text.strip() else []

    def _decode_ids(self, ids: np.ndarray) -> str:
        """Recover the chunk's text from the ids the voice layer handed back."""
        return self.tokenizer.decode([int(i) for i in np.asarray(ids).reshape(-1)])

    # ------------------------------------------------------------------
    # Reference clip
    # ------------------------------------------------------------------

    def _preprocess_reference(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
        """Upstream ``create_voice_clone_prompt``: resample, RMS-normalise, trim."""
        wav = np.asarray(audio, np.float32).reshape(1, -1)
        if sr != SAMPLE_RATE:
            wav = resample(wav.reshape(-1), sr, SAMPLE_RATE).reshape(1, -1)
        rms = float(np.sqrt(np.mean(wav ** 2)))
        if 0 < rms < TARGET_RMS:
            wav = wav * TARGET_RMS / rms

        trimmed = remove_silence(wav, SAMPLE_RATE, mid_sil=200, lead_sil=100, trail_sil=200)
        if trimmed.shape[-1] == 0:
            # Upstream raises here and tells the caller to disable preprocessing. A voice
            # assistant cannot act on that, so keep the untrimmed clip: it still clones.
            LOG.warning("OmniVoice: reference clip is all silence by the -50 dBFS gate; "
                        "using it untrimmed")
        else:
            wav = trimmed

        duration = wav.shape[-1] / SAMPLE_RATE
        if duration > MAX_REF_SECONDS:
            LOG.warning("OmniVoice: reference clip is %.1fs (>%.0fs); cloning is slower "
                        "and usually worse — 3-10s works best", duration, MAX_REF_SECONDS)

        clip = wav.shape[-1] % HOP_LENGTH
        if clip:
            wav = wav[:, :-clip]
        return wav.reshape(-1), rms

    def encode_reference(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
        """Reference clip -> ``(codes[8, T], original_rms)``, cached across chunks."""
        key = (hash(np.asarray(audio, np.float32).tobytes()), int(sr))
        if key == self._ref_cache_key:
            return self._ref_cache
        wav, rms = self._preprocess_reference(audio, sr)
        if wav.size == 0:
            raise RuntimeError("OmniVoice reference clip is empty")
        wav16 = resample(wav, SAMPLE_RATE, SEMANTIC_SAMPLE_RATE)

        acoustic = self._cloning_session("acoustic_encoder", "acoustic_encoder_path").run(
            ["acoustic_features"], {"waveform_24k": wav[None, None, :].astype(np.float32)})[0]
        semantic = self._cloning_session("semantic_encoder", "semantic_encoder_path").run(
            ["semantic_features"], {"waveform_16k": wav16[None, :].astype(np.float32)})[0]
        # the two branches can land one frame apart on a length that is not a clean
        # multiple of the hop; upstream trims to the shorter one
        frames = min(acoustic.shape[2], semantic.shape[2])
        codes = self._cloning_session("quantizer_encoder", "quantizer_encoder_path").run(
            ["codes"], {"acoustic_features": acoustic[:, :, :frames],
                        "semantic_features": semantic[:, :, :frames]})[0]
        result = (np.asarray(codes).reshape(NUM_CODEBOOKS, -1).astype(np.int64), rms)
        self._ref_cache_key, self._ref_cache = key, result
        return result

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_language(lang: Optional[str]) -> Optional[str]:
        """Map a phoonnx language tag or an English language name onto an OmniVoice ID.

        ``pt-br`` and ``pt`` both resolve to ``pt``; an unknown tag returns ``None``,
        which puts the model in its language-agnostic mode rather than failing.
        """
        if not lang or str(lang).lower() == "none":
            return None
        lang = str(lang).strip()
        for candidate in (lang, lang.replace("_", "-").split("-")[0]):
            if candidate in LANG_IDS:
                return candidate
            if candidate.lower() in LANG_NAME_TO_ID:
                return LANG_NAME_TO_ID[candidate.lower()]
        LOG.warning("OmniVoice does not know language %r — generating language-agnostic", lang)
        return None

    def build_prompt(self, text: str, target_len: int, ref_text: Optional[str] = None,
                     ref_codes: Optional[np.ndarray] = None, lang: Optional[str] = None,
                     instruct: Optional[str] = None,
                     denoise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Lay out ``(input_ids[1, 8, S], audio_mask[1, S])`` as upstream does.

        Every one of the eight rows carries the same text ids; ``audio_mask`` marks where
        the rows stop being a copy of the text and start being codec streams, which is
        what tells the embedding layer which table to look each position up in.
        """
        style = "<|denoise|>" if (denoise and ref_codes is not None) else ""
        style += f"<|lang_start|>{lang or 'None'}<|lang_end|>"
        style += f"<|instruct_start|>{instruct or 'None'}<|instruct_end|>"
        style_ids = self._encode(style)

        full_text = combine_text(text, ref_text)
        text_ids = tokenize_with_nonverbal_tags(
            f"<|text_start|>{full_text}<|text_end|>", self._encode)

        parts = [np.tile(np.asarray(style_ids, np.int64), (NUM_CODEBOOKS, 1)),
                 np.tile(np.asarray(text_ids, np.int64), (NUM_CODEBOOKS, 1))]
        if ref_codes is not None:
            parts.append(np.asarray(ref_codes, np.int64))
        parts.append(np.full((NUM_CODEBOOKS, target_len), AUDIO_MASK_ID, np.int64))

        input_ids = np.concatenate(parts, axis=1)[None]
        total = input_ids.shape[2]
        audio_start = total - target_len - (ref_codes.shape[1] if ref_codes is not None else 0)
        audio_mask = np.zeros((1, total), bool)
        audio_mask[0, audio_start:] = True
        return input_ids, audio_mask

    def estimate_target_len(self, text: str, ref_text: Optional[str],
                            num_ref_frames: Optional[int], length_scale: float = 1.0) -> int:
        """How many codec frames to generate. Upstream falls back to a fixed
        "Nice to meet you." / 25-frame pair when there is no reference to calibrate on."""
        if not num_ref_frames or not ref_text:
            ref_text, num_ref_frames = "Nice to meet you.", 25
        est = self._estimator.estimate_duration(text, ref_text, num_ref_frames)
        if length_scale and length_scale != 1.0:
            est = est * float(length_scale)
        return max(1, int(est))

    # ------------------------------------------------------------------
    # Sampler
    # ------------------------------------------------------------------

    @staticmethod
    def unmask_schedule(target_len: int, num_step: int, t_shift: float) -> List[int]:
        """How many of the ``8 x target_len`` slots to fill at each step.

        The time grid is warped by ``t_shift`` (upstream default 0.1), which front-loads
        the schedule so most slots are still masked late in the run, and the last step
        always takes whatever is left — a slot left at MASK would reach the codec decoder
        as id 1024, outside its 0..1023 range.
        """
        steps = np.linspace(0.0, 1.0, num_step + 1)
        steps = t_shift * steps / (1 + (t_shift - 1) * steps)
        total = int(target_len) * NUM_CODEBOOKS
        remaining, schedule = total, []
        for step in range(num_step):
            take = (remaining if step == num_step - 1
                    else min(int(math.ceil(total * (steps[step + 1] - steps[step]))), remaining))
            schedule.append(int(take))
            remaining -= int(take)
        return schedule

    def generate_codes(self, session: onnxruntime.InferenceSession, text: str, target_len: int,
                       ref_text: Optional[str] = None, ref_codes: Optional[np.ndarray] = None,
                       lang: Optional[str] = None, instruct: Optional[str] = None,
                       denoise: bool = True, num_step: int = 32, guidance_scale: float = 2.0,
                       t_shift: float = 0.1, layer_penalty_factor: float = 5.0,
                       position_temperature: float = 5.0, class_temperature: float = 0.0,
                       seed: Optional[int] = None) -> np.ndarray:
        """Run the masked-diffusion loop and return ``codes[8, target_len]``."""
        rng = np.random.default_rng(seed)
        target_len = int(target_len)
        cond_ids, cond_mask = self.build_prompt(text, target_len, ref_text, ref_codes,
                                                lang, instruct, denoise)
        cond_len = cond_ids.shape[2]
        # the unconditional row is the target span with no text, language or reference
        uncond_ids = cond_ids[:, :, -target_len:].copy()
        uncond_mask = cond_mask[:, -target_len:].copy()

        schedule = self.unmask_schedule(target_len, num_step, t_shift)
        tokens = np.full((NUM_CODEBOOKS, target_len), AUDIO_MASK_ID, np.int64)
        layer_ids = np.arange(NUM_CODEBOOKS, dtype=np.float32).reshape(-1, 1)

        for step, take in enumerate(schedule):
            if take <= 0:
                continue
            cond_logits = session.run(None, {"input_ids": cond_ids, "audio_mask": cond_mask}
                                      )[0][0, :, cond_len - target_len:cond_len, :]
            cond_logits = cond_logits.astype(np.float32)

            if guidance_scale != 0:
                uncond_logits = session.run(
                    None, {"input_ids": uncond_ids, "audio_mask": uncond_mask}
                )[0][0, :, :target_len, :].astype(np.float32)
                cond_lp, uncond_lp = _log_softmax(cond_logits), _log_softmax(uncond_logits)
                log_probs = _log_softmax(cond_lp + guidance_scale * (cond_lp - uncond_lp))
            else:
                log_probs = _log_softmax(cond_logits)
            log_probs[..., AUDIO_MASK_ID] = -np.inf   # never re-predict MASK

            if class_temperature > 0.0:
                predicted = _gumbel(_filter_top_k(log_probs), class_temperature, rng).argmax(-1)
            else:
                predicted = log_probs.argmax(-1)

            # confidence, biased towards the low codebooks (they carry more of the signal)
            scores = log_probs.max(-1) - layer_ids * layer_penalty_factor
            if position_temperature > 0.0:
                scores = _gumbel(scores, position_temperature, rng)
            scores = np.where(tokens != AUDIO_MASK_ID, -np.inf, scores)

            flat_scores = scores.ravel()
            chosen = (np.argpartition(-flat_scores, take - 1)[:take]
                      if take < flat_scores.size else np.arange(flat_scores.size))
            flat_tokens = tokens.ravel().copy()
            flat_tokens[chosen] = predicted.ravel()[chosen]
            tokens = flat_tokens.reshape(NUM_CODEBOOKS, target_len)

            cond_ids[0, :, cond_len - target_len:cond_len] = tokens
            uncond_ids[0, :, :target_len] = tokens

        return tokens

    def decode_codes(self, codes: np.ndarray) -> np.ndarray:
        if self.decoder is None:
            raise RuntimeError("OmniVoice voice missing decoder_path in engine_params")
        codes = np.asarray(codes, np.int64).reshape(NUM_CODEBOOKS, -1)
        return np.asarray(self.decoder.run(
            ["waveform_24k"], {"codes": codes[:, None, :]})[0]).reshape(-1)

    def post_process(self, audio: np.ndarray, ref_rms: Optional[float],
                     pad_duration: float = 0.1, fade_duration: float = 0.1) -> np.ndarray:
        """Trim long silences, restore the reference's loudness, fade and pad the edges."""
        wav = np.asarray(audio, np.float32).reshape(1, -1)
        wav = remove_silence(wav, SAMPLE_RATE, mid_sil=500, lead_sil=100, trail_sil=100)
        if wav.shape[-1] == 0:
            wav = np.asarray(audio, np.float32).reshape(1, -1)
        if ref_rms is not None and ref_rms < TARGET_RMS:
            wav = wav * ref_rms / TARGET_RMS
        elif ref_rms is None:
            peak = float(np.abs(wav).max())
            if peak > 1e-6:
                wav = wav / peak * 0.5
        wav = fade_and_pad_audio(wav, pad_duration, fade_duration, SAMPLE_RATE)
        return wav.reshape(-1).astype(np.float32)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(self, request: AdapterSynthesisRequest,
                   session: onnxruntime.InferenceSession) -> AdapterSynthesisResult:
        if self.tokenizer is None:
            raise RuntimeError("OmniVoice voice missing bpe_tokenizer_path in engine_params")
        params = request.params
        text = self._decode_ids(request.phoneme_ids)

        reference_text = params.get("speaker_reference_text")
        reference_text = add_punctuation(reference_text) if reference_text else None

        ref_codes = ref_rms = None
        reference = params.get("reference_audio")
        if reference is not None:
            ref_codes, ref_rms = self.encode_reference(reference[0], int(reference[1]))
        ref_text = reference_text if ref_codes is not None else None
        if ref_codes is None and reference_text:
            LOG.warning("OmniVoice: a reference transcription was given without a "
                        "reference clip — ignoring it")

        lang = self.resolve_language(params.get("lang") or params.get("lang_code"))
        target_len = self.estimate_target_len(
            text, ref_text, ref_codes.shape[1] if ref_codes is not None else None,
            float(params.get("length_scale") or 1.0))

        codes = self.generate_codes(
            session, text, target_len, ref_text=ref_text, ref_codes=ref_codes, lang=lang,
            instruct=params.get("instruct"), denoise=bool(params.get("denoise", True)),
            num_step=int(params.get("num_step", self.NUM_STEP)),
            guidance_scale=float(params.get("guidance_scale", self.GUIDANCE_SCALE)),
            t_shift=float(params.get("t_shift", self.T_SHIFT)),
            layer_penalty_factor=float(params.get("layer_penalty_factor",
                                                  self.LAYER_PENALTY_FACTOR)),
            position_temperature=float(params.get("position_temperature",
                                                  self.POSITION_TEMPERATURE)),
            class_temperature=float(params.get("class_temperature", self.CLASS_TEMPERATURE)),
            seed=params.get("seed"))

        audio = self.decode_codes(codes)
        audio = self.post_process(audio, ref_rms)
        return AdapterSynthesisResult(audio=audio, extras={"audio_codes": codes,
                                                           "target_frames": target_len})

    # build_feed_dict / parse_outputs are required by the ABC but unused — synthesize()
    # drives the multi-ONNX diffusion loop directly.
    def build_feed_dict(self, request: AdapterSynthesisRequest,
                        session: onnxruntime.InferenceSession) -> Dict[str, np.ndarray]:
        raise NotImplementedError("OmniVoice is iterative — use synthesize()")

    def parse_outputs(self, outputs: List[np.ndarray], request: AdapterSynthesisRequest,
                      output_names: Optional[List[str]] = None) -> AdapterSynthesisResult:
        raise NotImplementedError("OmniVoice is iterative — use synthesize()")

    @staticmethod
    def detect(config: Optional[Dict[str, Any]] = None,
               session: Optional[onnxruntime.InferenceSession] = None) -> bool:
        return bool(config and config.get("engine") == "omnivoice")
