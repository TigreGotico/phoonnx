"""Spark-TTS inference adapter — autoregressive codec-LM TTS with BiCodec tokens.

Spark-TTS (SparkAudio, ``SparkAudio/Spark-TTS-0.5B``) is a decoder-only language model
on a Qwen2.5-0.5B backbone. It does not predict audio directly: it predicts *BiCodec*
tokens, and BiCodec turns those tokens back into a waveform. BiCodec splits a voice into
two streams:

* **global tokens** — 32 tokens that carry the speaker (timbre, not content);
* **semantic tokens** — one stream at 50 Hz that carries what is said.

The language model reads a prompt made of control tokens, the text, and the 32 global
tokens, and then writes the semantic stream. This is the same autoregressive family as
:class:`phoonnx.engines.chatterbox.ChatterboxAdapter`, so it also drives the overridable
``BaseOnnxAdapter.synthesize`` rather than the single-graph ``build_feed_dict`` path.

Five ONNX graphs (HF: ``OpenVoiceOS/phoonnx-spark-tts``)::

    model.onnx                      Qwen2 LM, KV-cached AR step (= the voice's ``session``)
    bicodec_vocoder.onnx            semantic + global tokens -> waveform @ 16 kHz
    wav2vec2_model.onnx             reference wav -> 1024-dim features        (cloning only)
    bicodec_encoder_quantizer.onnx  features -> reference semantic tokens     (cloning only)
    speaker_encoder_tokenizer.onnx  reference magnitude spectrogram -> global tokens (cloning only)

plus ``tokenizer.json`` (the model's own subword BPE) and, for a preset voice, a small
JSON that holds the 32 global tokens of that speaker.

Voices come in two forms:

* **preset** — the 32 global tokens ship with the voice as an asset, so the speaker is
  fixed and no reference clip is needed;
* **zero-shot clone** — ``SynthesisConfig.speaker_reference`` gives a reference clip, and
  the three cloning graphs turn it into global tokens. Add
  ``speaker_reference_text`` (the transcription of that clip) for the in-context variant,
  which also feeds the clip's semantic tokens to the model and clones more closely.

The short-time Fourier transform in front of the speaker encoder stays outside the ONNX
graphs: ONNX has no complex dtype, so neither torch exporter can lower ``torch.stft``.
The mel filterbank projection *is* inside ``speaker_encoder_tokenizer.onnx``; only the
magnitude spectrogram is computed here, and it matches torchaudio to 5e-7.

Prompt layout, sampler order and the audio front end follow SparkAudio's own inference
code (Apache-2.0, ``SparkAudio/Spark-TTS``): the model is generated with HuggingFace
``generate``, whose warpers apply temperature, then top-k, then top-p, with no
repetition penalty.
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

SAMPLE_RATE = 16000
"""Every BiCodec stage works at 16 kHz."""

REF_SEGMENT_SAMPLES = 96000
"""6 s of reference audio, rounded down to a multiple of the 320-sample latent hop."""

N_FFT, WIN_LENGTH, HOP_LENGTH = 1024, 640, 320

MAX_NEW_TOKENS = 3000
"""Upstream's generation budget — about 60 s of audio at 50 tokens per second."""

MAX_CHUNK_CHARS = 200
"""Longest text handed to one autoregressive pass; longer input is split on sentences."""

_SPECIAL_TOKENS = (
    "<|task_tts|>", "<|start_content|>", "<|end_content|>",
    "<|start_global_token|>", "<|end_global_token|>", "<|start_semantic_token|>",
    "<|im_end|>", "<|bicodec_global_0|>", "<|bicodec_semantic_0|>",
)

N_GLOBAL_TOKENS = 32
"""The speaker stream is always exactly 32 tokens."""

GLOBAL_CODEBOOK, SEMANTIC_CODEBOOK = 4096, 8192


def volume_normalize(audio: np.ndarray, coeff: float = 0.2) -> np.ndarray:
    """Bring a clip to the loudness the BiCodec front end expects.

    Ported from SparkAudio's ``audio_volume_normalize`` (Apache-2.0). The scale comes
    from the mean of the loudest 10 % to 1 % of samples, which ignores both silence and
    isolated peaks.
    """
    audio = np.asarray(audio, np.float32).reshape(-1)
    temp = np.sort(np.abs(audio))
    if temp.size == 0:
        return audio
    if temp[-1] < 0.1:
        audio = audio / max(float(temp[-1]), 1e-3) * 0.1
    temp = temp[temp > 0.01]
    if temp.shape[0] <= 10:
        return audio
    volume = float(np.mean(temp[int(0.9 * temp.shape[0]):int(0.99 * temp.shape[0])]))
    audio = audio * np.clip(coeff / volume, 0.1, 10)
    peak = float(np.max(np.abs(audio)))
    if peak > 1:
        audio = audio / peak
    return audio.astype(np.float32)


def magnitude_spectrogram(wav: np.ndarray) -> np.ndarray:
    """Magnitude STFT of ``wav``, shaped ``(1, 1 + n_fft // 2, frames)``.

    Matches torchaudio ``Spectrogram(n_fft=1024, win_length=640, hop_length=320,
    power=1, center=True, pad_mode="reflect")``, whose output the exported
    ``speaker_encoder_tokenizer.onnx`` projects through the mel filterbank.
    """
    win = np.hanning(WIN_LENGTH + 1)[:-1].astype(np.float32)     # periodic Hann
    pad = (N_FFT - WIN_LENGTH) // 2
    win = np.pad(win, (pad, N_FFT - WIN_LENGTH - pad))
    x = np.pad(np.asarray(wav, np.float32).reshape(-1),
               (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = 1 + (len(x) - N_FFT) // HOP_LENGTH
    strided = np.lib.stride_tricks.as_strided(
        x, shape=(frames, N_FFT), strides=(x.strides[0] * HOP_LENGTH, x.strides[0]))
    spec = np.abs(np.fft.rfft(strided * win, n=N_FFT, axis=-1)).T
    return spec[None].astype(np.float32)


def resample(audio: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample a mono clip to ``target_sr``, preferring a polyphase filter."""
    audio = np.asarray(audio, np.float32).reshape(-1)
    if sr == target_sr:
        return audio
    try:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(sr, target_sr)
        return resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
    except ImportError:
        n = int(round(len(audio) * target_sr / sr))
        return np.interp(np.linspace(0, len(audio) - 1, n),
                         np.arange(len(audio)), audio).astype(np.float32)


def reference_clip(wav: np.ndarray) -> np.ndarray:
    """The fixed-length slice the speaker encoder reads, repeating a short clip."""
    wav = np.asarray(wav, np.float32).reshape(-1)
    if wav.size == 0:
        raise ValueError("Spark-TTS reference clip is empty")
    if wav.size < REF_SEGMENT_SAMPLES:
        wav = np.tile(wav, REF_SEGMENT_SAMPLES // wav.size + 1)
    return wav[:REF_SEGMENT_SAMPLES]


def sample_token(logits: np.ndarray, temperature: float, top_k: int,
                 top_p: float, rng: np.random.Generator) -> int:
    """Draw one token the way HuggingFace ``generate`` does for this model.

    The warpers run in the order HuggingFace builds them — temperature, then top-k, then
    top-p — and Spark-TTS sets no repetition penalty. Order matters: applying top-p
    before top-k gives a different nucleus, and a repetition penalty (which the sibling
    Chatterbox engine does need) would suppress the repeated codec tokens that normal
    speech contains.
    """
    x = np.asarray(logits, np.float64).reshape(-1)
    if temperature > 0:
        x = x / max(temperature, 1e-5)
    else:
        return int(x.argmax())
    if top_k and 0 < top_k < x.size:
        x = np.where(x < np.partition(x, -top_k)[-top_k], -np.inf, x)
    x = x - x.max()
    probs = np.exp(x)
    probs /= probs.sum()
    order = np.argsort(probs)[::-1]
    cutoff = int(np.searchsorted(np.cumsum(probs[order]), top_p)) + 1
    keep = order[:max(1, cutoff)]
    p = probs[keep] / probs[keep].sum()
    return int(rng.choice(keep, p=p))


def chunk_text(text: str, max_len: int = MAX_CHUNK_CHARS) -> List[str]:
    """Pack whole sentences into chunks of at most ``max_len`` characters.

    One chunk is one autoregressive pass. Sentence boundaries come from
    ``quebra_frases.sentence_tokenize``, the splitter the rest of phoonnx uses.
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


class SparkTTSAdapter(BaseOnnxAdapter):
    """Adapter for Spark-TTS (Qwen2 codec-LM + BiCodec, preset or cloned speakers)."""

    def __init__(self):
        self.vocoder: Optional[onnxruntime.InferenceSession] = None
        self.wav2vec2: Optional[onnxruntime.InferenceSession] = None
        self.speaker_tokenizer: Optional[onnxruntime.InferenceSession] = None
        self.semantic_tokenizer: Optional[onnxruntime.InferenceSession] = None
        self.tokenizer: Optional[BPETokenizer] = None
        self.preset_global_tokens: Optional[np.ndarray] = None
        self.special: Dict[str, int] = {}
        self._params: Dict[str, Any] = {}
        self._reference_text_ids: Optional[List[int]] = None
        self.past_names: List[str] = []
        self.num_kv_heads = 0
        self.head_dim = 0

    def default_params(self) -> Dict[str, float]:
        return {"temperature": 0.8, "top_k": 50.0, "top_p": 0.95}

    def param_labels(self) -> Dict[str, str]:
        return {"temperature": "Sampling temperature", "top_k": "Top-k",
                "top_p": "Nucleus (top-p)"}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def configure(self, voice_config: Any) -> None:
        """Load the vocoder, the model's own BPE and the preset speaker tokens.

        The three cloning graphs stay on disk until a call actually asks to clone. A
        preset voice never touches them, and the wav2vec2 front end alone costs about a
        gigabyte of memory to hold open.
        """
        ep = getattr(voice_config, "engine_params", None) or {}
        self._params = dict(ep)

        if self.vocoder is None and ep.get("vocoder_path"):
            self.vocoder = make_session(ep["vocoder_path"], providers=ep.get("providers"))
        if self.tokenizer is None and ep.get("bpe_tokenizer_path"):
            self.tokenizer = BPETokenizer(ep["bpe_tokenizer_path"])
            self.special = {t: self.tokenizer._tok.token_to_id(t) for t in _SPECIAL_TOKENS}
            missing = [t for t, i in self.special.items() if i is None]
            if missing:
                raise ValueError(f"Spark-TTS tokenizer.json is missing {missing}")
        if self.preset_global_tokens is None and ep.get("speaker_tokens_path"):
            self.preset_global_tokens = self._load_speaker_tokens(ep["speaker_tokens_path"])

    @staticmethod
    def _load_speaker_tokens(path: str) -> np.ndarray:
        """Read a preset speaker's 32 global tokens from its JSON asset."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokens = data["global_tokens"] if isinstance(data, dict) else data
        arr = np.asarray(tokens, np.int64).reshape(-1)
        if arr.size != N_GLOBAL_TOKENS:
            raise ValueError(f"Spark-TTS speaker tokens must be {N_GLOBAL_TOKENS} values, "
                             f"got {arr.size} from '{path}'")
        return arr

    def _read_kv_shape(self, session: onnxruntime.InferenceSession) -> None:
        """Read the KV-cache layout from the LM's own input signature."""
        for inp in session.get_inputs():
            if inp.name.startswith("past_key_values"):
                self.past_names.append(inp.name)
                if not self.num_kv_heads:
                    self.num_kv_heads = int(inp.shape[1])
                    self.head_dim = int(inp.shape[3])

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    def encode_text(self, text: str, voice: Any, syn_config: Any) -> List[List[int]]:
        """Turn text into one subword-id list per autoregressive pass.

        Only the *content* is tokenized here. The control tokens and the speaker stream
        are added in :meth:`synthesize`, which is where the speaker is known.

        A cloning reference's transcription is tokenized here too. It is text for this
        model, so it goes through the same subword BPE as the content — not through the
        shared phonemizer path, whose ``prompt_tokens`` are phoneme ids for the
        phoneme-based in-context engines and mean nothing to Spark-TTS. The whole text is
        encoded before the first chunk is synthesized, so one transcription is in place
        for every chunk of the call.
        """
        if self.tokenizer is None:
            raise RuntimeError("Spark-TTS voice missing bpe_tokenizer_path in engine_params")
        reference_text = getattr(syn_config, "speaker_reference_text", None)
        self._reference_text_ids = (self.tokenizer.tokenize(reference_text)
                                    if reference_text else None)
        return [self.tokenizer.tokenize(chunk) for chunk in chunk_text(text) if chunk.strip()]

    # ------------------------------------------------------------------
    # Speaker
    # ------------------------------------------------------------------

    def _cloning_session(self, attribute: str, key: str) -> onnxruntime.InferenceSession:
        """Open a cloning graph the first time a call needs it, then keep it."""
        session = getattr(self, attribute)
        if session is None:
            path = self._params.get(key)
            if not path:
                raise RuntimeError(f"Spark-TTS voice cannot clone: no {key} in engine_params")
            session = make_session(path, providers=self._params.get("providers"))
            setattr(self, attribute, session)
        return session

    def tokenize_reference(self, audio: np.ndarray, sr: int,
                           with_semantic: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Turn a reference clip into its global tokens (and semantic tokens on request).

        Returns ``(global_tokens, semantic_tokens)``; the second value is ``None`` unless
        ``with_semantic`` is set, because only the in-context cloning path needs it.
        """
        self.speaker_tokenizer = self._cloning_session(
            "speaker_tokenizer", "speaker_tokenizer_path")
        wav = volume_normalize(resample(audio, sr))
        spec = magnitude_spectrogram(reference_clip(wav))
        global_tokens = np.asarray(
            self.speaker_tokenizer.run(None, {"spec": spec})[0], np.int64).reshape(-1)

        if not with_semantic:
            return global_tokens, None
        self.wav2vec2 = self._cloning_session("wav2vec2", "wav2vec2_path")
        self.semantic_tokenizer = self._cloning_session(
            "semantic_tokenizer", "semantic_tokenizer_path")
        # wav2vec2 reads a zero-mean, unit-variance waveform (its feature extractor's
        # do_normalize), and Spark-TTS mixes hidden states 11/14/16 inside the graph.
        norm = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)
        feat = self.wav2vec2.run(None, {"wav": norm[None].astype(np.float32)})[0]
        semantic = np.asarray(
            self.semantic_tokenizer.run(None, {"feat": feat})[0], np.int64).reshape(-1)
        return global_tokens, semantic

    def _resolve_speaker(self, request: AdapterSynthesisRequest
                         ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Pick the speaker for this call: the reference clip if given, else the preset."""
        ref = request.params.get("reference_audio")
        if ref is not None:
            return self.tokenize_reference(np.asarray(ref[0]), int(ref[1]),
                                           with_semantic=bool(self._reference_text_ids))
        if self.preset_global_tokens is None:
            raise RuntimeError("Spark-TTS voice has neither a speaker_tokens_path preset "
                               "nor a speaker_reference clip to clone from")
        return self.preset_global_tokens, None

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def build_prompt(self, content_ids: List[int], global_tokens: np.ndarray,
                     prompt_text_ids: Optional[List[int]] = None,
                     prompt_semantic: Optional[np.ndarray] = None) -> np.ndarray:
        """Assemble the control-token prompt Spark-TTS was trained on.

        The bare form is ``task, content, speaker``. When a transcribed reference clip is
        available the prompt also carries that transcription and the clip's own semantic
        tokens, so the model continues a real utterance instead of starting cold.
        """
        s = self.special
        global_ids = (s["<|bicodec_global_0|>"] + np.asarray(global_tokens, np.int64)).tolist()
        ids: List[int] = [s["<|task_tts|>"], s["<|start_content|>"]]
        ids += list(prompt_text_ids or [])
        ids += list(content_ids)
        ids += [s["<|end_content|>"], s["<|start_global_token|>"]]
        ids += global_ids
        ids += [s["<|end_global_token|>"]]
        if prompt_text_ids and prompt_semantic is not None:
            ids += [s["<|start_semantic_token|>"]]
            ids += (s["<|bicodec_semantic_0|>"] + np.asarray(prompt_semantic, np.int64)).tolist()
        return np.asarray(ids, np.int64)[None]

    def _generate(self, session: onnxruntime.InferenceSession, prompt: np.ndarray,
                  temperature: float, top_k: int, top_p: float,
                  rng: np.random.Generator) -> List[int]:
        """Run the KV-cached autoregressive loop and return the emitted token ids."""
        if not self.past_names:
            self._read_kv_shape(session)
        names = [o.name for o in session.get_outputs()]
        past = {n: np.zeros((1, self.num_kv_heads, 0, self.head_dim), np.float32)
                for n in self.past_names}
        prompt_len = prompt.shape[1]
        feed = {"input_ids": prompt,
                "attention_mask": np.ones((1, prompt_len), np.int64),
                "position_ids": np.arange(prompt_len, dtype=np.int64)[None], **past}
        eos = self.special["<|im_end|>"]
        emitted: List[int] = []
        for step in range(MAX_NEW_TOKENS):
            outputs = dict(zip(names, session.run(None, feed)))
            token = sample_token(outputs["logits"][0, -1], temperature, top_k, top_p, rng)
            if token == eos:
                break
            emitted.append(token)
            past = {n: outputs[n.replace("past_key_values", "present")]
                    for n in self.past_names}
            position = prompt_len + step
            feed = {"input_ids": np.array([[token]], np.int64),
                    "attention_mask": np.ones((1, position + 1), np.int64),
                    "position_ids": np.array([[position]], np.int64), **past}
        return emitted

    def synthesize(self, request: AdapterSynthesisRequest,
                   session: onnxruntime.InferenceSession) -> AdapterSynthesisResult:
        if self.vocoder is None:
            raise RuntimeError("Spark-TTS voice missing vocoder_path in engine_params")
        if self.tokenizer is None:
            raise RuntimeError("Spark-TTS voice missing bpe_tokenizer_path in engine_params")

        global_tokens, prompt_semantic = self._resolve_speaker(request)
        prompt_text_ids = self._reference_text_ids
        prompt = self.build_prompt(
            np.asarray(request.phoneme_ids, np.int64).reshape(-1).tolist(),
            global_tokens, prompt_text_ids, prompt_semantic)

        p = request.params
        rng = np.random.default_rng(p.get("seed"))
        emitted = self._generate(session, prompt, float(p.get("temperature", 0.8)),
                                 int(p.get("top_k", 50)), float(p.get("top_p", 0.95)), rng)

        base = self.special["<|bicodec_semantic_0|>"]
        semantic = [t - base for t in emitted if base <= t < base + SEMANTIC_CODEBOOK]
        if not semantic:
            LOG.warning("Spark-TTS produced no semantic tokens for this chunk")
            return AdapterSynthesisResult(audio=np.zeros(0, np.float32))

        wav = self.vocoder.run(None, {
            "semantic_tokens": np.asarray([semantic], np.int64),
            "global_tokens": np.asarray(global_tokens, np.int64).reshape(1, 1, -1),
        })[0]
        audio = np.asarray(wav, np.float32).reshape(-1)
        return AdapterSynthesisResult(
            audio=audio, extras={"semantic_token_count": len(semantic)})

    # build_feed_dict / parse_outputs are required by the ABC but unused — synthesize()
    # drives the autoregressive pipeline directly.
    def build_feed_dict(self, request: AdapterSynthesisRequest,
                        session: onnxruntime.InferenceSession) -> Dict[str, np.ndarray]:
        raise NotImplementedError("Spark-TTS is autoregressive — use synthesize()")

    def parse_outputs(self, outputs: List[np.ndarray],
                      request: AdapterSynthesisRequest,
                      output_names: Optional[List[str]] = None) -> AdapterSynthesisResult:
        raise NotImplementedError("Spark-TTS is autoregressive — use synthesize()")

    @staticmethod
    def detect(config: Optional[Dict[str, Any]] = None,
               session: Optional[onnxruntime.InferenceSession] = None) -> bool:
        return bool(config and config.get("engine") == "sparktts")
