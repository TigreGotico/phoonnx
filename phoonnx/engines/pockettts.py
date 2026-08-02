"""Kyutai Pocket TTS inference adapter — 100M-parameter streaming codec TTS.

Pocket TTS (Kyutai, ``kyutai/pocket-tts``) is a compact multilingual text-to-speech
model that runs faster than real time on two CPU cores. It covers English, French,
German, Italian, Portuguese and Spanish. Each language is a separate weight bundle.

The model is a *flow-matching latent language model* on top of the Mimi neural codec:

* a SentencePiece tokenizer (4000 units, one per language) turns text into token ids;
* ``text_conditioner`` embeds the token ids;
* ``flow_lm_main`` is an autoregressive transformer with an explicit key/value state.
  Each step consumes the previous latent frame and returns a conditioning vector plus
  an end-of-speech logit;
* ``flow_lm_flow`` is the flow network. A short Euler loop over it turns a Gaussian
  sample into the next 32-dimensional latent frame;
* ``mimi_decoder`` turns latent frames into 24 kHz audio at 12.5 frames per second
  (1920 samples per frame). It also keeps an explicit state, so audio can be decoded
  in small chunks while generation continues;
* ``mimi_encoder`` turns a reference recording into latent frames for voice cloning.

Five ONNX graphs per bundle::

    text_conditioner : token_ids                              -> text_embeddings
    flow_lm_main     : sequence, text_embeddings, state_*      -> conditioning, eos_logit, out_state_*
    flow_lm_flow     : c, s, t, x                              -> flow
    mimi_decoder     : latent, state_*                         -> audio, out_state_*
    mimi_encoder     : audio                                   -> embeddings

plus ``bundle.json`` (sample rate, latent dim, chunk limit and the two state
manifests), ``tokenizer.model`` (SentencePiece) and ``bos_before_voice.npy`` (a
beginning-of-sequence embedding prepended to cloned voice embeddings).

The **state manifests** are the contract between the graphs and the caller. Each entry
names one input tensor (``state_N``), the matching output tensor (``out_state_N``), its
shape, its dtype and how to fill it at the start of a stream. The caller creates the
tensors, feeds them in, and copies the returned tensors back for the next step. The
manifest also maps a saved voice state (a safetensors file of module states) onto the
same tensors, which is how the published speaker voices load.

A **voice** is therefore a *state*, not an embedding vector: the transformer state after
the model has consumed the speaker's audio. Kyutai publishes one such state per speaker
per language. Cloning at run time follows the same path — encode a reference clip with
``mimi_encoder``, prepend the BOS embedding, and prime ``flow_lm_main`` with it.

This is a **grapheme/text-token engine** in the same family as
:class:`phoonnx.engines.supertonic.SuperTonicAdapter` and
:class:`phoonnx.engines.chatterbox.ChatterboxAdapter`: it owns text -> id conversion via
``encode_text``, so no phonemizer runs. The text frontend is Pocket TTS's own — capitalize
the first letter, add a final period, then split on sentence-final punctuation and pack
the pieces up to the bundle's token limit, because the model was trained on single
sentences and drifts on longer inputs.

The text frontend, the chunk packing and the sampling loop below follow Kyutai's
``pocket_tts`` reference implementation (Apache-2.0, Copyright Kyutai) and the ONNX
runtime contract published with the exported graphs.
"""
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter
from phoonnx.providers import make_session

# Languages Pocket TTS was trained on, as ISO-639-1 codes.
AVAILABLE_LANGS = ["en", "fr", "de", "it", "pt", "es"]

# Default sampling controls, from Kyutai's ``pocket_tts.default_parameters``.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_LSD_STEPS = 1
DEFAULT_EOS_THRESHOLD = -4.0

# Generation length guard, from the reference implementation: the model emits about
# three text tokens per second of speech, plus a two-second allowance.
TOKENS_PER_SECOND_ESTIMATE = 3.0
GEN_SECONDS_PADDING = 2.0

_NUMPY_DTYPES = {
    "float32": np.float32,
    "float16": np.float16,
    "int64": np.int64,
    "bool": np.bool_,
}


def prepare_text(text: str, remove_semicolons: bool = False,
                 pad_with_spaces_for_short_inputs: bool = False) -> Tuple[str, int]:
    """Apply Pocket TTS's text frontend to one chunk.

    The model was trained on capitalized sentences that end in punctuation, so text that
    does not look like one is repaired first. Returns the prepared text and the number of
    frames to keep generating after the end-of-speech logit fires: short inputs need more,
    because the model signals the end early on them.
    """
    text = text.strip()
    if not text:
        raise ValueError("Pocket TTS cannot synthesize empty text")
    text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
    if remove_semicolons:
        text = text.replace(";", ",")
    frames_after_eos = 3 if len(text.split()) <= 4 else 1
    if not text[0].isupper():
        text = text[0].upper() + text[1:]
    if text[-1].isalnum():
        text += "."
    if pad_with_spaces_for_short_inputs and len(text.split()) < 5:
        text = " " * 8 + text
    return text, frames_after_eos


def find_boundary_indices(tokens: List[int], boundary_tokens: set) -> List[int]:
    """Return the split points that end each run of boundary tokens.

    A boundary token is a sentence-final or clause-final piece such as ``.`` or ``,``.
    Consecutive boundary tokens (``...``) count as one split point, and the punctuation
    stays with the segment it closes.
    """
    indices = [0]
    previous_was_boundary = False
    for index, token in enumerate(tokens):
        if token in boundary_tokens:
            previous_was_boundary = True
        else:
            if previous_was_boundary:
                indices.append(index)
            previous_was_boundary = False
    indices.append(len(tokens))
    return indices


class PocketTTSAdapter(BaseOnnxAdapter):
    """Adapter for Kyutai Pocket TTS (five ONNX graphs, explicit stream state)."""

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE,
                 lsd_steps: int = DEFAULT_LSD_STEPS,
                 eos_threshold: float = DEFAULT_EOS_THRESHOLD,
                 seed: Optional[int] = None):
        self.text_conditioner: Optional[onnxruntime.InferenceSession] = None
        self.flow_lm_flow: Optional[onnxruntime.InferenceSession] = None
        self.mimi_decoder: Optional[onnxruntime.InferenceSession] = None
        self.mimi_encoder: Optional[onnxruntime.InferenceSession] = None
        self.tokenizer = None
        self.metadata: Dict[str, Any] = {}
        self.bos_before_voice: Optional[np.ndarray] = None
        self.voice_state: Optional[Dict[str, np.ndarray]] = None

        # Bundle geometry — replaced by bundle.json in configure().
        self.sample_rate = 24000
        self.frame_rate = 12.5
        self.latent_dim = 32
        self.conditioning_dim = 1024
        self.max_token_per_chunk = 50
        self.insert_bos_before_voice = True
        self.remove_semicolons = False
        self.pad_with_spaces_for_short_inputs = False
        self.model_recommended_frames_after_eos: Optional[int] = None
        self.flow_state_manifest: List[Dict[str, Any]] = []
        self.mimi_state_manifest: List[Dict[str, Any]] = []

        self.temperature = float(temperature)
        self.lsd_steps = int(lsd_steps)
        self.eos_threshold = float(eos_threshold)
        self.seed = seed

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def default_params(self) -> Dict[str, float]:
        return {"temperature": self.temperature,
                "lsd_steps": float(self.lsd_steps),
                "eos_threshold": self.eos_threshold}

    def param_labels(self) -> Dict[str, str]:
        return {"temperature": "Sampling temperature",
                "lsd_steps": "Flow integration steps",
                "eos_threshold": "End-of-speech threshold"}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def configure(self, voice_config: Any) -> None:
        """Load the bundle metadata, the four auxiliary graphs, the SentencePiece
        tokenizer, the BOS embedding and the speaker state from ``engine_params``.
        ``flow_lm_main`` is the voice's own primary session."""
        ep = getattr(voice_config, "engine_params", None) or {}

        if ep.get("bundle_path") and not self.metadata:
            with open(ep["bundle_path"], "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self._apply_metadata(self.metadata)

        def sess(path):
            return make_session(path, providers=ep.get("providers"))

        if self.text_conditioner is None and ep.get("text_conditioner_path"):
            self.text_conditioner = sess(ep["text_conditioner_path"])
        if self.flow_lm_flow is None and ep.get("flow_lm_flow_path"):
            self.flow_lm_flow = sess(ep["flow_lm_flow_path"])
        if self.mimi_decoder is None and ep.get("mimi_decoder_path"):
            self.mimi_decoder = sess(ep["mimi_decoder_path"])
        if self.mimi_encoder is None and ep.get("mimi_encoder_path"):
            self.mimi_encoder = sess(ep["mimi_encoder_path"])

        if self.tokenizer is None and ep.get("tokenizer_path"):
            import sentencepiece
            self.tokenizer = sentencepiece.SentencePieceProcessor()
            self.tokenizer.Load(ep["tokenizer_path"])

        if self.bos_before_voice is None and ep.get("bos_path"):
            self.bos_before_voice = np.load(ep["bos_path"]).astype(np.float32)

        if self.voice_state is None and ep.get("voice_state_path"):
            self.voice_state = self.load_voice_state(ep["voice_state_path"])

        if "temperature" in ep:
            self.temperature = float(ep["temperature"])
        if "lsd_steps" in ep:
            self.lsd_steps = int(ep["lsd_steps"])
        if "eos_threshold" in ep:
            self.eos_threshold = float(ep["eos_threshold"])
        if "seed" in ep:
            self.seed = ep["seed"]

    def _apply_metadata(self, metadata: Dict[str, Any]) -> None:
        """Read the bundle geometry that the graphs were exported with."""
        self.sample_rate = int(metadata["sample_rate"])
        self.frame_rate = float(metadata["frame_rate"])
        self.latent_dim = int(metadata["latent_dim"])
        self.conditioning_dim = int(metadata["conditioning_dim"])
        self.max_token_per_chunk = int(metadata.get("max_token_per_chunk", 50))
        self.insert_bos_before_voice = bool(metadata.get("insert_bos_before_voice", False))
        self.remove_semicolons = bool(metadata.get("remove_semicolons", False))
        self.pad_with_spaces_for_short_inputs = bool(
            metadata.get("pad_with_spaces_for_short_inputs", False))
        self.model_recommended_frames_after_eos = metadata.get(
            "model_recommended_frames_after_eos")
        self.flow_state_manifest = metadata["flow_lm_state_manifest"]
        self.mimi_state_manifest = metadata["mimi_state_manifest"]

    # ------------------------------------------------------------------
    # Stream state
    # ------------------------------------------------------------------

    @staticmethod
    def _filled(shape, dtype, fill: str) -> np.ndarray:
        if fill == "nan":
            return np.full(shape, np.nan, dtype=dtype)
        if fill == "ones":
            return np.ones(shape, dtype=dtype)
        return np.zeros(shape, dtype=dtype)

    def init_state(self, manifest: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Build the start-of-stream tensors named by a state manifest."""
        return {e["input_name"]: self._filled(e["shape"], _NUMPY_DTYPES[e["dtype"]], e["fill"])
                for e in manifest}

    @staticmethod
    def _update_state(state: Dict[str, np.ndarray], outputs: List[np.ndarray],
                      manifest: List[Dict[str, Any]], output_offset: int) -> None:
        """Copy the returned ``out_state_N`` tensors back onto their ``state_N`` inputs."""
        for entry in manifest:
            state[entry["input_name"]] = outputs[output_offset + entry["index"]]

    def _adapt_state_tensor(self, source: np.ndarray, entry: Dict[str, Any]) -> np.ndarray:
        """Fit a saved tensor onto the shape and dtype the graph declares.

        A published voice state was saved by the PyTorch model, whose caches are sized to
        the audio it consumed. The exported graph declares fixed cache shapes, so a
        shorter saved cache is copied into the leading part of a freshly filled tensor and
        a longer one is truncated.
        """
        target_shape = tuple(entry["shape"])
        target_dtype = _NUMPY_DTYPES[entry["dtype"]]
        source = np.asarray(source, dtype=target_dtype)
        if source.shape == target_shape:
            return source.copy()
        if source.size == int(np.prod(target_shape, dtype=np.int64)):
            return source.reshape(target_shape).copy()
        target = self._filled(list(target_shape), target_dtype, entry["fill"])
        if source.ndim != len(target_shape):
            return target
        slices = tuple(slice(0, min(src, dst)) for src, dst in zip(source.shape, target_shape))
        if all(s.start == s.stop for s in slices):
            return target
        target[slices] = source[slices]
        return target

    @staticmethod
    def _derive_step(module_state: Dict[str, np.ndarray]) -> np.ndarray:
        """Recover the transformer step counter from a saved module state.

        Exports differ in what they call the counter, so fall back through the names the
        published states use, and finally infer it from the cache length.
        """
        if "step" in module_state:
            return np.asarray(module_state["step"], dtype=np.int64).reshape(1)
        if "offset" in module_state and "end_offset" not in module_state:
            return np.asarray(module_state["offset"], dtype=np.int64).reshape(1)
        if "current_end" in module_state:
            return np.array([module_state["current_end"].shape[0]], dtype=np.int64)
        return np.array([0], dtype=np.int64)

    def load_voice_state(self, path: str) -> Dict[str, np.ndarray]:
        """Load a published speaker state (a safetensors file of module states) and map
        it onto the flow-LM state tensors through the bundle's state manifest."""
        from safetensors import safe_open

        model_state: Dict[str, Dict[str, np.ndarray]] = {}
        with safe_open(str(path), framework="np") as handle:
            for key in handle.keys():
                module_name, tensor_key = key.split("/", 1)
                model_state.setdefault(module_name, {})[tensor_key] = handle.get_tensor(key)

        state = self.init_state(self.flow_state_manifest)
        for entry in self.flow_state_manifest:
            module_state = model_state.get(entry["module"], {})
            tensor = module_state.get(entry["key"])
            if tensor is None and entry["key"] == "step":
                tensor = self._derive_step(module_state)
            if tensor is None:
                continue
            state[entry["input_name"]] = self._adapt_state_tensor(tensor, entry)
        return state

    def encode_reference(self, audio: np.ndarray) -> np.ndarray:
        """Turn a mono 24 kHz reference clip into Mimi latent frames for cloning."""
        if self.mimi_encoder is None:
            raise RuntimeError("Pocket TTS voice missing mimi_encoder_path in engine_params")
        audio = np.asarray(audio, dtype=np.float32).reshape(1, 1, -1)
        embeddings = self.mimi_encoder.run(None, {"audio": audio})[0]
        while embeddings.ndim > 3:
            embeddings = embeddings.squeeze(0)
        if embeddings.ndim < 3:
            embeddings = embeddings[None]
        return embeddings.astype(np.float32, copy=False)

    def state_from_reference(self, audio: np.ndarray,
                             session: onnxruntime.InferenceSession) -> Dict[str, np.ndarray]:
        """Prime a fresh flow-LM state with a cloned voice, so later steps speak in it."""
        embeddings = self.encode_reference(audio)
        if self.insert_bos_before_voice and self.bos_before_voice is not None:
            embeddings = np.concatenate([self.bos_before_voice, embeddings], axis=1)
        state = self.init_state(self.flow_state_manifest)
        outputs = session.run(None, {
            "sequence": np.zeros((1, 0, self.latent_dim), dtype=np.float32),
            "text_embeddings": embeddings, **state})
        self._update_state(state, outputs, self.flow_state_manifest, output_offset=2)
        return state

    # ------------------------------------------------------------------
    # Text frontend
    # ------------------------------------------------------------------

    def _segments(self, tokens: List[int], boundaries: List[int]) -> List[Tuple[int, str]]:
        return [(boundaries[i + 1] - boundaries[i],
                 self.tokenizer.Decode(tokens[boundaries[i]:boundaries[i + 1]]))
                for i in range(len(boundaries) - 1)]

    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into pieces the model can say in one pass.

        Split on sentence-final punctuation first. A sentence still longer than the
        bundle's token limit is split again on commas, semicolons and colons. Adjacent
        pieces are then packed back together while they fit under the limit, so short
        sentences are not synthesized one at a time.
        """
        prepared, _ = prepare_text(text, self.remove_semicolons,
                                   self.pad_with_spaces_for_short_inputs)
        tokens = self.tokenizer.Encode(prepared.strip())

        eos_tokens = set(self.tokenizer.Encode(".!...?")[1:])
        segments = self._segments(tokens, find_boundary_indices(tokens, eos_tokens))

        fallback_tokens = set(self.tokenizer.Encode(",;:")[1:])
        refined: List[Tuple[int, str]] = []
        for count, segment in segments:
            if count <= self.max_token_per_chunk:
                refined.append((count, segment))
                continue
            sub_tokens = self.tokenizer.Encode(segment.strip())
            sub_segments = self._segments(
                sub_tokens, find_boundary_indices(sub_tokens, fallback_tokens))
            refined.extend(sub_segments if len(sub_segments) > 1 else [(count, segment)])

        chunks: List[str] = []
        current, current_count = "", 0
        for count, segment in refined:
            if not current:
                current, current_count = segment, count
            elif current_count + count > self.max_token_per_chunk:
                chunks.append(current.strip())
                current, current_count = segment, count
            else:
                current += " " + segment
                current_count += count
        if current:
            chunks.append(current.strip())
        return [c for c in chunks if c]

    def encode_text(self, text: str, voice: Any, syn_config: Any) -> List[List[int]]:
        """Chunk the text Pocket TTS's way, then tokenize each chunk with the bundle's
        own SentencePiece model. One list of token ids per chunk."""
        if self.tokenizer is None:
            raise RuntimeError("Pocket TTS voice missing tokenizer_path in engine_params")
        ids = []
        for chunk in self.split_into_chunks(text):
            prepared, _ = prepare_text(chunk, self.remove_semicolons,
                                       self.pad_with_spaces_for_short_inputs)
            ids.append(list(self.tokenizer.Encode(prepared)))
        return ids

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _frames_after_eos(self, token_ids: np.ndarray) -> int:
        """How many frames to keep generating after the end-of-speech logit fires."""
        if self.model_recommended_frames_after_eos is not None:
            return int(self.model_recommended_frames_after_eos)
        if self.tokenizer is None:
            return 3
        _, guess = prepare_text(self.tokenizer.Decode([int(i) for i in token_ids.reshape(-1)]),
                                self.remove_semicolons,
                                self.pad_with_spaces_for_short_inputs)
        return guess + 2

    def _max_frames(self, token_count: int) -> int:
        """Hard cap on generated frames, from the text length. Guards against a stream
        that never signals the end."""
        seconds = token_count / TOKENS_PER_SECOND_ESTIMATE + GEN_SECONDS_PADDING
        return int(math.ceil(seconds * self.frame_rate))

    def generate_latents(self, token_ids: np.ndarray, voice_state: Dict[str, np.ndarray],
                         session: onnxruntime.InferenceSession, temperature: float,
                         lsd_steps: int, eos_threshold: float,
                         rng: np.random.Generator) -> np.ndarray:
        """Run the autoregressive loop for one chunk and return its latent frames.

        Each step asks ``flow_lm_main`` for a conditioning vector and an end-of-speech
        logit, then integrates ``flow_lm_flow`` from a Gaussian sample to the next latent
        frame. The loop stops a few frames after the logit crosses the threshold, or at
        the length cap.
        """
        state = {k: v.copy() for k, v in voice_state.items()}
        text_embeddings = self.text_conditioner.run(None, {"token_ids": token_ids})[0]
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings[None]

        empty_seq = np.zeros((1, 0, self.latent_dim), dtype=np.float32)
        empty_text = np.zeros((1, 0, self.conditioning_dim), dtype=np.float32)

        outputs = session.run(None, {"sequence": empty_seq,
                                     "text_embeddings": text_embeddings, **state})
        self._update_state(state, outputs, self.flow_state_manifest, output_offset=2)

        frames_after_eos = self._frames_after_eos(token_ids)
        frame_limit = self._max_frames(token_ids.shape[1])
        dt = 1.0 / lsd_steps
        st_buffers = [(np.array([[j / lsd_steps]], np.float32),
                       np.array([[j / lsd_steps + dt]], np.float32))
                      for j in range(lsd_steps)]

        curr = np.full((1, 1, self.latent_dim), np.nan, dtype=np.float32)
        latents: List[np.ndarray] = []
        eos_step: Optional[int] = None

        for step in range(frame_limit):
            outputs = session.run(None, {"sequence": curr,
                                         "text_embeddings": empty_text, **state})
            conditioning, eos_logit = outputs[0], outputs[1]
            self._update_state(state, outputs, self.flow_state_manifest, output_offset=2)

            if eos_step is None and eos_logit[0][0] > eos_threshold:
                eos_step = step
            if eos_step is not None and step >= eos_step + frames_after_eos:
                break

            if temperature > 0:
                x = rng.normal(0.0, math.sqrt(temperature),
                               (1, self.latent_dim)).astype(np.float32)
            else:
                x = np.zeros((1, self.latent_dim), dtype=np.float32)
            for s_arr, t_arr in st_buffers:
                x = x + self.flow_lm_flow.run(
                    None, {"c": conditioning, "s": s_arr, "t": t_arr, "x": x})[0] * dt

            curr = x.reshape(1, 1, self.latent_dim)
            latents.append(curr)

        if not latents:
            return np.zeros((1, 0, self.latent_dim), dtype=np.float32)
        return np.concatenate(latents, axis=1)

    def decode_latents(self, latents: np.ndarray, chunk_size: int = 15) -> np.ndarray:
        """Turn latent frames into audio, feeding the decoder in small chunks and
        carrying its state forward so the joins are seamless."""
        state = self.init_state(self.mimi_state_manifest)
        audio: List[np.ndarray] = []
        for index in range(0, latents.shape[1], chunk_size):
            outputs = self.mimi_decoder.run(
                None, {"latent": latents[:, index:index + chunk_size, :], **state})
            audio.append(np.asarray(outputs[0], np.float32).reshape(-1))
            self._update_state(state, outputs, self.mimi_state_manifest, output_offset=1)
        if not audio:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(audio)

    def synthesize(self, request: AdapterSynthesisRequest,
                   session: onnxruntime.InferenceSession) -> AdapterSynthesisResult:
        if self.text_conditioner is None or self.flow_lm_flow is None or self.mimi_decoder is None:
            raise RuntimeError("Pocket TTS voice missing text_conditioner_path / "
                               "flow_lm_flow_path / mimi_decoder_path in engine_params")
        if not self.flow_state_manifest or not self.mimi_state_manifest:
            raise RuntimeError("Pocket TTS voice missing bundle_path (bundle.json) "
                               "in engine_params")

        p = request.params
        temperature = float(p.get("temperature", self.temperature))
        lsd_steps = int(p.get("lsd_steps", self.lsd_steps))
        eos_threshold = float(p.get("eos_threshold", self.eos_threshold))
        if lsd_steps < 1:
            raise ValueError("pockettts lsd_steps must be >= 1")
        if temperature < 0:
            raise ValueError("pockettts temperature must be >= 0")

        seed = p.get("seed", self.seed)
        rng = np.random.default_rng(seed)

        reference = p.get("reference_audio")
        if reference is not None:
            voice_state = self.state_from_reference(reference, session)
        elif self.voice_state is not None:
            voice_state = self.voice_state
        else:
            raise RuntimeError("Pocket TTS voice missing voice_state_path in engine_params "
                               "and no speaker_reference was given")

        token_ids = np.asarray(request.phoneme_ids, np.int64).reshape(1, -1)
        latents = self.generate_latents(token_ids, voice_state, session, temperature,
                                        lsd_steps, eos_threshold, rng)
        audio = self.decode_latents(latents)
        return AdapterSynthesisResult(
            audio=audio, extras={"frames": int(latents.shape[1])})

    # build_feed_dict / parse_outputs are required by the ABC but unused — synthesize()
    # drives the multi-graph pipeline directly.
    def build_feed_dict(self, request: AdapterSynthesisRequest,
                        session: onnxruntime.InferenceSession) -> Dict[str, np.ndarray]:
        raise NotImplementedError("Pocket TTS is multi-graph — use synthesize()")

    def parse_outputs(self, outputs: List[np.ndarray],
                      request: AdapterSynthesisRequest,
                      output_names: Optional[List[str]] = None) -> AdapterSynthesisResult:
        raise NotImplementedError("Pocket TTS is multi-graph — use synthesize()")

    @staticmethod
    def detect(config: Optional[Dict[str, Any]] = None,
               session: Optional[onnxruntime.InferenceSession] = None) -> bool:
        if config and config.get("engine") == "pockettts":
            return True
        if session is not None:
            names = {i.name for i in session.get_inputs()}
            # flow_lm_main's distinctive input signature: the latent sequence, the text
            # embeddings and the numbered transformer state tensors.
            if {"sequence", "text_embeddings", "state_0", "state_1"} <= names:
                return True
        return False
