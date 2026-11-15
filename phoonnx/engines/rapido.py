"""Rápido inference adapter: a flow-matching DiT over the VoxCPM2 AudioVAE latent space.

Rápido (Apache 2.0) reads one token per letter or phoneme and writes 64-dim latents at
25 Hz, which the frozen VoxCPM2 AudioVAE v2 decoder turns into 48 kHz audio. The
engine has no text path of its own: the voice pipeline hands it token ids.

Four ONNX graphs per voice::

    text_encoder : ids [1,L] -> features [1,L,D], durations [1,L] (frames per token)
    aligner      : features, durations, token word, token word start, frame word,
                   frame position, speaker -> condition [1,T,D]
    model        : x [1,T,64], condition, t [1], guidance [1] -> velocity [1,T,64]
                   (classifier-free guidance folded in; the voice's own ``session``)
    decoder      : latents [1,64,T] -> audio [1,1,T*1920]

The word and frame layouts between the text encoder and the aligner are computed here
in numpy. ``phoonnx_train.rapido`` imports the same functions, so training and
inference cannot lay a sentence out differently.
"""
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter
from phoonnx.providers import make_session

CLAUSE_TOKENS = (".", "?", "!", ",", ":", ";")
MAX_WORD_FRAMES = 250


def layout_boundary_ids(phoneme_id_map: Mapping[str, Union[int, Sequence[int]]],
                        word_sep_token: str = " ", bos: Optional[str] = "^",
                        eos: Optional[str] = "$") -> frozenset:
    """The token ids that end a word: the word separator, the clause punctuation and
    BOS/EOS, each only when the vocabulary has it.

    Punctuation counts as a boundary because a phonemizer writes ``bẽj̃?ˈsɐ̃w̃`` with no
    separator after a sentence-final mark.
    """
    ids = set()
    for token in (word_sep_token, *CLAUSE_TOKENS, bos, eos):
        if token is None or token not in phoneme_id_map:
            continue
        value = phoneme_id_map[token]
        ids.add(int(value[0] if isinstance(value, (list, tuple)) else value))
    return frozenset(ids)


def word_layout(ids: Sequence[int], boundary_ids: Collection[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Token ids -> each token's word index and the index of that word's first token.

    A word starts at a non-boundary token that is first or follows a boundary token.
    A word owns the boundary tokens after it; leading boundary tokens join word 0.
    """
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    is_boundary = np.isin(ids, np.fromiter(boundary_ids, dtype=np.int64))
    after_boundary = np.concatenate([[True], is_boundary[:-1]])
    starts = ~is_boundary & after_boundary
    word = np.maximum(np.cumsum(starts) - 1, 0)
    first = np.concatenate([[0], np.flatnonzero(starts)[1:]]).astype(np.int64)
    return word.astype(np.int64), first[word]


def frame_layout(counts: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Frames per word -> each frame's word index and its position in that word, in [0, 1)."""
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    frame_word = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    start = (np.cumsum(counts) - counts)[frame_word]
    position = (np.arange(len(frame_word)) - start) / counts[frame_word].astype(np.float64)
    return frame_word, position.astype(np.float32)


def word_frame_counts(durations: np.ndarray, word: np.ndarray) -> np.ndarray:
    """Per-token durations in frames -> whole frames per word, at least one, at most 250."""
    totals = np.zeros(int(word.max()) + 1, dtype=np.float32)
    np.add.at(totals, word, np.asarray(durations, dtype=np.float32))
    return np.clip(np.round(totals), 1, MAX_WORD_FRAMES).astype(np.int64)


def ode_times(steps: int) -> np.ndarray:
    """The ``steps + 1`` time points of the midpoint ODE, from noise (0) to data (1)."""
    return np.linspace(0.0, 1.0, steps + 1).astype(np.float32)


class RapidoAdapter(BaseOnnxAdapter):
    """Adapter for Rápido (flow-matching DiT, 42 preset speakers, 48 kHz)."""

    def __init__(self, steps: int = 8, guidance: float = 4.0, seed: int = 0):
        self.text_encoder: Optional[onnxruntime.InferenceSession] = None
        self.aligner: Optional[onnxruntime.InferenceSession] = None
        self.decoder: Optional[onnxruntime.InferenceSession] = None
        self.latent_mean: Optional[np.ndarray] = None
        self.latent_std: Optional[np.ndarray] = None
        self.boundary_ids: frozenset = frozenset()
        self.num_speakers = 1
        self.steps = int(steps)
        self.guidance = float(guidance)
        self.seed = int(seed)

    def default_params(self) -> Dict[str, float]:
        return {"length_scale": 1.0, "steps": float(self.steps),
                "guidance": self.guidance, "seed": float(self.seed)}

    def configure(self, voice_config: Any) -> None:
        """Load the text encoder, aligner and decoder graphs and the latent statistics
        from ``engine_params``; the velocity model is the voice's own session."""
        ep = voice_config.engine_params or {}
        for attribute, key in (("text_encoder", "text_encoder_path"),
                               ("aligner", "aligner_path"),
                               ("decoder", "decoder_path")):
            if getattr(self, attribute) is None and ep.get(key):
                setattr(self, attribute, make_session(ep[key], providers=ep.get("providers")))
        if "latent_mean" in ep:
            self.latent_mean = np.asarray(ep["latent_mean"], dtype=np.float32)
            self.latent_std = np.asarray(ep["latent_std"], dtype=np.float32)
        for name in ("steps", "seed"):
            if name in ep:
                setattr(self, name, int(ep[name]))
        if "guidance" in ep:
            self.guidance = float(ep["guidance"])
        vocabulary = voice_config.tokenizer.vocabulary
        self.boundary_ids = layout_boundary_ids(vocabulary.char2idx, voice_config.word_sep_token or " ",
                                                vocabulary.bos, vocabulary.eos)
        self.num_speakers = int(voice_config.num_speakers or 1)

    def synthesize(self, request: AdapterSynthesisRequest,
                   session: onnxruntime.InferenceSession) -> AdapterSynthesisResult:
        if self.text_encoder is None or self.aligner is None or self.decoder is None \
                or self.latent_mean is None:
            raise RuntimeError("Rápido voice missing text_encoder_path / aligner_path / "
                               "decoder_path / latent_mean in engine_params")
        speaker = int(request.speaker_id or 0)
        if not 0 <= speaker < self.num_speakers:
            raise ValueError(f"speaker_id {speaker} out of range for {self.num_speakers} speakers")
        p = request.params
        steps = int(p.get("steps", self.steps))
        guidance = np.array([p.get("guidance", self.guidance)], dtype=np.float32)

        ids = np.asarray(request.phoneme_ids, dtype=np.int64).reshape(1, -1)
        features, durations = self.text_encoder.run(None, {"ids": ids})
        word, word_start = word_layout(ids[0], self.boundary_ids)
        counts = word_frame_counts(durations[0] * np.float32(p.get("length_scale", 1.0)), word)
        frame_word, frame_pos = frame_layout(counts)
        condition = self.aligner.run(None, {
            "features": features, "durations": durations,
            "word": word[None], "word_start": word_start[None],
            "frame_word": frame_word[None], "frame_pos": frame_pos[None],
            "speaker": np.array([speaker], dtype=np.int64)})[0]

        rng = np.random.default_rng(int(p.get("seed", self.seed)))
        x = rng.standard_normal((1, len(frame_word), len(self.latent_mean))).astype(np.float32)

        def velocity(x: np.ndarray, t: np.float32) -> np.ndarray:
            return session.run(None, {"x": x, "condition": condition,
                                      "t": np.array([t], dtype=np.float32), "guidance": guidance})[0]

        times = ode_times(steps)
        for a, b in zip(times[:-1], times[1:]):
            h = b - a
            x = x + h * velocity(x + h / 2 * velocity(x, a), (a + b) / 2)

        latents = (x[0] * self.latent_std + self.latent_mean).T[None].astype(np.float32)
        audio = self.decoder.run(None, {"latents": latents})[0]
        return AdapterSynthesisResult(audio=np.asarray(audio, dtype=np.float32).reshape(-1))

    # build_feed_dict / parse_outputs are required by the ABC but unused: synthesize()
    # drives the multi-graph ODE loop directly.
    def build_feed_dict(self, request: AdapterSynthesisRequest,
                        session: onnxruntime.InferenceSession) -> Dict[str, np.ndarray]:
        raise NotImplementedError("Rápido is iterative, use synthesize()")

    def parse_outputs(self, outputs: List[np.ndarray], request: AdapterSynthesisRequest,
                      output_names: Optional[List[str]] = None) -> AdapterSynthesisResult:
        raise NotImplementedError("Rápido is iterative, use synthesize()")

    @staticmethod
    def detect(config: Optional[Dict[str, Any]] = None,
               session: Optional[onnxruntime.InferenceSession] = None) -> bool:
        return bool(config and config.get("engine") == "rapido")
