"""Streaming VITS inference adapter (split encoder/decoder).

Standard Piper/VITS models are a **single** ONNX graph: phoneme IDs → waveform.
A *streaming* VITS model is published as **two** ONNX graphs so that audio can be
produced incrementally, lowering time-to-first-audio:

1. **Encoder** — phoneme IDs → latent ``z`` [B, 192, T] and ``y_mask`` [B, 1, T].
   Runs **once** per sentence. Samples the duration/noise internally.
2. **Decoder** — a slice of ``z`` (+ ``y_mask``) → waveform for that slice.
   Run **repeatedly** over the time axis, streaming chunks of audio.

These are the Sonata "+RT" fast voices (config carries ``"streaming": true``),
or any model exported with a split encoder/decoder. A combined single-graph model
CANNOT stream — use the plain :class:`VitsAdapter` for those.

The decoder is convolutional, so naively slicing ``z`` on the time axis
contaminates the chunk edges (~16-19 frames deep). This adapter therefore uses
**discard-and-stitch**: each chunk is decoded with a context margin of ``M``
frames on both sides, the contaminated margins are discarded, and only the clean
middle is kept. The stitched result is bit-identical to a one-shot decode.

engine_params (in the voice config)::

    "engine_params": {
        "decoder_path": "decoder.onnx",   # required: the second graph
        "streaming": {                     # optional: all have proven defaults
            "context_margin": 32,          # M: frames of context each side
            "first_chunk": 8,              # frames in the first (latency) chunk
            "next_chunk": 160,             # frames per subsequent chunk
            "fallback_frames": 128         # <= this many frames: one-shot decode
        }
    }

ONNX interface (encoder = primary session):
  IN  ``input``         int64    [B, T]   phoneme IDs
      ``input_lengths`` int64    [B]
      ``scales``        float32  [3]      [noise_scale, length_scale, noise_w]
      ``sid``           int64    [B]      speaker ID (multi-speaker only)
  OUT ``z``             float32  [B, 192, T]
      ``y_mask``        float32  [B, 1, T]

ONNX interface (decoder = auxiliary graph):
  IN  ``z``             float32  [B, 192, T_slice]
      ``y_mask``        float32  [B, 1, T_slice]
  OUT ``output``        float32  [B, 1, N] or [N]   waveform samples
"""
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime

from phoonnx.engines.base import (
    AdapterSynthesisRequest,
    AdapterSynthesisResult,
    BaseOnnxAdapter,
)


# Proven defaults from empirical tuning (see docs/streaming.md):
#   context_margin=32  bit-identical on every voice tested (24 often works and is
#                      ~6% faster, but 16 corrupts edges — 32 is the safe default).
#   first_chunk=8      small first slice → lowest time-to-first-audio.
#   next_chunk=160     balances per-chunk overhead against throughput.
#   fallback_frames=128 short sentences decode one-shot (streaming can't help).
_DEFAULT_STREAMING = {
    "context_margin": 32,
    "first_chunk": 8,
    "next_chunk": 160,
    "fallback_frames": 128,
}


class VitsStreamingAdapter(BaseOnnxAdapter):
    """Adapter for split-graph (streaming) VITS models.

    The primary ``session`` is the **encoder**; the **decoder** is loaded as an
    auxiliary graph from ``engine_params["decoder_path"]``. ``synthesize`` is
    overridden to run the discard-and-stitch chunk loop instead of a single
    ``session.run``.
    """

    MEMOIZED_WRITES = {
        # Samples per latent frame, measured once from the decoder itself.
        "_resolve_hop": frozenset({"_hop_length"}),
    }

    def __init__(self, decoder_session: Optional[onnxruntime.InferenceSession] = None):
        self.decoder = decoder_session
        self._engine_params: Dict[str, Any] = {}
        self._streaming: Dict[str, int] = dict(_DEFAULT_STREAMING)
        self._hop_length: Optional[int] = None  # resolved from config or measured

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, voice_config: Any) -> None:
        """Load the decoder graph and streaming parameters from the voice config."""
        engine_params = getattr(voice_config, "engine_params", None) or {}
        self.configure_from_params(engine_params)
        hop = getattr(voice_config, "hop_length", None)
        if hop:
            self._hop_length = int(hop)

    def configure_from_params(self, engine_params: Dict[str, Any]) -> None:
        self._engine_params = engine_params or {}
        user_streaming = self._engine_params.get("streaming") or {}
        if isinstance(user_streaming, dict):
            self._streaming = {**_DEFAULT_STREAMING, **user_streaming}
        threads = self._engine_params.get("intra_op_num_threads")
        decoder_path = self._engine_params.get("decoder_path")
        if self.decoder is None and decoder_path:
            self.decoder = self._build_decoder(decoder_path, threads)

    @staticmethod
    def _build_decoder(path: str, threads: Optional[int]) -> onnxruntime.InferenceSession:
        import os
        so = onnxruntime.SessionOptions()
        if threads is None:
            threads = max(1, (os.cpu_count() or 4) // 2)
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        return onnxruntime.InferenceSession(path, so)


    # ------------------------------------------------------------------
    # Control parameters (shared with the plain VITS adapter)
    # ------------------------------------------------------------------

    def default_params(self) -> Dict[str, float]:
        return {"noise_scale": 0.667, "length_scale": 1.0, "noise_w_scale": 0.8}

    def parse_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        inference = config.get("inference", {})
        defaults = self.default_params()
        return {
            "noise_scale": inference.get("noise_scale", defaults["noise_scale"]),
            "length_scale": inference.get("length_scale", defaults["length_scale"]),
            "noise_w_scale": inference.get("noise_w", defaults["noise_w_scale"]),
        }

    def param_labels(self) -> Dict[str, str]:
        return {"noise_scale": "Noise scale",
                "length_scale": "Length scale",
                "noise_w_scale": "Noise W"}

    @staticmethod
    def detect(config: Optional[Dict[str, Any]] = None,
               session: Optional[onnxruntime.InferenceSession] = None) -> bool:
        """A streaming VITS model declares ``streaming: true`` and ships a decoder.
        Both the config flag and a decoder graph are required — a bare
        ``streaming`` flag on a single-graph model is not enough to stream."""
        if not config:
            return False
        if not config.get("streaming"):
            return False
        engine_params = config.get("engine_params") or {}
        return bool(engine_params.get("decoder_path"))

    # ------------------------------------------------------------------
    # Encoder feed (mirrors VitsAdapter.build_feed_dict, encoder outputs z/y_mask)
    # ------------------------------------------------------------------

    def build_feed_dict(self, request: AdapterSynthesisRequest,
                        session: onnxruntime.InferenceSession) -> Dict[str, np.ndarray]:
        params = request.params
        d = self.default_params()
        noise = np.float32(params.get("noise_scale", d["noise_scale"]))
        length = np.float32(params.get("length_scale", d["length_scale"]))
        noise_w = np.float32(params.get("noise_w_scale", d["noise_w_scale"]))
        scales = np.array([noise, length, noise_w], dtype=np.float32)
        rank = self._input_rank(session, "scales")
        if rank is not None and rank >= 2:
            scales = scales.reshape(1, -1)
        args: Dict[str, np.ndarray] = {
            "input": request.phoneme_ids,
            "input_lengths": request.phoneme_lengths,
            "scales": scales,
        }
        if request.speaker_id is not None:
            args["sid"] = np.array([request.speaker_id], dtype=np.int64)
        return self._filter_inputs(args, session)

    def parse_outputs(self, outputs: List[np.ndarray],
                      request: AdapterSynthesisRequest) -> AdapterSynthesisResult:
        """Required by the base interface. Streaming synthesis is driven by
        ``synthesize`` (which loops over the decoder), so this is only used if a
        caller runs the encoder graph directly: it wraps the first output as audio."""
        audio = np.asarray(outputs[0]).squeeze().astype(np.float32)
        return AdapterSynthesisResult(audio=audio)


    # ------------------------------------------------------------------
    # Streaming synthesis: encode once, decode in discard-and-stitch chunks
    # ------------------------------------------------------------------

    def _resolve_hop(self, z_frames: int, sample_count: int) -> int:
        """hop_length = decoder samples per latent frame. Prefer the configured
        value; otherwise measure it from a one-shot decode (samples / frames)."""
        if self._hop_length:
            return self._hop_length
        if z_frames > 0:
            self._hop_length = max(1, round(sample_count / z_frames))
        return self._hop_length or 256

    @staticmethod
    def _run_encoder(session: onnxruntime.InferenceSession,
                     feed: Dict[str, np.ndarray]) -> tuple:
        """Run the encoder and return ``(z, y_mask)``.

        Two encoder interfaces exist: the pre-split "+RT" voices name their
        outputs ``z`` / ``y_mask``; an auto-split encoder (cut at the already-
        masked ``z * y_mask``) has a single, arbitrarily-named latent output and
        no separate mask. Outputs are therefore matched by name first, then by
        shape (``[B, 192, T]`` latent vs. ``[B, 1, T]`` mask). ``y_mask`` is
        ``None`` when the encoder does not expose one."""
        names = [o.name for o in session.get_outputs()]
        outs = session.run(None, feed)
        by_name = dict(zip(names, outs))
        if "z" in by_name:
            return by_name["z"], by_name.get("y_mask")
        z = None
        y_mask = None
        for arr in outs:
            a = np.asarray(arr)
            if a.ndim == 3 and a.shape[1] == 1 and y_mask is None:
                y_mask = a
            elif a.ndim == 3 and a.shape[1] > 1 and z is None:
                z = a
        return (z if z is not None else np.asarray(outs[0])), y_mask

    def _decode(self, z: np.ndarray, y_mask: Optional[np.ndarray],
                speaker_id: Optional[int]) -> np.ndarray:
        """Feed one latent chunk to the decoder, matching whatever inputs it
        declares (single-tensor auto-split, ``z``+``y_mask`` +RT, or a
        multi-speaker decoder that also wants ``sid``)."""
        feed: Dict[str, np.ndarray] = {}
        for inp in self.decoder.get_inputs():
            shape = inp.shape or []
            second = shape[1] if len(shape) > 1 else None
            name = inp.name
            if name in ("sid", "spk", "spks", "g", "speaker", "speaker_id"):
                if speaker_id is not None:
                    feed[name] = np.array([speaker_id], dtype=np.int64)
            elif name == "y_mask" or (len(shape) == 3 and second == 1):
                feed[name] = (y_mask if y_mask is not None
                              else np.ones((z.shape[0], 1, z.shape[2]), dtype=np.float32))
            else:
                # any other input is the latent: covers a single arbitrarily-named
                # auto-split tensor, "z", and inputs whose shape is unknown.
                feed[name] = z
        out = self.decoder.run(None, feed)[0]
        return np.asarray(out).squeeze()

    def synthesize(self, request: AdapterSynthesisRequest,
                   session: onnxruntime.InferenceSession) -> AdapterSynthesisResult:
        if self.decoder is None:
            raise RuntimeError(
                "VitsStreamingAdapter has no decoder graph. Set "
                "engine_params['decoder_path'] to the decoder .onnx.")

        # 1) Encoder runs once, producing the full latent sequence.
        feed = self.build_feed_dict(request, session)
        z, y_mask = self._run_encoder(session, feed)
        sid = request.speaker_id
        T = z.shape[2]

        M = self._streaming["context_margin"]
        first = self._streaming["first_chunk"]
        nxt = self._streaming["next_chunk"]
        fallback = self._streaming["fallback_frames"]

        # 2) Short sentence: one-shot decode (streaming overhead can't pay off).
        if T <= fallback:
            audio = self._decode(z, y_mask, sid)
            self._resolve_hop(T, len(audio))
            return AdapterSynthesisResult(audio=audio.astype(np.float32))

        # 3) Determine hop from a throwaway measure only if unknown. We avoid an
        #    extra full decode: use the first chunk to establish hop.
        segments: List[np.ndarray] = []
        start = 0
        is_first = True
        hop = self._hop_length or 256
        while start < T:
            size = first if is_first else nxt
            end = min(start + size, T)
            a = max(0, start - M)
            b = min(T, end + M)
            ym_slice = y_mask[:, :, a:b] if y_mask is not None else None
            chunk_audio = self._decode(z[:, :, a:b], ym_slice, sid)
            if is_first and not self._hop_length:
                # first full-context decode lets us pin hop precisely
                hop = self._resolve_hop(b - a, len(chunk_audio))
            lo = (start - a) * hop
            hi = min(lo + (end - start) * hop, len(chunk_audio))
            segments.append(chunk_audio[lo:hi])
            is_first = False
            start = end

        audio = np.concatenate(segments).astype(np.float32)
        return AdapterSynthesisResult(audio=audio)
