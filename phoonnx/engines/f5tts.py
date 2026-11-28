"""F5-TTS / Habibi-TTS inference adapter — flow-matching DiT with Euler ODE.

F5-TTS (`SWivid/F5-TTS <https://github.com/SWivid/F5-TTS>`_) is a
diffusion-transformer (DiT) TTS model using **flow matching** with Euler ODE
integration.  It is the architecture behind Habibi-TTS (Arabic dialectal) and
several community fine-tunes.

The ONNX export (e.g. `DakeQQ/F5-TTS-ONNX
<https://github.com/DakeQQ/F5-TTS-ONNX>`_) produces three graphs:

  preprocess : ref_audio + text_ids + max_duration
               → noise, rope_cos/sin, cat_mel_text, ref_signal_len
  transformer: noise, rope, cat_mel_text, time_step
               → denoised, time_step           (run in a loop, NFE steps)
  decode     : denoised, ref_signal_len
               → waveform                      (Vocos + ISTFT baked in)

The exporter's later runtime contract (``Inference_F5_TTS_ONNX.py``) changes the
graph signatures and binds the tensors by name::

  preprocess : audio, text_ids, max_duration
               → noise, rope_cos, rope_sin, cat_mel_text, cat_mel_text_drop,
                 ref_signal_len, rms_scale, ref_mel_tail
  transformer: noise, rope_cos, rope_sin, cat_mel_text, cat_mel_text_drop, time_step
               → denoised                      (time_step = 0 .. NFE-1, fed by the caller)
  decode     : denoised, ref_signal_len, rms_scale, ref_mel_tail
               → output_audio

A transformer with six inputs selects this contract.

The step count is fixed at export time on both contracts. The name-bound export
stores it in its metadata as ``nfe_step``: the adapter then runs exactly that many
steps and logs a warning when a call asks for another ``nfe``. On a graph without
that key, and on the older contract, ``nfe`` comes from ``engine_params`` or from
``SynthesisConfig.extra`` (default 32) and must equal the export's value.
The voice config's ``inference`` section does not set it.

This adapter overrides :meth:`synthesize` to drive the iterative ODE loop
across the three graphs.  The voice's primary model is the ``transformer``;
the ``preprocess`` and ``decode`` graphs, plus an optional standalone Vocos
vocoder, are loaded in :meth:`configure` from ``engine_params``.

**engine_params** keys (set in the voice config JSON)::

    "engine_params": {
        "preprocess_path": "F5_Preprocess.onnx",
        "decode_path":     "F5_Decode.onnx",         // optional if vocoder provided
        "vocoder_path":    "vocos-mel-24khz.onnx",   // optional, prefer over decode
        "vocoder_type":    "vocos",
        "nfe":             32,
        "cfg_strength":    2.0,
        "sway_coefficient": -1.0,
        "target_rms":      0.15,
        "sample_rate":     24000,
        "speed":           1.0,
        "dialect":         "UNK"    // Habibi Unified only: dialect control tag
    }

The adapter also works for **Habibi-TTS** and any other model built on the
F5-TTS architecture (same ONNX layout, different weights).
"""
import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import onnxruntime

from phoonnx.audio_fx import resample_rate
from phoonnx.engines.base import (
    AdapterSynthesisRequest,
    AdapterSynthesisResult,
    BaseOnnxAdapter,
)
from phoonnx.providers import make_session

LOG = logging.getLogger(__name__)


# Habibi-TTS dialect control tokens (Unified model only). Upstream wraps the
# generation text as f"{dialect_char}〈{text}〉" (habibi_tts/model/utils.py);
# the specialized per-dialect models take plain untagged text. Only the 8
# dialects with training data (UNK/MSA/SAU/UAE/ALG/IRQ/EGY/MAR) are meaningful;
# the remaining codes exist in the vocab but were not trained.
HABIBI_DIALECT_MAP = {
    "UNK": "⓪", "MSA": "①", "SAU": "②", "UAE": "③", "ALG": "④",
    "IRQ": "⑤", "EGY": "⑥", "MAR": "⑦", "OMN": "⑧", "TUN": "⑨",
    "LEV": "⑩", "SDN": "⑪", "LBY": "⑫",
}
_TAG_OPEN = "〈"
_TAG_CLOSE = "〉"


class F5TTSAdapter(BaseOnnxAdapter):
    """Adapter for F5-TTS / Habibi-TTS (flow-matching DiT, Euler ODE)."""

    SAMPLE_RATE = 24000

    def __init__(self):
        self.preprocess: Optional[onnxruntime.InferenceSession] = None
        self.decode: Optional[onnxruntime.InferenceSession] = None
        self.vocoder = None
        self._nfe: int = 32
        self._cfg_strength: float = 2.0
        self._sway_coefficient: float = -1.0
        self._target_rms: float = 0.15
        self._speed: float = 1.0
        self._dialect: Optional[str] = None
        self._char2idx: Dict[str, int] = {}
        self._idx2char: Dict[int, str] = {}
        self._text_tokens: Optional[Callable[[str], List[str]]] = None

    def default_params(self) -> Dict[str, float]:
        return {
            "nfe": float(self._nfe),
            "cfg_strength": self._cfg_strength,
            "sway_coefficient": self._sway_coefficient,
            "target_rms": self._target_rms,
            "speed": self._speed,
        }

    def length_scale_params(self, length_scale: float) -> Dict[str, float]:
        """F5-TTS's ``speed`` for a VITS-style ``length_scale``.

        The generated length is the reference's frames per character times the text's
        characters, divided by ``speed``, so it is a rate: twice as slow is half the speed.
        Without this, ``<prosody rate>``, the pace of ``<emphasis>`` and a ``length_scale``
        in a call or in the voice config never reached the model. A ``speed`` in a call's
        ``extra`` wins. A ``speed`` in the voice's ``engine_params`` (the published voices
        write 1.0) is the voice's normal pace: a call's rate scales it, the voice config's
        own ``length_scale`` does not (``TTSVoice``).
        """
        return {"speed": self._speed / length_scale}

    def configure(self, voice_config: Any) -> None:
        """Load auxiliary ONNX graphs and optional vocoder from engine_params."""
        ep = voice_config.engine_params
        sess = lambda p: make_session(p, providers=ep.get("providers"), threads=ep.get("threads"))

        if self.preprocess is None and ep.get("preprocess_path"):
            self.preprocess = sess(ep["preprocess_path"])

        if self.decode is None and ep.get("decode_path"):
            self.decode = sess(ep["decode_path"])

        if self.vocoder is None and ep.get("vocoder_path"):
            from phoonnx.engines.vocoders import build_vocoder
            self.vocoder = build_vocoder(
                ep["vocoder_path"], ep.get("vocoder_type"), ep
            )

        if ep.get("sample_rate"):
            self.SAMPLE_RATE = int(ep["sample_rate"])
        if "nfe" in ep:
            self._nfe = int(ep["nfe"])
        if "cfg_strength" in ep:
            self._cfg_strength = float(ep["cfg_strength"])
        if "sway_coefficient" in ep:
            self._sway_coefficient = float(ep["sway_coefficient"])
        if "target_rms" in ep:
            self._target_rms = float(ep["target_rms"])
        if "speed" in ep:
            self._speed = float(ep["speed"])
        if "dialect" in ep:
            self._dialect = str(ep["dialect"]).upper() or None

        # char->id map for the Habibi dialect control tokens (Unified model).
        # The tag must go through the same mapping as regular text.
        tokenizer = getattr(voice_config, "tokenizer", None)
        vocabulary = getattr(tokenizer, "vocabulary", None)
        self._char2idx = dict(getattr(vocabulary, "char2idx", None) or {})
        self._idx2char = {i: c for c, i in self._char2idx.items()}

        # The pinyin text function adds a space between the reference and the
        # generation text in some cases (see _join_space).
        self._text_tokens = None
        if voice_config.phoneme_type == "f5tts_pinyin":
            from scriptconv.phonemizers.f5tts import F5TTSPinyinPhonemizer
            self._text_tokens = F5TTSPinyinPhonemizer().text_tokens

    def synthesize(
        self,
        request: AdapterSynthesisRequest,
        session: onnxruntime.InferenceSession,
    ) -> AdapterSynthesisResult:
        """Run the full F5-TTS pipeline: preprocess → ODE loop → decode."""
        if self.preprocess is None:
            raise RuntimeError(
                "F5-TTS voice missing 'preprocess_path' in engine_params"
            )
        p = request.params
        ref_audio = p.get("reference_audio")
        # TTSVoice delivers the reference transcription token IDs as
        # "prompt_tokens" (the shared in-context cloning key, same as ZipVoice);
        # "ref_text_tokens" is accepted as an adapter-level alias.
        ref_text_tokens = p.get("prompt_tokens", p.get("ref_text_tokens"))
        if ref_audio is None or ref_text_tokens is None:
            raise RuntimeError(
                "F5-TTS needs a reference clip + transcription "
                "(params 'reference_audio' and 'prompt_tokens')"
            )

        speed = float(p.get("speed", self._speed))
        nfe = int(p.get("nfe", self._nfe))
        # cfg_strength / sway_coefficient are baked into the DakeQQ transformer
        # export (the CFG double-pass and the sway-sampled time schedule are
        # inside the graph) — they are kept as config params for documentation
        # and for future exports that expose them as inputs, but the current
        # graphs take no such tensors at runtime.

        # ref_audio: (audio_array, sample_rate) tuple
        ref_audio_arr = np.asarray(ref_audio[0], np.float32).reshape(1, 1, -1)
        if ref_audio[1] != self.SAMPLE_RATE:
            ref_audio_arr = np.asarray(
                resample_rate(ref_audio_arr.reshape(-1), ref_audio[1], self.SAMPLE_RATE),
                np.float32,
            ).reshape(1, 1, -1)

        gen_text_ids = request.phoneme_ids.astype(np.int32).reshape(1, -1)
        ref_text_ids = self._reference_ids(ref_text_tokens, p.get("speaker_reference_text"),
                                           gen_text_ids[0].tolist())
        text_ids = self._wrap_dialect(np.concatenate([ref_text_ids, gen_text_ids], axis=1),
                                      p.get("dialect", self._dialect))

        # max_duration: estimated from ref audio length and text length ratio
        ref_audio_len = ref_audio_arr.shape[-1]
        ref_text_len = ref_text_ids.shape[1]
        gen_text_len = gen_text_ids.shape[1]
        ref_audio_frames = ref_audio_len // 256 + 1  # HOP_LENGTH = 256
        max_duration = np.array(
            [ref_audio_frames + int(ref_audio_frames / max(ref_text_len, 1) * gen_text_len / speed)],
            dtype=np.int64,
        )

        # --- Preprocess ---
        # DakeQQ's default export takes the reference audio as int16 PCM (the
        # graph rescales internally); some exports use float32 directly. Feed
        # whichever dtype the graph actually declares.
        audio_input_type = self.preprocess.get_inputs()[0].type
        if "int16" in audio_input_type:
            audio_feed = np.clip(ref_audio_arr * 32767.0, -32768, 32767).astype(np.int16)
        else:
            audio_feed = ref_audio_arr.astype(np.float32)

        pre_out = self.preprocess.run(None, {
            "audio": audio_feed,
            "text_ids": text_ids,
            "max_duration": max_duration,
        })
        # outputs: noise, rope_cos_q, rope_sin_q, rope_cos_k, rope_sin_k,
        #          cat_mel_text, cat_mel_text_drop, ref_signal_len
        transformer_inputs = [inp.name for inp in session.get_inputs()]
        if len(transformer_inputs) == len(self._BY_NAME_TRANSFORMER_INPUTS):
            return self._synthesize_by_name(session, pre_out, transformer_inputs, nfe)
        noise = pre_out[0]
        rope_cos_q = pre_out[1]
        rope_sin_q = pre_out[2]
        rope_cos_k = pre_out[3]
        rope_sin_k = pre_out[4]
        cat_mel_text = pre_out[5]
        cat_mel_text_drop = pre_out[6]
        ref_signal_len = pre_out[7]

        # --- Flow-matching ODE loop (Euler method) ---
        # The transformer graph bakes one Euler step per call and advances its own
        # internal time_step counter; matching the DakeQQ reference inference script,
        # the loop runs (nfe - 1) times (the first denoising step happens inside the
        # graph itself on the initial time_step=0 call already folded into "nfe" steps
        # total, so nfe-1 further calls complete the schedule).
        # torch.onnx.export can rename an input that shares a name with an output
        # (e.g. "time_step" -> "time_step.1") — feed by declared input order rather
        # than by hardcoded name so the adapter survives that renaming across
        # export runs / PyTorch versions.
        time_step = np.array([0], dtype=np.int32)
        for _ in range(max(nfe - 1, 0)):
            values = [noise, rope_cos_q, rope_sin_q, rope_cos_k, rope_sin_k,
                      cat_mel_text, cat_mel_text_drop, time_step]
            noise, time_step = session.run(None, dict(zip(transformer_inputs, values)))

        # --- Decode ---
        if self.vocoder is not None:
            # mel path: strip ref prefix, transpose to [1, n_mels, T], vocoder
            ref_len = int(ref_signal_len) if np.ndim(ref_signal_len) == 0 else int(ref_signal_len.reshape(-1)[0])
            mel = noise[:, ref_len:].transpose(0, 2, 1).astype(np.float32)
            audio = np.asarray(self.vocoder.mel_to_audio(mel), np.float32).reshape(-1)
        elif self.decode is not None:
            audio = self.decode.run(None, {
                "denoised": noise,
                "ref_signal_len": ref_signal_len,
            })[0]
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            audio = audio.reshape(-1)
        else:
            raise RuntimeError(
                "F5-TTS requires either 'decode_path' or 'vocoder_path' "
                "in engine_params"
            )

        return AdapterSynthesisResult(audio=audio)

    #: transformer inputs of the name-bound contract; the state is the first one
    _BY_NAME_TRANSFORMER_INPUTS = ("noise", "rope_cos", "rope_sin", "cat_mel_text",
                                   "cat_mel_text_drop", "time_step")

    def _synthesize_by_name(self, session: onnxruntime.InferenceSession,
                            pre_out: List[np.ndarray], transformer_inputs: List[str],
                            nfe: int) -> AdapterSynthesisResult:
        """ODE loop and decode for the name-bound export contract.

        As in the exporter's own runner, every transformer and decode input that the
        preprocess graph outputs is fed under its name, the one transformer input it
        does not output is the step index (0 .. nfe-1), and the denoised state goes
        back into the transformer's first input and, at the end, into the one decode
        input that preprocess does not output.
        """
        if self.decode is None:
            raise RuntimeError("F5-TTS requires 'decode_path' in engine_params")
        export_nfe = session.get_modelmeta().custom_metadata_map.get("nfe_step")
        if export_nfe is not None:
            if int(export_nfe) != nfe:
                LOG.warning("F5-TTS export runs a fixed %s steps; ignoring nfe=%s",
                            export_nfe, nfe)
            nfe = int(export_nfe)
        pre = dict(zip((o.name for o in self.preprocess.get_outputs()), pre_out))
        step_input, = (i for i in session.get_inputs() if i.name not in pre)
        step_dtype = np.int64 if "int64" in step_input.type else np.int32
        state_name = transformer_inputs[0]
        feed = {name: pre[name] for name in transformer_inputs if name != step_input.name}
        for step in range(nfe):
            feed[step_input.name] = np.array([step], dtype=step_dtype)
            feed[state_name] = session.run(None, feed)[0]
        decode_feed = {i.name: pre.get(i.name, feed[state_name])
                       for i in self.decode.get_inputs()}
        audio = np.asarray(self.decode.run(None, decode_feed)[0])
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        return AdapterSynthesisResult(audio=audio.astype(np.float32).reshape(-1))

    # --- ABC stubs (unused — synthesize() drives the loop) ---

    def build_feed_dict(self, request, session):
        raise NotImplementedError("F5-TTS is iterative — use synthesize()")

    def parse_outputs(self, outputs, request, output_names=None):
        raise NotImplementedError("F5-TTS is iterative — use synthesize()")

    @staticmethod
    def detect(
        config: Optional[Dict[str, Any]] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> bool:
        if config is not None:
            if config.get("engine") in ("f5tts", "f5-tts", "habibi"):
                return True
            ep = config.get("engine_params") or {}
            if ep.get("preprocess_path") and ep.get("nfe") is not None:
                return True
        if session is not None:
            inputs = {inp.name for inp in session.get_inputs()}
            if {"noise", "rope_cos_q", "cat_mel_text", "time_step"}.issubset(inputs):
                return True
        return False

    def _reference_ids(self, ref_text_tokens: Any, reference_text: Optional[str],
                       gen_text_ids: List[int]) -> np.ndarray:
        """The ids of the reference text, with the end that the upstream code gives it.

        Upstream, ``preprocess_ref_audio_text`` ends the reference text in
        ``". "`` unless it ends in ``". "`` or ``"。"``, and
        ``infer_batch_process`` adds one more space when its last character is
        one byte (``src/f5_tts/infer/utils_infer.py``, SWivid/F5-TTS 600e5fd;
        Habibi-TTS and SILMA TTS do the same). The model then reads the
        reference text and the generation text as one string. The end is
        computed on *reference_text* without the whitespace at its end. The
        frontend keeps some of that whitespace as ids (a space, or a tab in
        the vocabularies that hold one) and drops the rest, so the whitespace
        ids at the end of the ids go first: the result is
        the same for a reference text with or without whitespace at its end.
        With no *reference_text*, the ids stay as given.
        """
        ids = list(ref_text_tokens)
        if reference_text:
            text = reference_text.rstrip()
            trailing = len(reference_text) - len(text)
            while trailing and ids and self._idx2char.get(ids[-1], "").isspace():
                ids.pop()
                trailing -= 1
            if text.endswith("。"):
                end = self._join_space(gen_text_ids)
            elif text.endswith("."):
                end = "  "
            else:
                end = ".  "
            ids += [self._char2idx.get(char, 0) for char in end]
        return np.asarray(ids, np.int32).reshape(1, -1)

    def _join_space(self, gen_text_ids: List[int]) -> str:
        """The space that the pinyin text function puts between a reference text that ends in ``"。"`` and the generation text.

        Upstream runs ``convert_char_to_pinyin`` on ``ref_text + gen_text``. A
        segment of two or more ASCII characters there follows a space unless
        the token before it is a space, ``:``, ``'`` or ``"``. Every other
        reference end is a space, so only a ``"。"`` end and an ASCII word at
        the start of the generation text give this space. The ASCII
        characters at the start of the generation text are read back from
        its ids, and the voice's own text function runs on ``"。"`` and those
        characters. The voices that read each character as one token get no
        space. The ids hold no hanzi, so a jieba dictionary word that joins
        those characters to the hanzi after them, such as ``IC卡``, still
        gets the space here, and upstream gives none.
        """
        if self._text_tokens is None:
            return ""
        start = ""
        for token_id in gen_text_ids:
            char = self._idx2char.get(token_id, "")
            if len(char) != 1 or not char.isascii() or char.isspace():
                break
            start += char
        joined = self._text_tokens("。" + start)
        return "".join(joined[1:len(joined) - len(self._text_tokens(start))])

    def _wrap_dialect(self, text_ids: np.ndarray,
                      dialect: Optional[str]) -> np.ndarray:
        """Wrap the text ids of one call, reference and generation text, in the
        Habibi dialect control tokens: ``{dialect_char}〈 ... 〉`` (Unified
        model only). Upstream wraps the whole ``ref_text + gen_text`` string
        (``text_list_formatter`` in ``src/habibi_tts/model/utils.py``,
        SWivid/Habibi-TTS 9274498).

        No-op when *dialect* is unset. Raises ``ValueError`` for an unknown
        dialect name. If the voice's vocab lacks the control tokens (the
        specialized per-dialect models, plain F5-TTS), the tag is skipped with
        a warning — those models were trained on untagged text.
        """
        if not dialect:
            return text_ids
        name = str(dialect).upper()
        if name not in HABIBI_DIALECT_MAP:
            raise ValueError(
                f"Unknown Habibi dialect {dialect!r}. "
                f"Valid values: {sorted(HABIBI_DIALECT_MAP)}"
            )
        chars = (HABIBI_DIALECT_MAP[name], _TAG_OPEN, _TAG_CLOSE)
        if any(c not in self._char2idx for c in chars):
            import logging
            logging.getLogger(__name__).warning(
                "Voice vocab lacks the Habibi dialect control tokens "
                "(%s) — 'dialect' only applies to the Unified model; "
                "synthesizing untagged text.", "".join(chars))
            return text_ids
        prefix = np.array([[self._char2idx[chars[0]], self._char2idx[chars[1]]]],
                          dtype=text_ids.dtype)
        suffix = np.array([[self._char2idx[chars[2]]]], dtype=text_ids.dtype)
        return np.concatenate([prefix, text_ids, suffix], axis=1)

