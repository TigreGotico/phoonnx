"""Adapter for tiny-voca ONNX models.

tiny-voca (https://huggingface.co/n4ze3m/tiny-voca, Apache-2.0) is a
4.5M-parameter American English model with a source-filter waveform head. It
reads the tokens of the dizione phonemizer, with one stress value per token:

  Inputs:  tokens       (int64, [1, N])  phoneme ids, <bos> and <eos> included
           stress       (int64, [1, N])  0 none, 1 primary, 2 secondary
           voice        (int64, [1])     speaker index
           speed        (float32, [1])   1.0 as predicted; larger is faster
           pitch_shift  (float32, [1])   added to log F0; 0.058 is about a semitone
           pitch_range  (float32, [1])   scales pitch movement around the mean
  Outputs: audio        (float32, [1, samples])  24 kHz
           durations    (int64, [1, N])  frames of 256 samples per token

Three properties of the model shape this adapter. Upstream's
``tiny_voca.infer.speak_text`` handles each one too, in a different way:

* A sentence that ends without a mark breaks off at its last word. The adapter
  adds a full stop token before ``<eos>`` when the last token, closing quotes
  aside, is no mark. Upstream adds "." to the text, so dizione can read it as
  part of an abbreviation ("etc." against "etc").
* One call reads at most 512 tokens. The adapter cuts a longer sentence into
  pieces of at most 500 tokens, after the last clause mark that fits, else
  between words. Upstream cuts the text after every clause mark and phonemizes
  each piece again.
* The audio can peak above 1.0. The adapter scales each sentence down to a
  peak of 0.99. phoonnx levels each sentence afterwards when
  ``normalize_audio`` is on, so the limit matters only with it off.

Durations come back for the request's tokens. The frames of an added full
stop go to ``<eos>``. A cut sentence returns no durations.
"""
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime

from phoonnx.engines.base import (
    AdapterSynthesisRequest,
    AdapterSynthesisResult,
    BaseOnnxAdapter,
)

PEAK = 0.99
MARKS = (".", "!", "?", "…", ",", ";", ":")
CLOSING = ('"', "'", ")")
CLAUSE_MARKS = (",", ";", ":", "—")


class TinyVocaAdapter(BaseOnnxAdapter):
    """Adapter for tiny-voca ONNX models."""

    DURATION_OUTPUT_NAMES = ["durations"]
    # upstream's limit for one call: max_tokens (512) less a margin of 12
    max_input_tokens = 500

    def configure(self, voice_config: Any) -> None:
        vocab = voice_config.tokenizer.vocabulary.char2idx
        self.bos, self.eos, self.word_gap, self.full_stop = (
            vocab["<bos>"], vocab["<eos>"], vocab["<sp>"], vocab["."])
        self.marks = {vocab[m] for m in MARKS}
        self.closing = {vocab[c] for c in CLOSING}
        self.clause_marks = {vocab[m] for m in CLAUSE_MARKS}

    def synthesize(
        self,
        request: AdapterSynthesisRequest,
        session: onnxruntime.InferenceSession,
    ) -> AdapterSynthesisResult:
        ids = request.phoneme_ids[0].tolist()
        stress = request.stress[0].tolist() if request.stress is not None else [0] * len(ids)
        finished = self._finished(ids, stress)
        result = super().synthesize(self._request(request, *finished), session)
        audio = result.audio
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > PEAK:
            audio = audio * (PEAK / peak)
        extras = dict(result.extras)
        durations = extras.get("phoneme_id_samples")
        if durations is not None and len(finished[0]) > len(ids):
            # the added full stop is a pause before <eos>
            durations = np.asarray(durations).copy()
            durations[-1] += durations[-2]
            extras["phoneme_id_samples"] = np.delete(durations, -2)
        return AdapterSynthesisResult(audio=audio.astype(np.float32), extras=extras)

    def _finished(self, ids: List[int], stress: List[int]) -> Tuple[List[int], List[int]]:
        """*ids* with a full stop before ``<eos>`` when the sentence ends without a mark."""
        if len(ids) < 3 or ids[-1] != self.eos:
            return ids, stress
        last = len(ids) - 2
        while last > 0 and ids[last] in self.closing:
            last -= 1
        if ids[last] in self.marks:
            return ids, stress
        return ids[:-1] + [self.full_stop, self.eos], stress[:-1] + [0, 0]

    @staticmethod
    def _request(request: AdapterSynthesisRequest, ids: List[int],
                 stress: List[int]) -> AdapterSynthesisRequest:
        return replace(request,
                       phoneme_ids=np.array([ids], dtype=np.int64),
                       phoneme_lengths=np.array([len(ids)], dtype=np.int64),
                       stress=np.array([stress], dtype=np.int64))

    def build_feed_dict(
        self,
        request: AdapterSynthesisRequest,
        session: onnxruntime.InferenceSession,
    ) -> Dict[str, np.ndarray]:
        params = request.params
        stress = request.stress if request.stress is not None else np.zeros_like(request.phoneme_ids)
        args = {
            "tokens": request.phoneme_ids,
            "stress": stress,
            "voice": np.array([int(request.speaker_id or 0)], dtype=np.int64),
            "speed": np.array([1.0 / float(params.get("length_scale", 1.0))], dtype=np.float32),
            "pitch_shift": np.array([float(params.get("pitch_shift", 0.0))], dtype=np.float32),
            "pitch_range": np.array([float(params.get("pitch_range", 1.0))], dtype=np.float32),
        }
        return self._filter_inputs(args, session)

    def parse_outputs(
        self,
        outputs: List[np.ndarray],
        request: AdapterSynthesisRequest,
        output_names: Optional[List[str]] = None,
    ) -> AdapterSynthesisResult:
        extras = {}
        durations = self._find_duration_output(outputs, output_names)
        if durations is not None:
            extras["phoneme_id_samples"] = np.asarray(durations).squeeze()
        return AdapterSynthesisResult(audio=np.asarray(outputs[0]).squeeze(), extras=extras)

    def default_params(self) -> Dict[str, float]:
        return {"length_scale": 1.0, "pitch_shift": 0.0, "pitch_range": 1.0}

    def param_labels(self) -> Dict[str, str]:
        return {"length_scale": "Length (1 / speed)",
                "pitch_shift": "Pitch shift (log F0)",
                "pitch_range": "Pitch range"}

    @staticmethod
    def detect(
        config: Optional[Dict[str, Any]] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> bool:
        if config is not None and config.get("engine") == "tinyvoca":
            return True
        if session is not None:
            inputs = {inp.name for inp in session.get_inputs()}
            return {"tokens", "stress", "voice", "pitch_range"} <= inputs
        return False
