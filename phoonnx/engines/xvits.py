"""x-vits inference adapter: a Piper/VITS voice with a flow-matching prosody model and word emphasis.

The ODE sampler, the emphasis template and the graph contracts are ported from the MIT-licensed
runtime of x-vits by Flavio De Melo (https://github.com/femelo/x-vits, commit
91a1890dba50a164862755f25c2c2e8defdd78d1: ``src/xvits/runtime.py`` and
``src/piper_train/prosody/emphasis.py``). The voice weights carry their own licence.

Five ONNX graphs per voice (the encoder is the voice's own ``session``)::

    encoder     : ids [1,T], lengths [1] -> x [1,C,T], x_mask [1,1,T]
    predictor   : x_t [1,4,T], mask, cond, t [1], style [1] -> v [1,4,T]   (one OT-CFM step)
    durations   : x, x_mask, z [1,2,T], h_add [1,C,T] -> logw [1,1,T]
    acoustic    : x, x_mask, prosody [1,3,T], w_ceil [1,1,T], noise [1,D,F] -> audio
    prominence  : x, x_mask -> logits [T]                                  (which words to stress)

``engine_params`` name the other files: ``durations_path``, ``acoustic_path``, ``predictor_path``,
``prominence_path``, ``dp_cond_path`` (the per-style duration offsets), ``emphasis_template_path``
and ``runtime_path`` (``runtime.json``: styles and default settings).

Per-call options go in ``SynthesisConfig.extra_params``: ``style``, ``emphasis`` (``auto``,
``marked`` or ``none``), ``emphasis_scale``, ``steps`` and ``seed``. ``*word*`` marks a stressed word.
"""
import difflib
import json
import math
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import onnxruntime

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter
from phoonnx.providers import make_session

N_TARGETS = 4
EMPHASIS_MODES = ("auto", "marked", "none")

_MARK = re.compile(r"\*([^*]+)\*")
_STRESS = ("ˈ", "ˌ")
_IGNORED = frozenset(("ˈ", "ˌ", " "))
_VOWELS = frozenset("aeiouyɑɐɒæəɘɚɛɜɝɞɨɪʊʌɔøœɯɤʏɵɶ")
# sonorants besides vowels that can carry lengthening without distortion
_SONORANTS = frozenset("mnŋɲɴlɫɹrɾɽʀʁjwʋɥʎʟː")
_OBSTRUENT_DURATION = 0.3


def parse_marked(text: str) -> Tuple[str, List[bool]]:
    """``'You *want* the shark?'`` -> ``('You want the shark?', [False, True, False, False])``.

    Only the marks leave the text, so its whitespace and paragraph breaks stay. Each word with a
    marked letter is flagged, so a span of several words flags each of them.
    """
    # the bounds of each span in the plain text: every span before it lost its two marks
    spans = [(match.start() - 2 * k, match.end() - 2 * k - 2) for k, match in enumerate(_MARK.finditer(text))]
    plain = _MARK.sub(r"\1", text)
    return plain, [any(start < word.end() and word.start() < end for start, end in spans)
                   for word in re.finditer(r"\S+", plain)]


def align_words(symbols: Sequence[str], words: Sequence[Sequence[str]]) -> List[int]:
    """The index of the word each of *symbols* belongs to, ``-1`` for spaces.

    *symbols* are the phonemes of a text read as a whole, where the phonemizer merges function
    words (``in the`` -> ``ɪnðə``). *words* are the phonemes of each word read alone. The two
    sequences are aligned without stress marks and spaces, so a merged word still gives its
    symbols to the words they came from.
    """
    reference: List[str] = []
    reference_word: List[int] = []
    for index, word in enumerate(words):
        for symbol in word:
            if symbol not in _IGNORED:
                reference.append(symbol)
                reference_word.append(index)
    kept = [i for i, symbol in enumerate(symbols) if symbol not in _IGNORED]
    hypothesis = [symbols[i] for i in kept]
    word_of = [-1] * len(symbols)
    opcodes = difflib.SequenceMatcher(a=reference, b=hypothesis, autojunk=False).get_opcodes()
    for _, a1, a2, b1, b2 in opcodes:
        for k in range(b2 - b1):
            if a2 > a1:
                source = a1 + min(a2 - a1 - 1, (k * (a2 - a1)) // max(1, b2 - b1))
                word_of[kept[b1 + k]] = reference_word[source]
            elif a1 > 0:
                word_of[kept[b1 + k]] = reference_word[a1 - 1]
    # The normaliser can move a word: "£3 billion" reads "three billion pounds". The symbols that
    # no equal block took are matched once more, so a moved run keeps its word. A run of one
    # symbol is too common to show a move.
    free_a = [i for tag, a1, a2, _, _ in opcodes if tag != "equal" for i in range(a1, a2)]
    free_b = [j for tag, _, _, b1, b2 in opcodes if tag != "equal" for j in range(b1, b2)]
    moved = difflib.SequenceMatcher(a=[reference[i] for i in free_a], b=[hypothesis[j] for j in free_b],
                                    autojunk=False)
    for a, b, size in moved.get_matching_blocks():
        if size > 1:
            for k in range(size):
                word_of[kept[free_b[b + k]]] = reference_word[free_a[a + k]]
    # a stress mark takes the word of the symbol after it
    for i in range(len(symbols) - 2, -1, -1):
        if symbols[i] in _STRESS and word_of[i] < 0:
            word_of[i] = word_of[i + 1]
    return word_of


def token_words(ids: Sequence[int], raw_ids: Sequence[int], raw_words: Sequence[int],
                bos_id: Optional[int], eos_id: Optional[int]) -> List[int]:
    """The word of every token in *ids*; ``-1`` for BOS, EOS and the tokens around spaces.

    *raw_ids* are the ids of the symbols alone and *raw_words* the word of each. A blank takes the
    word of the symbol before it.
    """
    out: List[int] = []
    position = 0
    previous = -1
    for token in ids:
        if token in (bos_id, eos_id):
            previous = -1
        elif position < len(raw_ids) and token == raw_ids[position]:
            previous = raw_words[position]
            position += 1
        out.append(previous if token != eos_id else -1)
    if position != len(raw_ids):
        raise ValueError(f"the ids hold {position} of the {len(raw_ids)} symbol tokens that were read")
    return out


def words_from_ids(ids: Sequence[int], symbols: Mapping[int, str], bos_id: Optional[int],
                   eos_id: Optional[int]) -> List[int]:
    """The word of every token when only the ids are known: a word runs up to the next space token."""
    out: List[int] = []
    word, inside = -1, False
    for token in ids:
        symbol = symbols.get(int(token))
        if token in (bos_id, eos_id) or symbol == " ":
            inside = False
            out.append(-1)
        elif symbol == "_" and not inside:
            out.append(-1)
        else:
            if not inside:
                word, inside = word + 1, True
            out.append(word)
    return out


def tonic_offsets(ids: Sequence[int], words: Sequence[int], symbols: Mapping[int, str],
                  flags: Sequence[bool]) -> np.ndarray:
    """Each token's signed offset from the stressed vowel of its word; NaN outside flagged words.

    The stressed vowel is the first vowel after the primary stress mark, else after a secondary
    mark, else the first vowel of the word.
    """
    tokens = np.asarray(words)
    offsets = np.full(len(ids), np.nan, np.float32)
    for word in sorted(set(tokens[tokens >= 0].tolist())):
        if not flags[word]:
            continue
        where = np.nonzero(tokens == word)[0]
        names = [symbols.get(int(ids[i]), "?") for i in where]
        marks = [k for k, s in enumerate(names) if s == "ˈ"] or [k for k, s in enumerate(names) if s == "ˌ"]
        start = marks[0] if marks else 0
        centre = next((k for k in range(start, len(names)) if names[k] in _VOWELS), None)
        if centre is None:
            centre = next((k for k, s in enumerate(names) if s in _VOWELS), 0)
        offsets[where] = np.arange(len(where)) - centre
    return offsets


def _bump(k: np.ndarray, mu: float, left: float, right: float) -> np.ndarray:
    width = np.where(k < mu, left, right)
    return np.exp(-0.5 * ((k - mu) / np.maximum(width, 1e-3)) ** 2)


def apply_template(offsets: np.ndarray, template: Mapping[str, Any], scale: float,
                   style: Optional[str], symbols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-token (pitch shift in semitones, duration multiplier) of an emphasis template.

    The pitch is an asymmetric bump at the stressed vowel minus a release after it; the log
    duration is a word-wide rise plus a bump. Obstruents take a fraction of the duration change,
    because a stretched closure sounds wrong; a pad takes the class of the symbol before it.
    NaN offsets (not emphasised) give a shift of 0 and a multiplier of 1.
    """
    params = template.get("styles", {}).get(style, template) if style else template
    k = np.nan_to_num(offsets, nan=0.0)
    off = np.isnan(offsets)
    pitch_p, dur_p = params["pitch"], params["dur"]
    bump = _bump(k, pitch_p["mu"], pitch_p["s_l"], pitch_p["s_r"])
    release = pitch_p["R"] / (1 + np.exp(-(k - pitch_p["k_r"]) / max(pitch_p["tau"], 1e-3)))
    pitch = np.where(off, 0.0, scale * (pitch_p["A"] * bump - release)).astype(np.float32)
    log_duration = scale * (dur_p["C"] + dur_p["B"] * _bump(k, dur_p["mu"], dur_p["s_l"], dur_p["s_r"]))
    weight = np.ones(len(symbols), np.float32)
    previous: Optional[str] = None
    for i, symbol in enumerate(symbols):
        kind = symbol if symbol != "_" else previous
        if kind is not None and kind not in _VOWELS and kind not in _SONORANTS and kind not in _STRESS:
            weight[i] = _OBSTRUENT_DURATION
        if symbol != "_":
            previous = symbol
    return pitch, np.where(off, 1.0, np.exp(log_duration * weight)).astype(np.float32)


class XVitsAdapter(BaseOnnxAdapter):
    """Adapter for x-vits: prosody sampled by flow matching, then a VITS acoustic model."""

    READS_TEXT = True

    def __init__(self):
        self.durations = self.acoustic = self.predictor = self.prominence = None
        self.style_offsets: Optional[np.ndarray] = None
        self.template: Optional[Dict[str, Any]] = None
        self.runtime: Optional[Dict[str, Any]] = None
        self.symbols: Dict[int, str] = {}
        self.bos_id: Optional[int] = None
        self.eos_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def configure(self, voice_config: Any) -> None:
        """Load the four other graphs and the voice files named in ``engine_params``.

        The inference block of a Piper config holds the base checkpoint's noise and length
        scales; ``runtime.json`` holds this voice's, so they replace the voice config's.
        """
        ep = voice_config.engine_params or {}
        for attribute, key in (("durations", "durations_path"), ("acoustic", "acoustic_path"),
                               ("predictor", "predictor_path"), ("prominence", "prominence_path")):
            if getattr(self, attribute) is None and ep.get(key):
                setattr(self, attribute, make_session(ep[key], providers=ep.get("providers"), threads=ep.get("threads")))
        if ep.get("dp_cond_path"):
            with np.load(ep["dp_cond_path"]) as archive:
                self.style_offsets = archive["style"].astype(np.float32)
        if ep.get("emphasis_template_path"):
            with open(ep["emphasis_template_path"], encoding="utf-8") as template:
                self.template = json.load(template)
        if ep.get("runtime_path"):
            with open(ep["runtime_path"], encoding="utf-8") as runtime:
                self.runtime = json.load(runtime)
            defaults = self.runtime["defaults"]
            voice_config.noise_scale = defaults["noise_scale"]
            voice_config.noise_w_scale = defaults["noise_scale_w"]
            voice_config.length_scale = defaults["length_scale"]
        vocabulary = voice_config.tokenizer.vocabulary
        self.symbols = vocabulary.idx2char
        self.bos_id, self.eos_id = vocabulary.bos_id, vocabulary.eos_id

    def default_params(self) -> Dict[str, float]:
        defaults = self.runtime["defaults"] if self.runtime else {}
        return {"noise_scale": defaults.get("noise_scale", 0.667),
                "noise_w_scale": defaults.get("noise_scale_w", 0.4),
                "length_scale": defaults.get("length_scale", 1.08),
                "steps": defaults.get("pp_steps", 16),
                "emphasis_scale": defaults.get("template_scale", 1.5)}

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    def read_text(self, text: str, voice: Any, config: Any, stages: Any) -> Iterator[Any]:
        """Read *text* with ``*word*`` marks into utterances that carry each token's word.

        The ids come from one frontend pass over the unmarked text. The word of each symbol is
        found by aligning that pass with the phonemes of each word read alone, so every token
        has a word however the phonemizer merged them, and the marks reach the model as flags.
        """
        from phoonnx.voice import Utterance

        marked = _MARK.search(text) is not None
        plain, flags = parse_marked(text) if marked else (text, [False] * len(text.split()))
        segments = list(voice.frontend.segments(plain, stages))
        words_of: Optional[List[int]] = None
        if config.extra_params.get("emphasis", "auto") != "none":
            alone = [voice.frontend.phonemize_flat(word, stages) for word in plain.split()]
            words_of = align_words([symbol for segment in segments for symbol in segment.phonemes], alone)
        tokenizer = voice.tokenizer
        start = 0
        for segment in segments:
            symbols = list(segment.phonemes)
            ids = tokenizer.tokenize(symbols)
            params: Dict[str, Any] = {}
            if words_of is not None:
                chunk_words = words_of[start:start + len(symbols)]
                params = {"token_words": self._token_words(tokenizer, symbols, chunk_words, ids),
                          "word_flags": flags if marked else None}
            start += len(symbols)
            if ids:
                yield Utterance(text=segment.text, symbols=tuple(symbols), ids=tuple(ids), params=params)

    def _token_words(self, tokenizer: Any, symbols: List[str], words: List[int], ids: List[int]) -> List[int]:
        """The word of each of *ids*, which *tokenizer* made from *symbols*, the word of each of them *words*."""
        raw_ids: List[int] = []
        raw_words: List[int] = []
        for symbol, word in zip(symbols, words):
            encoded = tokenizer.encode([symbol])
            raw_ids += encoded
            raw_words += [word] * len(encoded)
        if raw_ids != tokenizer.encode(symbols):
            raise ValueError("this voice's tokenizer reads symbols in groups, so its tokens "
                             "cannot be matched to words")
        return token_words(ids, raw_ids, raw_words, self.bos_id, self.eos_id)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _velocity(self, x_t: np.ndarray, mask: np.ndarray, cond: np.ndarray, t: float,
                  style: int) -> np.ndarray:
        return self.predictor.run(None, {"x_t": x_t, "mask": mask, "cond": cond,
                                         "t": np.array([t], np.float32),
                                         "style": np.array([style], np.int64)})[0]

    def _sample_prosody(self, x: np.ndarray, mask: np.ndarray, style: Optional[int],
                        temperature: np.ndarray, guidance: Any, steps: int,
                        rng: np.random.Generator) -> np.ndarray:
        """The OT-CFM ODE with a cosine time schedule, a midpoint step and style guidance.

        The velocity of the chosen style is pushed away from the unconditioned one by the
        ``guidance`` weight of each target. Returns the prosody [1, 4, T].
        """
        features = self.runtime["predictor"]["token_feats"]
        cond = x if not features else np.concatenate(
            [x, np.zeros((1, features, x.shape[2]), np.float32)], 1)
        y = rng.standard_normal((1, N_TARGETS, x.shape[2])).astype(np.float32) * temperature * mask
        weight = np.asarray(guidance, np.float32).reshape(1, -1, 1)
        guided = style is not None and bool((weight != 1.0).any())

        def flow(y_: np.ndarray, t: float) -> np.ndarray:
            v = self._velocity(y_, mask, cond, t, style or 0)
            if guided:
                free = self._velocity(y_, mask, cond, t, 0)
                v = free + weight * (v - free)
            return v * mask

        times = 1 - np.cos(0.5 * math.pi * np.linspace(0, 1, steps + 1, dtype=np.float32))
        for i in range(steps):
            t0, t1 = float(times[i]), float(times[i + 1])
            h = t1 - t0
            k1 = flow(y, t0)
            y = (y + h * flow(y + 0.5 * h * k1, t0 + 0.5 * h)) * mask
        return y

    @staticmethod
    def _decode(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """The model's [lf0, voicing, energy] input from the sampled targets."""
        voicing = np.clip(y[:, 2:3], 0.0, 1.0)
        lf0 = y[:, 1:2] * (voicing > 0.5)
        return np.concatenate([lf0, voicing, y[:, 3:4]], 1) * mask

    def _emphasis(self, request: AdapterSynthesisRequest, x: np.ndarray, mask: np.ndarray,
                  style: Optional[str], scale: float, mode: str) -> Tuple[np.ndarray, np.ndarray]:
        """Per-token (pitch shift in semitones, duration multiplier) from the marks or the prominence model."""
        ids = request.phoneme_ids[0]
        neutral = np.zeros(len(ids), np.float32), np.ones(len(ids), np.float32)
        if mode == "none" or self.template is None:
            return neutral
        words = request.params.get("token_words") or words_from_ids(ids, self.symbols, self.bos_id, self.eos_id)
        flags = request.params.get("word_flags")
        strength = np.ones(len(flags)) if flags is not None else np.zeros(0)
        if flags is None:
            if mode == "marked" or self.prominence is None:
                return neutral
            logits = self.prominence.run(None, {"x": x, "x_mask": mask})[0][0]
            word_logit = np.full(max(words) + 1, -1e4, np.float32)
            for token, word in enumerate(words):
                if word >= 0:
                    word_logit[word] = max(word_logit[word], logits[token])
            threshold = self.runtime["defaults"]["prominence_threshold"]
            probability = 1 / (1 + np.exp(-word_logit))
            strength = np.clip((probability - threshold) / max(1e-6, 1 - threshold), 0, 1)
            flags = list(strength > 0)
        offsets = tonic_offsets(ids, words, self.symbols, flags)
        pitch, stretch = apply_template(offsets, self.template, scale, style,
                                        [self.symbols.get(int(i), "?") for i in ids])
        token_strength = np.array([strength[w] if w >= 0 else 0.0 for w in words], np.float32)
        return pitch * token_strength, np.power(stretch, token_strength).astype(np.float32)

    def synthesize(self, request: AdapterSynthesisRequest,
                   session: onnxruntime.InferenceSession) -> AdapterSynthesisResult:
        if self.runtime is None or any(part is None for part in (
                self.durations, self.acoustic, self.predictor, self.style_offsets)):
            raise RuntimeError("x-vits voice missing durations_path / acoustic_path / predictor_path / "
                               "dp_cond_path / runtime_path in engine_params")
        p = request.params
        defaults = self.runtime["defaults"]
        styles = self.runtime["predictor"]["styles"]
        style = p.get("style")
        if style is not None and style not in styles:
            raise ValueError(f"unknown x-vits style {style!r}; the valid styles are {styles}")
        mode = p.get("emphasis", "auto")
        if mode not in EMPHASIS_MODES:
            raise ValueError(f"unknown x-vits emphasis mode {mode!r}; the valid modes are {list(EMPHASIS_MODES)}")
        rng = np.random.default_rng(p.get("seed"))

        ids = np.asarray(request.phoneme_ids, np.int64)
        x, mask = session.run(None, {"ids": ids, "lengths": np.asarray(request.phoneme_lengths, np.int64)})
        style_index = None if style is None else 1 + styles.index(style)
        temperatures = [float(v) for v in defaults["style_temperature"].get(
            style, defaults["pp_temperature"]).split(",")]
        t_dur, t_pros = (temperatures[0], temperatures[0]) if len(temperatures) == 1 else temperatures
        temperature = np.array([t_dur, t_pros, t_pros, t_pros], np.float32).reshape(1, 4, 1)
        weights = [float(v) for v in defaults["pp_guidance"].split(",")]
        guidance = weights[0] if len(weights) == 1 else [weights[0], weights[1], weights[1], weights[1]]
        steps = int(p.get("steps") or defaults["pp_steps"])
        y = self._sample_prosody(x, mask, style_index, temperature, guidance, steps, rng)
        prosody = self._decode(y, mask)

        pitch, stretch = self._emphasis(request, x, mask, style,
                                        float(p.get("emphasis_scale", defaults["template_scale"])), mode)
        shift = pitch * (math.log(2) / 12) / self.runtime["predictor"]["stats"]["lf0_std"]
        prosody[:, 0] = np.where(prosody[:, 1] > 0.5, prosody[:, 0] + shift[None], prosody[:, 0])

        length = ids.shape[1]
        durations_style = self.runtime["durations_styles"]
        offset = 1 + durations_style.index(style) if style in durations_style else 0
        h_add = np.zeros((1, x.shape[1], length), np.float32) + self.style_offsets[offset][None, :, None]
        z = (rng.standard_normal((1, 2, length)) * float(p.get("noise_w_scale", defaults["noise_scale_w"]))
             ).astype(np.float32)
        log_w = self.durations.run(None, {"x": x, "x_mask": mask, "z": z, "h_add": h_add})[0]
        w = np.exp(log_w) * mask * stretch.reshape(1, 1, -1)
        w_ceil = (np.ceil(w * float(p.get("length_scale", defaults["length_scale"]))) * mask).astype(np.float32)
        frames = int(w_ceil.sum())
        noise = (rng.standard_normal((1, self.runtime["inter_channels"], frames))
                 * float(p.get("noise_scale", defaults["noise_scale"]))).astype(np.float32)
        audio = self.acoustic.run(None, {"x": x, "x_mask": mask, "prosody": prosody.astype(np.float32),
                                         "w_ceil": w_ceil, "noise": noise})[0]
        return AdapterSynthesisResult(audio=np.asarray(audio, np.float32).reshape(-1),
                                      extras={"phoneme_id_samples": w_ceil[0, 0]})

    # build_feed_dict / parse_outputs are required by the ABC but unused: synthesize() runs the
    # five graphs itself.
    def build_feed_dict(self, request: AdapterSynthesisRequest,
                        session: onnxruntime.InferenceSession) -> Dict[str, np.ndarray]:
        raise NotImplementedError("x-vits runs five graphs, use synthesize()")

    def parse_outputs(self, outputs: List[np.ndarray], request: AdapterSynthesisRequest,
                      output_names: Optional[List[str]] = None) -> AdapterSynthesisResult:
        raise NotImplementedError("x-vits runs five graphs, use synthesize()")

    @staticmethod
    def detect(config: Optional[Dict[str, Any]] = None,
               session: Optional[onnxruntime.InferenceSession] = None) -> bool:
        if config is not None and config.get("engine") == "xvits":
            return True
        if session is None:
            return False
        inputs = {i.name for i in session.get_inputs()}
        outputs = {o.name for o in session.get_outputs()}
        return inputs == {"ids", "lengths"} and outputs == {"x", "x_mask"}
