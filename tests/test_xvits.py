"""The x-vits adapter: prosody sampling, durations, style checks and the word of each token.

The graphs are stand-ins whose output is known by hand, so every expected value below comes from
arithmetic on the stand-ins' numbers, not from the adapter: a constant velocity moves the prosody by
that velocity over t in [0, 1], a decaying velocity gives the closed form of the midpoint rule, and
the durations are ``ceil(exp(logw) * length_scale)`` of chosen log durations.
"""
import json
import math

import numpy as np
import pytest

from phoonnx.config import Alphabet, Engine, SynthesisConfig, VoiceConfig
from phoonnx.engines import detect_engine, get_adapter, list_engines
from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult
from phoonnx.engines.xvits import (XVitsAdapter, align_words, apply_template, parse_marked, token_words,
                                   tonic_offsets, words_from_ids)
from phoonnx.voice import TTSVoice
from tests.test_prepare_render import CONFIG

STYLES = ["confused", "default", "enunciated", "happy", "laughing", "sad"]
CHANNELS = 4
HOP = 3


class _Recorder:
    """A graph that records its feeds and answers with ``fn(feed)``."""

    def __init__(self, fn):
        self.fn, self.feeds = fn, []

    def run(self, names, feed):
        self.feeds.append(feed)
        return self.fn(feed)


def _runtime(**defaults):
    return {
        "inter_channels": 5,
        "predictor": {"stats": {"lf0_std": 0.5}, "styles": STYLES, "token_feats": 0},
        "durations_styles": STYLES,
        "defaults": {"noise_scale": 0.5, "noise_scale_w": 0.25, "length_scale": 1.5,
                     "pp_temperature": "0,0", "pp_guidance": "1.0,2.0", "style_temperature": {},
                     "template_scale": 1.0, "prominence_threshold": 0.5, "pp_steps": 4, **defaults}}


def _request(length=3, **params):
    ids = np.arange(5, 5 + length, dtype=np.int64)[None]
    return AdapterSynthesisRequest(phoneme_ids=ids, phoneme_lengths=np.array([length], np.int64),
                                   params=params)


def _encoder(length=3):
    return _Recorder(lambda feed: [np.ones((1, CHANNELS, feed["ids"].shape[1]), np.float32),
                                   np.ones((1, 1, feed["ids"].shape[1]), np.float32)])


def _adapter(velocity, log_w, runtime=None, template=None, prominence=None):
    """An adapter whose predictor answers ``velocity(feed)`` and whose duration graph gives *log_w*."""
    adapter = XVitsAdapter()
    adapter.runtime = runtime or _runtime()
    adapter.template = template
    adapter.predictor = _Recorder(lambda feed: [velocity(feed)])
    adapter.durations = _Recorder(lambda feed: [np.array(log_w, np.float32)[None, None, :feed["x"].shape[2]]])
    adapter.acoustic = _Recorder(lambda feed: [np.arange(feed["noise"].shape[2] * HOP, dtype=np.float32)[None, None]])
    adapter.prominence = prominence
    adapter.style_offsets = np.arange(7 * CHANNELS, dtype=np.float32).reshape(7, CHANNELS) / 10
    adapter.symbols = {0: "_", 1: "^", 2: "$"}
    adapter.bos_id, adapter.eos_id = 1, 2
    return adapter


def _constant(per_style):
    """A velocity that does not depend on the state: ``per_style[style]`` on each of the four targets."""
    def velocity(feed):
        row = np.array(per_style[int(feed["style"][0])], np.float32).reshape(1, 4, 1)
        return np.broadcast_to(row, feed["x_t"].shape).copy()
    return velocity


# ----------------------------------------------------------------------------- registration


def test_registered_and_detected():
    assert "xvits" in list_engines()
    assert isinstance(get_adapter("xvits"), XVitsAdapter)
    assert Engine("xvits") is Engine.XVITS
    assert isinstance(detect_engine(config={"engine": "xvits"}), XVitsAdapter)
    assert not XVitsAdapter.detect({"engine": "rapido"})
    assert not XVitsAdapter.detect(None)


def test_detects_the_encoder_graph_by_its_signature():
    class Named:
        def __init__(self, name):
            self.name = name

    class Session:
        def __init__(self, inputs, outputs):
            self.inputs, self.outputs = inputs, outputs

        def get_inputs(self):
            return [Named(n) for n in self.inputs]

        def get_outputs(self):
            return [Named(n) for n in self.outputs]

    assert XVitsAdapter.detect(session=Session(["ids", "lengths"], ["x", "x_mask"]))
    assert not XVitsAdapter.detect(session=Session(["ids"], ["features", "durations"]))
    assert not XVitsAdapter.detect(session=Session(["x", "x_lengths", "scales"], ["mel"]))


# ----------------------------------------------------------------------------- prosody ODE


def test_constant_velocity_moves_the_prosody_by_the_velocity_and_guidance_weights_the_style():
    # style "happy" is index 4: the predictor sees style 4, and style 0 for the unconditioned pass
    free = [0.1, 0.2, 0.3, 0.4]
    happy = [0.5, 0.5, 0.6, 0.1]
    adapter = _adapter(_constant({0: free, 4: happy}), [0.0] * 3)
    y = adapter._sample_prosody(np.ones((1, CHANNELS, 3), np.float32), np.ones((1, 1, 3), np.float32), 4,
                                np.zeros((1, 4, 1), np.float32), [1.0, 2.0, 2.0, 2.0], 5,
                                np.random.default_rng(0))
    # guided velocity = free + w * (happy - free), applied over t in [0, 1]
    expected = [0.5, 0.2 + 2 * (0.5 - 0.2), 0.3 + 2 * (0.6 - 0.3), 0.4 + 2 * (0.1 - 0.4)]
    np.testing.assert_allclose(y[0, :, 0], expected, atol=1e-6)
    assert {int(f["style"][0]) for f in adapter.predictor.feeds} == {0, 4}


def test_no_style_runs_the_predictor_unconditioned_and_unguided():
    adapter = _adapter(_constant({0: [0.1, 0.2, 0.3, 0.4]}), [0.0] * 3)
    y = adapter._sample_prosody(np.ones((1, CHANNELS, 3), np.float32), np.ones((1, 1, 3), np.float32), None,
                                np.zeros((1, 4, 1), np.float32), [1.0, 2.0, 2.0, 2.0], 3,
                                np.random.default_rng(0))
    np.testing.assert_allclose(y[0, :, 0], [0.1, 0.2, 0.3, 0.4], atol=1e-6)
    assert len(adapter.predictor.feeds) == 6  # two midpoint evaluations per step, no second pass


def test_midpoint_rule_on_the_cosine_schedule():
    # v = -y: one midpoint step of size h multiplies y by 1 - h + h^2 / 2. The cosine schedule
    # 1 - cos(pi t / 2) for two steps has times 0, 1 - cos(pi / 4), 1.
    adapter = _adapter(lambda feed: -feed["x_t"], [0.0] * 3)
    noise = np.random.default_rng(42).standard_normal((1, 4, 2))
    y = adapter._sample_prosody(np.ones((1, CHANNELS, 2), np.float32), np.ones((1, 1, 2), np.float32), None,
                                np.full((1, 4, 1), 2.0, np.float32), 1.0, 2, np.random.default_rng(42))
    h1 = 1 - math.cos(math.pi / 4)
    h2 = 1 - h1
    factor = (1 - h1 + h1 ** 2 / 2) * (1 - h2 + h2 ** 2 / 2)
    np.testing.assert_allclose(y, noise * 2.0 * factor, rtol=1e-5)
    assert factor != pytest.approx(math.exp(-1), rel=1e-2)  # not the exact flow, not Euler either


def test_padding_stays_zero_in_the_sampled_prosody():
    adapter = _adapter(_constant({0: [1.0, 1.0, 1.0, 1.0]}), [0.0] * 3)
    mask = np.array([[[1, 1, 0]]], np.float32)
    y = adapter._sample_prosody(np.ones((1, CHANNELS, 3), np.float32), mask, None,
                                np.ones((1, 4, 1), np.float32), 1.0, 2, np.random.default_rng(1))
    assert np.all(y[..., 2] == 0.0)


def test_decode_gates_pitch_by_voicing_and_clips_it():
    y = np.array([[[9.0, 9.0, 9.0], [1.0, 2.0, 3.0], [0.4, 0.6, 1.7], [5.0, 6.0, 7.0]]], np.float32)
    out = XVitsAdapter._decode(y, np.array([[[1, 1, 0]]], np.float32))
    # [lf0 where voicing > 0.5, voicing clipped to [0, 1], energy], masked
    np.testing.assert_allclose(out[0], [[0.0, 2.0, 0.0], [0.4, 0.6, 0.0], [5.0, 6.0, 0.0]])


# ----------------------------------------------------------------------------- whole synthesis


def test_synthesize_runs_the_graphs_with_the_durations_noise_and_style_of_the_request():
    # velocities put the happy prosody at [., 0.8, 0.9, -0.2] after guidance (see the constant test)
    velocity = _constant({0: [0.1, 0.2, 0.3, 0.4], 4: [0.5, 0.5, 0.6, 0.1]})
    adapter = _adapter(velocity, [math.log(0.4), math.log(1.2), math.log(2.5)])
    encoder = _encoder()
    result = adapter.synthesize(_request(style="happy", emphasis="none", seed=9, steps=3), encoder)

    ids = encoder.feeds[0]["ids"]
    assert ids.tolist() == [[5, 6, 7]] and encoder.feeds[0]["lengths"].tolist() == [3]

    # durations: z follows the prosody noise in the generator, scaled by the duration noise (0.25);
    # h_add is the happy row of the style offsets (index 4)
    draws = np.random.default_rng(9)
    draws.standard_normal((1, 4, 3))
    z = (draws.standard_normal((1, 2, 3)) * 0.25).astype(np.float32)
    feed = adapter.durations.feeds[0]
    np.testing.assert_array_equal(feed["z"], z)
    np.testing.assert_allclose(feed["h_add"][0, :, 0], np.arange(16, 20) / 10)

    # ceil([0.4, 1.2, 2.5] * 1.5) = ceil([0.6, 1.8, 3.75]) = [1, 2, 4]
    acoustic = adapter.acoustic.feeds[0]
    assert acoustic["w_ceil"].tolist() == [[[1.0, 2.0, 4.0]]]
    # the latent noise is the third draw, scaled by 0.5, one column per frame
    frames = 7
    noise = (draws.standard_normal((1, 5, frames)) * 0.5).astype(np.float32)
    np.testing.assert_array_equal(acoustic["noise"], noise)
    np.testing.assert_allclose(acoustic["prosody"][0], np.array([[0.8] * 3, [0.9] * 3, [-0.2] * 3]), atol=1e-6)
    assert result.audio.shape == (frames * HOP,) and result.audio.dtype == np.float32
    assert result.extras["phoneme_id_samples"].tolist() == [1.0, 2.0, 4.0]


def test_length_scale_of_the_request_replaces_the_default():
    adapter = _adapter(_constant({0: [0.0] * 4}), [math.log(2.0)] * 3)
    adapter.synthesize(_request(length_scale=2.4), _encoder())
    # ceil(2 * 2.4); the runtime default 1.5 would give ceil(2 * 1.5) = 3
    assert adapter.acoustic.feeds[0]["w_ceil"].tolist() == [[[5.0, 5.0, 5.0]]]


def test_default_style_row_of_the_durations_is_the_unconditioned_row_without_a_style():
    adapter = _adapter(_constant({0: [0.0] * 4}), [0.0] * 3)
    adapter.synthesize(_request(), _encoder())
    np.testing.assert_allclose(adapter.durations.feeds[0]["h_add"][0, :, 0], np.arange(0, 4) / 10)


def test_same_seed_gives_the_same_audio_and_another_seed_another_noise():
    adapter = _adapter(_constant({0: [0.0] * 4}), [0.0] * 3, runtime=_runtime(pp_temperature="1,1"))
    first = adapter.synthesize(_request(seed=1), _encoder())
    adapter.synthesize(_request(seed=1), _encoder())
    adapter.synthesize(_request(seed=2), _encoder())
    n = adapter.acoustic.feeds
    np.testing.assert_array_equal(n[0]["noise"], n[1]["noise"])
    assert not np.array_equal(n[0]["noise"], n[2]["noise"])
    assert first.audio.size == n[0]["noise"].shape[2] * HOP


def test_unknown_style_names_the_valid_ones():
    adapter = _adapter(_constant({0: [0.0] * 4}), [0.0] * 3)
    with pytest.raises(ValueError) as error:
        adapter.synthesize(_request(style="angry"), _encoder())
    message = str(error.value)
    assert "'angry'" in message and all(style in message for style in STYLES)
    assert adapter.predictor.feeds == []


def test_unknown_emphasis_mode_names_the_valid_ones():
    adapter = _adapter(_constant({0: [0.0] * 4}), [0.0] * 3)
    with pytest.raises(ValueError, match="auto.*marked.*none"):
        adapter.synthesize(_request(emphasis="loud"), _encoder())


def test_synthesize_requires_every_graph_and_the_runtime_file():
    with pytest.raises(RuntimeError, match="dp_cond_path"):
        XVitsAdapter().synthesize(_request(), _encoder())


def test_build_feed_dict_and_parse_outputs_are_not_used():
    with pytest.raises(NotImplementedError):
        XVitsAdapter().build_feed_dict(_request(), None)
    with pytest.raises(NotImplementedError):
        XVitsAdapter().parse_outputs([], _request())


# ----------------------------------------------------------------------------- emphasis maths


TEMPLATE = {"pitch": {"A": 2.0, "mu": 0.0, "s_l": 1.0, "s_r": 1.0, "R": 0.5, "k_r": 1.0, "tau": 1.0},
            "dur": {"C": 0.1, "B": 0.2, "mu": 0.0, "s_l": 1.0, "s_r": 1.0},
            "styles": {"happy": {"pitch": {"A": 4.0, "mu": 0.0, "s_l": 1.0, "s_r": 1.0, "R": 0.0,
                                           "k_r": 1.0, "tau": 1.0},
                                 "dur": {"C": 0.0, "B": 0.0, "mu": 0.0, "s_l": 1.0, "s_r": 1.0}}}}


def test_tonic_offsets_count_from_the_stressed_vowel_of_each_flagged_word():
    symbols = {5: "ɹ", 6: "ˈ", 7: "i", 8: "ə", 9: "l", 10: "a", 11: "b", 0: "_"}
    ids = [5, 6, 7, 8, 9, 10, 11]
    # word 0 = "ɹˈiəl", word 1 = "ab" (no stress mark: the first vowel)
    offsets = tonic_offsets(ids, [0, 0, 0, 0, 0, 1, 1], symbols, [True, True])
    assert offsets.tolist() == [-2, -1, 0, 1, 2, 0, 1]
    only_second = tonic_offsets(ids, [0, 0, 0, 0, 0, 1, 1], symbols, [False, True])
    assert np.isnan(only_second[:5]).all() and only_second[5:].tolist() == [0, 1]


def test_template_pitch_and_duration_at_known_offsets():
    offsets = np.array([0.0, 1.0, np.nan], np.float32)
    pitch, stretch = apply_template(offsets, TEMPLATE, 1.0, None, ["a", "a", "a"])
    bump = math.exp(-0.5)
    # pitch = A * bump(k) - R / (1 + exp(-(k - k_r) / tau)); log duration = C + B * bump(k)
    assert pitch.tolist() == pytest.approx([2.0 - 0.5 / (1 + math.exp(1)), 2.0 * bump - 0.5 / 2, 0.0], abs=1e-6)
    assert stretch.tolist() == pytest.approx([math.exp(0.3), math.exp(0.1 + 0.2 * bump), 1.0], abs=1e-6)
    scaled_pitch, scaled_stretch = apply_template(offsets, TEMPLATE, 2.0, None, ["a", "a", "a"])
    assert scaled_pitch[0] == pytest.approx(2 * pitch[0], abs=1e-6)
    assert scaled_stretch[0] == pytest.approx(math.exp(0.6), abs=1e-6)


def test_template_style_overrides_and_obstruents_take_a_fraction_of_the_stretch():
    offsets = np.array([0.0, 0.0, 0.0], np.float32)
    pitch, _ = apply_template(offsets, TEMPLATE, 1.0, "happy", ["a", "a", "a"])
    assert pitch[0] == pytest.approx(4.0, abs=1e-6)
    # "t" is an obstruent: 0.3 of the log duration; its pad takes its class, a pad after a vowel does not
    _, stretch = apply_template(offsets, TEMPLATE, 1.0, None, ["a", "_", "t"])
    assert stretch.tolist() == pytest.approx([math.exp(0.3)] * 2 + [math.exp(0.3 * 0.3)], abs=1e-6)
    _, stretch = apply_template(offsets, TEMPLATE, 1.0, None, ["t", "_", "a"])
    assert stretch.tolist() == pytest.approx([math.exp(0.09)] * 2 + [math.exp(0.3)], abs=1e-6)


def test_marked_word_is_shifted_and_stretched_and_the_others_are_not():
    adapter = _adapter(_constant({0: [0.0, 1.0, 1.0, 0.0]}), [0.0] * 4, template=TEMPLATE)
    adapter.symbols = {5: "a", 6: "b", 7: "a", 8: "b"}
    request = _request(length=4, emphasis="marked", token_words=[0, 0, 1, 1], word_flags=[False, True])
    encoder = _encoder()
    adapter.synthesize(request, encoder)
    # lf0 target 1.0 at voicing 1.0; the flagged word 1 gets pitch / 12 * ln 2 / lf0_std (0.5)
    pitch, stretch = apply_template(np.array([np.nan, np.nan, 0.0, 1.0], np.float32), TEMPLATE, 1.0, None,
                                    ["a", "b", "a", "b"])
    shift = pitch * (math.log(2) / 12) / 0.5
    lf0 = adapter.acoustic.feeds[0]["prosody"][0, 0]
    np.testing.assert_allclose(lf0, 1.0 + shift, atol=1e-6)
    assert lf0[:2].tolist() == [1.0, 1.0]
    # w = exp(0) * stretch, then ceil(w * 1.5)
    assert adapter.acoustic.feeds[0]["w_ceil"][0, 0].tolist() == [2.0, 2.0, math.ceil(stretch[2] * 1.5),
                                                                  math.ceil(stretch[3] * 1.5)]


def test_prominence_model_picks_the_words_when_nothing_is_marked():
    # words: token ids 5,6 -> word 0 (logits -9, -9), 7,8 -> word 1 (logits 0, ln 3): the word
    # probability is the max-logit one: sigmoid(ln 3) = 0.75, so strength (0.75 - 0.5) / 0.5 = 0.5
    logits = np.array([-9.0, -9.0, 0.0, math.log(3)], np.float32)
    prominence = _Recorder(lambda feed: [logits[None]])
    adapter = _adapter(_constant({0: [0.0, 1.0, 1.0, 0.0]}), [0.0] * 4, template=TEMPLATE, prominence=prominence)
    adapter.symbols = {5: "a", 6: "b", 7: "a", 8: "b"}
    adapter.synthesize(_request(length=4, token_words=[0, 0, 1, 1]), _encoder())
    pitch, stretch = apply_template(np.array([np.nan, np.nan, 0.0, 1.0], np.float32), TEMPLATE, 1.0, None,
                                    ["a", "b", "a", "b"])
    lf0 = adapter.acoustic.feeds[0]["prosody"][0, 0]
    np.testing.assert_allclose(lf0, 1.0 + 0.5 * pitch * (math.log(2) / 12) / 0.5, atol=1e-6)
    assert adapter.acoustic.feeds[0]["w_ceil"][0, 0, 2] == math.ceil(math.sqrt(stretch[2]) * 1.5)
    assert len(prominence.feeds) == 1


def test_marked_mode_without_marks_and_none_leave_the_prosody_alone():
    prominence = _Recorder(lambda feed: [np.full((1, 4), 9.0, np.float32)])
    for mode in ("marked", "none"):
        adapter = _adapter(_constant({0: [0.0, 1.0, 1.0, 0.0]}), [0.0] * 4, template=TEMPLATE,
                           prominence=prominence)
        adapter.synthesize(_request(length=4, emphasis=mode, token_words=[0, 0, 1, 1]), _encoder())
        assert adapter.acoustic.feeds[0]["prosody"][0, 0].tolist() == [1.0] * 4
    assert prominence.feeds == []


# ----------------------------------------------------------------------------- marks and words


def test_parse_marked_flags_each_word_of_a_span_and_keeps_punctuation_attached():
    assert parse_marked("You *want* the shark?") == ("You want the shark?", [False, True, False, False])
    assert parse_marked("*really big* house") == ("really big house", [True, True, False])
    assert parse_marked("Oh *no*, it is") == ("Oh no, it is", [False, True, False, False])
    assert parse_marked("no marks here") == ("no marks here", [False, False, False])


def test_parse_marked_keeps_the_whitespace_and_flags_a_word_marked_inside():
    assert parse_marked("one *two*\n\nthree four.") == ("one two\n\nthree four.", [False, True, False, False])
    assert parse_marked("un*believ*able news") == ("unbelievable news", [True, False])


def test_align_words_gives_a_moved_word_its_symbols():
    # "£3 billion" reads "three billion pounds": "pounds" of "£3" moves after "billion"
    alone = [list("θɹipaʊndz"), list("bɪljən")]
    sentence = list("θɹi") + [" "] + list("bɪljən") + [" "] + list("paʊndz")
    assert align_words(sentence, alone) == [0] * 3 + [-1] + [1] * 6 + [-1] + [0] * 6


def test_align_words_follows_merged_and_reduced_words():
    # "in the house": the phonemizer merges "in" and "the" and reduces the vowel of "the"
    alone = [["ɪ", "n"], ["ð", "i"], ["h", "ˈ", "a", "ʊ", "s"]]
    sentence = ["ɪ", "n", "ð", "ə", " ", "h", "ˈ", "a", "ʊ", "s"]
    assert align_words(sentence, alone) == [0, 0, 1, 1, -1, 2, 2, 2, 2, 2]


def test_align_words_gives_a_stress_mark_the_word_of_the_symbol_after_it():
    assert align_words(["ˈ", "a", " ", "ˈ", "b"], [["a"], ["b"]]) == [0, 0, -1, 1, 1]


def test_token_words_follow_the_piper_layout():
    # ^ _ a _ b _ ' ' _ c _ $ : BOS, pad, then each symbol with its pad, then EOS
    ids = [1, 0, 5, 0, 6, 0, 3, 0, 7, 0, 2]
    assert token_words(ids, [5, 6, 3, 7], [0, 0, -1, 1], 1, 2) == [-1, -1, 0, 0, 0, 0, -1, -1, 1, 1, -1]
    with pytest.raises(ValueError):
        token_words([1, 0, 5, 0, 2], [5, 6], [0, 0], 1, 2)


def test_words_from_ids_split_at_the_space_token():
    symbols = {0: "_", 1: "^", 2: "$", 3: " ", 5: "a", 6: "b", 7: "c"}
    ids = [1, 0, 5, 0, 6, 0, 3, 0, 7, 0, 2]
    assert words_from_ids(ids, symbols, 1, 2) == [-1, -1, 0, 0, 0, 0, -1, -1, 1, 1, -1]


class _CapturingAdapter(XVitsAdapter):
    """The real reader, with the engine call replaced by a record of the request."""

    def __init__(self):
        super().__init__()
        self.requests = []

    def synthesize(self, request, session):
        self.requests.append(request)
        return AdapterSynthesisResult(audio=np.zeros(4, np.float32))


def _voice():
    adapter = _CapturingAdapter()
    voice = TTSVoice(config=VoiceConfig.from_dict(dict(CONFIG)), session=object(), adapter=adapter)
    adapter.configure(voice.config)
    return voice, adapter


def test_marks_reach_the_tokens_of_the_marked_word_from_the_pass_that_made_the_ids():
    voice, _ = _voice()
    utterances = list(voice.prepare("so *really* it is. go now", SynthesisConfig()))
    first, second = utterances
    vocabulary = voice.config.tokenizer.vocabulary
    ids = vocabulary.char2idx
    # words: so, really, it, is., go, now ; the voice reads letters, so each word is its letters
    assert first.params["word_flags"] == [False, True, False, False, False, False]
    assert second.params["word_flags"] == first.params["word_flags"]
    spoken = first.params["token_words"]
    assert len(spoken) == len(first.ids)
    by_word = {}
    for token, word in zip(first.ids, spoken):
        by_word.setdefault(word, []).append(token)
    letters = {word: [t for t in tokens if t != ids["_"]] for word, tokens in by_word.items()}
    assert letters[0] == [ids[c] for c in "so"] and letters[1] == [ids[c] for c in "really"]
    assert letters[2] == [ids[c] for c in "it"] and letters[3] == [ids[c] for c in "is"]
    # BOS, EOS, the pad before the first symbol and the spaces belong to no word
    assert spoken[0] == spoken[1] == spoken[-1] == -1
    assert [t for t, w in zip(first.ids, spoken) if w == -1 and t not in (ids["_"], ids["^"], ids["$"])] \
        == [ids[" "]] * 3
    # the second sentence continues the word numbering: go = 4, now = 5
    assert {w for w in second.params["token_words"] if w >= 0} == {4, 5}


def _spelled(voice, utterance):
    """The letters of each word of *utterance*, keyed by word. This voice reads no punctuation."""
    letters = voice.config.tokenizer.vocabulary.idx2char
    words = {}
    for token, word in zip(utterance.ids, utterance.params["token_words"]):
        if word >= 0 and letters[token] != "_":
            words[word] = words.get(word, "") + letters[token]
    return words


def test_a_mark_follows_its_word_when_the_normaliser_moves_it():
    voice, _ = _voice()
    (utterance,) = voice.prepare("a £3 *billion* deal.", SynthesisConfig(extra_params={"emphasis": "marked"}))
    assert utterance.text == "a three billion pounds deal."
    assert utterance.params["word_flags"] == [False, False, True, False]
    assert _spelled(voice, utterance) == {0: "a", 1: "threepounds", 2: "billion", 3: "deal"}


def test_a_mark_keeps_the_paragraph_breaks():
    voice, _ = _voice()
    utterances = list(voice.prepare("one *two*\n\nthree four.", SynthesisConfig()))
    assert [u.text for u in utterances] == ["one two", "three four."]
    assert [_spelled(voice, u) for u in utterances] == [{0: "one", 1: "two"}, {2: "three", 3: "four"}]
    assert utterances[0].params["word_flags"] == [False, True, False, False]


def test_text_without_marks_still_gets_a_word_for_each_token_and_no_flags():
    voice, _ = _voice()
    (utterance,) = list(voice.prepare("hi there", SynthesisConfig()))
    assert utterance.params["word_flags"] is None
    assert {w for w in utterance.params["token_words"] if w >= 0} == {0, 1}


def test_emphasis_none_reads_no_words():
    voice, _ = _voice()
    (utterance,) = list(voice.prepare("hi *there*", SynthesisConfig(extra_params={"emphasis": "none"})))
    assert dict(utterance.params) == {}


def test_phonemic_input_is_left_to_the_voice():
    voice, _ = _voice()
    utterances = list(voice.prepare("hi there", SynthesisConfig(alphabet=Alphabet.UNICODE)))
    assert all(dict(u.params) == {} for u in utterances)


def test_utterance_params_join_the_request_params_of_the_engine_call():
    voice, adapter = _voice()
    list(voice.synthesize("so *really* it is", SynthesisConfig(extra_params={"style": "happy", "seed": 3})))
    request = adapter.requests[0]
    assert request.params["style"] == "happy" and request.params["seed"] == 3
    assert request.params["word_flags"] == [False, True, False, False]
    assert len(request.params["token_words"]) == request.phoneme_ids.shape[1]


# ----------------------------------------------------------------------------- loading


def test_configure_loads_the_graphs_files_and_replaces_the_scales_of_the_piper_config(tmp_path):
    runtime = _runtime()
    (tmp_path / "runtime.json").write_text(json.dumps(runtime))
    (tmp_path / "template.json").write_text(json.dumps(TEMPLATE))
    np.savez(tmp_path / "dp_cond.npz", style=np.ones((7, CHANNELS), np.float32))
    config = VoiceConfig.from_dict(dict(CONFIG), engine_params={
        "runtime_path": str(tmp_path / "runtime.json"), "emphasis_template_path": str(tmp_path / "template.json"),
        "dp_cond_path": str(tmp_path / "dp_cond.npz")})
    adapter = XVitsAdapter()
    adapter.configure(config)
    assert adapter.runtime == runtime and adapter.template == TEMPLATE
    assert adapter.style_offsets.shape == (7, CHANNELS)
    assert (config.noise_scale, config.noise_w_scale, config.length_scale) == (0.5, 0.25, 1.5)
    assert adapter.symbols[config.tokenizer.vocabulary.char2idx["a"]] == "a"
    assert (adapter.bos_id, adapter.eos_id) == (1, 2)
    assert adapter.default_params()["length_scale"] == 1.5 and adapter.default_params()["steps"] == 4
