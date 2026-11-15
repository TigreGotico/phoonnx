import numpy as np
import pytest

from phoonnx.config import Engine, VoiceConfig
from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.rapido import (RapidoAdapter, frame_layout, layout_boundary_ids, ode_times,
                                    word_frame_counts, word_layout)

# "<pad>"=0, " "=1, "a"=2, "b"=3, "!"=4, "^"=5, "$"=6
ID_MAP = {"<pad>": 0, " ": 1, "a": 2, "b": 3, "!": 4, "^": 5, "$": 6}


def _req(ids, speaker=0, **params):
    ids = np.array([ids], np.int64)
    return AdapterSynthesisRequest(phoneme_ids=ids, phoneme_lengths=np.array([ids.shape[1]], np.int64),
                                   speaker_id=speaker, language_id=0, params=params)


def test_rapido_registered():
    from phoonnx.engines import get_adapter, list_engines
    assert "rapido" in list_engines()
    assert isinstance(get_adapter("rapido"), RapidoAdapter)
    assert Engine("rapido") is Engine.RAPIDO


def test_rapido_detect():
    from phoonnx.engines import detect_engine
    assert RapidoAdapter.detect({"engine": "rapido"})
    assert not RapidoAdapter.detect({"engine": "zipvoice"})
    assert not RapidoAdapter.detect(None)
    assert isinstance(detect_engine(config={"engine": "rapido"}), RapidoAdapter)


def test_boundary_ids_are_separator_clause_marks_and_bos_eos_present_in_the_map():
    assert layout_boundary_ids(ID_MAP) == {1, 4, 5, 6}
    assert layout_boundary_ids(ID_MAP, bos=None, eos=None) == {1, 4}
    # piper-style list-valued ids; tokens absent from the map are skipped
    assert layout_boundary_ids({" ": [7], "?": [9], "x": [3]}) == {7, 9}


def test_word_layout_word_owns_its_trailing_boundaries():
    # "ab!a b": "ab!" | "a " | "b"
    word, start = word_layout([2, 3, 4, 2, 1, 3], {1, 4})
    assert word.tolist() == [0, 0, 0, 1, 1, 2]
    assert start.tolist() == [0, 0, 0, 3, 3, 5]


def test_word_layout_mark_without_a_following_separator_ends_the_word():
    # "ab!ba": the phonemizer writes no space after "!"
    word, start = word_layout([2, 3, 4, 3, 2], {1, 4})
    assert word.tolist() == [0, 0, 0, 1, 1]
    assert start.tolist() == [0, 0, 0, 3, 3]


def test_word_layout_leading_boundaries_join_the_first_word():
    # "^ ab b$" with BOS/EOS as boundaries
    word, start = word_layout([5, 1, 2, 3, 1, 3, 6], {1, 5, 6})
    assert word.tolist() == [0, 0, 0, 0, 0, 1, 1]
    assert start.tolist() == [0, 0, 0, 0, 0, 5, 5]


def test_word_layout_matches_space_only_rule_on_letter_text():
    # "olá tudo" with only the space as boundary: two words, the first owns the space
    word, start = word_layout([9, 9, 9, 1, 8, 8, 8, 8], {1})
    assert word.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert start.tolist() == [0, 0, 0, 0, 4, 4, 4, 4]


def test_word_layout_all_boundaries_is_one_word():
    word, start = word_layout([1, 4, 1], {1, 4})
    assert word.tolist() == [0, 0, 0] and start.tolist() == [0, 0, 0]


def test_frame_layout_positions_inside_each_word():
    frame_word, frame_pos = frame_layout([2, 3])
    assert frame_word.tolist() == [0, 0, 1, 1, 1]
    assert frame_word.dtype == np.int64 and frame_pos.dtype == np.float32
    np.testing.assert_array_equal(frame_pos, np.array([0, 1 / 2, 0, 1 / 3, 2 / 3], np.float32))


def test_word_frame_counts_round_and_clamp():
    durations = np.array([1.0, 1.2, 0.4, 0.1, 0.2, 400.0], np.float32)
    word = np.array([0, 0, 0, 1, 1, 2])
    # 2.6 -> 3, 0.3 -> at least 1, 400 -> at most 250
    assert word_frame_counts(durations, word).tolist() == [3, 1, 250]
    # round half to even, as torch.round
    assert word_frame_counts(np.array([2.5, 3.5], np.float32), np.array([0, 1])).tolist() == [2, 4]


def test_ode_times():
    np.testing.assert_array_equal(ode_times(4), np.array([0, 0.25, 0.5, 0.75, 1], np.float32))


class _Recorder:
    def __init__(self, fn):
        self.fn, self.feeds = fn, []

    def run(self, names, feed):
        self.feeds.append(feed)
        return self.fn(feed)


def _configured_adapter(durations, velocity=0.5, latent_dim=3, num_speakers=2):
    adapter = RapidoAdapter()
    adapter.text_encoder = _Recorder(lambda f: [np.zeros((1, f["ids"].shape[1], 4), np.float32),
                                                np.array([durations], np.float32)])
    adapter.aligner = _Recorder(lambda f: [np.zeros((1, f["frame_word"].shape[1], 4), np.float32)])
    adapter.decoder = _Recorder(lambda f: [f["latents"].copy()])
    adapter.latent_mean = np.array([1.0, -1.0, 0.5], np.float32)[:latent_dim]
    adapter.latent_std = np.array([2.0, 0.5, 1.0], np.float32)[:latent_dim]
    adapter.boundary_ids = frozenset({1, 4})
    adapter.num_speakers = num_speakers
    model = _Recorder(lambda f: [np.full_like(f["x"], velocity)])
    return adapter, model


def test_synthesize_runs_layout_ode_and_denormalization():
    # "ab!a b": words of 2.6, 2.6 and 3.4 frames, scaled by length_scale 2 -> 5, 5, 7 frames
    adapter, model = _configured_adapter([1.0, 1.2, 0.4, 2.0, 0.6, 3.4], velocity=0.5)
    result = adapter.synthesize(_req([2, 3, 4, 2, 1, 3], speaker=1, length_scale=2.0, steps=4,
                                     guidance=3.0, seed=11), model)

    feed = adapter.aligner.feeds[0]
    assert feed["word"].tolist() == [[0, 0, 0, 1, 1, 2]]
    assert feed["word_start"].tolist() == [[0, 0, 0, 3, 3, 5]]
    assert feed["frame_word"].tolist() == [[0] * 5 + [1] * 5 + [2] * 7]
    np.testing.assert_allclose(feed["frame_pos"][0, 10:], np.arange(7) / 7, atol=1e-7)
    assert feed["speaker"].tolist() == [1]
    # the aligner sees the unscaled durations, as the model was trained
    np.testing.assert_array_equal(feed["durations"], np.array([[1.0, 1.2, 0.4, 2.0, 0.6, 3.4]], np.float32))

    # midpoint ODE: two model calls per step, guidance passed through
    assert len(model.feeds) == 8
    assert all(f["guidance"].tolist() == [3.0] for f in model.feeds)
    assert [float(f["t"][0]) for f in model.feeds[:2]] == [0.0, 0.125]

    # a constant velocity of 0.5 moves the noise by 0.5 over t in [0, 1]
    noise = np.random.default_rng(11).standard_normal((1, 17, 3)).astype(np.float32)
    expected = (noise[0] + 0.5) * np.array([2.0, 0.5, 1.0], np.float32) + np.array([1.0, -1.0, 0.5], np.float32)
    np.testing.assert_allclose(adapter.decoder.feeds[0]["latents"][0], expected.T, atol=1e-5)
    assert result.audio.shape == (3 * 17,) and result.audio.dtype == np.float32


def test_synthesize_seed_comes_from_the_request():
    adapter, model = _configured_adapter([1.0, 1.0])
    first = adapter.synthesize(_req([2, 3], seed=1), model).audio
    second = adapter.synthesize(_req([2, 3], seed=2), model).audio
    again = adapter.synthesize(_req([2, 3], seed=1), model).audio
    assert not np.allclose(first, second)
    np.testing.assert_array_equal(first, again)


def test_synthesize_requires_aux_graphs():
    with pytest.raises(RuntimeError):
        RapidoAdapter().synthesize(_req([2, 3]), None)


def test_synthesize_rejects_unknown_speaker():
    adapter, model = _configured_adapter([1.0, 1.0], num_speakers=2)
    with pytest.raises(ValueError):
        adapter.synthesize(_req([2, 3], speaker=2), model)


def test_build_feed_dict_and_parse_outputs_not_implemented():
    with pytest.raises(NotImplementedError):
        RapidoAdapter().build_feed_dict(_req([2]), None)
    with pytest.raises(NotImplementedError):
        RapidoAdapter().parse_outputs([], _req([2]))


def test_configure_reads_vocabulary_and_engine_params():
    config = VoiceConfig.from_dict({
        "phoonnx_version": "1.0", "engine": "rapido", "phoneme_type": "graphemes", "alphabet": "unicode",
        "lang_code": "pt-BR", "num_speakers": 3, "phoneme_id_map": ID_MAP, "pad": "<pad>",
        "add_blank_char": False, "use_eos_bos": True, "blank_at_start": False, "blank_at_end": False,
        "word_sep_token": " ",
        "engine_params": {"latent_mean": [0.0, 1.0], "latent_std": [1.0, 2.0], "steps": 6,
                          "guidance": 2.5, "seed": 9}})
    adapter = RapidoAdapter()
    adapter.configure(config)
    assert adapter.boundary_ids == {1, 4, 5, 6}
    assert adapter.num_speakers == 3
    assert (adapter.steps, adapter.guidance, adapter.seed) == (6, 2.5, 9)
    np.testing.assert_array_equal(adapter.latent_std, [1.0, 2.0])
    assert adapter.default_params() == {"length_scale": 1.0, "steps": 6.0, "guidance": 2.5, "seed": 9.0}
