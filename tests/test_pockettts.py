import json

import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.pockettts import (
    AVAILABLE_LANGS, DEFAULT_EOS_THRESHOLD, PocketTTSAdapter,
    find_boundary_indices, prepare_text,
)

LATENT_DIM = 4
COND_DIM = 8


def _manifest():
    """A two-entry flow-LM state manifest in the published format: one float cache
    filled with NaN and one int64 step counter filled with zeros."""
    return [
        {"index": 0, "input_name": "state_0", "output_name": "out_state_0",
         "module": "transformer.layers.0.self_attn", "key": "cache",
         "dtype": "float32", "fill": "nan", "shape": [2, 1, 6, 2, 2]},
        {"index": 1, "input_name": "state_1", "output_name": "out_state_1",
         "module": "transformer.layers.0.self_attn", "key": "step",
         "dtype": "int64", "fill": "zeros", "shape": [1]},
    ]


def _mimi_manifest():
    return [
        {"index": 0, "input_name": "state_0", "output_name": "out_state_0",
         "module": "decoder.model.0", "key": "first",
         "dtype": "bool", "fill": "ones", "shape": [1]},
        {"index": 1, "input_name": "state_1", "output_name": "out_state_1",
         "module": "decoder.model.0", "key": "previous",
         "dtype": "float32", "fill": "zeros", "shape": [1, 2, 3]},
    ]


def _bundle():
    return {
        "sample_rate": 24000, "frame_rate": 12.5, "samples_per_frame": 1920,
        "latent_dim": LATENT_DIM, "conditioning_dim": COND_DIM,
        "max_token_per_chunk": 8, "insert_bos_before_voice": True,
        "remove_semicolons": False, "pad_with_spaces_for_short_inputs": False,
        "model_recommended_frames_after_eos": None,
        "flow_lm_state_manifest": _manifest(),
        "mimi_state_manifest": _mimi_manifest(),
    }


class _Cfg:
    def __init__(self, **engine_params):
        self.engine_params = engine_params


class _FakeTokenizer:
    """Whitespace/punctuation tokenizer standing in for SentencePiece. Ids are stable
    per piece so Encode/Decode round-trip, which is all the adapter relies on.

    Like SentencePiece, the first token of any encoding is a word-start marker, which
    the adapter drops when it builds its punctuation sets."""

    _PUNCT = ".!?,;:"
    _MARKER = "▁"

    def __init__(self):
        self._pieces = ["<unk>", self._MARKER]

    def _id(self, piece):
        if piece not in self._pieces:
            self._pieces.append(piece)
        return self._pieces.index(piece)

    def Encode(self, text):
        out = [self._id(self._MARKER)]
        word = ""
        for ch in text:
            if ch in self._PUNCT:
                if word:
                    out.append(self._id(word))
                    word = ""
                out.append(self._id(ch))
            elif ch.isspace():
                if word:
                    out.append(self._id(word))
                    word = ""
            else:
                word += ch
        if word:
            out.append(self._id(word))
        return out

    def Decode(self, ids):
        pieces = [self._pieces[i] if 0 <= i < len(self._pieces) else "<unk>" for i in ids]
        pieces = [p for p in pieces if p != self._MARKER]
        text = ""
        for piece in pieces:
            if piece in self._PUNCT or not text:
                text += piece
            else:
                text += " " + piece
        return text


class _FlowLMSession:
    """Stand-in for flow_lm_main. Returns conditioning, an end-of-speech logit that
    fires at ``eos_at``, and the two state tensors, exactly as the export does."""

    def __init__(self, eos_at=3):
        self.eos_at = eos_at
        self.calls = 0
        self.feeds = []

    def run(self, output_names, feed):
        self.feeds.append(feed)
        self.calls += 1
        step = self.calls - 2  # the priming call does not count as a frame
        logit = 10.0 if step >= self.eos_at else -20.0
        return [np.ones((1, COND_DIM), np.float32),
                np.array([[logit]], np.float32),
                feed["state_0"], feed["state_1"] + 1]

    def get_inputs(self):
        class _Inp:
            def __init__(self, name): self.name = name
        return [_Inp(n) for n in ("sequence", "text_embeddings", "state_0", "state_1")]


class _FlowSession:
    """Stand-in for flow_lm_flow: a constant velocity field."""

    def run(self, output_names, feed):
        return [np.full_like(feed["x"], 0.5)]


class _CondSession:
    """Stand-in for text_conditioner."""

    def __init__(self):
        self.last = None

    def run(self, output_names, feed):
        self.last = feed
        n = feed["token_ids"].shape[1]
        return [np.ones((1, n, COND_DIM), np.float32)]


class _DecoderSession:
    """Stand-in for mimi_decoder: 4 samples per latent frame, state carried forward."""

    def __init__(self):
        self.chunk_sizes = []

    def run(self, output_names, feed):
        frames = feed["latent"].shape[1]
        self.chunk_sizes.append(frames)
        # carry the latent values into the audio so different latents give
        # different waveforms
        audio = np.repeat(feed["latent"].mean(axis=2).reshape(1, 1, frames), 4, axis=2)
        audio = audio.astype(np.float32) + 0.25
        return [audio, feed["state_0"], feed["state_1"]]


class _EncoderSession:
    def run(self, output_names, feed):
        samples = feed["audio"].shape[-1]
        return [np.ones((1, max(samples // 8, 1), COND_DIM), np.float32)]


def _adapter(**overrides):
    a = PocketTTSAdapter(**overrides)
    a._apply_metadata(_bundle())
    a.tokenizer = _FakeTokenizer()
    a.text_conditioner = _CondSession()
    a.flow_lm_flow = _FlowSession()
    a.mimi_decoder = _DecoderSession()
    a.mimi_encoder = _EncoderSession()
    a.voice_state = a.init_state(a.flow_state_manifest)
    return a


def _req(ids=(1, 2, 3), **params):
    arr = np.array([list(ids)], np.int64)
    return AdapterSynthesisRequest(phoneme_ids=arr,
                                   phoneme_lengths=np.array([arr.shape[1]], np.int64),
                                   speaker_id=0, language_id=0, params=params)


# ---------------------------------------------------------------------------
# registration / detect
# ---------------------------------------------------------------------------

def test_pockettts_registered():
    from phoonnx.engines import list_engines
    assert "pockettts" in list_engines()


def test_pockettts_get_adapter_returns_instance():
    from phoonnx.engines import get_adapter
    assert isinstance(get_adapter("pockettts"), PocketTTSAdapter)


def test_detect_by_config():
    assert PocketTTSAdapter.detect({"engine": "pockettts"})
    assert not PocketTTSAdapter.detect({"engine": "supertonic"})
    assert not PocketTTSAdapter.detect(None)


def test_detect_by_session_signature():
    assert PocketTTSAdapter.detect(session=_FlowLMSession())


def test_detect_rejects_other_session():
    class _Inp:
        def __init__(self, name): self.name = name

    class _Other:
        def get_inputs(self):
            return [_Inp("noisy_latent"), _Inp("text_emb")]

    assert not PocketTTSAdapter.detect(session=_Other())


def test_engine_named_by_config_wins_detection():
    from phoonnx.engines import detect_engine
    assert isinstance(detect_engine(config={"engine": "pockettts"}), PocketTTSAdapter)


def test_voice_config_routes_pockettts_through_graphemes():
    from phoonnx.config import Engine, VoiceConfig
    from scriptconv.phonemizers.enums import Alphabet

    cfg = VoiceConfig.from_dict({"engine": "pockettts"})
    assert cfg.engine == Engine.POCKETTTS
    assert cfg.alphabet == Alphabet.GRAPHEMES
    assert cfg.sample_rate == 24000


def test_available_langs_are_the_six_trained_languages():
    assert AVAILABLE_LANGS == ["en", "fr", "de", "it", "pt", "es"]


# ---------------------------------------------------------------------------
# text frontend
# ---------------------------------------------------------------------------

def test_prepare_text_capitalizes_and_terminates():
    assert prepare_text("hello world")[0] == "Hello world."


def test_prepare_text_keeps_existing_punctuation():
    assert prepare_text("Hello world!")[0] == "Hello world!"


def test_prepare_text_short_input_asks_for_more_frames():
    assert prepare_text("Hi there.")[1] == 3
    assert prepare_text("One two three four five six.")[1] == 1


def test_prepare_text_rejects_empty():
    with pytest.raises(ValueError):
        prepare_text("   ")


def test_prepare_text_removes_semicolons_when_the_bundle_asks():
    assert prepare_text("a; b", remove_semicolons=True)[0] == "A, b."
    assert prepare_text("a; b")[0] == "A; b."


def test_prepare_text_pads_short_inputs_when_the_bundle_asks():
    assert prepare_text("Hi.", pad_with_spaces_for_short_inputs=True)[0] == " " * 8 + "Hi."


def test_prepare_text_collapses_newlines():
    assert prepare_text("Hello\nworld")[0] == "Hello world."


def test_find_boundary_indices_splits_after_punctuation_runs():
    # tokens: a . b -> one boundary after the '.', plus the start and the end
    assert find_boundary_indices([10, 1, 11], {1}) == [0, 2, 3]


def test_find_boundary_indices_treats_repeated_punctuation_as_one_split():
    assert find_boundary_indices([10, 1, 1, 1, 11], {1}) == [0, 4, 5]


def test_find_boundary_indices_without_boundaries():
    assert find_boundary_indices([10, 11], {1}) == [0, 2]


def test_split_into_chunks_splits_on_sentence_end():
    a = _adapter()
    a.max_token_per_chunk = 4
    assert a.split_into_chunks("One two. Three four.") == ["One two.", "Three four."]


def test_split_into_chunks_packs_short_sentences_together():
    a = _adapter()
    a.max_token_per_chunk = 50
    assert a.split_into_chunks("One. Two. Three.") == ["One. Two. Three."]


def test_split_into_chunks_falls_back_to_commas_over_the_token_limit():
    a = _adapter()
    a.max_token_per_chunk = 3
    chunks = a.split_into_chunks("alpha beta gamma, delta epsilon zeta.")
    assert len(chunks) > 1


def test_encode_text_returns_one_id_list_per_chunk():
    a = _adapter()
    a.max_token_per_chunk = 4
    ids = a.encode_text("One two. Three four.", None, None)
    assert len(ids) == 2
    assert all(isinstance(i, int) for chunk in ids for i in chunk)


def test_encode_text_without_tokenizer_is_an_error():
    a = _adapter()
    a.tokenizer = None
    with pytest.raises(RuntimeError, match="tokenizer_path"):
        a.encode_text("Hello.", None, None)


# ---------------------------------------------------------------------------
# stream state
# ---------------------------------------------------------------------------

def test_init_state_honours_fill_and_dtype():
    a = _adapter()
    state = a.init_state(a.flow_state_manifest)
    assert np.isnan(state["state_0"]).all()
    assert state["state_0"].dtype == np.float32
    assert state["state_1"].tolist() == [0]
    assert state["state_1"].dtype == np.int64


def test_init_state_fills_ones_for_boolean_flags():
    a = _adapter()
    state = a.init_state(a.mimi_state_manifest)
    assert state["state_0"].dtype == np.bool_
    assert state["state_0"].all()
    assert not state["state_1"].any()


def test_update_state_copies_outputs_onto_inputs():
    a = _adapter()
    state = a.init_state(a.flow_state_manifest)
    outputs = ["cond", "eos", np.zeros((2, 1, 6, 2, 2), np.float32),
               np.array([7], np.int64)]
    a._update_state(state, outputs, a.flow_state_manifest, output_offset=2)
    assert state["state_1"].tolist() == [7]
    assert not np.isnan(state["state_0"]).any()


def test_adapt_state_tensor_keeps_an_exact_match():
    a = _adapter()
    entry = a.flow_state_manifest[1]
    out = a._adapt_state_tensor(np.array([5], np.int64), entry)
    assert out.tolist() == [5]


def test_adapt_state_tensor_reshapes_a_flat_tensor():
    a = _adapter()
    entry = a.flow_state_manifest[0]
    flat = np.arange(2 * 1 * 6 * 2 * 2, dtype=np.float32)
    assert a._adapt_state_tensor(flat, entry).shape == tuple(entry["shape"])


def test_adapt_state_tensor_pads_a_shorter_cache():
    a = _adapter()
    entry = a.flow_state_manifest[0]
    short = np.ones((2, 1, 2, 2, 2), np.float32)
    out = a._adapt_state_tensor(short, entry)
    assert out.shape == tuple(entry["shape"])
    assert (out[:, :, :2] == 1).all()
    assert np.isnan(out[:, :, 2:]).all()


def test_adapt_state_tensor_truncates_a_longer_cache():
    a = _adapter()
    entry = a.flow_state_manifest[0]
    long = np.ones((2, 1, 99, 2, 2), np.float32)
    assert a._adapt_state_tensor(long, entry).shape == tuple(entry["shape"])


def test_derive_step_prefers_an_explicit_counter():
    assert PocketTTSAdapter._derive_step({"step": np.array([4])}).tolist() == [4]


def test_derive_step_falls_back_to_offset():
    assert PocketTTSAdapter._derive_step({"offset": np.array([9])}).tolist() == [9]


def test_derive_step_ignores_offset_when_end_offset_is_present():
    state = {"offset": np.array([9]), "end_offset": np.array([1])}
    assert PocketTTSAdapter._derive_step(state).tolist() == [0]


def test_derive_step_infers_from_the_cache_length():
    state = {"current_end": np.zeros((6, 2))}
    assert PocketTTSAdapter._derive_step(state).tolist() == [6]


def test_load_voice_state_maps_a_saved_state_onto_the_manifest(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    path = tmp_path / "voice.safetensors"
    save_file({"transformer.layers.0.self_attn/cache": np.ones((2, 1, 3, 2, 2), np.float32),
               "transformer.layers.0.self_attn/step": np.array([3], np.int64)}, str(path))

    a = _adapter()
    state = a.load_voice_state(str(path))
    assert state["state_1"].tolist() == [3]
    assert (state["state_0"][:, :, :3] == 1).all()
    assert np.isnan(state["state_0"][:, :, 3:]).all()


def test_load_voice_state_derives_a_missing_step(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    path = tmp_path / "voice.safetensors"
    save_file({"transformer.layers.0.self_attn/cache": np.ones((2, 1, 3, 2, 2), np.float32)},
              str(path))
    a = _adapter()
    assert a.load_voice_state(str(path))["state_1"].tolist() == [0]


# ---------------------------------------------------------------------------
# configure
# ---------------------------------------------------------------------------

def test_configure_reads_the_bundle_geometry(tmp_path):
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps(_bundle()))
    a = PocketTTSAdapter()
    a.configure(_Cfg(bundle_path=str(bundle)))
    assert a.latent_dim == LATENT_DIM
    assert a.conditioning_dim == COND_DIM
    assert a.max_token_per_chunk == 8
    assert a.flow_state_manifest == _manifest()


def test_configure_reads_the_bos_embedding(tmp_path):
    bos = tmp_path / "bos.npy"
    np.save(bos, np.ones((1, 1, COND_DIM), np.float32))
    a = PocketTTSAdapter()
    a.configure(_Cfg(bos_path=str(bos)))
    assert a.bos_before_voice.shape == (1, 1, COND_DIM)


def test_configure_overrides_sampling_controls():
    a = PocketTTSAdapter()
    a.configure(_Cfg(temperature=0.3, lsd_steps=4, eos_threshold=-2.0, seed=7))
    assert (a.temperature, a.lsd_steps, a.eos_threshold, a.seed) == (0.3, 4, -2.0, 7)


def test_default_params_expose_the_sampling_controls():
    params = PocketTTSAdapter().default_params()
    assert set(params) == {"temperature", "lsd_steps", "eos_threshold"}
    assert params["eos_threshold"] == DEFAULT_EOS_THRESHOLD


def test_param_labels_cover_every_param():
    a = PocketTTSAdapter()
    assert set(a.param_labels()) == set(a.default_params())


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------

def test_synthesize_returns_audio_for_the_generated_frames():
    a = _adapter()
    session = _FlowLMSession(eos_at=3)
    result = a.synthesize(_req(), session)
    # eos fires at frame 3 and a short input asks for 5 more frames
    assert result.extras["frames"] == 8
    assert result.audio.shape == (8 * 4,)
    assert result.audio.dtype == np.float32


def test_synthesize_primes_the_transformer_with_the_text_embeddings():
    a = _adapter()
    session = _FlowLMSession()
    a.synthesize(_req(), session)
    priming = session.feeds[0]
    assert priming["sequence"].shape == (1, 0, LATENT_DIM)
    assert priming["text_embeddings"].shape[1] == 3


def test_synthesize_feeds_empty_text_after_priming():
    a = _adapter()
    session = _FlowLMSession()
    a.synthesize(_req(), session)
    assert session.feeds[1]["text_embeddings"].shape == (1, 0, COND_DIM)
    assert session.feeds[1]["sequence"].shape == (1, 1, LATENT_DIM)


def test_synthesize_starts_the_sequence_as_nan():
    a = _adapter()
    session = _FlowLMSession()
    a.synthesize(_req(), session)
    assert np.isnan(session.feeds[1]["sequence"]).all()


def test_synthesize_feeds_the_previous_latent_back_in():
    a = _adapter(temperature=0.0)
    session = _FlowLMSession()
    a.synthesize(_req(), session)
    # temperature 0 plus a constant velocity field: every frame is 0.5
    assert np.allclose(session.feeds[2]["sequence"], 0.5)


def test_synthesize_is_deterministic_at_temperature_zero():
    a = _adapter(temperature=0.0)
    first = a.synthesize(_req(), _FlowLMSession()).audio
    second = a.synthesize(_req(), _FlowLMSession()).audio
    assert np.array_equal(first, second)


def test_synthesize_is_reproducible_with_a_seed():
    a = _adapter()
    first = a.synthesize(_req(temperature=0.7, seed=11), _FlowLMSession()).audio
    second = a.synthesize(_req(temperature=0.7, seed=11), _FlowLMSession()).audio
    assert np.array_equal(first, second)


def test_synthesize_different_seeds_give_different_audio():
    a = _adapter()
    first = a.synthesize(_req(temperature=0.7, seed=1), _FlowLMSession()).audio
    second = a.synthesize(_req(temperature=0.7, seed=2), _FlowLMSession()).audio
    assert not np.array_equal(first, second)


def test_synthesize_does_not_mutate_the_voice_state():
    a = _adapter()
    before = a.voice_state["state_1"].copy()
    a.synthesize(_req(), _FlowLMSession())
    assert a.voice_state["state_1"].tolist() == before.tolist()


def test_synthesize_stops_at_the_length_cap_when_eos_never_fires():
    a = _adapter()
    result = a.synthesize(_req(), _FlowLMSession(eos_at=10_000))
    assert result.extras["frames"] == a._max_frames(3)


def test_synthesize_rejects_zero_lsd_steps():
    a = _adapter()
    with pytest.raises(ValueError, match="lsd_steps"):
        a.synthesize(_req(lsd_steps=0), _FlowLMSession())


def test_synthesize_rejects_negative_temperature():
    a = _adapter()
    with pytest.raises(ValueError, match="temperature"):
        a.synthesize(_req(temperature=-1.0), _FlowLMSession())


def test_synthesize_without_auxiliary_graphs_is_an_error():
    a = _adapter()
    a.flow_lm_flow = None
    with pytest.raises(RuntimeError, match="flow_lm_flow_path"):
        a.synthesize(_req(), _FlowLMSession())


def test_synthesize_without_a_bundle_is_an_error():
    a = _adapter()
    a.flow_state_manifest = []
    with pytest.raises(RuntimeError, match="bundle_path"):
        a.synthesize(_req(), _FlowLMSession())


def test_synthesize_without_a_voice_is_an_error():
    a = _adapter()
    a.voice_state = None
    with pytest.raises(RuntimeError, match="voice_state_path"):
        a.synthesize(_req(), _FlowLMSession())


def test_more_lsd_steps_integrate_the_flow_further():
    a = _adapter(temperature=0.0)
    session = _FlowLMSession()
    a.synthesize(_req(lsd_steps=2), session)
    # two half-steps of a constant 0.5 field still integrate to 0.5
    assert np.allclose(session.feeds[2]["sequence"], 0.5)


def test_eos_threshold_is_honoured():
    a = _adapter()
    late = a.synthesize(_req(eos_threshold=100.0), _FlowLMSession(eos_at=1))
    early = a.synthesize(_req(eos_threshold=-100.0), _FlowLMSession(eos_at=1))
    assert late.extras["frames"] > early.extras["frames"]


def test_build_feed_dict_is_not_supported():
    with pytest.raises(NotImplementedError):
        _adapter().build_feed_dict(_req(), _FlowLMSession())


def test_parse_outputs_is_not_supported():
    with pytest.raises(NotImplementedError):
        _adapter().parse_outputs([], _req())


# ---------------------------------------------------------------------------
# decoding / cloning
# ---------------------------------------------------------------------------

def test_decode_latents_feeds_the_decoder_in_chunks():
    a = _adapter()
    latents = np.zeros((1, 7, LATENT_DIM), np.float32)
    audio = a.decode_latents(latents, chunk_size=3)
    assert a.mimi_decoder.chunk_sizes == [3, 3, 1]
    assert audio.shape == (7 * 4,)


def test_decode_latents_of_nothing_is_empty():
    a = _adapter()
    audio = a.decode_latents(np.zeros((1, 0, LATENT_DIM), np.float32))
    assert audio.shape == (0,)


def test_encode_reference_shapes_the_clip_for_the_encoder():
    a = _adapter()
    embeddings = a.encode_reference(np.zeros(160, np.float32))
    assert embeddings.ndim == 3
    assert embeddings.shape[-1] == COND_DIM


def test_encode_reference_without_the_encoder_is_an_error():
    a = _adapter()
    a.mimi_encoder = None
    with pytest.raises(RuntimeError, match="mimi_encoder_path"):
        a.encode_reference(np.zeros(160, np.float32))


def test_state_from_reference_prepends_the_bos_embedding():
    a = _adapter()
    a.bos_before_voice = np.ones((1, 1, COND_DIM), np.float32)
    session = _FlowLMSession()
    a.state_from_reference(np.zeros(160, np.float32), session)
    # 160 samples -> 20 encoder frames, plus the BOS frame
    assert session.feeds[0]["text_embeddings"].shape[1] == 21


def test_state_from_reference_without_bos_passes_the_clip_through():
    a = _adapter()
    a.insert_bos_before_voice = False
    session = _FlowLMSession()
    a.state_from_reference(np.zeros(160, np.float32), session)
    assert session.feeds[0]["text_embeddings"].shape[1] == 20


def test_synthesize_with_a_reference_clone_uses_the_cloned_state():
    a = _adapter()
    a.voice_state = None
    result = a.synthesize(_req(reference_audio=np.zeros(160, np.float32)), _FlowLMSession())
    assert result.audio.size > 0


# ---------------------------------------------------------------------------
# voice index
# ---------------------------------------------------------------------------

def _index():
    from phoonnx.model_manager import TTSModelManager
    path = TTSModelManager.voice_index_path() / "pockettts.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_voice_index_is_bundled():
    from phoonnx.model_manager import TTSModelManager
    files = [p.name for p in TTSModelManager.voice_index_files()]
    assert "pockettts.json" in files


def test_voice_index_covers_the_six_languages():
    langs = {entry["lang"] for entry in _index().values()}
    assert langs == set(AVAILABLE_LANGS)


def test_voice_index_entries_are_self_consistent():
    for voice_id, entry in _index().items():
        assert entry["voice_id"] == voice_id
        assert entry["engine"] == "pockettts"
        assert entry["alphabet"] == "graphemes"
        assert entry["config_url"] is None


def test_voice_index_entries_carry_every_auxiliary_graph():
    required = {"bundle_path", "tokenizer_path", "bos_path", "text_conditioner_path",
                "flow_lm_flow_path", "mimi_decoder_path", "mimi_encoder_path",
                "voice_state_path"}
    for entry in _index().values():
        assert required <= set(entry["aux_model_urls"])


def test_voice_index_urls_point_at_the_mirror():
    for entry in _index().values():
        urls = [entry["model_url"], *entry["aux_model_urls"].values()]
        assert all(u.startswith("https://huggingface.co/OpenVoiceOS/phoonnx-pocket-tts/")
                   for u in urls)


def test_voice_index_auxiliary_filenames_do_not_collide():
    for entry in _index().values():
        names = [u.rsplit("/", 1)[-1] for u in entry["aux_model_urls"].values()]
        assert len(names) == len(set(names))


def test_voice_index_voice_state_matches_the_voice_id():
    for voice_id, entry in _index().items():
        speaker = voice_id.rsplit("/", 1)[-1]
        assert entry["aux_model_urls"]["voice_state_path"].endswith(f"/{speaker}.safetensors")
