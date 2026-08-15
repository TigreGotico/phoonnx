"""Tests for the OmniVoice adapter (masked-diffusion codec LM).

The sampler is the risky part: it is a bounded unmasking loop over an ``8 x T`` grid with
classifier-free guidance, a codebook-order penalty and a warped time schedule, and every
one of those has a way of failing silently (leftover MASK ids reaching the codec, the
unconditional branch carrying the text after all, a schedule that never finishes). A fake
session with the real ``InferenceSession`` contract drives the loop so those are asserted
directly, in the style of the sparktts / neutts fakes.
"""
import sys

import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.omnivoice import (AUDIO_MASK_ID, AUDIO_VOCAB_SIZE, FRAME_RATE,
                                       HOP_LENGTH, NUM_CODEBOOKS, SAMPLE_RATE,
                                       OmniVoiceAdapter, _filter_top_k, _log_softmax)
from phoonnx.thirdparty.omnivoice import (RuleDurationEstimator, add_punctuation,
                                          combine_text, fade_and_pad_audio, remove_silence,
                                          resample, tokenize_with_nonverbal_tags)


class _IOSpec:
    def __init__(self, name, shape, type="tensor(float)"):
        self.name, self.shape, self.type = name, shape, type


class _FakeTokenizer:
    """Byte-ish stand-in for the Qwen3 BPE: one id per character, decodable."""

    def __init__(self):
        self._tok = self

    def tokenize(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


class _FakeBackbone:
    """Fake OmniVoice backbone with the real session contract.

    ``preferred`` fixes which codebook entry wins at every position, so the decoded grid
    is fully predictable. Feeds are recorded and shape-checked, because the adapter is
    responsible for keeping ``input_ids`` and ``audio_mask`` in step.
    """

    def __init__(self, preferred=7, vocab=AUDIO_VOCAB_SIZE):
        self.preferred, self.vocab, self.calls = preferred, vocab, []

    def get_inputs(self):
        return [_IOSpec("input_ids", ["batch", NUM_CODEBOOKS, "seq"], "tensor(int64)"),
                _IOSpec("audio_mask", ["batch", "seq"], "tensor(bool)")]

    def get_outputs(self):
        return [_IOSpec("logits", ["batch", NUM_CODEBOOKS, "seq", self.vocab])]

    def run(self, output_names, feed):
        input_ids, audio_mask = feed["input_ids"], feed["audio_mask"]
        assert input_ids.ndim == 3 and input_ids.shape[1] == NUM_CODEBOOKS
        assert audio_mask.shape == (input_ids.shape[0], input_ids.shape[2])
        assert input_ids.dtype == np.int64 and audio_mask.dtype == bool
        self.calls.append({"input_ids": input_ids.copy(), "audio_mask": audio_mask.copy()})
        seq = input_ids.shape[2]
        logits = np.full((1, NUM_CODEBOOKS, seq, self.vocab), -20.0, np.float32)
        logits[..., self.preferred] = 10.0
        logits[..., AUDIO_MASK_ID] = 50.0     # MASK is the argmax unless the adapter bans it
        return [logits]


class _FakeDecoder:
    def __init__(self):
        self.calls = []

    def get_inputs(self):
        return [_IOSpec("codes", [NUM_CODEBOOKS, 1, "frames"], "tensor(int64)")]

    def get_outputs(self):
        return [_IOSpec("waveform_24k", [1, 1, "samples"])]

    def run(self, output_names, feed):
        codes = feed["codes"]
        assert codes.shape[0] == NUM_CODEBOOKS and codes.shape[1] == 1
        # a MASK id reaching the codec is out of its 0..1023 range; catch it here
        assert codes.min() >= 0 and codes.max() < AUDIO_MASK_ID, "MASK token reached the codec"
        self.calls.append(codes.copy())
        n = codes.shape[2] * HOP_LENGTH
        return [np.linspace(-0.5, 0.5, n, dtype=np.float32).reshape(1, 1, n)]


def _adapter(decoder=None):
    ad = OmniVoiceAdapter()
    ad.tokenizer = _FakeTokenizer()
    ad.decoder = decoder
    return ad


def _req(ids, **params):
    ids = np.asarray(ids, np.int64).reshape(1, -1)
    return AdapterSynthesisRequest(phoneme_ids=ids,
                                   phoneme_lengths=np.array([ids.shape[1]], np.int64),
                                   speaker_id=0, language_id=0, params=params)


# ---------------------------------------------------------------------------
# Registration and config
# ---------------------------------------------------------------------------

def test_omnivoice_registered_at_priority_21():
    from phoonnx.engines import _PRIORITIES, list_engines
    assert "omnivoice" in list_engines()
    assert _PRIORITIES["omnivoice"] == 21


def test_detect_only_claims_its_own_voices():
    assert OmniVoiceAdapter.detect({"engine": "omnivoice"})
    assert not OmniVoiceAdapter.detect({"engine": "zipvoice"})
    assert not OmniVoiceAdapter.detect({})
    assert not OmniVoiceAdapter.detect(None)


def test_engine_enum_and_config_route_through_the_adapter():
    from phoonnx.config import Alphabet, Engine, VoiceConfig
    cfg = VoiceConfig.from_dict({"engine": "omnivoice"}, engine=Engine.OMNIVOICE, lang_code="en")
    assert cfg.engine == Engine.OMNIVOICE
    # graphemes means TTSVoice never runs a phonemizer for this engine
    assert cfg.alphabet == Alphabet.GRAPHEMES
    assert cfg.sample_rate == SAMPLE_RATE


def test_default_params_match_upstream_generation_config():
    # upstream OmniVoiceGenerationConfig defaults
    p = OmniVoiceAdapter().default_params()
    assert p["num_step"] == 32.0
    assert p["guidance_scale"] == 2.0
    assert p["t_shift"] == 0.1
    assert p["layer_penalty_factor"] == 5.0
    assert p["position_temperature"] == 5.0
    assert p["class_temperature"] == 0.0


def test_every_param_has_a_label():
    ad = OmniVoiceAdapter()
    assert set(ad.param_labels()) == set(ad.default_params())


def test_codec_constants_match_the_checkpoint():
    assert (NUM_CODEBOOKS, AUDIO_VOCAB_SIZE, AUDIO_MASK_ID) == (8, 1025, 1024)
    assert SAMPLE_RATE // HOP_LENGTH == FRAME_RATE == 25


def test_single_graph_helpers_are_refused():
    ad = _adapter()
    with pytest.raises(NotImplementedError):
        ad.build_feed_dict(_req([1]), _FakeBackbone())
    with pytest.raises(NotImplementedError):
        ad.parse_outputs([], _req([1]))


# ---------------------------------------------------------------------------
# Language resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("given,expected", [
    ("en", "en"), ("English", "en"), ("english", "en"),
    ("pt-BR", "pt"), ("pt_br", "pt"), ("zh", "zh"), ("yue", "yue"),
    (None, None), ("", None), ("None", None),
])
def test_resolve_language(given, expected):
    assert OmniVoiceAdapter.resolve_language(given) == expected


def test_unknown_language_falls_back_to_agnostic_rather_than_raising():
    assert OmniVoiceAdapter.resolve_language("klingon") is None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def test_prompt_layout_marks_only_the_audio_region():
    ad = _adapter()
    ids, mask = ad.build_prompt("hello", target_len=6, lang="en")
    assert ids.shape == (1, NUM_CODEBOOKS, ids.shape[2])
    assert mask.shape == (1, ids.shape[2])
    # exactly the 6 generated slots are audio, and they all start masked
    assert mask.sum() == 6
    assert mask[0, -6:].all()
    assert (ids[0, :, -6:] == AUDIO_MASK_ID).all()
    # every codebook row carries the same text ids
    assert (ids[0, 0, :-6] == ids[0, 5, :-6]).all()


def test_reference_codes_sit_between_the_text_and_the_masks_and_are_not_regenerated():
    ad = _adapter()
    ref = np.arange(NUM_CODEBOOKS * 4, dtype=np.int64).reshape(NUM_CODEBOOKS, 4)
    ids, mask = ad.build_prompt("hi", target_len=5, ref_text="Ref.", ref_codes=ref, lang="en")
    assert (ids[0, :, -9:-5] == ref).all()
    assert (ids[0, :, -5:] == AUDIO_MASK_ID).all()
    # the reference frames are audio too, so the mask covers them
    assert mask.sum() == 9


def test_denoise_token_only_appears_when_cloning():
    ad = _adapter()
    ref = np.zeros((NUM_CODEBOOKS, 2), np.int64)
    with_ref, _ = ad.build_prompt("hi", 3, ref_codes=ref, denoise=True)
    without, _ = ad.build_prompt("hi", 3, denoise=True)
    assert "<|denoise|>" in ad._decode_ids(with_ref[0, 0])
    assert "<|denoise|>" not in ad._decode_ids(without[0, 0])


def test_prompt_carries_language_instruct_and_joined_text():
    ad = _adapter()
    ids, _ = ad.build_prompt("world", 2, ref_text="Hello.", lang="pt",
                             instruct="a calm voice", ref_codes=None)
    prompt = ad._decode_ids(ids[0, 0, :-2])
    assert "<|lang_start|>pt<|lang_end|>" in prompt
    assert "<|instruct_start|>a calm voice<|instruct_end|>" in prompt
    # the reference transcription is *joined to* the target text, not a separate field
    assert "<|text_start|>Hello. world<|text_end|>" in prompt


def test_missing_language_becomes_the_literal_none_slot():
    ad = _adapter()
    ids, _ = ad.build_prompt("x", 1, lang=None)
    assert "<|lang_start|>None<|lang_end|>" in ad._decode_ids(ids[0, 0, :-1])


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target_len,num_step", [(1, 1), (5, 32), (100, 32), (37, 7), (250, 16)])
def test_schedule_fills_every_slot_exactly_once(target_len, num_step):
    sched = OmniVoiceAdapter.unmask_schedule(target_len, num_step, 0.1)
    assert len(sched) == num_step
    assert all(n >= 0 for n in sched)
    # every one of the 8*T slots must be unmasked by the end -- a leftover MASK id
    # would be passed to the codec decoder, outside its 0..1023 range
    assert sum(sched) == target_len * NUM_CODEBOOKS


def test_schedule_front_loads_less_work_with_the_default_shift():
    sched = OmniVoiceAdapter.unmask_schedule(100, 32, 0.1)
    # t_shift 0.1 warps the grid so the bulk lands at the end
    assert sched[0] < sched[-1]
    assert sched[-1] == max(sched)


def test_schedule_with_shift_one_is_close_to_uniform():
    sched = OmniVoiceAdapter.unmask_schedule(64, 8, 1.0)
    assert sum(sched) == 64 * NUM_CODEBOOKS
    assert max(sched) - min(sched) <= 1


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def test_sampler_leaves_no_mask_and_returns_the_right_shape():
    ad, sess = _adapter(), _FakeBackbone(preferred=42)
    codes = ad.generate_codes(sess, "hello", target_len=9, num_step=6, seed=0)
    assert codes.shape == (NUM_CODEBOOKS, 9)
    assert (codes != AUDIO_MASK_ID).all()
    assert (codes == 42).all()      # the fake's peak, so MASK really was banned


def test_sampler_never_predicts_the_mask_token_even_when_it_is_the_argmax():
    # the fake gives MASK a logit of 50 against 10 for the real token; if the adapter
    # forgot to set the MASK column to -inf, every slot would come back as 1024
    ad, sess = _adapter(), _FakeBackbone(preferred=3)
    codes = ad.generate_codes(sess, "x", target_len=4, num_step=3, seed=1)
    assert (codes == 3).all()


def test_sampler_runs_two_forwards_per_step_under_guidance():
    ad, sess = _adapter(), _FakeBackbone()
    ad.generate_codes(sess, "hello", target_len=5, num_step=4, guidance_scale=2.0, seed=0)
    assert len(sess.calls) == 8


def test_sampler_runs_one_forward_per_step_without_guidance():
    ad, sess = _adapter(), _FakeBackbone()
    ad.generate_codes(sess, "hello", target_len=5, num_step=4, guidance_scale=0.0, seed=0)
    assert len(sess.calls) == 4


def test_unconditional_branch_drops_the_text_and_keeps_only_the_target_span():
    ad, sess = _adapter(), _FakeBackbone()
    target = 6
    ad.generate_codes(sess, "a long prompt here", target_len=target, num_step=2,
                      guidance_scale=2.0, lang="en", seed=0)
    cond, uncond = sess.calls[0], sess.calls[1]
    assert cond["input_ids"].shape[2] > target        # prompt + masks
    assert uncond["input_ids"].shape[2] == target     # masks only
    assert uncond["audio_mask"].all()                 # all of it is audio
    # no text survives into the unconditional row
    assert (uncond["input_ids"][0, 0] == AUDIO_MASK_ID).all()


def test_state_threads_back_into_both_branches_between_steps():
    """Slots decided at step N must be visible as real codes at step N+1 in *both* the
    conditional and unconditional rows, otherwise each step re-decides from scratch."""
    ad, sess = _adapter(), _FakeBackbone(preferred=11)
    target = 8
    ad.generate_codes(sess, "hello", target_len=target, num_step=4,
                      guidance_scale=2.0, seed=0)
    # calls come in (cond, uncond) pairs
    for step in range(1, 4):
        cond = sess.calls[2 * step]["input_ids"][0, :, -target:]
        uncond = sess.calls[2 * step + 1]["input_ids"][0, :, :target]
        assert (cond == uncond).all(), "branches disagree about already-decoded slots"
        prev_cond = sess.calls[2 * (step - 1)]["input_ids"][0, :, -target:]
        decided_before = prev_cond != AUDIO_MASK_ID
        # nothing already decided may be reverted or changed
        assert (cond[decided_before] == prev_cond[decided_before]).all()
        assert decided_before.sum() < (cond != AUDIO_MASK_ID).sum()   # progress each step


def test_audio_mask_stays_constant_across_steps():
    ad, sess = _adapter(), _FakeBackbone()
    ad.generate_codes(sess, "hello", target_len=5, num_step=3, guidance_scale=0.0, seed=0)
    masks = [c["audio_mask"] for c in sess.calls]
    assert all((m == masks[0]).all() for m in masks)


def test_greedy_sampling_is_deterministic_and_ignores_the_seed():
    ad, sess = _adapter(), _FakeBackbone(preferred=5)
    kw = dict(target_len=6, num_step=4, position_temperature=0.0, class_temperature=0.0)
    a = ad.generate_codes(sess, "hello", seed=0, **kw)
    b = ad.generate_codes(_FakeBackbone(preferred=5), "hello", seed=999, **kw)
    assert (a == b).all()


def test_seed_makes_the_stochastic_path_reproducible():
    kw = dict(target_len=6, num_step=4, position_temperature=5.0)
    a = _adapter().generate_codes(_FakeBackbone(), "hello", seed=7, **kw)
    b = _adapter().generate_codes(_FakeBackbone(), "hello", seed=7, **kw)
    assert (a == b).all()


def test_one_step_decodes_everything():
    ad, sess = _adapter(), _FakeBackbone(preferred=1)
    codes = ad.generate_codes(sess, "hello", target_len=12, num_step=1, seed=0)
    assert (codes == 1).all()
    assert len(sess.calls) == 2


# ---------------------------------------------------------------------------
# Sampler maths
# ---------------------------------------------------------------------------

def test_log_softmax_normalises():
    x = np.array([[1.0, 2.0, 3.0]], np.float32)
    assert np.allclose(np.exp(_log_softmax(x)).sum(-1), 1.0)


def test_filter_top_k_keeps_a_tenth_of_the_vocabulary():
    lp = _log_softmax(np.arange(100, dtype=np.float32).reshape(1, 100))
    kept = np.isfinite(_filter_top_k(lp, ratio=0.1))
    assert kept.sum() == 10
    assert kept[0, -10:].all()          # the ten largest survive


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------

def test_duration_scales_with_text_length():
    ad = _adapter()
    short = ad.estimate_target_len("Hi.", "Nice to meet you.", 25)
    long = ad.estimate_target_len("Hi. " * 20, "Nice to meet you.", 25)
    assert 1 <= short < long


def test_duration_without_a_reference_uses_the_upstream_fallback_pair():
    ad = _adapter()
    assert ad.estimate_target_len("Hello there.", None, None) == \
           ad.estimate_target_len("Hello there.", "Nice to meet you.", 25)


def test_length_scale_stretches_the_estimate():
    ad = _adapter()
    base = ad.estimate_target_len("Hello there friend.", "Nice to meet you.", 25)
    slower = ad.estimate_target_len("Hello there friend.", "Nice to meet you.", 25,
                                    length_scale=2.0)
    assert slower > base


def test_duration_is_at_least_one_frame():
    assert _adapter().estimate_target_len("", None, None) >= 1


def test_cjk_text_is_slower_per_character_than_latin():
    est = RuleDurationEstimator()
    # one Han character is a whole syllable; one Latin letter is not
    assert est.calculate_total_weight("好") > est.calculate_total_weight("a")


# ---------------------------------------------------------------------------
# Vendored text helpers
# ---------------------------------------------------------------------------

def test_combine_text_joins_reference_and_target():
    assert combine_text("world", "Hello.") == "Hello. world"
    assert combine_text("  world  ") == "world"


def test_combine_text_collapses_whitespace_and_strips_newlines():
    assert combine_text("a  \t b\nc") == "a bc"


def test_combine_text_removes_spaces_around_han_characters():
    assert combine_text("你好 世界") == "你好世界"
    assert combine_text("hello 你好") == "hello你好"


def test_combine_text_normalises_full_width_parentheses():
    assert combine_text("a（b）c") == "a(b)c"


def test_nonverbal_tags_tokenize_standalone():
    calls = []

    def enc(s):
        calls.append(s)
        return [len(s)]

    tokenize_with_nonverbal_tags("I know [laughter] really", enc)
    assert "[laughter]" in calls
    assert not any(c != "[laughter]" and "[laughter]" in c for c in calls)


def test_text_without_tags_is_encoded_in_one_piece():
    assert tokenize_with_nonverbal_tags("plain text", lambda s: [len(s)]) == [10]


def test_add_punctuation_respects_script():
    assert add_punctuation("hello") == "hello."
    assert add_punctuation("hello.") == "hello."
    assert add_punctuation("你好") == "你好。"
    assert add_punctuation("") == ""


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def test_resample_changes_length_by_the_ratio():
    sr, n = 24000, 24000
    w = np.sin(2 * np.pi * 220 * np.arange(n) / sr).astype(np.float32)
    out = resample(w, 24000, 16000)
    assert abs(len(out) - 16000) <= 1


def test_resample_is_a_no_op_at_the_same_rate():
    w = np.random.default_rng(0).standard_normal(100).astype(np.float32)
    assert np.array_equal(resample(w, 16000, 16000), w)


def test_resample_preserves_a_tone_amplitude():
    sr, n = 24000, 24000
    w = np.sin(2 * np.pi * 220 * np.arange(n) / sr).astype(np.float32)
    out = resample(w, 24000, 16000)
    # ignore the filter ramp at the edges
    assert 0.9 < np.abs(out[500:-500]).max() < 1.1


def test_fade_and_pad_adds_silence_and_zeroes_the_edges():
    audio = np.ones((1, 24000), np.float32)
    out = fade_and_pad_audio(audio, pad_duration=0.1, fade_duration=0.1,
                             sample_rate=SAMPLE_RATE)
    assert out.shape[-1] == 24000 + 2 * 2400
    assert out[0, 0] == 0.0 and out[0, -1] == 0.0
    assert out[0, out.shape[-1] // 2] == pytest.approx(1.0)


def test_fade_and_pad_leaves_empty_audio_alone():
    empty = np.zeros((1, 0), np.float32)
    assert fade_and_pad_audio(empty).shape[-1] == 0


def _silence_padded_tone(sr=16000, lead_ms=300, tone_ms=500, trail_ms=300, freq=220):
    """(1, T) signal: near-silence, then a loud tone, then near-silence.

    The pauses use a tiny amount of noise (not exact zeros) so the RMS-window
    fallback, which measures energy rather than testing for literal zero,
    still classifies them as silent.
    """
    rng = np.random.default_rng(0)
    lead = rng.standard_normal(int(sr * lead_ms / 1000)).astype(np.float32) * 1e-5
    trail = rng.standard_normal(int(sr * trail_ms / 1000)).astype(np.float32) * 1e-5
    n = int(sr * tone_ms / 1000)
    tone = np.sin(2 * np.pi * freq * np.arange(n) / sr).astype(np.float32)
    audio = np.concatenate([lead, tone, trail])[None, :]
    voice_start, voice_end = len(lead), len(lead) + len(tone)
    return audio, sr, voice_start, voice_end


def test_remove_silence_pydub_path_trims_the_padded_edges():
    """Locks the upstream pydub branch: silent lead/trail shrink, the tone stays."""
    audio, sr, voice_start, voice_end = _silence_padded_tone()
    out = remove_silence(audio, sr, mid_sil=300, lead_sil=50, trail_sil=50,
                         silence_threshold=-50.0)

    assert out.shape[0] == 1
    # most of the padding got cut, but the tone (200ms in) must survive
    assert out.shape[-1] < audio.shape[-1]
    assert out.shape[-1] > voice_end - voice_start
    # the loud tone is still present somewhere in the trimmed output
    assert np.abs(out).max() == pytest.approx(1.0, abs=0.05)
    # a contiguous loud region close to the tone's own length remains
    loud = np.abs(out[0]) > 0.1
    assert loud.sum() >= (voice_end - voice_start) * 0.9


def test_remove_silence_falls_back_to_numpy_when_pydub_is_unavailable(monkeypatch):
    """Locks the numpy fallback: same contract as the pydub path when pydub is absent.

    ``_remove_silence_pydub`` does ``from pydub import AudioSegment`` at call time, so
    forcing ``sys.modules["pydub"] = None`` makes that import raise ImportError and
    ``remove_silence`` falls through to the pure-numpy RMS-window implementation
    (phoonnx/thirdparty/omnivoice/audio.py:215-245).
    """
    audio, sr, voice_start, voice_end = _silence_padded_tone()
    kwargs = dict(mid_sil=300, lead_sil=50, trail_sil=50, silence_threshold=-50.0)

    reference = remove_silence(audio, sr, **kwargs)

    monkeypatch.setitem(sys.modules, "pydub", None)
    monkeypatch.setitem(sys.modules, "pydub.silence", None)
    fallback = remove_silence(audio, sr, **kwargs)

    # the fallback actually ran its own code path, not a cached pydub result
    assert fallback.shape[0] == 1
    assert fallback.shape[-1] > 0
    # both paths trim the vast majority of the padding
    assert fallback.shape[-1] < audio.shape[-1]
    assert reference.shape[-1] < audio.shape[-1]
    # the documented contract: trim boundaries may shift a little between the
    # two implementations, but not by more than the ms-resolution of the
    # RMS-window scan (chunk_ms=10 -> tens of ms of samples at this rate)
    tolerance = int(sr * 0.05)  # 50ms
    assert abs(fallback.shape[-1] - reference.shape[-1]) <= tolerance
    # the voiced tone itself must still be intact and full-amplitude in the
    # fallback output, not silently dropped or attenuated
    assert np.abs(fallback).max() == pytest.approx(1.0, abs=0.05)
    loud = np.abs(fallback[0]) > 0.1
    assert loud.sum() >= (voice_end - voice_start) * 0.9


# ---------------------------------------------------------------------------
# End to end through synthesize()
# ---------------------------------------------------------------------------

def test_synthesize_produces_audio_and_reports_the_codes():
    ad = _adapter(decoder=_FakeDecoder())
    sess = _FakeBackbone(preferred=9)
    text = "hello"
    result = ad.synthesize(_req([ord(c) for c in text], num_step=3, position_temperature=0.0),
                           sess)
    assert result.audio.ndim == 1 and result.audio.size > 0
    assert result.audio.dtype == np.float32
    codes = result.extras["audio_codes"]
    assert codes.shape == (NUM_CODEBOOKS, result.extras["target_frames"])
    assert (codes == 9).all()
    assert ad.decoder.calls[0].shape[0] == NUM_CODEBOOKS


def test_synthesize_uses_the_language_from_the_voice_params():
    ad = _adapter(decoder=_FakeDecoder())
    sess = _FakeBackbone()
    ad.synthesize(_req([ord(c) for c in "hi"], num_step=1, lang="pt-BR"), sess)
    prompt = ad._decode_ids(sess.calls[0]["input_ids"][0, 0])
    assert "<|lang_start|>pt<|lang_end|>" in prompt


def test_synthesize_without_a_tokenizer_says_what_is_missing():
    ad = OmniVoiceAdapter()
    with pytest.raises(RuntimeError, match="bpe_tokenizer_path"):
        ad.synthesize(_req([1, 2]), _FakeBackbone())


def test_decode_codes_without_a_decoder_says_what_is_missing():
    ad = _adapter()
    with pytest.raises(RuntimeError, match="decoder_path"):
        ad.decode_codes(np.zeros((NUM_CODEBOOKS, 3), np.int64))


def test_encode_text_does_not_stash_reference_state_on_the_adapter():
    # the reference transcription travels on request.params, not on self -- this
    # adapter instance is shared across concurrent requests
    ad = _adapter()

    class _Syn:
        speaker_reference_text = "Reference clip"

    ids = ad.encode_text("target", voice=None, syn_config=_Syn())
    assert ids == [[ord(c) for c in "target"]]
    assert not hasattr(ad, "_reference_text")


def test_synthesize_uses_the_reference_text_from_request_params_not_from_encode_text():
    """Regression: encode_text used to stash the reference transcription on self, which
    the shared adapter (one per voice_id under the threaded server) could serve to a
    different, concurrently-running request's synthesize() call. It must come from
    request.params instead."""
    ad = _adapter(decoder=_FakeDecoder())
    sess = _FakeBackbone()
    ad.encode_reference = lambda audio, sr: (np.zeros((NUM_CODEBOOKS, 2), np.int64), 0.1)

    class _SynA:
        speaker_reference_text = "Request A reference"

    class _SynB:
        speaker_reference_text = "Request B reference"

    # simulate interleaving: request A's encode_text runs, then request B's encode_text
    # runs on the same shared adapter (as would happen under a threaded server), and
    # only then does request A's synthesize() run.
    ad.encode_text("a text", voice=None, syn_config=_SynA())
    ad.encode_text("b text", voice=None, syn_config=_SynB())

    ref_audio = (np.zeros(SAMPLE_RATE, np.float32), SAMPLE_RATE)
    req_a = _req([ord(c) for c in "a text"], num_step=1,
                speaker_reference_text="Request A reference", reference_audio=ref_audio)
    ad.synthesize(req_a, sess)

    prompt = ad._decode_ids(sess.calls[0]["input_ids"][0, 0])
    assert "Request A reference" in prompt
    assert "Request B reference" not in prompt


def test_empty_text_yields_no_chunks():
    ad = _adapter()

    class _Syn:
        speaker_reference_text = None

    assert ad.encode_text("   ", voice=None, syn_config=_Syn()) == []


def test_post_process_restores_a_quiet_reference_loudness():
    ad = _adapter()
    loud = np.ones(SAMPLE_RATE, np.float32) * 0.5
    out = ad.post_process(loud, ref_rms=0.01, pad_duration=0.0, fade_duration=0.0)
    assert np.abs(out).max() < 0.5


def test_post_process_normalises_when_there_is_no_reference():
    ad = _adapter()
    quiet = np.ones(SAMPLE_RATE, np.float32) * 0.01
    out = ad.post_process(quiet, ref_rms=None, pad_duration=0.0, fade_duration=0.0)
    assert np.abs(out).max() == pytest.approx(0.5, abs=1e-3)


def test_voice_index_entries_are_well_formed():
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "phoonnx" / "voice_index" / "omnivoice.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    assert len(index) > 500          # the checkpoint advertises 600+ languages
    for vid, entry in index.items():
        assert entry["voice_id"] == vid
        assert entry["engine"] == "omnivoice"
        assert entry["alphabet"] == "graphemes"
        assert entry["model_url"].endswith("omnivoice_backbone.onnx")
        assert set(entry["aux_model_urls"]) == {
            "acoustic_encoder_path", "semantic_encoder_path",
            "quantizer_encoder_path", "decoder_path", "bpe_tokenizer_path"}
    assert "omnivoice/en" in index and "omnivoice/zh" in index
