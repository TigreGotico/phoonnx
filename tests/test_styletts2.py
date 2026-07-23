"""Tests for the StyleTTS2 / Kokoro adapter."""
import numpy as np
import pytest

from phoonnx.engines import get_adapter, detect_engine
from phoonnx.engines.styletts2 import StyleTTS2Adapter
from phoonnx.engines.base import AdapterSynthesisRequest


class _In:
    def __init__(self, name): self.name = name


class _Sess:
    def __init__(self, names): self._i = [_In(n) for n in names]
    def get_inputs(self): return self._i


def _req(n=5, **p):
    return AdapterSynthesisRequest(phoneme_ids=np.arange(1, n + 1, dtype=np.int64)[None, :],
                                   phoneme_lengths=np.array([n], np.int64), params=p)


def test_registered_and_detect():
    assert isinstance(get_adapter("styletts2"), StyleTTS2Adapter)
    assert StyleTTS2Adapter.detect(config={"engine": "styletts2"}) is True
    assert StyleTTS2Adapter.detect(config={"engine": "kokoro"}) is True  # same family
    assert isinstance(detect_engine(config={"engine": "kokoro"}), StyleTTS2Adapter)


def test_styletts2_feed_pads_and_filters():
    # baked-ref StyleTTS2: input_ids + attention_mask + speed (no style)
    sess = _Sess(["input_ids", "attention_mask", "speed"])
    feed = StyleTTS2Adapter().build_feed_dict(_req(5, speed=1.2), sess)
    assert set(feed) == {"input_ids", "attention_mask", "speed"}
    # plain StyleTTS2 (no multi-row style pack) pads the START only; a trailing
    # pad makes the model decode a noise burst at the end.
    assert feed["input_ids"].shape == (1, 6)        # $-padded at start only
    assert feed["input_ids"][0, 0] == 0 and feed["input_ids"][0, -1] != 0
    assert feed["speed"][0] == pytest.approx(1.2)
    assert feed["attention_mask"].shape == (1, 6)


def test_kokoro_style_pack_length_indexed():
    # Kokoro: a [510, 256] style pack, indexed by token length
    pack = np.arange(510 * 256, dtype=np.float32).reshape(510, 256)
    sess = _Sess(["input_ids", "style", "speed"])
    feed = StyleTTS2Adapter(style_pack=pack).build_feed_dict(_req(5), sess)
    assert "style" in feed and feed["style"].shape == (1, 256)
    # 5 tokens (unpadded) -> style_pack[5]; upstream kokoro-onnx indexes
    # voices[voice][len(tokens)] BEFORE padding, so the padded length (7) is wrong
    assert np.allclose(feed["style"][0], pack[5])
    assert "attention_mask" not in feed   # filtered (not a model input)


def test_length_scale_falls_back_to_speed():
    # SynthesisConfig.length_scale is the canonical speed knob; honour it when
    # the adapter's native "speed" key is not explicitly set.
    sess = _Sess(["input_ids", "attention_mask", "speed"])
    feed = StyleTTS2Adapter().build_feed_dict(_req(5, length_scale=1.3), sess)
    assert feed["speed"][0] == pytest.approx(1.3)


def test_speed_takes_precedence_over_length_scale():
    sess = _Sess(["input_ids", "attention_mask", "speed"])
    feed = StyleTTS2Adapter().build_feed_dict(_req(5, speed=0.7, length_scale=1.3), sess)
    assert feed["speed"][0] == pytest.approx(0.7)


def test_parse_outputs_picks_waveform():
    r = StyleTTS2Adapter().parse_outputs([np.zeros((1, 512, 8), np.float32), np.ones(20000, np.float32)], _req())
    assert r.audio.ndim == 1 and r.audio.size == 20000


# --- phoneme alignment (durations) ---------------------------------------

def test_parse_outputs_no_durations_by_default():
    """Standard StyleTTS2/Kokoro exports emit only the waveform."""
    r = StyleTTS2Adapter().parse_outputs(
        [np.zeros((1, 512, 8), np.float32), np.ones(20000, np.float32)],
        _req(), output_names=["decoder_hidden", "audio"],
    )
    assert "phoneme_id_samples" not in r.extras


def test_parse_outputs_picks_up_named_durations():
    durs = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    r = StyleTTS2Adapter().parse_outputs(
        [np.ones(20000, np.float32), durs],
        _req(), output_names=["audio", "pred_dur"],
    )
    np.testing.assert_array_equal(r.extras["phoneme_id_samples"], [1, 2, 3, 4, 5])


def test_configure_loads_style_pack_from_engine_params(tmp_path):
    """Kokoro: the manager downloads a style blob; configure() reshapes it to [N,256]."""
    import numpy as np
    blob = tmp_path / "style.bin"
    np.arange(510 * 256, dtype=np.float32).tofile(blob)

    class _Cfg:
        engine_params = {"style_path": str(blob)}

    a = StyleTTS2Adapter()
    a.configure(_Cfg())
    assert a.style_pack.shape == (510, 256)


def test_styletts2_cloning_splits_ref_and_s():
    """A cloning StyleTTS2 model takes ref[128]+s[128]; the adapter splits the
    256-d style from the speaker encoder."""
    import numpy as np
    class _Enc:
        def encode(self, audio, sr): return np.arange(256, dtype=np.float32)
    sess = _Sess(["input_ids", "attention_mask", "ref", "s", "speed"])
    a = StyleTTS2Adapter(speaker_encoder=_Enc())
    feed = a.build_feed_dict(_req(5, reference_audio=(np.zeros(24000, np.float32), 24000)), sess)
    assert feed["ref"].shape == (1, 128) and feed["s"].shape == (1, 128)
    assert np.allclose(feed["ref"][0], np.arange(128))
    assert np.allclose(feed["s"][0], np.arange(128, 256))


def test_styletts2_style_encoder_registered():
    from phoonnx.engines.speaker_encoders import list_speaker_encoders
    assert "styletts2_style" in list_speaker_encoders()


def test_missing_style_raises_a_clear_error():
    """The graph conditions on a style vector; without one onnxruntime only
    reported a missing required input."""
    import types, numpy as np, pytest
    from phoonnx.engines.styletts2 import StyleTTS2Adapter
    from phoonnx.engines.base import AdapterSynthesisRequest
    sess = type("S", (), {"get_inputs": lambda self: [
        types.SimpleNamespace(name=n) for n in ("input_ids", "attention_mask", "speed", "style")]})()
    req = AdapterSynthesisRequest(phoneme_ids=np.array([[1, 2]], dtype=np.int64),
                                  phoneme_lengths=np.array([2], dtype=np.int64), params={})
    with pytest.raises(ValueError, match="style"):
        StyleTTS2Adapter().build_feed_dict(req, sess)



def test_kokoro_style_row_ignores_the_padding():
    """The style row must come from the unpadded token count. Selecting it from
    the padded ids shifted every utterance's row, worst on short ones where
    adjacent rows differ most."""
    pack = np.arange(510 * 256, dtype=np.float32).reshape(510, 256)
    sess = _Sess(["input_ids", "style", "speed"])
    for n in (1, 3, 5, 17):
        feed = StyleTTS2Adapter(style_pack=pack).build_feed_dict(_req(n), sess)
        assert np.allclose(feed["style"][0], pack[n]), f"{n} tokens picked the wrong row"
