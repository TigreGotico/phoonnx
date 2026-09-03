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
        tokenizer = None
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


# --- ProxectoNos Galician StyleTTS2 (Celtia / Brais) -------------------------

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "scripts" / "conversion" / "styletts2"))
from gl_vocab import build_phoneme_id_map  # noqa: E402

# the upstream training table, vendored so the vocabulary check runs offline
_GL_TOKEN_MAP = json.loads(
    (Path(__file__).parent / "proxectonos_gl_phoneme_token_maps.json").read_text())
_VOICE_INDEX = Path(__file__).resolve().parent.parent / "phoonnx" / "voice_index" / "styletts2.json"


def _gl_index_entry(voice):
    return json.loads(_VOICE_INDEX.read_text())[f"proxectonos/{voice}-styletts2"]


@pytest.mark.parametrize("voice", ["celtia", "brais"])
def test_galician_index_entry_is_a_cloning_cotovia_voice(voice):
    e = _gl_index_entry(voice)
    assert e["engine"] == "styletts2"
    assert e["lang"] == "gl-ES"
    # these checkpoints were trained on Cotovia notation, not espeak IPA
    assert e["phoneme_type"] == "cotovia" and e["alphabet"] == "cotovia"
    # single-speaker: a default style ships, and a reference clip can override it
    assert e["style_url"].endswith(f"proxectonos-gl-{voice}/style.bin")
    assert e["speaker_encoder_url"].endswith(f"proxectonos-gl-{voice}/style_encoder.onnx")
    assert e["speaker_encoder_type"] == "styletts2_style"
    for url in (e["model_url"], e["config_url"], e["style_url"], e["speaker_encoder_url"]):
        assert f"/proxectonos-gl-{voice}/" in url


def test_galician_voices_are_the_only_cotovia_styletts2_entries():
    idx = json.loads(_VOICE_INDEX.read_text())
    cotovia = {k for k, v in idx.items() if v.get("alphabet") == "cotovia"}
    assert cotovia == {"proxectonos/celtia-styletts2", "proxectonos/brais-styletts2"}


def test_galician_vocab_is_the_69_symbol_cotovia_phoneset():
    """The checkpoints declare ``n_token: 69`` -- not the yl4579 178-symbol set.

    Rebuilt here the way the exporter does, so a wrong vocabulary in the shipped
    config.json cannot pass silently.
    """
    vocab = build_phoneme_id_map(_GL_TOKEN_MAP)
    assert set(vocab.values()) == set(range(69))
    # the ids the upstream token handling depends on
    assert vocab["X"] == 0          # unknown == the token training pads with
    assert vocab[" "] == 1          # word separator (a trained speech symbol)
    # Cotovia surface forms fold onto the trained single-symbol ids
    assert vocab["rr"] == vocab["R"]
    assert vocab["tS"] == vocab["W"]
    assert vocab["a^"] == vocab["á"] and vocab["o^"] == vocab["ó"]


def test_galician_config_pads_with_the_token_training_used():
    """The Galician voices pad with "X" (id 0), the token upstream's
    ``meldataset.py`` inserts and appends on every training sample
    (``text.insert(0, 0)``/``text.append(0)``).

    Upstream's own ``inference.py`` instead prepends the word separator
    (id 1). That is a trained speech symbol, so it makes the voice speak an
    extra syllable: on 52 Galician sentences it costs Brais 0.196 WER against
    0.122 for id 0.
    """
    from phoonnx.tokenizer import TTSTokenizer, Vocabulary

    vocab = build_phoneme_id_map(_GL_TOKEN_MAP)
    tok = TTSTokenizer(vocabulary=Vocabulary(char2idx=vocab, pad="X"),
                       add_blank_char=False, add_blank_word=False, use_eos_bos=False,
                       blank_at_start=False, blank_at_end=False)
    assert tok.pad_id == 0

    class _Cfg:
        tokenizer = tok
        engine_params = {}

    adapter = StyleTTS2Adapter()
    adapter.configure(_Cfg())
    feed = adapter.build_feed_dict(_req(5), _Sess(["input_ids", "speed"]))
    assert feed["input_ids"][0, 0] == 0
    # the word separator must NOT be what the sequence starts with
    assert feed["input_ids"][0, 0] != vocab[" "]


def test_configurable_pad_still_honours_a_non_zero_pad():
    """The pad id is read off the voice's tokenizer, not hardcoded -- a
    StyleTTS2 lineage that reorders its vocabulary must still pad correctly."""
    from phoonnx.tokenizer import TTSTokenizer, Vocabulary

    tok = TTSTokenizer(vocabulary=Vocabulary(char2idx={"a": 0, "b": 1, "$": 2}, pad="$"),
                       add_blank_char=False, add_blank_word=False, use_eos_bos=False,
                       blank_at_start=False, blank_at_end=False)

    class _Cfg:
        tokenizer = tok
        engine_params = {}

    adapter = StyleTTS2Adapter()
    adapter.configure(_Cfg())
    assert adapter.build_feed_dict(_req(5), _Sess(["input_ids", "speed"]))["input_ids"][0, 0] == 2


def test_default_pad_id_is_unchanged_for_the_dollar_vocabularies():
    """ddatt / bsc / kokoro all pad with "$" == id 0; the configurable pad must
    not move them."""
    from phoonnx.tokenizer import TTSTokenizer, Vocabulary

    tok = TTSTokenizer(vocabulary=Vocabulary(char2idx={"$": 0, "a": 1, "b": 2}, pad="$"),
                       add_blank_char=False, add_blank_word=False, use_eos_bos=False,
                       blank_at_start=False, blank_at_end=False)

    class _Cfg:
        tokenizer = tok
        engine_params = {}

    adapter = StyleTTS2Adapter()
    adapter.configure(_Cfg())
    assert adapter.build_feed_dict(_req(5), _Sess(["input_ids", "speed"]))["input_ids"][0, 0] == 0


def test_cotovia_phonemizer_covers_the_galician_vocabulary():
    """Every symbol a stress-marked Cotovia transcription emits must be in the
    voice's vocabulary -- otherwise the tokenizer silently drops phonemes."""
    pytest.importorskip("pycotovia")
    from scriptconv.phonemizers.gl import CotoviaPhonemizer
    from phoonnx.config import Alphabet

    vocab = build_phoneme_id_map(_GL_TOKEN_MAP)
    compounds = sorted((k for k in vocab if len(k) > 1), key=len, reverse=True)
    text = ("Este é un sistema de conversión de texto a voz en lingua galega. "
            "O carro do labrego cruzaba a corredoira e chovía moito.")
    ps = CotoviaPhonemizer(alphabet=Alphabet.COTOVIA, model="stress").phonemize_string(text, "gl")
    i, unknown = 0, []
    while i < len(ps):
        for c in compounds:
            if ps.startswith(c, i):
                i += len(c)
                break
        else:
            if ps[i] not in vocab:
                unknown.append(ps[i])
            i += 1
    assert not unknown, f"phonemes outside the voice vocabulary: {sorted(set(unknown))}"


# --- BSC-LT multispeaker StyleTTS2 (Spanish / Catalan named speakers) --------

_BSC_CA_SPEAKERS = ["bet", "eli", "eva", "jan", "mar", "ona",
                    "pau", "pep", "pol", "teo", "uri"]
_BSC_ES_SPEAKERS = ["3946", "8882", "9972", "10246", "11797", "12367"]


def _index():
    return json.loads(_VOICE_INDEX.read_text())


@pytest.mark.parametrize("lang,speaker,voice", (
    [("ca", s, f"bsc/ca-{s}") for s in _BSC_CA_SPEAKERS]
    + [("es", s, f"bsc/es-cml{s}") for s in _BSC_ES_SPEAKERS]))
def test_bsc_speaker_entry_reuses_the_shared_checkpoint(lang, speaker, voice):
    """Every named BSC speaker is the SAME graph plus its own style blob.

    A per-speaker model_url would mean 545 MB downloaded 17 times over.
    """
    e = _index()[voice]
    d = f"bsc-{lang}-styletts2"
    assert e["voice_id"] == voice
    assert e["engine"] == "styletts2" and e["lang"] == lang
    # BSC trained on espeak IPA, same front-end as the zero-shot parent voice
    assert e["phoneme_type"] == "espeak" and e["alphabet"] == "ipa"
    assert e["model_url"].endswith(f"{d}/model.onnx")
    assert e["config_url"].endswith(f"{d}/config.json")
    # the ONLY thing that differs between speakers
    assert e["style_url"].endswith(f"{d}/{speaker}.bin")
    # the reference-clip path stays available, so a caller can still clone
    assert e["speaker_encoder_url"].endswith(f"{d}/style_encoder.onnx")
    assert e["speaker_encoder_type"] == "styletts2_style"


@pytest.mark.parametrize("lang,speakers", [("ca", _BSC_CA_SPEAKERS), ("es", _BSC_ES_SPEAKERS)])
def test_bsc_speaker_styles_are_distinct_per_speaker(lang, speakers):
    """No two speakers may share a style blob -- that would silently ship the
    same voice under several names."""
    idx = _index()
    prefix = "bsc/ca-" if lang == "ca" else "bsc/es-cml"
    urls = [idx[f"{prefix}{s}"]["style_url"] for s in speakers]
    assert len(set(urls)) == len(urls)


def test_bsc_parent_voices_stay_reference_only():
    """``bsc/<lang>-styletts2`` is the zero-shot cloning entry: it must NOT gain a
    default style, or callers lose the "reference clip required" error."""
    for voice in ("bsc/es-styletts2", "bsc/ca-styletts2"):
        e = _index()[voice]
        assert "style_url" not in e or e["style_url"] is None
        assert e["speaker_encoder_type"] == "styletts2_style"


def test_bsc_speaker_ids_match_the_exporter():
    """The voice index and the exporter must not drift apart."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                           / "scripts" / "conversion" / "styletts2"))
    from export_bsc_speakers import SPEAKERS

    assert sorted(SPEAKERS["ca"]) == sorted(_BSC_CA_SPEAKERS)
    assert sorted(SPEAKERS["es"]) == sorted(_BSC_ES_SPEAKERS)
    idx = _index()
    assert {k for k in idx if k.startswith("bsc/ca-") and k != "bsc/ca-styletts2"} == \
        {f"bsc/ca-{s}" for s in SPEAKERS["ca"]}
    assert {k for k in idx if k.startswith("bsc/es-") and k != "bsc/es-styletts2"} == \
        {f"bsc/es-cml{s}" for s in SPEAKERS["es"]}


def test_bsc_style_blob_reshapes_to_a_single_256_row(tmp_path):
    """A preset StyleTTS2 style is one 256-d row (ref_p ++ ref_s), unlike Kokoro's
    length-indexed pack -- so the adapter must not add a trailing pad for it."""
    blob = tmp_path / "pau.bin"
    np.arange(256, dtype=np.float32).tofile(blob)

    class _Cfg:
        tokenizer = None
        engine_params = {"style_path": str(blob)}

    a = StyleTTS2Adapter()
    a.configure(_Cfg())
    assert a.style_pack.shape == (1, 256)
    sess = _Sess(["input_ids", "style", "speed"])
    feed = a.build_feed_dict(_req(5), sess)
    assert feed["style"].shape == (1, 256)
    assert np.allclose(feed["style"][0], np.arange(256))
    # single-row pack -> leading pad only
    assert feed["input_ids"].shape == (1, 6) and feed["input_ids"][0, 0] == 0


def test_bsc_reference_clip_overrides_the_preset_style(tmp_path):
    """A named BSC voice still accepts a reference clip; cloning wins over the
    shipped preset."""
    blob = tmp_path / "ona.bin"
    np.zeros(256, dtype=np.float32).tofile(blob)

    class _Enc:
        def encode(self, audio, sr):
            return np.full(256, 0.5, dtype=np.float32)

    class _Cfg:
        tokenizer = None
        engine_params = {"style_path": str(blob)}

    a = StyleTTS2Adapter(speaker_encoder=_Enc())
    a.configure(_Cfg())
    sess = _Sess(["input_ids", "style", "speed"])
    feed = a.build_feed_dict(
        _req(5, reference_audio=(np.zeros(24000, np.float32), 24000)), sess)
    assert np.allclose(feed["style"][0], 0.5)
