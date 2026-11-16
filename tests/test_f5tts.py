import types

import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.f5tts import F5TTSAdapter


def _req(**params):
    return AdapterSynthesisRequest(phoneme_ids=np.array([[4, 5, 6]], np.int64),
                                   phoneme_lengths=np.array([3], np.int64),
                                   speaker_id=0, language_id=0, params=params)


class _Inp:
    def __init__(self, type_=None, name=None):
        self.type = type_
        self.name = name


class _FakePreprocess:
    """Mimics the DakeQQ F5_Preprocess.onnx graph:
    audio, text_ids, max_duration -> noise, rope_cos_q, rope_sin_q, rope_cos_k,
    rope_sin_k, cat_mel_text, cat_mel_text_drop, ref_signal_len
    """
    def get_inputs(self):
        return [_Inp(type_="tensor(float)", name="audio")]

    def run(self, output_names, feed):
        max_duration = int(feed["max_duration"][0])
        n_mels = 8
        noise = np.random.default_rng(0).standard_normal((1, max_duration, n_mels)).astype(np.float32)
        rope = np.zeros((1, max_duration, n_mels), np.float32)
        cat_mel_text = np.zeros((1, max_duration, n_mels), np.float32)
        cat_mel_text_drop = np.zeros((1, max_duration, n_mels), np.float32)
        ref_signal_len = np.array(feed["audio"].shape[-1] // 256 + 1, np.int64)
        return [noise, rope, rope, rope, rope, cat_mel_text, cat_mel_text_drop, ref_signal_len]


class _FakeTransformer:
    """Mimics F5_Transformer.onnx: one Euler step per call, advances time_step.

    Exposes ``get_inputs()`` with the last input renamed to ``time_step.1`` —
    torch.onnx.export does this in the real DakeQQ export because "time_step"
    collides with an output name — so the adapter must feed by position, not by
    the literal string "time_step".
    """
    def __init__(self):
        self.calls = 0
        self._names = ["noise", "rope_cos_q", "rope_sin_q", "rope_cos_k",
                        "rope_sin_k", "cat_mel_text", "cat_mel_text_drop", "time_step.1"]

    def get_inputs(self):
        return [_Inp(type_="tensor(float)", name=n) for n in self._names[:-1]] + \
               [_Inp(type_="tensor(int32)", name=self._names[-1])]

    def run(self, output_names, feed):
        self.calls += 1
        noise = feed[self._names[0]]  # pass through unchanged (deterministic for the test)
        time_step = feed[self._names[-1]] + 1
        return [noise, time_step]


class _FakeDecode:
    """Mimics F5_Decode.onnx: denoised, ref_signal_len -> output_audio."""
    def run(self, output_names, feed):
        denoised = feed["denoised"]
        n_frames = denoised.shape[1]
        audio = np.linspace(-0.5, 0.5, n_frames * 256, dtype=np.float32).reshape(1, 1, -1)
        return [audio]


def _configured_adapter(nfe=4):
    ad = F5TTSAdapter()
    ad.preprocess = _FakePreprocess()
    ad.decode = _FakeDecode()
    ad._nfe = nfe
    return ad


def test_f5tts_registered():
    from phoonnx.engines import list_engines
    assert "f5tts" in list_engines()


def test_f5tts_detect_by_config():
    assert F5TTSAdapter.detect({"engine": "f5tts"})
    assert F5TTSAdapter.detect({"engine": "habibi"})
    assert F5TTSAdapter.detect({"engine_params": {"preprocess_path": "x.onnx", "nfe": 32}})
    assert not F5TTSAdapter.detect({"engine": "vits"})
    assert not F5TTSAdapter.detect(None)


def test_f5tts_detect_by_session_inputs():
    class _Inp:
        def __init__(self, name):
            self.name = name

    class _Sess:
        def get_inputs(self):
            return [_Inp(n) for n in ("noise", "rope_cos_q", "cat_mel_text", "time_step")]

    assert F5TTSAdapter.detect(session=_Sess())


def test_f5tts_requires_preprocess():
    # not configured -> clear error rather than an opaque None.run crash
    with pytest.raises(RuntimeError):
        F5TTSAdapter().synthesize(_req(reference_audio=(np.zeros(1000, np.float32), 24000),
                                       ref_text_tokens=[1, 2]), None)


def test_f5tts_requires_reference():
    ad = _configured_adapter()
    with pytest.raises(RuntimeError):
        ad.synthesize(_req(), None)  # no reference_audio / ref_text_tokens


def test_f5tts_full_pipeline_with_decode():
    ad = _configured_adapter(nfe=4)
    transformer = _FakeTransformer()
    ref_audio = (np.zeros(24000, np.float32), 24000)  # 1s of silence @ 24kHz
    result = ad.synthesize(
        _req(reference_audio=ref_audio, ref_text_tokens=[1, 2, 3]),
        transformer,
    )
    # (nfe - 1) Euler steps: matches the DakeQQ reference inference loop
    assert transformer.calls == 3
    assert result.audio.ndim == 1
    assert result.audio.size > 0
    assert np.any(result.audio != 0)  # non-silent


def test_f5tts_accepts_prompt_tokens_key():
    """TTSVoice hands the reference transcription over as 'prompt_tokens'
    (the shared in-context cloning key) — the adapter must accept it."""
    ad = _configured_adapter(nfe=2)
    result = ad.synthesize(
        _req(reference_audio=(np.zeros(24000, np.float32), 24000),
             prompt_tokens=[1, 2, 3]),
        _FakeTransformer(),
    )
    assert result.audio.size > 0


def test_f5tts_pipeline_resamples_reference_audio():
    ad = _configured_adapter(nfe=2)
    transformer = _FakeTransformer()
    ref_audio = (np.zeros(16000, np.float32), 16000)  # needs resample to 24kHz
    result = ad.synthesize(
        _req(reference_audio=ref_audio, ref_text_tokens=[1, 2]),
        transformer,
    )
    assert result.audio.size > 0


def test_f5tts_requires_decode_or_vocoder():
    ad = F5TTSAdapter()
    ad.preprocess = _FakePreprocess()  # decode and vocoder left unset
    with pytest.raises(RuntimeError):
        ad.synthesize(
            _req(reference_audio=(np.zeros(1000, np.float32), 24000),
                 ref_text_tokens=[1, 2]),
            _FakeTransformer(),
        )


def test_f5tts_param_labels_and_defaults():
    ad = F5TTSAdapter()
    labels = ad.param_labels()
    defaults = ad.default_params()
    for key in ("nfe", "cfg_strength", "sway_coefficient", "target_rms", "speed"):
        assert key in labels
        assert key in defaults


def test_f5tts_dialect_wrapping():
    """Habibi Unified dialect control: gen ids wrapped as {dialect}〈 text 〉,
    through the same char->id mapping (upstream habibi_tts text_list_formatter)."""
    from phoonnx.engines.f5tts import HABIBI_DIALECT_MAP

    class _RecordingPreprocess(_FakePreprocess):
        def __init__(self):
            self.text_ids = None

        def run(self, output_names, feed):
            self.text_ids = feed["text_ids"].copy()
            return super().run(output_names, feed)

    ad = _configured_adapter(nfe=2)
    pre = _RecordingPreprocess()
    ad.preprocess = pre
    # mimic the unified vocab: dialect chars + brackets present
    ad._char2idx = {"⑥": 2713, "〈": 2728, "〉": 2729}

    ad.synthesize(
        _req(reference_audio=(np.zeros(24000, np.float32), 24000),
             prompt_tokens=[7, 8], dialect="EGY"),
        _FakeTransformer(),
    )
    ids = pre.text_ids[0].tolist()
    # ref tokens first, then ⑥ 〈 gen 〉
    assert ids == [7, 8, 2713, 2728, 4, 5, 6, 2729]
    assert HABIBI_DIALECT_MAP["EGY"] == "⑥"


def test_f5tts_dialect_unknown_raises():
    ad = _configured_adapter(nfe=2)
    ad._char2idx = {"⓪": 1, "〈": 2, "〉": 3}
    with pytest.raises(ValueError):
        ad.synthesize(
            _req(reference_audio=(np.zeros(24000, np.float32), 24000),
                 prompt_tokens=[1], dialect="XXX"),
            _FakeTransformer(),
        )


def test_f5tts_dialect_skipped_without_vocab_tokens():
    """Specialized/plain vocabs lack the control tokens -> tag skipped, not crash."""
    class _RecordingPreprocess(_FakePreprocess):
        def __init__(self):
            self.text_ids = None

        def run(self, output_names, feed):
            self.text_ids = feed["text_ids"].copy()
            return super().run(output_names, feed)

    ad = _configured_adapter(nfe=2)
    pre = _RecordingPreprocess()
    ad.preprocess = pre
    ad._char2idx = {"a": 1}  # no dialect tokens
    ad.synthesize(
        _req(reference_audio=(np.zeros(24000, np.float32), 24000),
             prompt_tokens=[7], dialect="EGY"),
        _FakeTransformer(),
    )
    assert pre.text_ids[0].tolist() == [7, 4, 5, 6]  # untagged


def test_f5tts_voice_index_catalog():
    """The bundled catalog entries must construct valid TTSModelInfo objects
    with aux_model_urls pointing at the preprocess/decode graphs."""
    import json
    import os
    from phoonnx.model_manager import TTSModelInfo
    index = os.path.join(os.path.dirname(__file__), "..", "phoonnx",
                         "voice_index", "f5tts.json")
    with open(index, "r", encoding="utf-8") as f:
        entries = json.load(f)
    assert "f5tts/v1-base" in entries
    assert "habibi/ar-unified" in entries
    assert "silma/v1" in entries
    for voice_id, entry in entries.items():
        info = TTSModelInfo(**entry)
        assert info.engine == "f5tts"
        assert set(info.aux_model_urls) == {"preprocess_path", "decode_path"}
        for url in info.aux_model_urls.values():
            assert url.startswith("https://huggingface.co/Phoonnx/phoonnx-f5tts/")


def test_aux_model_urls_resolved_in_engine_params(tmp_path, monkeypatch):
    """download_aux_models() fetches each graph and engine_params() exposes
    the local paths under the same keys."""
    from phoonnx.model_manager import TTSModelInfo

    info = TTSModelInfo(
        voice_id="f5tts/test", lang="mul", model_url="https://example.com/model.onnx",
        engine="f5tts",
        aux_model_urls={"preprocess_path": "https://example.com/F5_Preprocess.onnx",
                        "decode_path": "https://example.com/F5_Decode.onnx"})
    # these graphs are self-hosted, so they take the direct-download path;
    # keep it inside the test's own cache root
    import phoonnx.model_manager as mm
    monkeypatch.setattr(mm, "HF_HUB_CACHE", str(tmp_path))
    expected = mm._direct_dir("https://example.com/F5_Preprocess.onnx")

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def iter_content(self, chunk_size): return [b"onnx-bytes"]
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(mm.requests, "get", lambda *a, **k: _Resp())

    params = info.engine_params()
    assert params["preprocess_path"] == str(expected / "F5_Preprocess.onnx")
    assert params["decode_path"] == str(expected / "F5_Decode.onnx")
    assert (expected / "F5_Preprocess.onnx").read_bytes() == b"onnx-bytes"


def test_f5tts_resample():
    a = np.zeros(1000, np.float32)
    assert F5TTSAdapter._resample(a, 24000, 24000) is a  # no-op
    out = F5TTSAdapter._resample(a, 22050, 24000)
    assert abs(len(out) - 1000 * 24000 / 22050) <= 2


def test_silma_voice_index_entries():
    """SILMA TTS v1 ships as two catalog listings (Arabic primary + English)
    pointing at the same silma-tts-v1 ONNX export."""
    import json
    import os
    from phoonnx.model_manager import TTSModelInfo

    index = os.path.join(os.path.dirname(__file__), "..", "phoonnx",
                         "voice_index", "f5tts.json")
    with open(index, "r", encoding="utf-8") as f:
        entries = json.load(f)

    ar = TTSModelInfo(**entries["silma/v1"])
    en = TTSModelInfo(**entries["silma/v1-en"])
    assert ar.lang == "ar"
    assert en.lang == "en"
    for info in (ar, en):
        assert info.engine == "f5tts"
        assert info.phoneme_type == "graphemes"
        assert "/silma-tts-v1/" in info.model_url
        assert info.model_url.endswith("model.onnx")
        assert info.aux_model_urls["preprocess_path"].endswith("F5_Preprocess.onnx")
        assert info.aux_model_urls["decode_path"].endswith("F5_Decode.onnx")
    # both listings resolve to the exact same artifacts
    assert ar.model_url == en.model_url
    assert ar.aux_model_urls == en.aux_model_urls


def test_silma_listed_for_arabic_and_english():
    """The model manager surfaces the silma voices for ar/en lookups."""
    from phoonnx.model_manager import TTSModelManager

    manager = TTSModelManager()
    manager.cache.clear()
    manager.merge_default_voices()
    assert "silma/v1" in manager.voices
    assert "silma/v1-en" in manager.voices
    ar_ids = [v.voice_id for v in manager.get_lang_voices("ar")]
    en_ids = [v.voice_id for v in manager.get_lang_voices("en-US")]
    assert "silma/v1" in ar_ids
    assert "silma/v1-en" in en_ids


def test_namaa_saudi_voice_index_entry():
    """NAMAA Saudi TTS V2 ships as a Saudi-Arabic F5-TTS catalog listing."""
    import json
    import os
    from phoonnx.model_manager import TTSModelInfo

    index = os.path.join(os.path.dirname(__file__), "..", "phoonnx",
                         "voice_index", "f5tts.json")
    with open(index, "r", encoding="utf-8") as f:
        entries = json.load(f)

    assert "namaa/ar-sa-v2" in entries
    info = TTSModelInfo(**entries["namaa/ar-sa-v2"])
    assert info.engine == "f5tts"
    assert info.lang == "ar-SA"
    assert info.phoneme_type == "graphemes"
    assert "/namaa-saudi-tts-v2/" in info.model_url
    assert info.model_url.endswith("model.onnx")
    assert info.aux_model_urls["preprocess_path"].endswith("F5_Preprocess.onnx")
    assert info.aux_model_urls["decode_path"].endswith("F5_Decode.onnx")


def test_namaa_saudi_listed_for_arabic():
    """The model manager surfaces the NAMAA Saudi voice for ar lookups."""
    from phoonnx.model_manager import TTSModelManager

    manager = TTSModelManager()
    manager.cache.clear()
    manager.merge_default_voices()
    assert "namaa/ar-sa-v2" in manager.voices
    ar_ids = [v.voice_id for v in manager.get_lang_voices("ar")]
    assert "namaa/ar-sa-v2" in ar_ids


class _NamedIO:
    def __init__(self, name, type_="tensor(float)"):
        self.name = name
        self.type = type_


class _FakeNamedPreprocess:
    """F5_Preprocess.onnx of the exporter's name-bound contract."""
    NAMES = ["noise", "rope_cos", "rope_sin", "cat_mel_text", "cat_mel_text_drop",
             "ref_signal_len", "rms_scale", "ref_mel_tail"]

    def get_inputs(self):
        return [_NamedIO("audio")]

    def get_outputs(self):
        return [_NamedIO(n) for n in self.NAMES]

    def run(self, output_names, feed):
        frames = int(feed["max_duration"][0])
        values = {name: np.zeros((1, frames, 4), np.float32) for name in self.NAMES[:5]}
        values.update(ref_signal_len=np.array(7, np.int64),
                      rms_scale=np.array([0.5], np.float32),
                      ref_mel_tail=np.full((1, 1, 4), 3.0, np.float32))
        return [values[name] for name in self.NAMES]


class _FakeNamedTransformer:
    """F5 transformer of the name-bound contract: one Euler step per call, the caller
    feeds the step index, and the only output is the new state."""

    def __init__(self, metadata=None, step_dtype=np.int32):
        self.steps = []
        self.metadata = metadata or {}
        self.step_dtype = step_dtype

    def get_inputs(self):
        return [_NamedIO(n) for n in _FakeNamedPreprocess.NAMES[:5]] + \
               [_NamedIO("time_step", f"tensor({np.dtype(self.step_dtype).name})")]

    def get_outputs(self):
        return [_NamedIO("denoised")]

    def get_modelmeta(self):
        return types.SimpleNamespace(custom_metadata_map=self.metadata)

    def run(self, output_names, feed):
        assert feed["time_step"].dtype == self.step_dtype
        self.steps.append(int(feed["time_step"][0]))
        return [feed["noise"] + 1.0]


class _FakeNamedDecode:
    def __init__(self, output=None):
        self.feed = None
        self.output = np.full((1, 1, 10), 0.25, np.float32) if output is None else output

    def get_inputs(self):
        return [_NamedIO(n) for n in ("denoised", "ref_signal_len", "rms_scale", "ref_mel_tail")]

    def run(self, output_names, feed):
        self.feed = feed
        return [self.output]


def _run_name_bound(transformer, decode=None, **params):
    ad = F5TTSAdapter()
    ad.preprocess = _FakeNamedPreprocess()
    ad.decode = decode or _FakeNamedDecode()
    return ad.synthesize(_req(reference_audio=(np.zeros(24000, np.float32), 24000),
                              ref_text_tokens=[1, 2, 3], **params), transformer)


def test_f5tts_runs_the_name_bound_export_contract():
    """DakeQQ's Inference_F5_TTS_ONNX.py feeds the step indices 0 .. NFE-1, binds the
    preprocess outputs by name, and decodes the last state with ref_signal_len,
    rms_scale and ref_mel_tail. The Twi Asante OpenBible export uses this contract."""
    ad = F5TTSAdapter()
    ad.preprocess = _FakeNamedPreprocess()
    ad.decode = decode = _FakeNamedDecode()
    transformer = _FakeNamedTransformer()
    result = ad.synthesize(_req(reference_audio=(np.zeros(24000, np.float32), 24000),
                                ref_text_tokens=[1, 2, 3], nfe=5), transformer)
    assert transformer.steps == [0, 1, 2, 3, 4]
    assert np.all(decode.feed["denoised"] == 5.0), "each of the five steps adds one"
    assert decode.feed["ref_signal_len"] == 7
    assert decode.feed["rms_scale"].tolist() == [0.5]
    assert np.all(decode.feed["ref_mel_tail"] == 3.0)
    assert result.audio.tolist() == [0.25] * 10


@pytest.mark.parametrize("requested", [8, 40])
def test_name_bound_export_runs_the_step_count_in_its_metadata(caplog, requested):
    """The exporter bakes the time schedule into the graph and records its length as
    the ``nfe_step`` metadata; upstream refuses any other NFE. A request for fewer or
    more steps than a 32-step export still runs the 32 steps and says so."""
    transformer = _FakeNamedTransformer(metadata={"nfe_step": "32"})
    decode = _FakeNamedDecode()
    with caplog.at_level("WARNING", logger="phoonnx.engines.f5tts"):
        _run_name_bound(transformer, decode, nfe=requested)
    assert transformer.steps == list(range(32))
    assert np.all(decode.feed["denoised"] == 32.0)
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "32" in warnings[0] and f"nfe={requested}" in warnings[0]


def test_name_bound_export_without_step_metadata_runs_the_requested_nfe(caplog):
    transformer = _FakeNamedTransformer(metadata={"in_sample_rate": "24000"})
    with caplog.at_level("WARNING", logger="phoonnx.engines.f5tts"):
        _run_name_bound(transformer, nfe=6)
    assert transformer.steps == list(range(6))
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_name_bound_export_scales_int16_decoder_output():
    """The exporter can emit int16 PCM from F5_Decode; the voice returns float audio."""
    pcm = np.array([[[-32768, -16384, 0, 8192, 32767]]], np.int16)
    result = _run_name_bound(_FakeNamedTransformer(), _FakeNamedDecode(pcm), nfe=2)
    assert result.audio.dtype == np.float32
    assert result.audio.tolist() == [-1.0, -0.5, 0.0, 0.25, 32767 / 32768]


def test_name_bound_export_feeds_an_int64_step_index():
    transformer = _FakeNamedTransformer(step_dtype=np.int64)
    _run_name_bound(transformer, nfe=3)
    assert transformer.steps == [0, 1, 2]


def _f5_index():
    import json
    import os
    index = os.path.join(os.path.dirname(__file__), "..", "phoonnx",
                         "voice_index", "f5tts.json")
    with open(index, "r", encoding="utf-8") as f:
        return json.load(f)


def test_twi_asante_openbible_voice_index_entry():
    """The Twi (Asante) OpenBible export in the F5 mirror has a catalog entry."""
    from phoonnx.model_manager import TTSModelInfo

    entries = _f5_index()
    assert "openbible/tw-asante" in entries
    info = TTSModelInfo(**entries["openbible/tw-asante"])
    folder = "https://huggingface.co/Phoonnx/phoonnx-f5tts/resolve/main/tw-asante-openbible/"
    assert info.engine == "f5tts"
    assert info.phoneme_type == "graphemes"
    assert info.model_url == folder + "model.onnx"
    assert info.config_url == folder + "config.json"
    assert info.aux_model_urls == {"preprocess_path": folder + "F5_Preprocess.onnx",
                                   "decode_path": folder + "F5_Decode.onnx"}


# voice id -> (language of the entry, lang_code of the published config.json, the
# number 3 as the voice's text front end must read it)
_SOURCE_LANGUAGES = {
    # silma-ai/silma-tts model card: "language: en, ar", a "bilingual (Arabic/English)"
    # model; one config.json (lang_code "ar") serves both entries
    "silma/v1": ("ar", "ar", "ثلاثة"),
    "silma/v1-en": ("en", "ar", "three"),
    # multilingual-tts/F5-TTS-OpenBible-Twi-Asante, trained on the "Twi (Asante)"
    # config of davidguzmanr/open-bible-resources
    "openbible/tw-asante": ("tw", "tw", None),
}


@pytest.mark.parametrize("voice_id", sorted(_SOURCE_LANGUAGES))
def test_entry_language_matches_the_source_and_reaches_the_front_end(voice_id):
    """The entry's language, not the shared config's, is what the voice reads text as."""
    from phoonnx.model_manager import TTSModelInfo
    from phoonnx.voice import TTSVoice

    lang, published, three = _SOURCE_LANGUAGES[voice_id]
    info = TTSModelInfo(**_f5_index()[voice_id])
    assert info.lang == lang
    info.download_config = lambda: {"engine": "f5tts", "phoneme_type": "graphemes",
                                    "alphabet": "unicode", "lang_code": published}
    assert info.config.lang_code == lang
    if three is not None:
        voice = TTSVoice(config=info.config, session=None)
        assert "".join(voice.phonemize("3")[0]) == three
