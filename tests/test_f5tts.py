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
            assert url.startswith("https://huggingface.co/OpenVoiceOS/phoonnx-f5tts/")


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
