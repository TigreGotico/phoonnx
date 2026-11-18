import functools
import re
import types

import numpy as np
import pytest

from phoonnx.audio_fx import resample_rate
from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.f5tts import F5TTSAdapter
from tests.conftest import retry_download


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
    # ⑥ 〈 ref gen 〉: upstream wraps the whole ref_text + gen_text string
    assert ids == [2713, 2728, 7, 8, 4, 5, 6, 2729]
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
        assert info.model_url.startswith(_F5_REPOS[voice_id])
        for url in info.aux_model_urls.values():
            assert url.startswith(_F5_REPOS[voice_id])


# voice id -> the per-voice repo of its files
_F5_REPOS = {voice_id: f"https://huggingface.co/Phoonnx/f5tts-{name}/resolve/" for voice_id, name in {
    "f5tts/v1-base": "multi-base-v1-graphemes",
    "habibi/ar-unified": "ar-habibi_unified-graphemes",
    "habibi/ar-alg": "ar_dz-habibi_alg-graphemes",
    "habibi/ar-egy": "ar_eg-habibi_egy-graphemes",
    "habibi/ar-irq": "ar_iq-habibi_irq-graphemes",
    "habibi/ar-mar": "ar_ma-habibi_mar-graphemes",
    "habibi/ar-msa": "ar-habibi_msa-graphemes",
    "habibi/ar-sau": "ar_sa-habibi_sau-graphemes",
    "habibi/ar-uae": "ar_ae-habibi_uae-graphemes",
    "namaa/ar-sa-v2": "ar_sa-namaa_space-v2-graphemes",
    "silma/v1": "multi-silma_ai-v1-graphemes",
    "silma/v1-en": "multi-silma_ai-v1-graphemes",
    "openbible/tw-asante": "tw-asante_openbible-graphemes",
}.items()}


@pytest.mark.parametrize("voice_id", sorted(_F5_REPOS))
def test_entry_loads_the_full_graphs_and_offers_the_int8_graphs(voice_id):
    """Each voice loads its full graphs unless int8 is asked for. The int8 variant
    names the three int8 graphs of the same repo and commit, and keeps the config."""
    from phoonnx.model_manager import TTSModelInfo

    info = TTSModelInfo(**_f5_index()[voice_id])
    pin = re.fullmatch(re.escape(_F5_REPOS[voice_id]) + r"([0-9a-f]{40})/model\.onnx", info.model_url)
    assert pin, info.model_url
    folder = f"{_F5_REPOS[voice_id]}{pin[1]}/"
    assert info.precision is None
    assert info.config_url == folder + "config.json"
    assert info.aux_model_urls == {"preprocess_path": folder + "F5_Preprocess.onnx",
                                   "decode_path": folder + "F5_Decode.onnx"}
    assert info.with_precision("auto", ["CPUExecutionProvider"]) is info
    int8 = info.with_precision("int8")
    assert int8.model_url == folder + "model_int8.onnx"
    assert int8.config_url == info.config_url
    assert int8.aux_model_urls == {"preprocess_path": folder + "F5_Preprocess_int8.onnx",
                                   "decode_path": folder + "F5_Decode_int8.onnx"}


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
    assert resample_rate(a, 24000, 24000) is a  # no-op
    out = resample_rate(a, 22050, 24000)
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
        assert info.phoneme_type == "f5tts_char"
        assert info.model_url.startswith("https://huggingface.co/Phoonnx/f5tts-multi-silma_ai-v1-graphemes/")
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
    assert info.phoneme_type == "f5tts_char"
    assert info.model_url.startswith("https://huggingface.co/Phoonnx/f5tts-ar_sa-namaa_space-v2-graphemes/")
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


# Whole model calls recorded with the upstream code: the reference text ends as
# preprocess_ref_audio_text and infer_batch_process leave it (SWivid/F5-TTS
# 600e5fd, src/f5_tts/infer/utils_infer.py), the text function reads
# ref_text + gen_text (text_list_formatter of SWivid/Habibi-TTS 9274498, or
# convert_char_to_pinyin of F5-TTS), and each token maps with get(c, 0)
# through the voice's vocab.txt. prompt_tokens and phoneme_ids are the ids of
# the reference and generation text alone, as TTSVoice gives them.
UPSTREAM_CALLS = [
    {"reference_text": "قال.", "dialect": "EGY", "prompt_tokens": [1739, 2550, 1741, 14],
     "phoneme_ids": [1743, 1738, 1742],
     "text_ids": [2713, 2728, 1739, 2550, 1741, 14, 0, 0, 1743, 1738, 1742, 2729],
     "char2idx": {" ": 0, ".": 14, "ا": 2550, "ع": 1738, "ق": 1739, "ل": 1741, "م": 1742,
                  "ن": 1743, "⑥": 2713, "〈": 2728, "〉": 2729}},
    {"reference_text": "قال", "dialect": "UNK", "prompt_tokens": [1739, 2550, 1741],
     "phoneme_ids": [1743, 1738, 1742],
     "text_ids": [2707, 2728, 1739, 2550, 1741, 14, 0, 0, 1743, 1738, 1742, 2729],
     "char2idx": {" ": 0, ".": 14, "ا": 2550, "ع": 1738, "ق": 1739, "ل": 1741, "م": 1742,
                  "ن": 1743, "⓪": 2707, "〈": 2728, "〉": 2729}},
    {"reference_text": "هل؟", "dialect": None, "prompt_tokens": [1744, 1741, 0],
     "phoneme_ids": [1743, 1738, 1742], "text_ids": [1744, 1741, 0, 14, 0, 0, 1743, 1738, 1742],
     "char2idx": {" ": 0, ".": 14, "ع": 1738, "ل": 1741, "م": 1742, "ن": 1743, "ه": 1744}},
    {"reference_text": "好。", "dialect": None, "prompt_tokens": [0, 458, 1895],
     "phoneme_ids": [0, 458], "text_ids": [0, 458, 1895, 0, 458],
     "char2idx": {" ": 0, "hao3": 458, "。": 1895}},
]


@pytest.mark.parametrize("call", UPSTREAM_CALLS, ids=lambda c: c["reference_text"])
def test_f5tts_text_ids_equal_the_upstream_call(call):
    """The reference text ends in ". " plus a space unless it ends in 。, and the
    dialect tag wraps the reference and the generation text together."""
    class _RecordingPreprocess(_FakePreprocess):
        def run(self, output_names, feed):
            self.text_ids = feed["text_ids"].copy()
            return super().run(output_names, feed)

    ad = _configured_adapter(nfe=2)
    ad.preprocess = pre = _RecordingPreprocess()
    ad._char2idx = call["char2idx"]
    params = {"reference_audio": (np.zeros(24000, np.float32), 24000),
              "prompt_tokens": call["prompt_tokens"], "speaker_reference_text": call["reference_text"]}
    if call["dialect"]:
        params["dialect"] = call["dialect"]
    request = AdapterSynthesisRequest(phoneme_ids=np.array([call["phoneme_ids"]], np.int64),
                                      phoneme_lengths=np.array([len(call["phoneme_ids"])], np.int64),
                                      speaker_id=0, language_id=0, params=params)
    ad.synthesize(request, _FakeTransformer())
    assert pre.text_ids[0].tolist() == call["text_ids"]


# Whole model calls recorded with the upstream code through TTSVoice inputs
# (fixes/record_test_rows.py and fixes/record_r2_rows.py in the f5-parity
# audit): preprocess_ref_audio_text and the one-byte rule of
# infer_batch_process (SWivid/F5-TTS 600e5fd) end the reference text, convert_char_to_pinyin (F5-TTS v1 Base) or the Habibi-TTS
# 9274498 text function reads ref_text + gen_text, and list_str_to_idx maps the
# tokens through vocab.txt. A reference text with whitespace at its end gives
# the ids of upstream on the reference text without it. Each map holds the
# vocab.txt entries that the texts can give, with their vocab.txt ids.
JOINED_CALLS = [
    {'phoneme_type': 'f5tts_pinyin', 'reference_text': '这就是我。', 'text': 'GPU和CPU不一样。', 'lang': 'zh',
     'text_ids': [0, 1341, 0, 546, 0, 1030, 0, 1178, 1895, 0, 38, 47, 52, 0, 461, 0, 34, 47, 52, 0, 142, 0, 1259, 0, 1249, 1895],
     'phoneme_id_map': {" ": 0, ".": 14, "C": 34, "G": 38, "P": 47, "U": 52, "bu4": 142, "he2": 461, "jiu4": 546, "shi4": 1030, "wo3": 1178, "yang4": 1249, "yi1": 1259, "yi2": 1260, "zhe4": 1341, "。": 1895}},
    {'phoneme_type': 'f5tts_pinyin', 'reference_text': '这就是我。', 'text': 'Python是一种编程语言，对吧？', 'lang': 'zh',
     'text_ids': [0, 1341, 0, 546, 0, 1030, 0, 1178, 1895, 0, 47, 1236, 1082, 441, 827, 765, 0, 1030, 0, 1262, 0, 1356, 0, 118, 0, 190, 0, 1282, 0, 1243, 2541, 0, 317, 0, 79, 2542],
     'phoneme_id_map': {" ": 0, ".": 14, "P": 47, "ba": 79, "bian1": 118, "cheng2": 190, "dui4": 317, "h": 441, "jiu4": 546, "n": 765, "o": 827, "shi4": 1030, "t": 1082, "wo3": 1178, "y": 1236, "yan2": 1243, "yi4": 1262, "yu3": 1282, "zhe4": 1341, "zhong3": 1356, "。": 1895, "，": 2541, "？": 2542}},
    {'phoneme_type': 'f5tts_pinyin', 'reference_text': '这就是我。', 'text': 'I am here.', 'lang': 'en',
     'text_ids': [0, 1341, 0, 546, 0, 1030, 0, 1178, 1895, 40, 0, 62, 704, 0, 441, 325, 940, 325, 14],
     'phoneme_id_map': {" ": 0, ".": 14, "I": 40, "a": 62, "e": 325, "h": 441, "jiu4": 546, "m": 704, "r": 940, "shi4": 1030, "wo3": 1178, "zhe4": 1341, "。": 1895}},
    {'phoneme_type': 'f5tts_pinyin', 'reference_text': '这就是我。', 'text': '你好。', 'lang': 'zh',
     'text_ids': [0, 1341, 0, 546, 0, 1030, 0, 1178, 1895, 0, 793, 0, 458, 1895],
     'phoneme_id_map': {" ": 0, ".": 14, "hao3": 458, "jiu4": 546, "ni2": 793, "shi4": 1030, "wo3": 1178, "zhe4": 1341, "。": 1895}},
    {'phoneme_type': 'f5tts_pinyin', 'reference_text': '这就是我。', 'text': 'OK.', 'lang': 'en',
     'text_ids': [0, 1341, 0, 546, 0, 1030, 0, 1178, 1895, 0, 46, 42, 14],
     'phoneme_id_map': {" ": 0, ".": 14, "K": 42, "O": 46, "jiu4": 546, "shi4": 1030, "wo3": 1178, "zhe4": 1341, "。": 1895}},
    {'phoneme_type': 'f5tts_pinyin', 'reference_text': 'Mother nature.', 'text': 'GPU和CPU不一样。', 'lang': 'zh',
     'text_ids': [44, 827, 1082, 441, 325, 940, 0, 765, 62, 1082, 1147, 940, 325, 14, 0, 0, 38, 47, 52, 0, 461, 0, 34, 47, 52, 0, 142, 0, 1259, 0, 1249, 1895],
     'phoneme_id_map': {" ": 0, ".": 14, "C": 34, "G": 38, "M": 44, "P": 47, "U": 52, "a": 62, "bu4": 142, "e": 325, "h": 441, "he2": 461, "n": 765, "o": 827, "r": 940, "t": 1082, "u": 1147, "yang4": 1249, "yi1": 1259, "yi2": 1260, "。": 1895}},
    {'phoneme_type': 'f5tts_char', 'reference_text': 'قال。', 'text': 'OK.', 'lang': 'en',
     'text_ids': [1739, 2550, 1741, 1895, 46, 42, 14],
     'phoneme_id_map': {" ": 0, ".": 14, "K": 42, "O": 46, "ا": 2550, "ق": 1739, "ل": 1741, "。": 1895}},
    {'phoneme_type': 'f5tts_pinyin', 'reference_text': '这就是我。 ', 'text': 'GPU和CPU不一样。', 'lang': 'zh',
     'text_ids': [0, 1341, 0, 546, 0, 1030, 0, 1178, 1895, 0, 38, 47, 52, 0, 461, 0, 34, 47, 52, 0, 142, 0, 1259, 0, 1249, 1895],
     'phoneme_id_map': {" ": 0, ".": 14, "C": 34, "G": 38, "P": 47, "U": 52, "bu4": 142, "he2": 461, "jiu4": 546, "shi4": 1030, "wo3": 1178, "yang4": 1249, "yi1": 1259, "yi2": 1260, "zhe4": 1341, "。": 1895}},
    {'phoneme_type': 'f5tts_pinyin', 'reference_text': 'mother nature ', 'text': 'Hello there.', 'lang': 'en',
     'text_ids': [704, 827, 1082, 441, 325, 940, 0, 765, 62, 1082, 1147, 940, 325, 14, 0, 0, 39, 325, 615, 615, 827, 0, 1082, 441, 325, 940, 325, 14],
     'phoneme_id_map': {" ": 0, ".": 14, "H": 39, "a": 62, "e": 325, "h": 441, "l": 615, "m": 704, "n": 765, "o": 827, "r": 940, "t": 1082, "u": 1147, "。": 1895}},
    {'phoneme_type': 'f5tts_char', 'reference_text': 'قال \n', 'text': 'نعم.', 'lang': 'ar',
     'text_ids': [1739, 2550, 1741, 14, 0, 0, 1743, 1738, 1742, 14],
     'phoneme_id_map': {" ": 0, ".": 14, "ا": 2550, "ع": 1738, "ق": 1739, "ل": 1741, "م": 1742, "ن": 1743, "。": 1895}},
    {'phoneme_type': 'f5tts_pinyin', 'reference_text': '这就是我。', 'text': '"GPU"很好。', 'lang': 'zh',
     'text_ids': [0, 1341, 0, 546, 0, 1030, 0, 1178, 1895, 2, 38, 47, 52, 2, 0, 465, 0, 458, 1895],
     'phoneme_id_map': {" ": 0, "\"": 2, ".": 14, "G": 38, "P": 47, "U": 52, "hao3": 458, "hen3": 465, "jiu4": 546, "shi4": 1030, "wo3": 1178, "zhe4": 1341, "。": 1895}},
    {'phoneme_type': 'f5tts_pinyin', 'reference_text': '这就是我。', 'text': '+86电话。', 'lang': 'zh',
     'text_ids': [0, 1341, 0, 546, 0, 1030, 0, 1178, 1895, 11, 0, 24, 22, 0, 289, 0, 484, 1895],
     'phoneme_id_map': {" ": 0, "+": 11, ".": 14, "6": 22, "8": 24, "dian4": 289, "hua4": 484, "jiu4": 546, "shi4": 1030, "wo3": 1178, "zhe4": 1341, "。": 1895}},
    {'phoneme_type': 'f5tts_pinyin', 'reference_text': '这就是我。', 'text': "O'Neil来了。", 'lang': 'zh',
     'text_ids': [0, 1341, 0, 546, 0, 1030, 0, 1178, 1895, 46, 7, 45, 325, 507, 615, 0, 621, 0, 634, 1895],
     'phoneme_id_map': {" ": 0, "'": 7, ".": 14, "N": 45, "O": 46, "e": 325, "i": 507, "jiu4": 546, "l": 615, "lai2": 621, "le": 634, "shi4": 1030, "wo3": 1178, "zhe4": 1341, "。": 1895}},
    {'phoneme_type': 'f5tts_pinyin', 'reference_text': '这就是我！', 'text': 'GPU很好。', 'lang': 'zh',
     'text_ids': [0, 1341, 0, 546, 0, 1030, 0, 1178, 2540, 14, 0, 0, 38, 47, 52, 0, 465, 0, 458, 1895],
     'phoneme_id_map': {" ": 0, ".": 14, "G": 38, "P": 47, "U": 52, "hao3": 458, "hen3": 465, "jiu4": 546, "shi4": 1030, "wo3": 1178, "zhe4": 1341, "。": 1895, "！": 2540}},
    {'phoneme_type': 'f5tts_pinyin', 'reference_text': '这就是我？', 'text': '你好。', 'lang': 'zh',
     'text_ids': [0, 1341, 0, 546, 0, 1030, 0, 1178, 2542, 14, 0, 0, 0, 793, 0, 458, 1895],
     'phoneme_id_map': {" ": 0, ".": 14, "hao3": 458, "jiu4": 546, "ni2": 793, "shi4": 1030, "wo3": 1178, "zhe4": 1341, "。": 1895, "？": 2542}},
    {'phoneme_type': 'f5tts_char', 'reference_text': 'قال\t', 'text': 'نعم.', 'lang': 'ar',
     'text_ids': [1739, 2550, 1741, 14, 0, 0, 1743, 1738, 1742, 14],
     'phoneme_id_map': {"\t": 2666, " ": 0, ".": 14, "ا": 2550, "ع": 1738, "ق": 1739, "ل": 1741, "م": 1742, "ن": 1743, "。": 1895}},
    {'phoneme_type': 'f5tts_char', 'reference_text': 'قال \t', 'text': 'نعم.', 'lang': 'ar',
     'text_ids': [1739, 2550, 1741, 14, 0, 0, 1743, 1738, 1742, 14],
     'phoneme_id_map': {"\t": 2666, " ": 0, ".": 14, "ا": 2550, "ع": 1738, "ق": 1739, "ل": 1741, "م": 1742, "ن": 1743, "。": 1895}},
    {'phoneme_type': 'f5tts_pinyin', 'reference_text': '我㗑', 'text': '你好。', 'lang': 'zh',
     'text_ids': [0, 1178, 0, 0, 14, 0, 0, 0, 793, 0, 458, 1895],
     'phoneme_id_map': {" ": 0, ".": 14, "hao3": 458, "ni2": 793, "wo3": 1178, "。": 1895}},
]


class _RecordingPreprocess(_FakePreprocess):
    def __init__(self):
        self.text_ids, self.max_duration = [], []

    def run(self, output_names, feed):
        self.text_ids.append(feed["text_ids"][0].tolist())
        self.max_duration.append(int(feed["max_duration"][0]))
        return super().run(output_names, feed)


def _f5_voice(phoneme_type, phoneme_id_map, lang):
    from phoonnx.config import VoiceConfig
    from phoonnx.voice import TTSVoice
    config = VoiceConfig.from_dict({
        "phoonnx_version": "1.0", "engine": "f5tts", "phoneme_type": phoneme_type, "alphabet": "unicode",
        "lang_code": lang, "audio": {"sample_rate": 24000}, "phoneme_id_map": phoneme_id_map,
        "add_blank_char": False, "blank_at_start": False, "blank_at_end": False, "use_eos_bos": False,
        "fold_compounds": False, "engine_params": {"nfe": 2}})
    voice = TTSVoice(session=_FakeTransformer(), config=config, adapter=F5TTSAdapter())
    voice.adapter.preprocess, voice.adapter.decode = _RecordingPreprocess(), _FakeDecode()
    return voice


def _synthesize(voice, reference_text, text):
    from phoonnx.config import SynthesisConfig
    list(voice.synthesize(text, SynthesisConfig(
        speaker_reference=(np.zeros(24000, np.float32), 24000), speaker_reference_text=reference_text)))
    return voice.adapter.preprocess


@pytest.mark.parametrize("call", JOINED_CALLS, ids=lambda c: f"{c['reference_text']!r}+{c['text']}")
def test_f5tts_call_ids_equal_upstream_on_the_joined_text(call):
    """After a reference text that ends in 。, upstream puts a space before a
    generation text that starts with an ASCII word of two or more characters,
    for the pinyin text function only. Whitespace at the end of the reference
    text changes no id."""
    voice = _f5_voice(call["phoneme_type"], call["phoneme_id_map"], call["lang"])
    assert _synthesize(voice, call["reference_text"], call["text"]).text_ids == [call["text_ids"]]


def test_f5tts_length_estimate_counts_the_reference_end():
    """max_duration: the reference frames, and as many again per generation id
    as the reference took per id, with the reference end ".  " counted."""
    call = JOINED_CALLS[8]  # "mother nature " + "Hello there."
    preprocess = _synthesize(_f5_voice(call["phoneme_type"], call["phoneme_id_map"], call["lang"]),
                             call["reference_text"], call["text"])
    frames = 24000 // 256 + 1
    reference_ids = len("mother nature.  ")
    generation_ids = len("Hello there.")
    assert preprocess.max_duration == [frames + int(frames / reference_ids * generation_ids)]


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
    pin = re.match(r"https://huggingface\.co/Phoonnx/f5tts-tw-asante_openbible-graphemes/resolve/([0-9a-f]{40})/",
                   info.model_url)
    assert pin, info.model_url
    folder = f"https://huggingface.co/Phoonnx/f5tts-tw-asante_openbible-graphemes/resolve/{pin[1]}/"
    assert info.engine == "f5tts"
    assert info.phoneme_type == "f5tts_pinyin"
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


# The text function of each voice's upstream inference code: SWivid/F5-TTS
# 600e5fd runs convert_char_to_pinyin (F5-TTS v1 Base, and the Twi fine-tune
# that runs its code); SWivid/Habibi-TTS 9274498 and silma-ai/silma-tts 2816b5a
# read the text one character per token (NAMAA v2 runs the Habibi-TTS code).
_UPSTREAM_TEXT_FUNCTION = {
    "f5tts/v1-base": "f5tts_pinyin", "openbible/tw-asante": "f5tts_pinyin",
    **{f"habibi/ar-{d}": "f5tts_char" for d in ("unified", "alg", "egy", "irq", "mar", "msa", "sau", "uae")},
    "namaa/ar-sa-v2": "f5tts_char", "silma/v1": "f5tts_char", "silma/v1-en": "f5tts_char",
}


@pytest.mark.parametrize("voice_id", sorted(_UPSTREAM_TEXT_FUNCTION))
def test_entry_selects_the_text_function_of_its_upstream_code(voice_id):
    """TTSModelInfo.load gives the voice the entry's phoneme_type over the config's."""
    assert _f5_index()[voice_id]["phoneme_type"] == _UPSTREAM_TEXT_FUNCTION[voice_id]


@functools.lru_cache(maxsize=None)
def _pinned_config(config_url, cache_dir):
    """Download (with retry) and parse one pinned config.json, once per URL."""
    import json
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    repo, revision, filename = re.fullmatch(
        r"https://huggingface\.co/([^/]+/[^/]+)/resolve/([0-9a-f]{40})/(.+)", config_url).groups()
    path = retry_download(hf_hub_download, repo, filename, revision=revision, cache_dir=cache_dir)
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def _pinned_config_cache(tmp_path_factory):
    """A module-wide cache dir so each pinned config is downloaded once."""
    return str(tmp_path_factory.mktemp("f5tts-config"))


@pytest.mark.parametrize("voice_id", sorted(_UPSTREAM_TEXT_FUNCTION))
def test_pinned_config_reads_text_as_the_entry_does(voice_id, _pinned_config_cache):
    """The config.json at the entry's pinned commit selects the entry's text function
    and maps each token on its own, as the upstream code does, so a voice loaded
    from its files alone reads text as the index load does. A config that serves
    one entry also carries that entry's language."""
    index = _f5_index()
    entry = index[voice_id]
    config = _pinned_config(entry["config_url"], _pinned_config_cache)
    assert config["phoneme_type"] == entry["phoneme_type"]
    assert config["fold_compounds"] is False
    if sum(e["config_url"] == entry["config_url"] for e in index.values()) == 1:
        assert config["lang_code"] == entry["lang"]


def test_base_install_brings_the_pinyin_text_function_dependencies():
    """The pinyin text function imports rjieba and pypinyin, and the index
    selects it for voices of every language, so a base install needs both."""
    import tomllib
    from pathlib import Path
    from packaging.requirements import Requirement

    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    base = {Requirement(r).name.lower() for r in pyproject["project"]["dependencies"]}
    pinyin_voices = {v for v, e in _f5_index().items() if e["phoneme_type"] == "f5tts_pinyin"}
    assert {"f5tts/v1-base", "openbible/tw-asante"} <= pinyin_voices
    assert {"rjieba", "pypinyin"} <= base


def test_base_entry_reads_hanzi_as_pinyin_over_a_graphemes_config():
    """The config on the Hub says graphemes, which reads every hanzi as id 0.
    The entry's phoneme_type wins, as in TTSModelInfo.load, so the voice reads
    the pinyin ids of convert_char_to_pinyin (SWivid/F5-TTS 600e5fd) through
    the F5-TTS v1 Base vocab.txt ids, recorded with the upstream code."""
    from phoonnx.model_manager import TTSModelInfo
    from phoonnx.voice import TTSVoice

    vocab = {" ": 0, "wo3": 1178, "zuo2": 1410, "tian1": 1108, "mai3": 711, "le": 634, "yi1": 1259,
             "yi4": 1262, "tai2": 1088, "xin1": 1208, "dian4": 289, "nao3": 783}
    info = TTSModelInfo(**_f5_index()["f5tts/v1-base"])
    info.download_config = lambda: {"engine": "f5tts", "phoneme_type": "graphemes", "alphabet": "unicode",
                                    "lang_code": "zh", "phoneme_id_map": vocab, "fold_compounds": False,
                                    "add_blank_char": False, "blank_at_start": False, "blank_at_end": False,
                                    "use_eos_bos": False}
    voice = TTSVoice(config=info.config, session=None)
    ids = [voice.phonemes_to_ids(chunk) for chunk in voice.phonemize("我昨天买了一台新电脑")]
    assert ids == [[0, 1178, 0, 1410, 0, 1108, 0, 711, 0, 634, 0, 1259, 0, 1088, 0, 1208, 0, 289, 0, 783]]
