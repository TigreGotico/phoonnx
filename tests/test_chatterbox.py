import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest, BaseOnnxAdapter
from phoonnx.engines.chatterbox import ChatterboxAdapter, _apply_repetition_penalty


def _req(**params):
    return AdapterSynthesisRequest(phoneme_ids=np.array([[1, 2, 3]], np.int64),
                                   phoneme_lengths=np.array([3], np.int64),
                                   speaker_id=0, language_id=0, params=params)


def test_chatterbox_registered():
    from phoonnx.engines import list_engines
    assert "chatterbox" in list_engines()


def test_chatterbox_detect():
    assert ChatterboxAdapter.detect({"engine": "chatterbox"})
    assert not ChatterboxAdapter.detect({"engine": "vits"})
    assert not ChatterboxAdapter.detect(None)


def test_chatterbox_tokenizes_raw_text():
    # the flag that makes TTSVoice skip phonemization for this engine
    assert ChatterboxAdapter.tokenizes_raw_text is True
    assert BaseOnnxAdapter.tokenizes_raw_text is False


def test_chatterbox_requires_aux_graphs():
    with pytest.raises(RuntimeError):
        ChatterboxAdapter().synthesize(_req(reference_audio=(np.zeros(100, np.float32), 24000)), None)


def test_chatterbox_requires_reference():
    ad = ChatterboxAdapter()
    ad.embed_tokens = ad.speech_encoder = ad.cond_decoder = object()   # pretend configured
    with pytest.raises(RuntimeError):
        ad.synthesize(_req(), None)                                    # no reference_audio


def test_repetition_penalty():
    scores = np.array([[1.0, 2.0, -1.0, 4.0]])
    prev = np.array([[1, 2]])                       # tokens 1 (+) and 2 (-) already emitted
    out = _apply_repetition_penalty(prev, scores, 2.0)
    assert out[0, 1] == pytest.approx(1.0)          # 2.0 / 2  (positive -> divide)
    assert out[0, 2] == pytest.approx(-2.0)         # -1.0 * 2 (negative -> multiply)
    assert out[0, 0] == pytest.approx(1.0)          # untouched


def test_bpe_tokenizer_joins_units():
    from phoonnx.tokenizer import BPETokenizer
    bpe = BPETokenizer.__new__(BPETokenizer)        # bypass __init__ (no tokenizer.json on disk)

    class _Enc:
        def __init__(self, ids): self.ids = ids

    class _Tok:
        def encode(self, text): return _Enc([ord(c) for c in text])

    bpe._tok = _Tok()
    # a list of char units (from the UNICODE-style path) is joined back to text first
    assert bpe.tokenize(["a", "b", "c"]) == [97, 98, 99]
    assert bpe.tokenize("abc") == [97, 98, 99]
