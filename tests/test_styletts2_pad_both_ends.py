"""The padding a StyleTTS2/Kokoro voice needs is stated, not guessed.

The adapter has always read the lineage off the style pack: a multi-row
``[N, 256]`` pack means Kokoro, which pads both ends, and anything else pads
the start only. That proxy fails for a Kokoro fine-tune whose style is baked
into the weights as a single vector. ``agbalu/Matoub-82M`` is such a model: it
registers one ``(1, 256)`` ``voice`` buffer, and its own tokenizer wraps every
sequence in the pad symbol on both sides because ``meldataset`` wrapped every
training target that way.

The oracle here is that tokenizer's rule, not this adapter's behaviour:

    boundary = [vocab["$"]]
    merged = boundary + token_ids_0 + boundary

so a Matoub sequence of N ids reaches the graph as N + 2.
"""
import numpy as np

from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.styletts2 import StyleTTS2Adapter


class _In:
    def __init__(self, name):
        self.name = name


class _Sess:
    def __init__(self, names):
        self._i = [_In(n) for n in names]

    def get_inputs(self):
        return self._i


class _Cfg:
    def __init__(self, engine_params):
        self.tokenizer = None
        self.engine_params = engine_params


def _req(n=5, **p):
    return AdapterSynthesisRequest(
        phoneme_ids=np.arange(1, n + 1, dtype=np.int64)[None, :],
        phoneme_lengths=np.array([n], np.int64), params=p)


def _ids(adapter, n=5):
    sess = _Sess(["input_ids", "attention_mask", "speed"])
    return adapter.build_feed_dict(_req(n), sess)["input_ids"]


def test_a_baked_style_voice_still_pads_both_ends_when_it_says_so():
    # The Matoub case. Without the flag this voice is padded on one side only,
    # which is the defect: the sequence is then off the distribution the
    # durations were fitted on.
    adapter = StyleTTS2Adapter()
    adapter.configure(_Cfg({"pad_both_ends": True}))
    ids = _ids(adapter, 5)
    assert ids.shape == (1, 7), "5 ids must reach the graph as 5 + 2"
    assert ids[0, 0] == adapter._pad_id
    assert ids[0, -1] == adapter._pad_id


def test_the_same_voice_without_the_flag_is_padded_on_one_side():
    # Fail-before control. This is what the style-pack proxy does to a
    # single-style Kokoro descendant, and it is why the flag exists. If this
    # ever starts giving 7, the default changed and the flag's reason is gone.
    ids = _ids(StyleTTS2Adapter(), 5)
    assert ids.shape == (1, 6)
    assert ids[0, 0] == 0 and ids[0, -1] != 0


def test_the_flag_can_also_turn_both_ends_off_for_a_kokoro_pack():
    # The setting is not a one-way door: it overrides the proxy in both
    # directions, so a multi-row pack can be told to pad the start only. A flag
    # that only ever agrees with the proxy in one direction would prove nothing.
    pack = np.arange(510 * 256, dtype=np.float32).reshape(510, 256)
    adapter = StyleTTS2Adapter(style_pack=pack)
    adapter.configure(_Cfg({"pad_both_ends": False}))
    assert _ids(adapter, 5).shape == (1, 6)
    # and the proxy still gives both ends for the same pack when nothing is said
    assert _ids(StyleTTS2Adapter(style_pack=pack), 5).shape == (1, 7)


def test_the_constructor_argument_and_engine_params_agree():
    adapter = StyleTTS2Adapter(pad_both_ends=True)
    assert _ids(adapter, 5).shape == (1, 7)
    # engine_params does not overwrite an explicit constructor argument
    adapter.configure(_Cfg({"pad_both_ends": False}))
    assert _ids(adapter, 5).shape == (1, 7)


def test_an_absent_flag_changes_nothing():
    # Every voice that does not carry the key keeps the behaviour it had.
    plain = StyleTTS2Adapter()
    plain.configure(_Cfg({}))
    assert _ids(plain, 5).shape == (1, 6)
    pack = np.arange(510 * 256, dtype=np.float32).reshape(510, 256)
    kokoro = StyleTTS2Adapter(style_pack=pack)
    kokoro.configure(_Cfg({}))
    assert _ids(kokoro, 5).shape == (1, 7)


def test_the_matoub_tokenizer_rule_is_what_this_reproduces():
    # The oracle, restated independently of the adapter: Matoub's tokenizer
    # wraps the ids in the pad symbol on both sides, so N ids become N + 2.
    # This is the arithmetic the flag has to satisfy for any N.
    adapter = StyleTTS2Adapter()
    adapter.configure(_Cfg({"pad_both_ends": True}))
    for n in (1, 3, 5, 12, 40):
        assert _ids(adapter, n).shape == (1, n + 2)
