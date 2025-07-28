from typing import Union

from phoonnx.phonemizers.base import BasePhonemizer, RawPhonemes, GraphemePhonemizer, TextChunks, RawPhonemizedChunks
from phoonnx.phonemizers.en import DeepPhonemizer, OpenPhonemizer, G2P_EN
from phoonnx.phonemizers.gl import CotoviaPhonemizer
from phoonnx.phonemizers.vi import VIPhonemePhonemizer
from phoonnx.phonemizers.ko import G2PKPhonemizer
from phoonnx.phonemizers.he import PhonikudPhonemizer
from phoonnx.phonemizers.ar import MantoqPhonemizer
from phoonnx.phonemizers.mul import (EspeakPhonemizer, EpitranPhonemizer,
                                     GruutPhonemizer, ByT5Phonemizer, CharsiuPhonemizer)

Phonemizer = Union[
    ByT5Phonemizer,
    CharsiuPhonemizer,
    EspeakPhonemizer,
    GruutPhonemizer,
    EpitranPhonemizer,
    VIPhonemePhonemizer,
    G2PKPhonemizer,
    PhonikudPhonemizer,
    CotoviaPhonemizer,
    MantoqPhonemizer,
    GraphemePhonemizer,
    RawPhonemes,
    OpenPhonemizer,
    G2P_EN,
    DeepPhonemizer
]
