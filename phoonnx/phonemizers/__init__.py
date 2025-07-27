from typing import Union

from phoonnx.phonemizers.base import BasePhonemizer, TextChunks, RawPhonemizedChunks
from phoonnx.phonemizers.en import DeepPhonemizer, OpenPhonemizer, G2P_EN
from phoonnx.phonemizers.gl import CotoviaPhonemizer
from phoonnx.phonemizers.vi import VIPhonemePhonemizer
from phoonnx.phonemizers.ko import G2PKPhonemizer
from phoonnx.phonemizers.mul import (GraphemePhonemizer, EspeakPhonemizer, EpitranPhonemizer,
                                     GruutPhonemizer, ByT5Phonemizer, CharsiuPhonemizer)

Phonemizer = Union[
    ByT5Phonemizer,
    CharsiuPhonemizer,
    EspeakPhonemizer,
    GruutPhonemizer,
    EpitranPhonemizer,
    VIPhonemePhonemizer,
    G2PKPhonemizer,
    CotoviaPhonemizer,
    GraphemePhonemizer,
    OpenPhonemizer,
    G2P_EN,
    DeepPhonemizer
]
