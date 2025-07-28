from typing import Union

from phoonnx.phonemizers.base import BasePhonemizer, RawPhonemes, GraphemePhonemizer, TextChunks, RawPhonemizedChunks
from phoonnx.phonemizers.en import DeepPhonemizer, OpenPhonemizer, G2PEnPhonemizer
from phoonnx.phonemizers.gl import CotoviaPhonemizer
from phoonnx.phonemizers.vi import VIPhonemePhonemizer
from phoonnx.phonemizers.he import PhonikudPhonemizer
from phoonnx.phonemizers.ar import MantoqPhonemizer
from phoonnx.phonemizers.ko import KoG2PPhonemizer, G2PKPhonemizer
from phoonnx.phonemizers.zh import (G2pCPhonemizer, G2pMPhonemizer, PypinyinPhonemizer,
                                    XpinyinPhonemizer, JiebaPhonemizer)
from phoonnx.phonemizers.mul import (EspeakPhonemizer, EpitranPhonemizer, MisakiPhonemizer,
                                     GruutPhonemizer, ByT5Phonemizer, CharsiuPhonemizer)

Phonemizer = Union[
    MisakiPhonemizer,
    ByT5Phonemizer,
    CharsiuPhonemizer,
    EspeakPhonemizer,
    GruutPhonemizer,
    EpitranPhonemizer,
    VIPhonemePhonemizer,
    G2PKPhonemizer,
    KoG2PPhonemizer,
    G2pCPhonemizer,
    G2pMPhonemizer,
    PypinyinPhonemizer,
    XpinyinPhonemizer,
    JiebaPhonemizer,
    PhonikudPhonemizer,
    CotoviaPhonemizer,
    MantoqPhonemizer,
    GraphemePhonemizer,
    RawPhonemes,
    OpenPhonemizer,
    G2PEnPhonemizer,
    DeepPhonemizer
]
