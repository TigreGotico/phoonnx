from typing import Union

from phoonnx.phonemizers.base import BasePhonemizer, UnicodeCodepointPhonemizer, RawPhonemes, GraphemePhonemizer, TextChunks, RawPhonemizedChunks
from phoonnx.phonemizers.en import DeepPhonemizer, OpenPhonemizer, G2PEnPhonemizer
from phoonnx.phonemizers.gl import CotoviaPhonemizer
from phoonnx.phonemizers.vi import VIPhonemePhonemizer
from phoonnx.phonemizers.he import PhonikudPhonemizer
from phoonnx.phonemizers.ar import MantoqPhonemizer
from phoonnx.phonemizers.fa import PersianPhonemizer
from phoonnx.phonemizers.ja import PyKakasiPhonemizer, CutletPhonemizer, OpenJTaklPhonemizer
from phoonnx.phonemizers.ko import KoG2PPhonemizer, G2PKPhonemizer
from phoonnx.phonemizers.zh import (G2pCPhonemizer, G2pMPhonemizer, PypinyinPhonemizer,
                                    XpinyinPhonemizer, JiebaPhonemizer)
from phoonnx.phonemizers.mul import (EspeakPhonemizer, EpitranPhonemizer, MisakiPhonemizer, GoruutPhonemizer,
                                     GruutPhonemizer, ByT5Phonemizer, CharsiuPhonemizer, TransphonePhonemizer)
from phoonnx.phonemizers.mwl import MirandesePhonemizer

Phonemizer = Union[
    MisakiPhonemizer,
    ByT5Phonemizer,
    UnicodeCodepointPhonemizer,
    CharsiuPhonemizer,
    EspeakPhonemizer,
    GruutPhonemizer,
    GoruutPhonemizer,
    EpitranPhonemizer,
    TransphonePhonemizer,
    MirandesePhonemizer,
    OpenJTaklPhonemizer,
    CutletPhonemizer,
    PyKakasiPhonemizer,
    PersianPhonemizer,
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
