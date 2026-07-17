from typing import Union

from phoonnx.phonemizers.base import BasePhonemizer, UnicodeCodepointPhonemizer, GraphemePhonemizer, TextChunks, RawPhonemizedChunks
from phoonnx.phonemizers.en import DeepPhonemizer, OpenPhonemizer, G2PEnPhonemizer
from phoonnx.phonemizers.gl import CotoviaPhonemizer
from phoonnx.phonemizers.eu import AhoTTSPhonemizer, EuskaphonePhonemizer
from phoonnx.phonemizers.vi import VIPhonemePhonemizer
from phoonnx.phonemizers.he import PhonikudPhonemizer
from phoonnx.phonemizers.ar import MantoqPhonemizer, ArbtokPhonemizer
from phoonnx.phonemizers.shami import ShamiPhonemizer
from phoonnx.phonemizers.fa import PersianPhonemizer
from phoonnx.phonemizers.pt import TugaphonePhonemizer, BarranquenhoPhonemizer
from phoonnx.phonemizers.o2ipa import Orthography2IPAPhonemizer
from phoonnx.phonemizers.ja import PyKakasiPhonemizer, CutletPhonemizer, OpenJTaklPhonemizer
from phoonnx.phonemizers.ko import KoG2PPhonemizer, G2PKPhonemizer
from phoonnx.phonemizers.zh import (G2pCPhonemizer, G2pMPhonemizer, PypinyinPhonemizer,
                                    XpinyinPhonemizer, JiebaPhonemizer)
from phoonnx.phonemizers.mul import (EspeakPhonemizer, EpitranPhonemizer, MisakiPhonemizer, GoruutPhonemizer,
                                     GruutPhonemizer, ByT5Phonemizer, CharsiuPhonemizer, TransphonePhonemizer,
                                     MisakiEnPhonemizer, MisakiJaPhonemizer, MisakiZhPhonemizer,
                                     MisakiKoPhonemizer, MisakiViPhonemizer)
from phoonnx.phonemizers.mwl import MirandesePhonemizer

Phonemizer = Union[
    MisakiPhonemizer,
    ByT5Phonemizer,
    TugaphonePhonemizer,
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
    AhoTTSPhonemizer,
    MantoqPhonemizer,
    ArbtokPhonemizer,
    ShamiPhonemizer,
    EuskaphonePhonemizer,
    BarranquenhoPhonemizer,
    Orthography2IPAPhonemizer,
    GraphemePhonemizer,
    OpenPhonemizer,
    G2PEnPhonemizer,
    DeepPhonemizer
]
