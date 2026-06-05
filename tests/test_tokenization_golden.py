"""
Cross-framework tokenization golden tests.

For each supported TTS framework, verify phoonnx's text -> phoneme_ids pipeline
matches the *original* framework's tokenizer. Each test is guarded on the
original library being importable, so CI (phoonnx core only) skips them while a
full/dev environment runs them and proves the match.

These guard the class of bug where a vocab order / phonemizer mismatch yields the
right timbre but the wrong words.
"""
import tempfile
import pytest


def _manager():
    from phoonnx.model_manager import TTSModelManager
    m = TTSModelManager(cache_path=tempfile.mktemp(suffix=".json"))
    m.merge_default_voices()
    return m


def _phoonnx_ids(voice, text):
    ids = []
    for chunk in voice.phonemize(text):
        ids.extend(voice.phonemes_to_ids(chunk))
    return ids


def _phoonnx_phoneme_str(voice, text):
    return "".join(p for chunk in voice.phonemize(text) for p in chunk).replace(" ", "")


def _pick(m, pred, why):
    hit = [k for k in m.voices if pred(k)]
    if not hit:
        pytest.skip(why)
    return hit[0]


# --- transformers (Meta MMS / HF-VITS): VitsTokenizer is the literal original ---

def test_tokenization_transformers_mms():
    transformers = pytest.importorskip("transformers")
    text = "hello world"
    tok = transformers.VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
    orig = tok(text)["input_ids"]
    if hasattr(orig, "tolist"):
        orig = orig.tolist()
    if orig and isinstance(orig[0], list):
        orig = orig[0]
    m = _manager()
    voice = m.voices["facebook/mms-tts-eng-English"].load()
    assert _phoonnx_ids(voice, text) == orig


# --- piper: piper_phonemize espeak ids are the original ---

def test_tokenization_piper():
    pp = pytest.importorskip("piper_phonemize")
    m = _manager()
    vid = _pick(m, lambda k: "piper" in k.lower() and ("en-us" in k.lower() or "en_us" in k.lower()),
               "no en-US piper voice in index")
    text = "hello world"
    phonemes = pp.phonemize_espeak(text, "en-us")
    flat = [p for sentence in phonemes for p in sentence]
    orig = list(pp.phoneme_ids_espeak(flat))
    voice = m.voices[vid].load()
    assert _phoonnx_ids(voice, text) == orig


# --- gruut: Larynx (GlowTTS) and Mimic3 share gruut. phoonnx splits gruut's
#     multi-char clusters into the model's individual IPA symbols, so compare the
#     phoneme *content* (joined), which must be identical. ---

def _gruut_str(text, lang):
    import gruut
    out = []
    for sent in gruut.sentences(text, lang=lang):
        for word in sent:
            if word.phonemes:
                out.extend(word.phonemes)
    return "".join(out)


def test_tokenization_larynx_gruut():
    pytest.importorskip("gruut")
    m = _manager()
    vid = _pick(m, lambda k: k.startswith("larynx/en-us"), "no en-US Larynx voice in index")
    voice = m.voices[vid].load()
    text = "hello world"
    assert _phoonnx_phoneme_str(voice, text) == _gruut_str(text, "en-us")


def test_tokenization_mimic3_gruut():
    pytest.importorskip("gruut")
    m = _manager()
    vid = _pick(m, lambda k: k.startswith("mimic3/en_"), "no en mimic3 voice in index")
    voice = m.voices[vid].load()
    text = "hello world"
    assert _phoonnx_phoneme_str(voice, text) == _gruut_str(text, "en-us")


# --- coqui: coqui-tts won't install in a transformers>=5 env, so we assert the
#     bridge reproduces coqui's BaseCharacters._create_vocab order exactly:
#     [pad, eos, bos, blank] + characters + punctuations (blank at id 3). ---

def test_tokenization_coqui_vocab_order():
    from phoonnx.engines.glowtts_config import voice_config_from_coqui
    cfg = {"use_phonemes": False, "add_blank": True, "audio": {"sample_rate": 22050},
           "characters": {"pad": "_", "eos": "~", "bos": "^", "blank": "<BLNK>",
                          "characters": "abc", "punctuations": "!?"}}
    vc = voice_config_from_coqui(cfg, lang_code="en")
    assert list(vc.tokenizer.vocabulary.char2idx) == ["_", "~", "^", "<BLNK>", "a", "b", "c", "!", "?"]


def test_tokenization_coqui_vocab_order_no_blank():
    from phoonnx.engines.glowtts_config import voice_config_from_coqui
    cfg = {"use_phonemes": False, "add_blank": False, "audio": {"sample_rate": 22050},
           "characters": {"pad": "_", "eos": "~", "bos": "^",
                          "characters": "abc", "punctuations": "!?"}}
    vc = voice_config_from_coqui(cfg, lang_code="en")
    assert list(vc.tokenizer.vocabulary.char2idx) == ["_", "~", "^", "a", "b", "c", "!", "?"]


# --- OptiSpeech: its IPATokenizer is the literal original ---

def test_tokenization_optispeech():
    optispeech = pytest.importorskip("optispeech")
    from optispeech.text.tokenizers import IPATokenizer
    text = "hello world"
    # the emily/mike models embed text_processor: add_blank=False, add_bos_eos=False
    tok = IPATokenizer(add_blank=False, add_bos_eos=False, normalize_text=True)
    orig, _ = tok(text, "en-us")
    if orig and isinstance(orig[0], list):
        orig = orig[0]
    m = _manager()
    vid = "hf_community/mush42/optispeech-lightspeech-en-us-emily"
    if vid not in m.voices:
        pytest.skip("optispeech voice not in index")
    voice = m.voices[vid].load()
    assert _phoonnx_ids(voice, text) == orig


# --- Matcha: matcha.text.text_to_sequence is the original (guarded; matcha-tts
#     ships a broken top-level import in some envs, so this skips until usable) ---

def test_tokenization_matcha():
    pytest.importorskip("matcha")
    from matcha.text import text_to_sequence  # noqa: F401
    m = _manager()
    vid = _pick(m, lambda k: "matxa" in k.lower() or (
        m.voices[k].engine and str(m.voices[k].engine).endswith("matcha")), "no matcha voice in index")
    voice = m.voices[vid].load()
    cleaners = ["catalan_cleaners"] if "cat" in vid.lower() else ["english_cleaners2"]
    orig = text_to_sequence("hello", cleaners)
    assert _phoonnx_ids(voice, "hello") == orig
