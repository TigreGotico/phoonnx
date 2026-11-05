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

from tests.conftest import retry_download


def _manager():
    from phoonnx.model_manager import TTSModelManager
    m = TTSModelManager(cache_path=tempfile.mktemp(suffix=".json"))
    m.merge_default_voices()
    return m


def _load_voice(model_info):
    """Load a voice, retrying on transient network errors from model download."""
    return retry_download(model_info.load)


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
    import transformers
    text = "hello world"
    tok = retry_download(transformers.VitsTokenizer.from_pretrained, "facebook/mms-tts-eng")
    orig = tok(text)["input_ids"]
    if hasattr(orig, "tolist"):
        orig = orig.tolist()
    if orig and isinstance(orig[0], list):
        orig = orig[0]
    m = _manager()
    voice = _load_voice(m.voices["facebook/mms-tts-eng-English"])
    assert _phoonnx_ids(voice, text) == orig


# --- piper: pinned espeak-id regression (piper_phonemize has no cp312/manylinux
# wheel on PyPI, so it cannot be installed here or in CI; this pins phoonnx's
# own known-good ids for a fixed voice/text instead of a live cross-check) ---

def test_tokenization_piper():
    m = _manager()
    vid = _pick(m, lambda k: "piper" in k.lower() and ("en-us" in k.lower() or "en_us" in k.lower()),
               "no en-US piper voice in index")
    text = "hello world"
    voice = _load_voice(m.voices[vid])
    expected = [1, 0, 20, 0, 59, 0, 24, 0, 120, 0, 27, 0, 100, 0, 3, 0,
                35, 0, 120, 0, 62, 0, 122, 0, 24, 0, 17, 0, 2]
    assert _phoonnx_ids(voice, text) == expected


# --- piper: native reimplementation of piper_phonemize.phoneme_ids_espeak.
#     phoonnx builds a piper voice's ids from the voice's own phoneme_id_map,
#     so it needs no GPL-linked piper_phonemize wrapper. This is a self-
#     contained fixture (no voice download, no espeak): a piper-format config
#     plus a fixed espeak phoneme string, asserting phoonnx reproduces piper's
#     documented [BOS, PAD, id, PAD, ..., id, PAD, EOS] espeak id scheme. ---

def test_tokenization_piper_native_id_scheme():
    from phoonnx.tokenizer import TTSTokenizer
    # piper's phoneme_id_map format: {symbol: [id, ...]}. The special-token ids
    # (pad "_"=0, bos "^"=1, eos "$"=2, space " "=3) and the phoneme ids below
    # are piper's documented espeak values (the same ids piper_phonemize would
    # emit); the espeak phoneme string is what "hello world" phonemizes to.
    cfg = {"phoneme_id_map": {
        "_": [0], "^": [1], "$": [2], " ": [3],
        "h": [20], "ə": [59], "l": [24], "ˈ": [120], "o": [27], "ʊ": [100],
        "w": [35], "ɜ": [62], "ː": [122], "d": [17],
    }}
    tok = TTSTokenizer.from_piper_config(cfg)
    # piper_phonemize.phoneme_ids_espeak(list("həlˈoʊ")) -> this exact sequence
    assert tok.tokenize("həlˈoʊ") == [1, 0, 20, 0, 59, 0, 24, 0, 120, 0, 27, 0, 100, 0, 2]
    assert tok.tokenize("həlˈoʊ wˈɜːld") == [
        1, 0, 20, 0, 59, 0, 24, 0, 120, 0, 27, 0, 100, 0, 3, 0,
        35, 0, 120, 0, 62, 0, 122, 0, 24, 0, 17, 0, 2]


# --- gruut: Larynx (GlowTTS) and Mimic3 share gruut. phoonnx splits gruut's
#     multi-char clusters into the model's individual IPA symbols, so compare the
#     phoneme *content* (joined), which must be identical. ---

def _gruut_stream(text, lang):
    """gruut phonemes concatenated (clusters like ˈoʊ kept whole)."""
    import gruut
    out = []
    for sent in gruut.sentences(text, lang=lang):
        for word in sent:
            if word.phonemes:
                out.append("".join(word.phonemes))
    return "".join(out)


def _greedy_vocab_phonemes(stream, vocab):
    """Map a phoneme stream to vocab symbols, greedy longest-match — so
    multi-phoneme tokens (aɪ, aʊ, d͡ʒ, t͡ʃ, …) resolve to a single id rather than
    being split into characters."""
    compounds = sorted((k for k in vocab if len(k) > 1 and not k.startswith("<")),
                       key=len, reverse=True)
    out, i = [], 0
    while i < len(stream):
        hit = next((c for c in compounds if stream.startswith(c, i)), None)
        if hit:
            out.append(hit); i += len(hit)
        elif stream[i] in vocab:
            out.append(stream[i]); i += 1
        else:
            i += 1
    return out


def _phoonnx_phonemes_decoded(voice, text):
    """phoonnx phoneme ids with blanks/word-seps stripped, decoded to symbols."""
    tok = voice.config.tokenizer
    inv = {i: s for s, i in tok.vocabulary.char2idx.items()}
    skip = {tok.blank_id, getattr(tok, "blank_word_id", None)}
    ids = []
    for chunk in voice.phonemize(text):
        ids.extend(voice.phonemes_to_ids(chunk))
    return [inv[i] for i in ids if i not in skip and inv.get(i, " ").strip()]


def _assert_gruut_ids_match(voice, text, lang):
    # gruut is the original phonemizer; the model's vocab (with its multi-phoneme
    # clusters) is the original id map. phoonnx must agree on both.
    ref = _greedy_vocab_phonemes(_gruut_stream(text, lang), voice.config.tokenizer.vocabulary.char2idx)
    assert _phoonnx_phonemes_decoded(voice, text) == ref


def test_tokenization_larynx_gruut_ids():
    m = _manager()
    vid = _pick(m, lambda k: k.startswith("larynx/en-us"), "no en-US Larynx voice in index")
    # diphthong + affricate clusters exercise multi-phoneme tokens
    _assert_gruut_ids_match(_load_voice(m.voices[vid]), "hello my joyful child now", "en-us")


def test_tokenization_mimic3_gruut_ids():
    m = _manager()
    vid = _pick(m, lambda k: k.startswith("mimic3/en_"), "no en mimic3 voice in index")
    _assert_gruut_ids_match(_load_voice(m.voices[vid]), "hello my joyful child now", "en-us")


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


# --- Matcha: pinned-golden-fixture (mirrors the piper test). matcha-tts's
#     text.cleaned_text_to_sequence maps each IPA symbol through matcha's
#     _symbol_to_id table; importing it instantiates a global espeak-ng backend
#     via GPL-linked wrappers (phonemizer / espeakng_loader), so instead of a
#     live cross-check we pin matcha's symbol->id sequence for a fixed phoneme
#     string as an in-repo fixture and assert phoonnx reproduces it. No matcha /
#     phonemizer / espeakng_loader import, no espeak, no skip. ---

# IPA phoneme string for "hello world" (what matcha's cleaner emits downstream).
_MATCHA_PHONEMES = "həlˈoʊ wˈɚld"
# Golden ids: matcha-tts's _symbol_to_id applied to _MATCHA_PHONEMES. Captured
# from the Matxa Catalan voice's shipped phoneme_id_map, whose symbol->id table
# is byte-identical to upstream matcha-tts's text.symbols._symbol_to_id (the
# previous live cross-check via cleaned_text_to_sequence confirmed the match).
_MATCHA_GOLDEN_IDS = [50, 83, 54, 156, 57, 135, 16, 65, 156, 85, 54, 46]


def test_tokenization_matcha():
    m = _manager()
    # a *phonemes* matxa voice ships matcha's IPA symbol table
    vid = _pick(m, lambda k: "matxa" in k.lower() and "grapheme" not in k.lower(),
                "no matcha (phonemes) voice in index")
    voice = _load_voice(m.voices[vid])
    assert voice.config.tokenizer.encode(_MATCHA_PHONEMES) == _MATCHA_GOLDEN_IDS
