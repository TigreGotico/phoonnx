"""A voice takes another text frontend only when it gives the symbols the voice was trained on.

The voice reads graphemes, so every frontend here needs no model files. Its
vocabulary holds the letters a to p and the space, with ids that do not equal
their positions.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
from scriptconv.frontend import FrontendConfig, TextFrontend
from scriptconv.phonemizers.enums import Alphabet, Phonemizer

from phoonnx.config import SynthesisConfig, VoiceConfig
from phoonnx.engines import get_adapter
from phoonnx.errors import FrontendMismatchError
from phoonnx.voice import TTSVoice

LETTERS = "abcdefghijklmnop "
CONFIG = {"phoonnx_version": "1.0.0", "engine": "vits", "phoneme_type": "graphemes", "alphabet": "unicode", "lang_code": "en-US",
          "audio": {"sample_rate": 22050},
          "phoneme_id_map": {c: [100 + 3 * i] for i, c in enumerate(LETTERS)}}


def ids_of(text):
    return [100 + 3 * LETTERS.index(c) for c in text]


@pytest.fixture
def voice():
    return TTSVoice(session=MagicMock(), config=VoiceConfig.from_dict(dict(CONFIG)), adapter=get_adapter("vits"))


def frontend(voice, **changes):
    """A grapheme frontend with the voice's own settings, except *changes*."""
    settings = dict(lang="en-US", phonemizer=Phonemizer.GRAPHEMES, alphabet=Alphabet.UNICODE,
                    keep_punctuation=voice.frontend.config.keep_punctuation)
    settings.update(changes)
    return FrontendConfig(**settings)


def test_a_frontend_with_another_alphabet_is_refused(voice):
    with pytest.raises(FrontendMismatchError, match="alphabet"):
        voice.with_frontend(frontend(voice, alphabet=Alphabet.IPA))


def test_a_frontend_that_treats_punctuation_otherwise_is_refused(voice):
    keep = voice.frontend.config.keep_punctuation
    with pytest.raises(FrontendMismatchError, match="keep_punctuation"):
        voice.with_frontend(frontend(voice, keep_punctuation=not keep))


def test_a_probe_symbol_outside_the_vocabulary_is_refused(voice):
    with pytest.raises(FrontendMismatchError, match=r"\['q', 'u'\]"):
        voice.with_frontend(frontend(voice), probe_text="a quick")
    # the same frontend passes a probe that stays inside the vocabulary
    voice.with_frontend(frontend(voice), probe_text="hello")


def test_the_probe_leaves_the_voice_tokenizer_as_it_was(voice):
    before = set(voice.tokenizer.not_found_characters)
    with pytest.raises(FrontendMismatchError):
        voice.with_frontend(frontend(voice), probe_text="a quick")
    assert voice.tokenizer.not_found_characters == before


def test_a_frontend_config_takes_the_voice_normalisation(voice):
    """A digit reads as a word in the new frontend, as it does in the voice's own."""
    assert voice.phonemize("3 bad") == [list("three bad")]
    assert voice.with_frontend(frontend(voice)).phonemize("3 bad") == [list("three bad")]


def test_the_new_voice_reads_through_the_given_frontend_and_keeps_it(voice):
    given = TextFrontend(frontend(voice))
    own = voice.frontend
    new = voice.with_frontend(given)
    assert new.frontend is given
    assert voice.frontend is own
    assert (new.session, new.tokenizer, new.adapter) == (voice.session, voice.tokenizer, voice.adapter)
    assert [list(u.ids) for u in new.prepare("bad", SynthesisConfig())] == [ids_of("bad")]
    # a later phonemizer change does not reach the frozen frontend
    new.phonemizer = voice.phonemizer
    assert new.frontend is given


def test_load_takes_a_frontend_and_checks_it(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx")
    (tmp_path / "model.onnx.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    ipa = FrontendConfig(lang="en-US", phonemizer=Phonemizer.GRAPHEMES, alphabet=Alphabet.IPA)
    with patch("phoonnx.voice.make_session", return_value=MagicMock()):
        with pytest.raises(FrontendMismatchError, match="alphabet"):
            TTSVoice.load(str(model), frontend=ipa)
        plain = TTSVoice.load(str(model))
        given = TextFrontend(frontend(plain))
        assert TTSVoice.load(str(model), frontend=given).frontend is given
