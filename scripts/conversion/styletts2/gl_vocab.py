"""The ProxectoNos Galician StyleTTS2 vocabulary, rebuilt from the training table.

Shared by ``export_proxectonos_gl.py`` (which writes it into the voice's
``config.json``) and the test-suite, so the shipped vocabulary is correct by
construction rather than by inspection.

The source of truth is ``phoneme_token_maps.json``, the table the upstream
``Utils/ASR/AuxiliaryASR/text_utils_gal.TextCleanerGal`` reads: symbol order is
id order, and a fixed run of punctuation is appended after it. The checkpoints
declare ``n_token: 69``.
"""
from typing import Dict

# text_utils_gal._special_tokens_phonemes, appended after the table's symbols
_SPECIAL_TOKENS = list('!"(),./:;?[]{}¡ª°´º¿\'')

# Cotovia surface form -> the single symbol the model was trained on.
#   rr / tS  : upstream ``phonemize.normalize_word`` digraph rules
#   V^       : upstream ``phonemize.clean_output`` stress rule (vowel + "^")
_COMPOUNDS = {"rr": "R", "tS": "W"}
for _plain, _accented in zip("aEeiOou", "áÉéíÓóú"):
    _COMPOUNDS[_plain + "^"] = _accented
_COMPOUNDS["j^"] = "j́"
_COMPOUNDS["w^"] = "ẃ"


def build_phoneme_id_map(token_map: Dict[str, Dict]) -> Dict[str, int]:
    """Return ``{phoneme: id}`` for the 69-symbol Galician Cotovia phoneset.

    ``token_map`` is the parsed ``phoneme_token_maps.json``. The returned map is
    extended with the multi-character Cotovia surface forms phoonnx's tokenizer
    folds back into single ids, so a raw
    ``CotoviaPhonemizer(model="stress", alphabet=COTOVIA)`` string tokenises to
    exactly the ids the checkpoint was trained with.
    """
    symbols = []
    for entry in token_map.values():
        if entry["phoneme"] not in symbols:
            symbols.append(entry["phoneme"])
    for special in _SPECIAL_TOKENS:
        if special not in symbols:
            symbols.append(special)
    vocab = {s: i for i, s in enumerate(symbols)}
    if len(vocab) != 69:
        raise ValueError(f"expected the 69-token Galician phoneset, got {len(vocab)}")
    for surface, target in _COMPOUNDS.items():
        if target in vocab:
            vocab[surface] = vocab[target]
    return vocab
