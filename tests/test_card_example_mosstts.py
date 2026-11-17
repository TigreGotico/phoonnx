import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from card_examples import UnsupportedExample, index_files, standalone_example  # noqa: E402
REPO = "Phoonnx/phoonnx-moss"


def _entry():
    index = json.loads((ROOT / "phoonnx/voice_index/mosstts.json").read_text())
    return next(e for e in index.values() if isinstance(e, dict) and e["voice_id"] == "mosstts/nano/en")


def _code(text="Hello there."):
    block = standalone_example(_entry(), index_files(_entry()), repo_id=REPO, text=text)
    return block.removeprefix("```python\n").removesuffix("\n```")


def test_mosstts_example_compiles():
    compile(_code(), "mosstts", "exec")
    compile(_code(None), "mosstts", "exec")


def test_mosstts_prompt_is_phoonnx_prompt():
    from phoonnx.engines import mosstts
    code = _code()
    for ids in (mosstts.USER_PROMPT_PREFIX_TOKEN_IDS, mosstts.USER_PROMPT_AFTER_REFERENCE_TOKEN_IDS,
                mosstts.ASSISTANT_PROMPT_PREFIX_TOKEN_IDS):
        assert str(list(ids)) in code
    assert (mosstts.AUDIO_START_TOKEN_ID, mosstts.AUDIO_END_TOKEN_ID, mosstts.AUDIO_USER_SLOT_TOKEN_ID,
            mosstts.AUDIO_ASSISTANT_SLOT_TOKEN_ID, mosstts.AUDIO_PAD_TOKEN_ID) == (6, 7, 8, 9, 1024)


def test_mosstts_tokenizer_matches_sentencepiece(tmp_path):
    """The written-out tokenizer gives the ids of the sentencepiece package, on a BPE model with byte fallback."""
    import sentencepiece as spm  # the test extra installs it
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(["the quick brown fox jumps over the lazy dog", "aaaa bbbb aaaa abab",
                                 "مُحَمَّدٌ رَسُولُ اللَّهِ", "ﬁne café über straße"] * 50), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / "sp"), model_type="bpe",
                                   vocab_size=340, byte_fallback=True, character_coverage=0.98,
                                   normalization_rule_name="nmt_nfkc")
    model = str(tmp_path / "sp.model")
    processor = spm.SentencePieceProcessor(model_file=model)
    code = _code()
    tokenizer = code[code.index("# the SentencePiece tokenizer"):code.index("with wave.open")]
    for text in ["the quick brown fox", "aaaaaaa  aaa", "مُحَمَّدٌ رَسُولُ", "ﬁne ＣＡＦÉ　x", "zebra 😀 Ω", "ab\tab"]:
        names = {"hf_hub_download": lambda repo, name: model, "repo": REPO, "text": text}
        exec("import struct\n" + code[code.index("def varint"):code.index("# the SentencePiece tokenizer")] + tokenizer,
             names)
        assert names["ids"] == processor.encode(text), text


def test_mosstts_refuses_a_graph_with_other_names():
    files = index_files(_entry())
    files["local_fixed_frame_path"].update(inputs={"global_hidden": 2}, outputs={"logits": 2})
    with pytest.raises(UnsupportedExample):
        standalone_example(_entry(), files, repo_id=REPO, text="Hello.")
