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


def test_mosstts_tokenizer_matches_sentencepiece():
    """The example's tokenizer gives the ids of the sentencepiece package, on a BPE model with byte fallback
    and nmt_nfkc (tests/fixtures/sentencepiece_reference.json)."""
    model = str(ROOT / "tests/fixtures/sp_bpe_nfkc.model")
    rows = json.loads((ROOT / "tests/fixtures/sentencepiece_reference.json").read_text(encoding="utf-8"))
    rows = rows["models"]["sp_bpe_nfkc.model"]["encode"]
    code = _code()
    tokenizer = code[code.index("# the SentencePiece tokenizer"):code.index("with wave.open")]
    assert "from phoonnx._sentencepiece import SentencePieceProcessor" in code
    for row in rows:
        names = {"hf_hub_download": lambda repo, name: model, "repo": REPO, "text": row["text"]}
        exec("from phoonnx._sentencepiece import SentencePieceProcessor\n" + tokenizer, names)
        assert names["ids"] == row["ids"], row["text"]


def test_mosstts_refuses_a_graph_with_other_names():
    files = index_files(_entry())
    files["local_fixed_frame_path"].update(inputs={"global_hidden": 2}, outputs={"logits": 2})
    with pytest.raises(UnsupportedExample):
        standalone_example(_entry(), files, repo_id=REPO, text="Hello.")
