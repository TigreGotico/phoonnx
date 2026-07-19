"""Phonemize corpora for StyleTTS2 auxiliary-model training.

Two jobs, both using phoonnx's own phonemizers and the shared 178-symbol
StyleTTS2 ``TextCleaner`` table:

- ``phonemize_list_file`` — turn a raw-text ``train_list.txt``
  (``filename|text|speaker``) into a phoneme list for the
  ``styletts2-aligner`` / ``styletts2-pitch`` / ``styletts2`` engines.
- ``phonemize_text_corpus`` — turn a plain-text corpus (one sentence per
  line) into the token-level PL-BERT dataset (``data.jsonl`` +
  ``token_maps.json``): per word, its phoneme string and a corpus-built word
  id (replaces upstream PL-BERT's deprecated transfo-xl tokenizer +
  ``token_maps.pkl``).

CLI::

    python -m phoonnx_train.styletts2.phonemize_corpus list  IN OUT --lang pt [--phonemizer espeak]
    python -m phoonnx_train.styletts2.phonemize_corpus plbert CORPUS OUT_DIR --lang pt
"""
import json
import logging
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from phoonnx_train.styletts2.meldataset import symbols

LOG = logging.getLogger(__name__)

# word-id space reserved entries (word_separator doubles as padding)
WORD_SEP_ID = 0
WORD_UNK_ID = 1
_RESERVED_WORDS = {"<sep>": WORD_SEP_ID, "<unk>": WORD_UNK_ID}

_SYMBOL_SET = set(symbols)


def get_phonemizer(name: str = "espeak"):
    import phoonnx.phonemizers as P
    aliases = {
        "espeak": "EspeakPhonemizer",
        "gruut": "GruutPhonemizer",
        "epitran": "EpitranPhonemizer",
        "goruut": "GoruutPhonemizer",
        "byt5": "ByT5Phonemizer",
        "charsiu": "CharsiuPhonemizer",
        "transphone": "TransphonePhonemizer",
        "grapheme": "GraphemePhonemizer",
        "unicode": "UnicodeCodepointPhonemizer",
    }
    cls_name = aliases.get(name, name)
    cls = getattr(P, cls_name, None)
    if cls is None:
        raise ValueError(f"Unknown phonemizer '{name}' — use one of "
                         f"{sorted(aliases)} or a phoonnx phonemizer class name")
    return cls()


def check_symbol_coverage(phonemes: str, context: str = "") -> str:
    """Drop (and warn about) symbols outside the StyleTTS2 table."""
    bad = {ch for ch in phonemes if ch not in _SYMBOL_SET}
    if bad:
        LOG.warning("symbols not in the StyleTTS2 table, dropped: %r %s",
                    "".join(sorted(bad)), f"({context})" if context else "")
        phonemes = "".join(ch for ch in phonemes if ch in _SYMBOL_SET)
    return phonemes


def phonemize_list_file(in_list: Path, out_list: Path, lang: str,
                        phonemizer: str = "espeak") -> None:
    """``filename|text|speaker`` (raw text) -> ``filename|phonemes|speaker``."""
    pho = get_phonemizer(phonemizer)
    out_lines = []
    for i, line in enumerate(Path(in_list).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        parts = line.split("|")
        text = parts[1] if len(parts) > 1 else parts[0]
        ipa = pho.phonemize_string(text, lang)
        ipa = check_symbol_coverage(ipa, f"{in_list}:{i + 1}")
        speaker = parts[2] if len(parts) > 2 else "0"
        out_lines.append(f"{parts[0]}|{ipa}|{speaker}")
    Path(out_list).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    LOG.info("wrote %d phonemized lines -> %s", len(out_lines), out_list)


def _phonemize_words(pho, words: List[str], lang: str) -> List[Tuple[str, str]]:
    """[(word, phonemes)] — one phonemizer call per word keeps a strict
    word<->phoneme alignment (upstream PL-BERT does the same per token)."""
    out = []
    for w in words:
        ipa = check_symbol_coverage(pho.phonemize_string(w, lang), w)
        if ipa:
            out.append((w, ipa))
    return out


def phonemize_text_corpus(in_txt: Path, out_dir: Path, lang: str,
                          phonemizer: str = "espeak",
                          min_words: int = 3,
                          vocab_size: int = 100000) -> Path:
    """Plain-text corpus (one sentence per line) -> PL-BERT dataset dir:

    - ``data.jsonl``: ``{"phonemes": [str per word], "words": [str per word]}``
    - ``token_maps.json``: word -> id (0=<sep>/pad, 1=<unk>)
    - ``dataset_config.json``: lang/phonemizer/counts
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pho = get_phonemizer(phonemizer)

    counter: Counter = Counter()
    n_lines = 0
    with open(out_dir / "data.jsonl", "w", encoding="utf-8") as fh:
        for line in Path(in_txt).read_text(encoding="utf-8").splitlines():
            words = [w.lower() for w in line.split() if w.strip()]
            if len(words) < min_words:
                continue
            pairs = _phonemize_words(pho, words, lang)
            if len(pairs) < min_words:
                continue
            counter.update(w for w, _ in pairs)
            fh.write(json.dumps({"phonemes": [p for _, p in pairs],
                                 "words": [w for w, _ in pairs]},
                                ensure_ascii=False) + "\n")
            n_lines += 1

    token_maps = dict(_RESERVED_WORDS)
    for w, _count in counter.most_common(vocab_size - len(token_maps)):
        token_maps[w] = len(token_maps)
    (out_dir / "token_maps.json").write_text(
        json.dumps(token_maps, ensure_ascii=False), encoding="utf-8")
    (out_dir / "dataset_config.json").write_text(json.dumps({
        "lang": lang, "phonemizer": phonemizer, "num_sentences": n_lines,
        "num_words": len(token_maps)}, ensure_ascii=False), encoding="utf-8")
    LOG.info("PL-BERT corpus: %d sentences, %d-word vocab -> %s",
             n_lines, len(token_maps), out_dir)
    return out_dir


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("list", help="phonemize a filename|text|speaker list file")
    p1.add_argument("in_list", type=Path)
    p1.add_argument("out_list", type=Path)
    p1.add_argument("--lang", required=True)
    p1.add_argument("--phonemizer", default="espeak")

    p2 = sub.add_parser("plbert", help="build a PL-BERT dataset from a text corpus")
    p2.add_argument("corpus", type=Path)
    p2.add_argument("out_dir", type=Path)
    p2.add_argument("--lang", required=True)
    p2.add_argument("--phonemizer", default="espeak")
    p2.add_argument("--vocab-size", type=int, default=100000)

    args = ap.parse_args()
    if args.cmd == "list":
        phonemize_list_file(args.in_list, args.out_list, args.lang, args.phonemizer)
    else:
        phonemize_text_corpus(args.corpus, args.out_dir, args.lang,
                              args.phonemizer, vocab_size=args.vocab_size)


if __name__ == "__main__":
    main()
