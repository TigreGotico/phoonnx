"""Write the reference data for tests/test_sentencepiece_reader.py.

phoonnx reads SentencePiece models without the ``sentencepiece`` package.
This script is the only place that package is used: it trains small models
into tests/fixtures/ and records what the package gives for every model there.

    uv venv /path/to/venv && uv pip install -p /path/to/venv/bin/python sentencepiece
    /path/to/venv/bin/python scripts/make_sentencepiece_fixtures.py
"""
import glob
import json
import os
import random
import tempfile

import sentencepiece as spm

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures")
REFERENCE = os.path.join(FIXTURES, "sentencepiece_reference.json")

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "Hello there. How are you today? Fine, thanks!",
    "O rato roeu a roupa do rei de Roma.",
    "L'élève va à l'école, n'est-ce pas ?",
    "Ich weiß, dass zwei plus zwei vier ist. Straße, Größe, Übung.",
    "¿Qué tal? ¡Muy bien, gracias! año, niño, señor",
    "Perché no? La città è bellissima!",
    "мир дружба жвачка, привет мир",
    "Ελληνικά: αβγ δεζ ΑΒΓ",
    "مُحَمَّدٌ رَسُولُ اللَّهِ مرحبا بالعالم",
    "我有三本书。你有几本？太好了！世界你好",
    "日本語のテキストです。カタカナとひらがな。",
    "한국어 텍스트 안녕하세요",
    "ﬁne ﬂow ① ② ③ ² ³ ™ ℃ Ⅻ ½ ﬀ",
    "ＦＵＬＬ　ｗｉｄｔｈ　１２３ ｶﾀｶﾅ",
    "café café naïve naïve",
    "emoji 😀 🎉 🌍 👍🏽 and flags 🇵🇹",
    "numbers 123 4567 0.5 -3 1,000,000 42%",
    "tabs\tand  double  spaces   here",
    "aaaa bbbb aaaa abab abba baba",
    "<|im_start|>user hello<|im_end|> <|audio_start|> <user_inst>fix ﬁx ﬁxed</user_inst>",
]

# MOSS-TTS-Nano carries chat tokens as user-defined pieces. The last three hold
# characters that nmt_nfkc would change, so the normalizer must skip them.
USER_DEFINED = ["<|im_start|>", "<|im_end|>", "<|audio_start|>", "<user_inst>", "</user_inst>",
                "ﬁx", "ＡＢ", "café"]

TRAINED = {
    "sp_unigram_identity": dict(model_type="unigram", normalization_rule_name="identity",
                                byte_fallback=True, vocab_size=600),
    "sp_bpe_nfkc": dict(model_type="bpe", normalization_rule_name="nmt_nfkc",
                        byte_fallback=True, vocab_size=600),
    "sp_bpe_identity_no_byte_fallback": dict(model_type="bpe", normalization_rule_name="identity",
                                             byte_fallback=False, vocab_size=400),
    "sp_bpe_nfkc_user_defined": dict(model_type="bpe", normalization_rule_name="nmt_nfkc",
                                     byte_fallback=True, vocab_size=600, user_defined_symbols=USER_DEFINED),
    "sp_unigram_identity_user_defined": dict(model_type="unigram", normalization_rule_name="identity",
                                             byte_fallback=True, vocab_size=600,
                                             user_defined_symbols=USER_DEFINED),
}

TEXTS = [
    "", " ", "  ", "   ", "\n", "\t", "\r\n", "a",
    "hello world", "Hello, World! How are you today?",
    "a  b   c", "text with     five spaces",
    "  leading and trailing  ",
    "new\nline\ttab\r\ncrlf",
    "O rato roeu a roupa do rei de Roma.",
    "L'élève va à l'école, n'est-ce pas ?",
    "¿Qué tal? ¡Muy bien, gracias!",
    "Ich weiß, dass zwei plus zwei vier ist.",
    "Perché no? La città è bellissima!",
    "olá, tudo bem? çãõ ñ àéîõü",
    "The quick brown fox jumps over the lazy dog 12345.",
    "MiXeD CaSe UPPER lower",
    "123 4567 0.5 -3 1,000,000 42%",
    "sub-word hyphen_underscore (parens) [brackets] {braces} <angle>",
    "...!!!???;;;:::",
    "ΑΒΓ αβγ ⟨angle⟩ ½ ¾ № — – ’ “ ”",
    "мир 世界 한국어 🌍 emoji test",
    "الْعَرَبِيَّة مرحبا بالعالم",
    "你好世界", "😀🎉🇵🇹", "我有三本书。你有几本？太好了！",
    "日本語のテキスト ｶﾀｶﾅ カタカナ",
    "ﬁne ＣＡＦÉ　x", "ﬁ ﬂ ﬀ ① ² ™ ℃ Ⅻ", "ＦＵＬＬ　ｗｉｄｔｈ　１２３",
    "café vs café", "naïve", "́ leading combining mark",
    "zebra 😀 Ω", "ab\tab", "aaaaaaa  aaa", "abab abba baba",
    "👍🏽 skin tone", " no-break space", "zero​width",
    "▁", "▁▁", "hallo▁welt",
    "a" * 200,
    "<|im_start|>", "<|im_start|>user\nhello<|im_end|>", " <|audio_start|> ", "x<|im_end|>y",
    "<|im_start|><|im_end|><|im_start|>", "<user_inst>speak</user_inst>", "<user_inst",
    "ﬁx ﬁxed ﬁ x fix", "ＡＢＣ ＡＢ AB", "café cafe\u0301 CAFÉ", "  <|im_start|>  two  spaces  ",
    # strings whose unigram lattice ties, for the Pocket TTS vocabularies
    "ü1'\"ob0Ha\"Lñ kQuAc.ßLbbb",
    "ãzñqm5pppliNínöYßbZ6U",
    "ú-SyHAßúKgUa'f\"k9ú úlll",
]

FUZZ_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    " \t\n\r.,!?;:'\"-_()[]{}<>/@#$%^&*+=~`|\\"
    "áéíóúñçãõäöüßΑΒΓαβγмир שלום العربية 世界日本語한국어🌍🎉½№’—▁ﬁ①ＡＢ　ｶ́"
)


def train(name, options):
    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "corpus.txt")
        with open(corpus, "w", encoding="utf-8") as handle:
            handle.write("\n".join(CORPUS * 20) + "\n")
        spm.SentencePieceTrainer.train(input=corpus, model_prefix=os.path.join(FIXTURES, name),
                                       character_coverage=0.98, hard_vocab_limit=False, **options)
    os.remove(os.path.join(FIXTURES, name + ".vocab"))


def reference(path, rng):
    processor = spm.SentencePieceProcessor(model_file=path)
    size = processor.get_piece_size()
    texts = TEXTS + ["".join(rng.choice(FUZZ_ALPHABET) for _ in range(rng.randint(0, 60)))
                     for _ in range(40)]
    encode = []
    for text in texts:
        ids = processor.encode(text)
        encode.append({"text": text, "ids": ids, "decoded": processor.decode(ids)})
    sequences = [[rng.randrange(size) for _ in range(rng.randint(1, 25))] for _ in range(40)]
    sequences += [[1, 2, 3], [0], [0, 4], [4, 5, 6], [3, 4], [0xE4 % size]]
    return {"pieces": [processor.id_to_piece(i) for i in range(size)],
            "encode": encode,
            "decode": [{"ids": ids, "text": processor.decode(ids)} for ids in sequences]}


def main():
    for name, options in TRAINED.items():
        train(name, options)
    rng = random.Random(20260814)
    models = sorted(glob.glob(os.path.join(FIXTURES, "*sp_*.model")))
    data = {"sentencepiece": spm.__version__,
            "models": {os.path.basename(path): reference(path, rng) for path in models}}
    with open(REFERENCE, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=0)
        handle.write("\n")
    print(f"{len(models)} models in, {len(data['models'])} references out: {REFERENCE}")


if __name__ == "__main__":
    main()
