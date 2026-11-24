"""Write the ``features.json`` feature table of a voice's phoneme id map.

The file gives each id of a ``phoneme_id_map`` a distinctive-feature vector,
in the ``phoonnx-input-features/1`` format that a feature-input text encoder
reads. Seven prosody features for space, pauses, question, exclamation and
stress, then four tone features, end every vector; punctuation of any script sets them by a rule over its
Unicode name. A phone written with IPA tone diacritics (``á``) gets the phone features of
its base and the tone block of its marks. Pad, BOS, EOS and other ids that are not phones get ``null``. A symbol that looks like a phone
and that the feature source cannot read stops the run, with the list.

The feature source runs here, once, and never at training or synthesis time,
so phonematcher and orthography2ipa are in the ``train-features`` extra only.

Example::

    python -m phoonnx_train.features_json config.json \\
        --source phonematcher --out features.json
"""
import hashlib
import json
import sys
import unicodedata
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import click

FORMAT = "phoonnx-input-features/1"

#: The prosody features that end every vector. A punctuation character
#: (category P) sets a row when its Unicode name contains one of the row's
#: words and none of its excluded words; a space (Zs) sets word_boundary; the
#: stress rows take their symbols. A phone is 0 on all of them, and a prosody
#: symbol is 0 on every phone feature.
PROSODY: Dict[str, dict] = {
    "pros_word_boundary": {"category": "Zs", "symbols": ("\u1361",)},
    "pros_pause_minor": {"names": ("COMMA", "SEMICOLON", "COLON", "ANO TELEIA", "SECTION MARK")},
    "pros_pause_major": {"names": ("FULL STOP", "QUESTION", "EXCLAMATION", "INTERROBANG", "DANDA", "ELLIPSIS"),
                         "not": ("INVERTED",), "symbols": ("‖",)},
    "pros_question": {"names": ("QUESTION", "INTERROBANG")},
    "pros_exclamation": {"names": ("EXCLAMATION", "INTERROBANG")},
    "pros_stress_primary": {"symbols": ("ˈ",)},
    "pros_stress_secondary": {"symbols": ("ˌ",)},
}

#: Symbols that are never phones and get null, with the reason printed for each.
NON_PHONE_SYMBOLS: Dict[str, str] = {
    "_": "pad",
    "^": "BOS",
    "$": "EOS",
    "|": "null by default; rule it a pause with --pause |",
}
#: Unicode categories that are never phones: punctuation with no prosody row,
#: and digits.
NON_PHONE_CATEGORIES: Dict[str, str] = {
    "Pc": "punctuation", "Pd": "punctuation", "Ps": "punctuation", "Pe": "punctuation",
    "Pi": "punctuation", "Pf": "punctuation", "Po": "punctuation",
    "Nd": "digit",
}

#: orthography2ipa feats.py has no list of names. These are the names of its
#: feature indices, from the feats.py module docstring (0-20) and the comment
#: above ``phone_features`` (21 click, 22 nasal_vowel).
O2I_FEATURES: Tuple[str, ...] = (
    "syllabic", "sonorant", "consonantal", "continuant", "delayed_release",
    "lateral", "nasal", "strident", "voice", "spread_glottis",
    "constricted_glottis", "anterior", "coronal", "distributed", "labial",
    "high", "low", "back", "round", "tense", "long", "click", "nasal_vowel",
)

#: The tone features after the prosody block: a tone id has tone_is 1 and its
#: pitch at the start, middle and end, each Chao level 1-5 as (level - 3) / 2.
TONE = ("tone_is", "tone_start", "tone_mid", "tone_end")

#: The IPA tie bars above and below, which join two symbols into one segment.
TIE_BARS = {"\u0361", "\u035c"}

_TERNARY = {True: 1, False: -1, None: 0}


def load_source(name: str) -> Tuple[str, Tuple[str, ...], Callable[[str], Sequence[Optional[bool]]]]:
    """The installed version, the feature names and the vectorizer of a source."""
    if name == "phonematcher":
        from phonematcher import FEATURES, vectorize_phones
        return version("phonematcher"), tuple(FEATURES), vectorize_phones
    from orthography2ipa.feats import NUM_FEATURES, vectorize_phones
    if NUM_FEATURES != len(O2I_FEATURES):
        raise RuntimeError(f"orthography2ipa feats.py has {NUM_FEATURES} features; "
                           f"this script names {len(O2I_FEATURES)}")
    return version("orthography2ipa"), O2I_FEATURES, vectorize_phones


def map_sha256(phoneme_id_map: dict) -> str:
    text = json.dumps(phoneme_id_map, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def id_symbols(phoneme_id_map: dict) -> Dict[int, str]:
    """id -> symbol. A map value is one id or a list of ids."""
    symbols: Dict[int, str] = {}
    for symbol, ids in phoneme_id_map.items():
        for i in [ids] if isinstance(ids, int) else ids:
            if i in symbols:
                raise click.ClickException(f"id {i} is mapped by both {symbols[i]!r} and {symbol!r}")
            symbols[i] = symbol
    return symbols


def tone_block(symbol: str) -> Optional[List[float]]:
    """The tone features of *symbol*, or None for a symbol that is not a tone.

    phonematcher reads the levels from tone letters, superscript numerals and
    left-stem tone letters; the pitch at the start, middle and end of the
    syllable is a linear interpolation of those levels.
    """
    from phonematcher import tone_levels
    try:
        levels = tone_levels(symbol)
    except ValueError:
        return None
    if len(levels) == 1:
        contour = levels * 3
    else:
        contour = []
        for k in range(3):
            x = k * (len(levels) - 1) / 2
            i = min(int(x), len(levels) - 2)
            contour.append(levels[i] + (levels[i + 1] - levels[i]) * (x - i))
    points = [(level - 3) / 2 for level in contour]
    return [1] + [int(p) if p.is_integer() else p for p in points]


def split_tone_marks(symbol: str) -> Optional[Tuple[str, str]]:
    """The base and the tone letters of a phone written with IPA tone diacritics
    (``á`` is ``a`` and ``˦``), or None when *symbol* has no tone diacritic or
    its base is neither a vowel nor a syllabic consonant, as phonematcher's
    ``is_vowel_phone`` reads it (``ḿ̩`` is ``m̩`` and ``˦``; ``ś`` is not ``s``
    with a tone). phonematcher's diacritic table gives the tone letters of each mark.
    """
    from phonematcher.features import is_vowel_phone
    from phonematcher.tone import DIACRITIC_TONES
    decomposed = unicodedata.normalize("NFD", symbol)
    letters = "".join(DIACRITIC_TONES.get(ch, "") for ch in decomposed)
    base = unicodedata.normalize("NFC", "".join(ch for ch in decomposed if ch not in DIACRITIC_TONES))
    if not letters or not base or not is_vowel_phone(base):
        return None
    return base, letters


def prosody_rows(symbol: str, pauses: Sequence[str] = ()) -> List[str]:
    """The prosody rows *symbol* sets to 1. A symbol in *pauses* is a minor pause."""
    if symbol in pauses:
        return ["pros_pause_minor"]
    category = unicodedata.category(symbol) if len(symbol) == 1 else ""
    name = unicodedata.name(symbol, "") if category.startswith("P") else ""
    rows = []
    for row, rule in PROSODY.items():
        if (symbol in rule.get("symbols", ())
                or category == rule.get("category")
                or (any(w in name for w in rule.get("names", ()))
                    and not any(w in name for w in rule.get("not", ())))):
            rows.append(row)
    return rows


def non_phone_reason(symbol: str, extra: Dict[str, str]) -> Optional[str]:
    if symbol in extra:
        return extra[symbol]
    if symbol in NON_PHONE_SYMBOLS:
        return NON_PHONE_SYMBOLS[symbol]
    categories = {unicodedata.category(c) for c in symbol}
    if len(categories) == 1 and categories <= NON_PHONE_CATEGORIES.keys():
        return NON_PHONE_CATEGORIES[categories.pop()]
    return None


def build_features(phoneme_id_map: dict, source: str,
                   non_phones: Sequence[str] = (), pauses: Sequence[str] = ()
                   ) -> Tuple[dict, List[Tuple[int, str, str]], List[Tuple[int, str, List[str]]]]:
    """The features.json object, the null ids as (id, symbol, reason) and the
    punctuation and stress ids as (id, symbol, prosody rows).

    Raises ``click.ClickException`` listing every symbol that looks like a
    phone and does not read.
    """
    src_version, names, vectorize = load_source(source)
    extra = {s: "ruled a non-phone by --non-phone" for s in non_phones}
    vectors: Dict[str, dict] = {}
    nulls: List[Tuple[int, str, str]] = []
    prosody: List[Tuple[int, str, List[str]]] = []
    unread: List[str] = []
    for i, symbol in sorted(id_symbols(phoneme_id_map).items()):
        vector = None
        reason = extra.get(symbol)
        tone = tone_block(symbol) if reason is None else None
        rows = prosody_rows(symbol, pauses) if reason is None and tone is None else []
        if tone:
            vector = [0] * (len(names) + len(PROSODY)) + tone
        elif rows:
            vector = [0] * len(names) + [int(row in rows) for row in PROSODY] + [0] * len(TONE)
            if not symbol.isspace():
                prosody.append((i, symbol, rows))
        elif reason is None:
            reason = non_phone_reason(symbol, extra)
            if reason == "punctuation" and len(symbol) == 1:
                reason = f"punctuation {unicodedata.name(symbol, '?')}, no prosody rule"
            if reason is None:
                phone, tone = symbol, [0] * len(TONE)
                if marked := split_tone_marks(symbol):
                    phone, tone = marked[0], tone_block(marked[1])
                try:
                    vector = [_TERNARY[v] for v in vectorize(phone)] + [0] * len(PROSODY) + tone
                except ValueError as e:
                    if source == "orthography2ipa" and TIE_BARS & set(symbol):
                        e = ("orthography2ipa feats.py does not read tie bars; a tied affricate or "
                             "double articulation reads with --source phonematcher")
                    unread.append(f"  id {i} {symbol!r} ({' '.join(f'U+{ord(c):04X}' for c in symbol)}): {e}")
                    continue
        if vector is None:
            nulls.append((i, symbol, reason))
        vectors[str(i)] = {"symbol": symbol, "vector": vector}
    if unread:
        raise click.ClickException(
            f"{len(unread)} symbols look like phones and {source} cannot read them:\n"
            + "\n".join(unread)
            + "\nFix the source, or pass --non-phone SYMBOL for a symbol that is not a phone.")
    table = {
        "format": FORMAT,
        "source": {"name": source, "version": src_version,
                   "tone": {"name": "phonematcher", "version": version("phonematcher")}},
        "feature_names": list(names) + list(PROSODY) + list(TONE),
        "phoneme_id_map_sha256": map_sha256(phoneme_id_map),
        "vectors": vectors,
    }
    return table, nulls, prosody


@click.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--source", required=True, type=click.Choice(["phonematcher", "orthography2ipa"]),
              help="Feature source")
@click.option("--out", required=True, type=click.Path(dir_okay=False), help="features.json to write")
@click.option("--non-phone", "non_phones", multiple=True,
              help="A symbol ruled a non-phone; it gets null. Repeatable")
@click.option("--pause", "pauses", multiple=True,
              help="A symbol ruled a minor pause for this voice, such as U+00B7. Repeatable")
def cli(config: str, source: str, out: str, non_phones: Tuple[str, ...],
        pauses: Tuple[str, ...]) -> None:
    """Write features.json for CONFIG, a voice config.json or a phoneme_id_map JSON."""
    for flag, values in (("--non-phone", non_phones), ("--pause", pauses)):
        if "" in values:
            raise click.BadParameter("an empty symbol", param_hint=flag)
    if long := [p for p in pauses if len(p) != 1]:
        raise click.BadParameter(f"a pause is one character: {', '.join(map(repr, long))}",
                                 param_hint="--pause")
    data = json.loads(Path(config).read_text(encoding="utf-8"))
    phoneme_id_map = data.get("phoneme_id_map", data)
    n_in = len(id_symbols(phoneme_id_map))
    click.echo(f"ids in: {n_in}")
    click.echo("non-phone allowlist: " + ", ".join(
        [f"{s!r} {r}" for s, r in NON_PHONE_SYMBOLS.items()]
        + [f"{c} {r}" for c, r in NON_PHONE_CATEGORIES.items()]
        + [f"{s!r} --non-phone" for s in non_phones]))
    table, nulls, prosody = build_features(phoneme_id_map, source, non_phones, pauses)
    click.echo(f"prosody ids: {len(prosody)}")
    for i, symbol, rows in prosody:
        click.echo(f"  {i} {symbol!r} {unicodedata.name(symbol, '?')}: {' '.join(rows)}")
    click.echo(f"null ids: {len(nulls)}")
    for i, symbol, reason in nulls:
        click.echo(f"  {i} {symbol!r} {reason}")
    click.echo(f"ids out: {len(table['vectors'])}")
    if len(table["vectors"]) != n_in:
        sys.exit(f"id count mismatch: {n_in} in, {len(table['vectors'])} out")
    partial = Path(f"{out}.partial")
    partial.write_text(json.dumps(table, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    partial.replace(out)
    click.echo(f"wrote {out} ({source} {table['source']['version']}, "
               f"{len(table['feature_names'])} features, {len(PROSODY)} prosody, {len(TONE)} tone)")


if __name__ == "__main__":
    cli()
