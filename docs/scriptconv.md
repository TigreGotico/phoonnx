# scriptconv integration

phoonnx delegates notation transcoding and script-tag normalisation to
[scriptconv](https://github.com/TigreGotico/scriptconv), the org-wide
zero-dependency notation library.

## What phoonnx delegates to scriptconv

| Area | phoonnx call site | scriptconv symbol |
|------|-------------------|-------------------|
| ARPABET → IPA lookup table | `phoonnx/thirdparty/arpa2ipa.py` | `scriptconv.notation._ARPA_TO_IPA` (re-exported as `arpa_to_ipa_lookup`) |
| ARPABET → IPA function | same file | `scriptconv.notation.arpa_to_ipa` |
| Arabic script ↔ Buckwalter | `phoonnx/thirdparty/mantoq/buck/phonetise_buckwalter.py` | `scriptconv.notation.arabic_to_buckwalter`, `scriptconv.notation.buckwalter_to_arabic` |
| MMS script-label → ISO-15924 | `phoonnx/util.normalize_lang` | `scriptconv.scripts.normalize_script_tag` |

### Publishing prerequisite

scriptconv is a private repository and is not yet on PyPI.
**Publishing scriptconv to PyPI is a merge prerequisite for this PR.**

## What intentionally stays vendored in phoonnx

### Mantoq phoneme notation → IPA (`phoonnx/thirdparty/bw2ipa.py`)

`mantoq_to_ipa` (and its tokenizer) converts Mantoq phoneme tokens
(`b`, `aa`, `_dbl_`, …) to IPA strings.  This is a *phoneme-level* mapping
specific to phoonnx's Arabic pipeline, not a script-level conversion.
scriptconv has no concept of Mantoq tokens; the file stays.

### Hangul phonology (`phoonnx/thirdparty/hangul2ipa.py`)

Hangul → IPA involves Korean phonological rules: palatalization,
tensification, coda neutralisation, resyllabification.  These are
language-specific *phonological* transforms, not script-to-script
transcoding.  scriptconv's `translit` module handles Hangul → IPA via the
same rule tables, but phoonnx imports `hangul2ipa` directly to keep the
Korean phonemizer self-contained and avoid a circular dependency.

### Arabic G2P logic (`phoonnx/thirdparty/mantoq/`)

The full Mantoq G2P pipeline (MSA phonetisation rules, fixed-word
exceptions, emphatic-context tracking) is phoonnx-specific and stays
vendored.  Only the two Buckwalter ↔ Arabic script functions inside
`phonetise_buckwalter.py` were replaced by scriptconv calls.
