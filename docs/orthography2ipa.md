# orthography2ipa phonemizer backend

`Orthography2IPAPhonemizer` (in `phoonnx/phonemizers/o2ipa.py`) wraps the
[orthography2ipa](https://github.com/TigreGotico/orthography2ipa) library to
provide IPA transcription for 387+ language codes.

## When to choose this backend

| Strength | Limitation |
|---|---|
| 387-language coverage out of the box | Rule-based: quality varies by language |
| No external models or binaries | Not tuned for any specific TTS voice |
| Works offline, zero inference cost | May not match a model's training transcription |
| Good for regular-orthography languages (Spanish, Portuguese, Galician, Arabic, Hebrew, …) | Complex languages (tonal, highly irregular) may need a dedicated phonemizer |

Use this backend when:
- No dedicated phonemizer exists for the target language
- Broad multilingual coverage matters more than per-voice accuracy
- You need a lightweight offline fallback

## Output alphabet

Always `Alphabet.IPA`.

## Installation

```bash
pip install "orthography2ipa>=1.3.0a1"
```

## Usage

```python
from phoonnx.phonemizers.o2ipa import Orthography2IPAPhonemizer

p = Orthography2IPAPhonemizer()

print(p.phonemize_string("hola mundo", "es"))     # → 'ˈola ˈmundo'
print(p.phonemize_string("olá mundo", "pt"))      # → 'oˈla ˈmundo'
print(p.phonemize_string("língua galega", "gl"))  # → 'ˈlinɡa ɡaˈleɣa'
print(p.phonemize_string("שלום", "he"))           # → 'ˈʃlvm'
print(p.phonemize_string("مرحبا", "ar"))          # → 'mrħbʔ'
```

## Factory dispatch

```python
from phoonnx.config import PhonemeType, get_phonemizer
p = get_phonemizer(PhonemeType.ORTHOGRAPHY2IPA)
```

## Language resolution

`Orthography2IPAPhonemizer.get_lang(lang)` resolves BCP-47 tags to the closest
supported code via exact match → bare-language fallback → `langcodes`
distance search.  Raises `ValueError` for unsupported languages.

## Supported codes

```python
from phoonnx.phonemizers.o2ipa import Orthography2IPAPhonemizer
print(len(Orthography2IPAPhonemizer.supported_langs()))  # 387+
```
