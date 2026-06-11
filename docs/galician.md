# Galician phonemizer (pycotovia)

The `CotoviaPhonemizer` class in `phoonnx/phonemizers/gl.py` produces Galician
phonemes using [pycotovia](https://github.com/TigreGotico/pycotovia), a pure-Python
port of the [Cotovia](http://webs.uvigo.es/gtm_voz) G2P engine.

## Output alphabets

| `Alphabet` value | Output format | Use case |
|---|---|---|
| `Alphabet.IPA` | IPA string via `pycotovia.cotovia_to_ipa` | Most phonemizer pipelines |
| `Alphabet.COTOVIA` | Raw Cotovia notation (e.g. `este e uN sistema`) | Voices trained on Cotovia alphabet output |

Voices trained on Cotovia-alphabet output (e.g. ProxectoNos, Sabela, Celtia)
receive strings identical to the original Cotovia C binary's: pycotovia is
binary-parity-tested against it (see `pycotovia/docs/parity.md`).

## Installation

```bash
pip install phoonnx[gl]
# or directly:
pip install pycotovia
```

## Usage

```python
from phoonnx.phonemizers.gl import CotoviaPhonemizer
from phoonnx.config import Alphabet

p_ipa = CotoviaPhonemizer(alphabet=Alphabet.IPA)
p_cot = CotoviaPhonemizer(alphabet=Alphabet.COTOVIA)

text = "Este é un sistema de conversión de texto a voz en lingua galega."

print(p_ipa.phonemize_string(text, "gl"))   # IPA
print(p_cot.phonemize_string(text, "gl"))   # Cotovia notation
```

## Accepted language codes

`CotoviaPhonemizer` accepts any language tag that matches `gl-ES` within a
`langcodes` distance of ≤ 10 (e.g. `"gl"`, `"gl-ES"`).

## Factory dispatch

```python
from phoonnx.config import PhonemeType, Alphabet, get_phonemizer
p = get_phonemizer(PhonemeType.COTOVIA, alphabet=Alphabet.IPA)
```
