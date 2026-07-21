# Third-party license notice — mantoq

This directory vendors the mantoq Arabic G2P pipeline. Its phonetisation core
(`buck/phonetise_buckwalter.py`) is adapted from Nawar Halabi's
Arabic-Phonetiser and is licensed under the
**Creative Commons Attribution-NonCommercial 4.0 International License**
(https://creativecommons.org/licenses/by-nc/4.0/) — NOT the Apache-2.0 license
of the rest of this repository.

It is kept solely for compatibility with published models trained on mantoq
phoneme sequences, is never selected automatically, and is only constructed
when a voice or caller explicitly requests `PhonemeType.MANTOQ`. The Arabic
default is arbtok. Arabic ↔ Buckwalter transliteration itself is NOT part of
this notice: it is delegated to `scriptconv.notation`, an independent
Apache-2.0 implementation of Buckwalter's published mapping.
