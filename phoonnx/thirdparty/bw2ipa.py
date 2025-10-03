# -*- coding: UTF8 -*-
"""
This script translates Buckwalter-transcribed Modern Standard Arabic to IPA.

This relies on Mantoq tokenization that uses separate tokens for vowel length and consonant gemination.

by Casimiro Ferreira with help of Gemini July 2025.
"""

import re

# This dictionary maps a single Buckwalter character to its most common IPA equivalent.
char_dict = {
    'a': 'a', 'A': 'aː', 'b': 'b', 'c': 'x', 'd': 'd', 'D': 'dˤ', 'e': 'e', 'E': 'ʕ',
    'f': 'f', 'g': 'ɣ', 'h': 'h', 'H': 'ħ', 'i': 'i', 'I': 'iː', 'j': 'ʒ', 'k': 'k',
    'l': 'l', 'm': 'm', 'n': 'n', 'p': 'p', 'q': 'q', 'r': 'r', 'R': 'r', 's': 's',
    'S': 'sˤ', 't': 't', 'T': 'tˤ', 'u': 'u', 'U': 'uː', 'v': 'v', 'w': 'w', 'x': 'x',
    'y': 'j', 'z': 'z', 'Z': 'ðˤ', '\'': 'ʔ', '<': 'ʔ', 'o': 'o', '-': ' ',
    '*': 'ð', '$': 'ʃ'
}
_vowels = {'a', 'i', 'u', 'aː', 'iː', 'uː'}


def translate(buckwalter_text: str) -> str:
    """
    Translates a Buckwalter-transcribed string into an IPA string.

    Args:
        buckwalter_text (str): The Buckwalter string to translate.

    Returns:
        str: The translated IPA string.
    """
    ipa_list = []
    i = 0
    while i < len(buckwalter_text):
        # Check for the longest token first. The new Mantoq tokenization
        # seems to use a 5-character token.
        token = buckwalter_text[i:i + 5]

        # If the previous character was a vowel, we assume it's a long vowel
        # marker (ː). Otherwise, we assume it's a geminated consonant.
        if token == '_dbl_':
            if ipa_list and ipa_list[-1] in _vowels:
                ipa_list.append('ː')  # Add length marker for long vowels
            elif ipa_list:
                ipa_list.append(ipa_list[-1])  # Duplicate the consonant
            i += 5
            continue

        # Check for multi-character mappings from char_dict
        two_char_token = buckwalter_text[i:i + 2]
        if two_char_token in char_dict:
            ipa_list.append(char_dict[two_char_token])
            i += 2
            continue

        # Handle single characters
        single_char = buckwalter_text[i]
        if single_char in char_dict:
            ipa_list.append(char_dict[single_char])
        else:
            ipa_list.append(single_char)
        i += 1

    return ''.join(ipa_list)
