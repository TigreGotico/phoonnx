#!/usr/bin/python
# -*- coding: UTF8 -*-

# adapted from: https://github.com/nawarhalabi/Arabic-Phonetiser/blob/master/phonetise-Buckwalter.py
# license: Creative Commons Attribution-NonCommercial 4.0 International License.
# https://creativecommons.org/licenses/by-nc/4.0/

import re

arabic_to_buckw_dict = {  # mapping from Arabic script to Buckwalter
    "\u0628": "b",
    "\u0630": "*",
    "\u0637": "T",
    "\u0645": "m",
    "\u062a": "t",
    "\u0631": "r",
    "\u0638": "Z",
    "\u0646": "n",
    "\u062b": "^",
    "\u0632": "z",
    "\u0639": "E",
    "\u0647": "h",
    "\u062c": "j",
    "\u0633": "s",
    "\u063a": "g",
    "\u062d": "H",
    "\u0642": "q",
    "\u0641": "f",
    "\u062e": "x",
    "\u0635": "S",
    "\u0634": "$",
    "\u062f": "d",
    "\u0636": "D",
    "\u0643": "k",
    "\u0623": ">",
    "\u0621": "'",
    "\u0626": "}",
    "\u0624": "&",
    "\u0625": "<",
    "\u0622": "|",
    "\u0627": "A",
    "\u0649": "Y",
    "\u0629": "p",
    "\u064a": "y",
    "\u0644": "l",
    "\u0648": "w",
    "\u064b": "F",
    "\u064c": "N",
    "\u064d": "K",
    "\u064e": "a",
    "\u064f": "u",
    "\u0650": "i",
    "\u0651": "~",
    "\u0652": "o",
}

buckw_to_arabic_dict = {  # mapping from Buckwalter to Arabic script
    "b": "\u0628",
    "*": "\u0630",
    "T": "\u0637",
    "m": "\u0645",
    "t": "\u062a",
    "r": "\u0631",
    "Z": "\u0638",
    "n": "\u0646",
    "^": "\u062b",
    "z": "\u0632",
    "E": "\u0639",
    "h": "\u0647",
    "j": "\u062c",
    "s": "\u0633",
    "g": "\u063a",
    "H": "\u062d",
    "q": "\u0642",
    "f": "\u0641",
    "x": "\u062e",
    "S": "\u0635",
    "$": "\u0634",
    "d": "\u062f",
    "D": "\u0636",
    "k": "\u0643",
    ">": "\u0623",
    "'": "\u0621",
    "}": "\u0626",
    "&": "\u0624",
    "<": "\u0625",
    "|": "\u0622",
    "A": "\u0627",
    "Y": "\u0649",
    "p": "\u0629",
    "y": "\u064a",
    "l": "\u0644",
    "w": "\u0648",
    "F": "\u064b",
    "N": "\u064c",
    "K": "\u064d",
    "a": "\u064e",
    "u": "\u064f",
    "i": "\u0650",
    "~": "\u0651",
    "o": "\u0652",
}


def arabic_to_buckwalter(word):  # Convert input string to Buckwalter
    res = ""
    for letter in word:
        if letter in arabic_to_buckw_dict:
            res += arabic_to_buckw_dict[letter]
        else:
            res += letter
    return res


def buckwalter_to_arabic(word):  # Convert input string to Arabic
    res = ""
    for letter in word:
        if letter in buckw_to_arabic_dict:
            res += buckw_to_arabic_dict[letter]
        else:
            res += letter
    return res


# ----------------------------------------------------------------------------
# Grapheme to Phoneme mappings------------------------------------------------
# ----------------------------------------------------------------------------
unambiguousConsonantMap = {
    "b": "b",
    "*": "*",
    "T": "T",
    "m": "m",
    "t": "t",
    "r": "r",
    "Z": "Z",
    "n": "n",
    "^": "^",
    "z": "z",
    "E": "E",
    "h": "h",
    "j": "j",
    "s": "s",
    "g": "g",
    "H": "H",
    "q": "q",
    "f": "f",
    "x": "x",
    "S": "S",
    "$": "$",
    "d": "d",
    "D": "D",
    "k": "k",
    ">": "<",
    "'": "<",
    "}": "<",
    "&": "<",
    "<": "<",
}

ambiguousConsonantMap = {
    # These consonants are only unambiguous in certain contexts
    "l": ["l", ""],
    "w": "w",
    "y": "y",
    "p": ["t", ""],
}

maddaMap = {"|": [["<", "aa"], ["<", "AA"]]}

vowelMap = {
    "A": [["aa", ""], ["AA", ""]],
    "Y": [["aa", ""], ["AA", ""]],
    "w": [["uu0", "uu1"], ["UU0", "UU1"]],
    "y": [["ii0", "ii1"], ["II0", "II1"]],
    "a": ["a", "A"],
    "u": [["u0", "u1"], ["U0", "U1"]],
    "i": [["i0", "i1"], ["I0", "I1"]],
}

nunationMap = {
    "F": [["a", "n"], ["A", "n"]],
    "N": [["u1", "n"], ["U1", "n"]],
    "K": [["i1", "n"], ["I1", "n"]],
}

diacritics = ["o", "a", "u", "i", "F", "N", "K", "~"]
diacriticsWithoutShadda = ["o", "a", "u", "i", "F", "N", "K"]
emphatics = ["D", "S", "T", "Z", "g", "x", "q"]
forwardEmphatics = ["g", "x"]
consonants = [
    ">",
    "<",
    "}",
    "&",
    "'",
    "b",
    "t",
    "^",
    "j",
    "H",
    "x",
    "d",
    "*",
    "r",
    "z",
    "s",
    "$",
    "S",
    "D",
    "T",
    "Z",
    "E",
    "g",
    "f",
    "q",
    "k",
    "l",
    "m",
    "n",
    "h",
    "|",
]

punctuation = [".", ",", "?", "!"]

# ------------------------------------------------------------------------------------
# Words with fixed irregular pronunciations-------------------------------------------
# ------------------------------------------------------------------------------------
fixedWords = {
    "h*A": [
        "h aa * aa",
        "h aa * a",
    ],
    "h*h": ["h aa * i0 h i0", "h aa * i1 h"],
    "h*An": ["h aa * aa n i0", "h aa * aa n"],
    "h&lA'": ["h aa < u0 l aa < i0", "h aa < u0 l aa <"],
    "*lk": ["* aa l i0 k a", "* aa l i0 k"],
    "k*lk": ["k a * aa l i0 k a", "k a * aa l i1 k"],
    "*lkm": "* aa l i0 k u1 m",
    ">wl}k": ["< u0 l aa < i0 k a", "< u0 l aa < i1 k"],
    "Th": "T aa h a",
    "lkn": ["l aa k i0 nn a", "l aa k i1 n"],
    "lknh": "l aa k i0 nn a h u0",
    "lknhm": "l aa k i0 nn a h u1 m",
    "lknk": ["l aa k i0 nn a k a", "l aa k i0 nn a k i0"],
    "lknkm": "l aa k i0 nn a k u1 m",
    "lknkmA": "l aa k i0 nn a k u0 m aa",
    "lknnA": "l aa k i0 nn a n aa",
    "AlrHmn": ["rr a H m aa n i0", "rr a H m aa n"],
    "Allh": ["ll aa h i0", "ll aa h", "ll AA h u0", "ll AA h a", "ll AA h", "ll A"],
    "h*yn": ["h aa * a y n i0", "h aa * a y n"],
    "nt": "n i1 t",
    "fydyw": "v i0 d y uu1",
    "lndn": "l A n d u1 n",
}


def isFixedWord(word, results, orthography, pronunciations):
    lastLetter = ""
    if len(word) > 0:
        lastLetter = word[-1]
    if lastLetter == "a":
        lastLetter = ["a", "A"]
    elif lastLetter == "A":
        lastLetter = ["aa"]
    elif lastLetter == "u":
        lastLetter = ["u0"]
    elif lastLetter == "i":
        lastLetter = ["i0"]
    elif lastLetter in unambiguousConsonantMap:
        lastLetter = [unambiguousConsonantMap[lastLetter]]
    # Remove all dacritics from word
    wordConsonants = re.sub(r"[^h*Ahn\'>wl}kmyTtfd]", "", word)
    if wordConsonants in fixedWords:  # check if word is in the fixed word lookup table
        if isinstance(fixedWords[wordConsonants], list):
            for pronunciation in fixedWords[wordConsonants]:
                if pronunciation.split(" ")[-1] in lastLetter:
                    # add each pronunciation to the pronunciation dictionary
                    results += word + " " + pronunciation + "\n"
                    pronunciations.append(pronunciation.split(" "))
        else:
            # add pronunciation to the pronunciation dictionary
            results += word + " " + fixedWords[wordConsonants] + "\n"
            pronunciations.append(fixedWords[wordConsonants].split(" "))
    return results


def preprocess_utterance(utterance):
    # Do some normalisation work and split utterance to words
    utterance = utterance.replace("AF", "F")
    utterance = utterance.replace("\u0640", "")
    utterance = utterance.replace("o", "")
    utterance = utterance.replace("aA", "A")
    utterance = utterance.replace("aY", "Y")
    utterance = utterance.replace(" A", " ")
    utterance = utterance.replace("F", "an")
    utterance = utterance.replace("N", "un")
    utterance = utterance.replace("K", "in")
    utterance = utterance.replace("|", ">A")

    utterance = utterance.replace("i~", "~i")
    utterance = utterance.replace("a~", "~a")
    utterance = utterance.replace("u~", "~u")

    # Deal with Hamza types that when not followed by a short vowel letter,
    # this short vowel is added automatically
    utterance = re.sub(r"Ai", "<i", utterance)
    utterance = re.sub(r"Aa", ">a", utterance)
    utterance = re.sub(r"Au", ">u", utterance)
    utterance = re.sub(r"^>([^auAw])", ">a\\1", utterance)
    utterance = re.sub(r" >([^auAw ])", " >a\\1", utterance)
    utterance = re.sub(r"<([^i])", "<i\\1", utterance)

    utterance = re.sub(r"(\S)(\.|\?|,|!)", "\\1 \\2", utterance)

    utterance = utterance.split(" ")

    return utterance


def process_word(word):

    if word in punctuation:
        return word

    pronunciations = (
        []
    )  # Start with empty set of possible pronunciations of current word
    # Add fixed irregular pronuncations if possible
    isFixedWord(word, "", word, pronunciations)

    # Indicates whether current character is in an emphatic context or not. Starts with False
    emphaticContext = False
    # This is the end/beginning of word symbol. just for convenience
    word = "bb" + word + "ee"

    phones = []  # Empty list which will hold individual possible word's pronunciation

    # -----------------------------------------------------------------------------------
    # MAIN LOOP: here is where the Modern Standard Arabic phonetisation rule-set starts--
    # -----------------------------------------------------------------------------------
    for index in range(2, len(word) - 2):
        letter = word[index]  # Current Character
        letter1 = word[index + 1]  # Next Character
        letter2 = word[index + 2]  # Next-Next Character
        letter_1 = word[index - 1]  # Previous Character
        letter_2 = word[index - 2]  # Before Previous Character
        # ----------------------------------------------------------------------------------------------------------------
        if letter in consonants + ["w", "y"] and not letter in emphatics + [
            "r" """, u'l'"""
        ]:  # non-emphatic consonants (except for Lam and Ra) change emphasis back to False
            emphaticContext = False
        if letter in emphatics:  # Emphatic consonants change emphasis context to True
            emphaticContext = True
        # If following letter is backward emphatic, emphasis state is set to True
        if letter1 in emphatics and not letter1 in forwardEmphatics:
            emphaticContext = True
        # ----------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------
        # Unambiguous consonant phones. These map to a predetermined phoneme
        if letter in unambiguousConsonantMap:
            phones += [unambiguousConsonantMap[letter]]
        # ----------------------------------------------------------------------------------------------------------------
        if letter == "l":  # Lam is a consonant which requires special treatment
            # Lam could be omitted in definite article (sun letters)
            if (not letter1 in diacritics and not letter1 in vowelMap) and letter2 in [
                "~"
            ]:
                phones += [ambiguousConsonantMap["l"][1]]  # omit
            else:
                # do not omit
                phones += [ambiguousConsonantMap["l"][0]]
        # ----------------------------------------------------------------------------------------------------------------
        # shadda just doubles the letter before it
        if letter == "~" and not letter_1 in ["w", "y"] and len(phones) > 0:
            phones[-1] += phones[-1]
        # ----------------------------------------------------------------------------------------------------------------
        if letter == "|":  # Madda only changes based in emphaticness
            if emphaticContext:
                phones += [maddaMap["|"][1]]
            else:
                phones += [maddaMap["|"][0]]
        # ----------------------------------------------------------------------------------------------------------------
        if (
            letter == "p"
        ):  # Ta' marboota is determined by the following if it is a diacritic or not
            if letter1 in diacritics:
                phones += [ambiguousConsonantMap["p"][0]]
            else:
                phones += [ambiguousConsonantMap["p"][1]]
        # ----------------------------------------------------------------------------------------------------------------
        if letter in vowelMap:
            # Waw and Ya are complex they could be consonants or vowels and their gemination is complex as it could be a combination of a vowel and consonants
            if letter in ["w", "y"]:
                if (
                    letter1 in diacriticsWithoutShadda + ["A", "Y"]
                    or (
                        letter1 in ["w", "y"]
                        and not letter2 in diacritics + ["A", "w", "y"]
                    )
                    or (
                        letter_1 in diacriticsWithoutShadda
                        and letter1 in consonants + ["e"]
                    )
                ):
                    if (
                        letter in ["w"]
                        and letter_1 in ["u"]
                        and not letter1 in ["a", "i", "A", "Y"]
                    ) or (
                        letter in ["y"]
                        and letter_1 in ["i"]
                        and not letter1 in ["a", "u", "A", "Y"]
                    ):
                        if emphaticContext:
                            phones += [vowelMap[letter][1][0]]
                        else:
                            phones += [vowelMap[letter][0][0]]
                    else:
                        if letter1 in ["A"] and letter in ["w"] and letter2 in ["e"]:
                            phones += [
                                [ambiguousConsonantMap[letter], vowelMap[letter][0][0]]
                            ]
                        else:
                            phones += [ambiguousConsonantMap[letter]]
                elif letter1 in ["~"]:
                    if (
                        letter_1 in ["a"]
                        or (letter in ["w"] and letter_1 in ["i", "y"])
                        or (letter in ["y"] and letter_1 in ["w", "u"])
                    ):
                        phones += [
                            ambiguousConsonantMap[letter],
                            ambiguousConsonantMap[letter],
                        ]
                    else:
                        phones += [
                            vowelMap[letter][0][0],
                            ambiguousConsonantMap[letter],
                        ]
                else:  # Waws and Ya's at the end of the word could be shortened
                    if emphaticContext:
                        if letter_1 in consonants + ["u", "i"] and letter1 in ["e"]:
                            phones += [
                                [vowelMap[letter][1][0], vowelMap[letter][1][0][1:]]
                            ]
                        else:
                            phones += [vowelMap[letter][1][0]]
                    else:
                        if letter_1 in consonants + ["u", "i"] and letter1 in ["e"]:
                            phones += [
                                [vowelMap[letter][0][0], vowelMap[letter][0][0][1:]]
                            ]
                        else:
                            phones += [vowelMap[letter][0][0]]
            # Kasra and Damma could be mildened if before a final silent consonant
            if letter in ["u", "i"]:
                if emphaticContext:
                    if (
                        (letter1 in unambiguousConsonantMap or letter1 == "l")
                        and letter2 == "e"
                        and len(word) > 7
                    ):
                        phones += [vowelMap[letter][1][1]]
                    else:
                        phones += [vowelMap[letter][1][0]]
                else:
                    if (
                        (letter1 in unambiguousConsonantMap or letter1 == "l")
                        and letter2 == "e"
                        and len(word) > 7
                    ):
                        phones += [vowelMap[letter][0][1]]
                    else:
                        phones += [vowelMap[letter][0][0]]
            # Alif could be ommited in definite article and beginning of some words
            if letter in ["a", "A", "Y"]:
                if letter in ["A"] and letter_1 in ["w", "k"] and letter_2 == "b":
                    phones += [["a", vowelMap[letter][0][0]]]
                elif letter in ["A"] and letter_1 in ["u", "i"]:
                    temp = True  # do nothing
                # Waw al jama3a: The Alif after is optional
                elif letter in ["A"] and letter_1 in ["w"] and letter1 in ["e"]:
                    phones += [[vowelMap[letter][0][0], vowelMap[letter][0][1]]]
                elif letter in ["A", "Y"] and letter1 in ["e"]:
                    if emphaticContext:
                        phones += [[vowelMap[letter][1][0], vowelMap["a"][1]]]
                    else:
                        phones += [[vowelMap[letter][0][0], vowelMap["a"][0]]]
                else:
                    if emphaticContext:
                        phones += [vowelMap[letter][1][0]]
                    else:
                        phones += [vowelMap[letter][0][0]]
    # -------------------------------------------------------------------------------------------------------------------------
    # End of main loop---------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------
    possibilities = 1  # Holds the number of possible pronunciations of a word

    # count the number of possible pronunciations
    for letter in phones:
        if isinstance(letter, list):
            possibilities = possibilities * len(letter)

    # Generate all possible pronunciations
    for i in range(0, possibilities):
        pronunciations.append([])
        iterations = 1
        for index, letter in enumerate(phones):
            if isinstance(letter, list):
                curIndex = int(i / iterations) % len(letter)
                if letter[curIndex] != "":
                    pronunciations[-1].append(letter[curIndex])
                iterations = iterations * len(letter)
            else:
                if letter != "":
                    pronunciations[-1].append(letter)

    # Iterate through each pronunciation to perform some house keeping. And append pronunciation to dictionary
    # 1- Remove duplicate vowels
    # 2- Remove duplicate y and w
    for pronunciation in pronunciations:
        prevLetter = ""
        toDelete = []
        for i in range(0, len(pronunciation)):
            letter = pronunciation[i]
            # Delete duplicate consecutive vowels
            if (
                letter in ["aa", "uu0", "ii0", "AA", "UU0", "II0"]
                and prevLetter.lower() == letter[1:].lower()
            ):
                toDelete.append(i - 1)
                pronunciation[i] = pronunciation[i - 1][0] + pronunciation[i - 1]
            # Delete duplicates
            if letter in ["u0", "i0"] and prevLetter.lower() == letter.lower():
                toDelete.append(i - 1)
                pronunciation[i] = pronunciation[i - 1]
            if letter in ["y", "w"] and prevLetter == letter:  # delete duplicate
                pronunciation[i - 1] += pronunciation[i - 1]
                toDelete.append(i)

            prevLetter = letter
        for i in reversed(range(0, len(toDelete))):
            del pronunciation[toDelete[i]]

    return pronunciations[0]


def process_utterance(utterance):

    utterance = preprocess_utterance(utterance)
    phonemes = []

    for word in utterance:
        if word in ["-", "sil"]:
            phonemes.append(["sil"])
            continue

        phonemes_word = process_word(word)
        if phonemes_word in punctuation and phonemes:
            phonemes[-1] += phonemes_word
        else:
            phonemes.append(phonemes_word)

    final_sequence = " + ".join(
        " ".join(phon for phon in phones) for phones in phonemes
    )

    return final_sequence
