import logging
import re

from ovos_number_parser import pronounce_number, is_fractional, pronounce_fraction
from ovos_number_parser.util import is_numeric
from unicode_rbnf import RbnfEngine, FormatPurpose

LOG = logging.getLogger("normalize")

# A dictionary of common contractions and their expanded forms.
# This list is very comprehensive for English.
CONTRACTIONS = {
    "en": {
        "I'd": "I would",
        "I'll": "I will",
        "I'm": "I am",
        "I've": "I have",
        "ain't": "is not",
        "aren't": "are not",
        "can't": "can not",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "gonna": "going to",
        "gotta": "got to",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "how'd": "how did",
        "how'll": "how will",
        "how's": "how is",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "might've": "might have",
        "mightn't": "might not",
        "must've": "must have",
        "mustn't": "must not",
        "needn't": "need not",
        "oughtn't": "ought not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "somebody's": "somebody is",
        "someone'd": "someone would",
        "someone'll": "someone will",
        "someone's": "someone is",
        "that'd": "that would",
        "that'll": "that will",
        "that's": "that is",
        "there'd": "there would",
        "there're": "there are",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'd": "what did",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "whats": "what is",
        "when'd": "when did",
        "when's": "when is",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'd": "who would",
        "who'd've": "who would have",
        "who'll": "who will",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "why'd": "why did",
        "why're": "why are",
        "why's": "why is",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'ain't": "you are not",
        "y'aint": "you are not",
        "y'all": "you all",
        "ya'll": "you all",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
        "I'm'a": "I am going to",
        "I'm'o": "I am going to",
        "I'll've": "I will have",
        "I'd've": "I would have",
        "Whatcha": "What are you",
        "amn't": "am not",
        "'cause": "because",
        "can't've": "cannot have",
        "couldn't've": "could not have",
        "daren't": "dare not",
        "daresn't": "dare not",
        "dasn't": "dare not",
        "everyone's": "everyone is",
        "gimme": "give me",
        "gon't": "go not",
        "hadn't've": "had not have",
        "he've": "he would have",
        "he'll've": "he will have",
        "he'd've": "he would have",
        "here's": "here is",
        "how're": "how are",
        "how'd'y": "how do you do",
        "howd'y": "how do you do",
        "howdy": "how do you do",
        "'tis": "it is",
        "'twas": "it was",
        "it'll've": "it will have",
        "it'd've": "it would have",
        "kinda": "kind of",
        "let's": "let us",
        "ma'am": "madam",
        "may've": "may have",
        "mayn't": "may not",
        "mightn't've": "might not have",
        "mustn't've": "must not have",
        "needn't've": "need not have",
        "ol'": "old",
        "oughtn't've": "ought not have",
        "sha'n't": "shall not",
        "shan't": "shall not",
        "shalln't": "shall not",
        "shan't've": "shall not have",
        "she'd've": "she would have",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "something's": "something is",
        "that're": "that are",
        "that'd've": "that would have",
        "there'll": "there will",
        "there'd've": "there would have",
        "these're": "these are",
        "they'll've": "they will have",
        "they'd've": "they would have",
        "this's": "this is",
        "this'll": "this will",
        "this'd": "this would",
        "those're": "those are",
        "to've": "to have",
        "wanna": "want to",
        "we'll've": "we will have",
        "we'd've": "we would have",
        "what'll've": "what will have",
        "when've": "when have",
        "where're": "where are",
        "which's": "which is",
        "who'll've": "who will have",
        "why've": "why have",
        "will've": "will have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "you'll've": "you will have"
    }
}

# Dictionaries for titles, units, and their full word equivalents.
TITLES = {
    "en": {
        "Dr.": "Doctor",
        "Mr.": "Mister",
        "Prof.": "Professor"
    },
    "ca": {
        "Dr.": "Doctor",
        "Sr.": "Senyor",
        "Sra.": "Senyora",
        "Prof.": "Professor"
    },
    "es": {
        "Dr.": "Doctor",
        "Sr.": "Señor",
        "Sra.": "Señora",
        "Prof.": "Profesor",
        "D.": "Don",
        "Dña.": "Doña"
    },
    "pt": {
        "Dr.": "Doutor",
        "Sr.": "Senhor",
        "Sra.": "Senhora",
        "Prof.": "Professor",
        "Drª.": "Doutora",
        "Eng.": "Engenheiro",
        "D.": "Dom",
        "Dª": "Dona"
    },
    "gl": {
        "Dr.": "Doutor",
        "Sr.": "Señor",
        "Sra.": "Señora",
        "Prof.": "Profesor",
        "Srta.": "Señorita"
    },
    "fr": {
        "Dr.": "Docteur",
        "M.": "Monsieur",
        "Mme": "Madame",
        "Mlle": "Mademoiselle",
        "Prof.": "Professeur",
        "Pr.": "Professeur"
    },
    "it": {
        "Dr.": "Dottore",
        "Sig.": "Signore",
        "Sig.ra": "Signora",
        "Prof.": "Professore",
        "Dott.ssa": "Dottoressa",
        "Sig.na": "Signorina"
    },
    "nl": {
        "Dr.": "Dokter",
        "Dhr.": "De Heer",
        "Mevr.": "Mevrouw",
        "Prof.": "Professor",
        "Drs.": "Dokterandus",
        "Ing.": "Ingenieur"
    },
    "de": {
        "Dr.": "Doktor",
        "Prof.": "Professor"
    }
}

UNITS = {
    "en": {
        "€": "euros",
        "%": "per cent",
        "ºC": "degrees celsius",
        "ºF": "degrees fahrenheit",
        "ºK": "degrees kelvin",
        "°": "degrees",
        "$": "dollars",
        "£": "pounds",
        "km": "kilometers",
        "m": "meters",
        "cm": "centimeters",
        "mm": "millimeters",
        "ft": "feet",
        "in": "inches",
        "yd": "yards",
        "mi": "miles",
        "kg": "kilograms",
        "g": "grams",
        "lb": "pounds",
        "oz": "ounces",
        "L": "liters",
        "mL": "milliliters",
        "gal": "gallons",
        "qt": "quarts",
        "pt": "pints",
        "hr": "hours",
        "min": "minutes",
        "s": "seconds"
    },
    "pt": {
        "€": "euros",
        "%": "por cento",
        "ºC": "graus celsius",
        "ºF": "graus fahrenheit",
        "ºK": "graus kelvin",
        "°": "graus",
        "$": "dólares",
        "£": "libras",
        "km": "quilômetros",
        "m": "metros",
        "cm": "centímetros",
        "mm": "milímetros",
        "kg": "quilogramas",
        "g": "gramas",
        "L": "litros",
        "mL": "mililitros",
        "h": "horas",
        "min": "minutos",
        "s": "segundos"
    },
    "es": {
        "€": "euros",
        "%": "por ciento",
        "ºC": "grados celsius",
        "ºF": "grados fahrenheit",
        "ºK": "grados kelvin",
        "°": "grados",
        "$": "dólares",
        "£": "libras",
        "km": "kilómetros",
        "m": "metros",
        "cm": "centímetros",
        "kg": "kilogramos",
        "g": "gramos",
        "L": "litros",
        "mL": "millilitros"
    },
    "fr": {
        "€": "euros",
        "%": "pour cent",
        "ºC": "degrés celsius",
        "ºF": "degrés fahrenheit",
        "ºK": "degrés kelvin",
        "°": "degrés",
        "$": "dollars",
        "£": "livres",
        "km": "kilomètres",
        "m": "mètres",
        "cm": "centimètres",
        "kg": "kilogrammes",
        "g": "grammes",
        "L": "litres",
        "mL": "millilitres"
    },
    "de": {
        "€": "Euro",
        "%": "Prozent",
        "ºC": "Grad Celsius",
        "ºF": "Grad Fahrenheit",
        "ºK": "Grad Kelvin",
        "°": "Grad",
        "$": "Dollar",
        "£": "Pfund",
        "km": "Kilometer",
        "m": "Meter",
        "cm": "Zentimeter",
        "kg": "Kilogramm",
        "g": "Gramm",
        "L": "Liter",
        "mL": "Milliliter"
    }
}


def is_fraction(word: str) -> bool:
    """Checks if a word is a fraction like '3/3'."""
    if "/" in word:
        parts = word.split("/")
        if len(parts) == 2:
            n1, n2 = parts
            return n1.isdigit() and n2.isdigit()
    return False


def normalize(text: str, lang: str) -> str:
    """
    Normalizes a text string by expanding contractions, titles, and pronouncing
    numbers, units, and fractions.
    """
    lang, full_lang = lang.split("-")[0], lang
    dialog = text
    try:
        rbnf_engine = RbnfEngine.for_language(lang)
    except:  # Does not support the language
        rbnf_engine = None

    # Step 1: Pre-process with regex to handle English am/pm times
    if lang == "en":
        # Fix for DeprecationWarning by moving (?i) flag to the start of the pattern.
        dialog = re.sub(r"(?i)(\d+)(am|pm)", r"\1 \2", dialog)
        # Handle the pronunciation for TTS
        dialog = dialog.replace("am", "A M")
        dialog = dialog.replace("pm", "P M")

    # Step 2: Pre-process with regex to add spaces and expand units
    if lang in UNITS:
        # Create a list of all units to use in the regex pattern, sorted by length descending
        sorted_units = sorted(UNITS[lang].keys(), key=len, reverse=True)
        unit_pattern = "|".join(re.escape(unit) for unit in sorted_units)

        # Use a negative lookahead (?!\w) instead of a word boundary (\b)
        # to allow for more flexible unit matching without capturing parts of other words.
        pattern = re.compile(r"(\d+\.?\d*)\s*(" + unit_pattern + r")(?!\w)", re.IGNORECASE)

        # We need a function to handle the replacement dynamically
        def replace_unit(match):
            number = match.group(1)
            unit_symbol = match.group(2)
            unit_word = UNITS[lang][unit_symbol]
            return f"{number} {unit_word}"

        dialog = pattern.sub(replace_unit, dialog)

    # Step 3: Handle dates and times
    # A more robust implementation would use a ovos-date-parser
    # to parse and format dates and times for natural language pronunciation.
    # For example, "03/08/2025" could become "August third, twenty twenty-five".
    #
    # TODO: Add more general date and time normalization logic here.

    words = dialog.split()
    normalized_words = []
    for word in words:
        # Step 4: Expand contractions
        if word in CONTRACTIONS.get(lang, {}):
            normalized_words.append(CONTRACTIONS[lang][word])
            continue

        # Step 5: Expand titles
        if word in TITLES.get(lang, {}):
            normalized_words.append(TITLES[lang][word])
            continue

        # Step 6: Pronounce numbers and fractions
        if is_numeric(word):
            try:
                num = float(word) if "." in word else int(word)
                normalized_words.append(pronounce_number(num, lang=full_lang))
            except Exception as e:
                # The ovos-number-parser library may raise an error for some languages/numbers.
                # This could be due to a missing language pack or a bug in the library.
                # We will log the error and fall back to the original word.
                LOG.error(f"ovos-number-parser failed to pronounce number: {word} - ({e})")
                normalized_words.append(word)  # Fallback to original word

        elif is_fraction(word):
            try:
                # Handle fractions
                normalized_words.append(pronounce_fraction(word, full_lang))
            except Exception as e:
                # The ovos-number-parser library may raise an error for some languages/fractions.
                # We will log the error and fall back to the original word.
                LOG.error(f"ovos-number-parser failed to pronounce number: {word} - ({e})")
                normalized_words.append(word)  # Fallback to original word

        # Step 7: Fallback for digits (handles single digits not caught by other rules)
        elif rbnf_engine and word.isdigit():
            try:
                normalized_words.append(rbnf_engine.format_number(word, FormatPurpose.CARDINAL).text)
            except Exception as e:
                LOG.error(f"unicode-rbnf failed to pronounce number: {word} - ({e})")
                normalized_words.append(word)  # Fallback to original word
        else:
            normalized_words.append(word)

    dialog = " ".join(normalized_words)

    LOG.debug(f"normalized dialog: '{text}' -> '{dialog}'")
    return dialog


if __name__ == "__main__":
    # Example usage for demonstration purposes
    print("English example: " + normalize('I\'m Dr. Prof. 3/3 0.5% of 12345€, 5ft, and 10kg', 'en'))
    print(f"Portuguese example: {normalize('Dr. Prof. 3/3 0.5% de 12345€, 5m, e 10kg', 'pt')}")
    # Example with a simple number
    print(f"English simple number: {normalize('The number is 42', 'en')}")
    # Example with a fraction
    print(f"English fraction: {normalize('The fraction is 1/2', 'en')}")
    # Example with a plural fraction
    print(f"English plural fraction: {normalize('There are 3/4 of a cup', 'pt')}")
    # Example with a new unit
    print(f"Spanish example with units: {normalize('The temperature is 25ºC', 'es')}")
    print("French example with units: " + normalize('C\'est 10kg de sucre', 'fr'))
    print(f"German example with units: {normalize('Das kostet 50€', 'de')}")
    # Example with number attached to a unit
    print(f"English attached unit: {normalize('The box weighs 10kg', 'en')}")
    print(f"Portuguese attached unit: {normalize('Ele tem 10m de altura', 'pt')}")
    # Example demonstrating the fix for your reported bug
    print(f"Spanish without overlapping: {normalize('Los grados centígrados', 'es')}")
    # Example demonstrating the fix for your reported bug (incorrect input)
    print(f"Spanish incorrect input: {normalize('La temperatura es 25 gradosC', 'es')}")
    # Example demonstrating the fix for the Portuguese comma issue
    print(f"Portuguese with punctuation: {normalize('12345€, 5m e 10kg', 'pt')}")
    # New examples for am/pm
    print(f"English AM time: {normalize('The meeting is at 10am', 'en')}")
    print(f"English PM time: {normalize('The party is at 7pm', 'en')}")
