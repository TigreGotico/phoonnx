"""``<say-as>`` for the quantities a skill kit writes: times, dates, currency, units, numbers.

Each is read through ``parse_ssml`` in the language of the text, as a voice receives it. The plain
text normalizer read ``14 h 30`` as "quatorze h trente", ``3,50 €`` as "trois virgule cinq euros" and
``1 200`` as "un deux cents"; the say-as path must not.
"""
from __future__ import annotations

import pytest

from phoonnx.ssml import parse_ssml


def said(kind: str, value: str, lang: str, extra: str = "") -> str:
    (piece,) = parse_ssml(f'<speak><say-as interpret-as="{kind}"{extra}>{value}</say-as></speak>', lang)
    return piece.text


@pytest.mark.parametrize("value,lang,expected", [
    # English: the abbreviation's own full stop must not meet the sentence's ("p.m..")
    ("9 pm", "en", "nine P M"), ("9 am", "en", "nine A M"), ("9 a.m.", "en", "nine A M"),
    ("9am", "en", "nine A M"), ("9PM", "en", "nine P M"), ("9:15 p.m.", "en", "quarter past nine P M"),
    ("2:30pm", "en", "half past two P M"), ("12 pm", "en", "noon"), ("12 am", "en", "midnight"),
    # French: heure(s) is feminine ("demie"), midi and minuit are not
    ("7:30 pm", "fr", "sept heures et demie du soir"), ("7:30 am", "fr", "sept heures et demie du matin"),
    ("1:30 pm", "fr", "une heure et demie de l'après-midi"), ("12:30 pm", "fr", "midi et demi"),
    ("12:30 am", "fr", "minuit et demi"), ("7:15 pm", "fr", "sept heures et quart du soir"),
    ("7:45 pm", "fr", "huit heures moins le quart du soir"),
    ("9 pm", "fr", "neuf heures du soir"), ("9 am", "fr", "neuf heures du matin"),
    ("9am", "fr", "neuf heures du matin"), ("9 a.m.", "fr", "neuf heures du matin"),
    ("14 h 30", "fr", "quatorze heures trente"), ("14h30", "fr", "quatorze heures trente"),
    ("14 h", "fr", "quatorze heures"), ("14:30", "fr", "quatorze heures trente"),
    ("14.30 Uhr", "de", "vierzehn uhr dreißig"), ("14 Uhr", "de", "vierzehn uhr"),
    ("9 pm", "es", "las nueve de la noche"), ("9 pm", "de", "neun uhr abends"),
])
def test_a_time_is_read_in_the_words_of_its_language(value, lang, expected):
    assert said("time", value, lang) == expected


@pytest.mark.parametrize("value", ["9", "25:00", "14:61", "13 pm", "0 am", "soon"])
def test_a_time_that_is_not_one_is_kept_as_written(value):
    assert said("time", value, "en") == value


def test_a_time_ending_a_sentence_has_one_full_stop_not_two():
    (piece,) = parse_ssml('<speak>See you at <say-as interpret-as="time">9 p.m.</say-as>.</speak>', "en")
    assert piece.text == "See you at nine P M." and ".." not in piece.text


@pytest.mark.parametrize("value,lang,expected", [
    ("3,50 €", "fr", "trois euros et cinquante centimes"), ("3,50 €", "en", "three euros and fifty cents"),
    ("1.200,50 €", "de", "eintausendzweihundert Euro und fünfzig Cent"),
    ("1.200,50 €", "es", "mil doscientos euros con cincuenta céntimos"),
    ("1.200,50 €", "nl", "duizendtweehonderd euro en vijftig cent"),
    ("1.200,50 €", "pt", "mil e duzentos euros e cinquenta cêntimos"),
    ("1.200,50 €", "it", "milleduecento euro e cinquanta centesimi"),
    ("£1.200,50", "es", "mil doscientas libras con cincuenta peniques"), ("£1,00", "es", "una libra"),
    ("£2,00", "pt", "duas libras"), ("£1,00", "it", "una sterlina"), ("£1.200,50", "it", "milleduecento sterline e cinquanta penny"),
    ("1,200.50 CHF", "en", "one thousand, two hundred francs and fifty centimes"),
    ("CHF 12,50", "it", "dodici CHF e cinquanta"),  # no word for the currency: its code, and "and"
])
def test_a_currency_reads_its_decimal_mark_and_its_words(value, lang, expected):
    assert said("currency", value, lang) == expected


@pytest.mark.parametrize("value,lang,expected", [
    ("1 200", "fr", "mille deux cents"), ("1 200", "fr", "mille deux cents"), ("1 200", "fr", "mille deux cents"),
    ("1.200", "es", "mil doscientos"), ("1.200", "nl", "duizendtweehonderd"), ("1.200", "pt", "mil e duzentos"),
    ("1.200", "it", "milleduecento"), ("1.200,50", "de", "eintausendzweihundert Komma fünf null"),
    ("3,5", "fr", "trois virgule cinq"), ("-12", "fr", "moins douze"),
])
def test_a_number_with_a_group_or_decimal_mark_is_one_number(value, lang, expected):
    assert said("cardinal", value, lang) == expected


@pytest.mark.parametrize("value,lang,expected", [
    ("1,5 L", "de", "eins Komma fünf Liter"), ("1,5 L", "en", "one point five liters"),
    ("1,5 L", "fr", "un virgule cinq litres"), ("1 200 km", "fr", "mille deux cents kilomètres"),
    ("1 200 km", "de", "eintausendzweihundert Kilometer"), ("1 kg", "en", "one kilogram"),
    ("1 kg", "fr", "un kilogramme"), ("1 kg", "de", "ein Kilogramm"), ("1 kg", "es", "un kilogramo"),
    ("1 km/h", "en", "one kilometer per hour"), ("2 kg", "en", "two kilograms"), ("1 l", "fr", "un litre"),
    ("5 km/h", "fr", "cinq kilomètres par heure"), ("12 °C", "fr", "douze degrés celsius"),
    ("1 °C", "en", "one degree celsius"), ("50 %", "fr", "cinquante pour cent"),
])
def test_a_quantity_reads_its_number_and_the_unit_agrees(value, lang, expected):
    assert said("unit", value, lang) == expected


@pytest.mark.parametrize("value,lang,expected", [
    ("9 novembre 2026", "fr", "lundi neuf novembre deux-mille-vingt-six"),
    ("1er mai 2026", "fr", "vendredi premier mai deux-mille-vingt-six"),
    ("November 9, 2026", "en", "monday, november ninth, twenty twenty six"),
    ("9 November 2026", "en", "monday, november ninth, twenty twenty six"),
    ("9. November 2026", "de", "neunte november zweitausendsechsundzwanzig"),
    ("9 de noviembre de 2026", "es", "nueve de noviembre de dos mil veintiséis"),
    ("9 de novembro de 2026", "pt", "nove de novembro de dois mil e vinte e seis"),
    ("9 novembre 2026", "it", "nove novembre duemilaventisei"),
    ("9 november 2026", "nl", "negen november tweeduizendzesentwintig"),
    ("09/11/2026", "fr", "lundi neuf novembre deux-mille-vingt-six"),
    ("2026-11-09", "es", "nueve de noviembre de dos mil veintiséis"),
])
def test_a_date_with_its_month_written_out_is_read_as_a_date(value, lang, expected):
    assert said("date", value, lang) == expected


@pytest.mark.parametrize("value,lang,expected", [
    ("1er", "fr", "premier"), ("2e", "fr", "deuxième"), ("21st", "en", "twenty-first"),
    ("1.", "de", "erste"), ("3º", "pt", "terceiro"), ("1", "es", "primero"),
])
def test_an_ordinal_with_its_suffix_is_read(value, lang, expected):
    assert said("ordinal", value, lang) == expected
