import unittest

from phoonnx.util import _normalize_dates_and_times


class TestAmPmNormalization(unittest.TestCase):
    """Regression tests for digit-anchored am/pm normalization.

    Previously a global text.replace("am", "A M").replace("pm", "P M")
    corrupted any ordinary word containing "am" or "pm", e.g. "am", "spam",
    "program", "campus". The fix anchors the substitution to a preceding
    digit so only actual time expressions are affected.
    """

    def test_am_word_untouched(self):
        result = _normalize_dates_and_times("I am happy today", "en-US")
        self.assertEqual(result, "I am happy today")

    def test_spam_untouched(self):
        result = _normalize_dates_and_times("this is spam", "en-US")
        self.assertEqual(result, "this is spam")

    def test_campus_program_untouched(self):
        result = _normalize_dates_and_times("campus program schedule", "en-US")
        self.assertEqual(result, "campus program schedule")

    def test_am_radio_casing_untouched(self):
        result = _normalize_dates_and_times("tune in to AM radio", "en-US")
        self.assertEqual(result, "tune in to AM radio")

    def test_combined_words_and_real_time(self):
        result = _normalize_dates_and_times(
            "I am happy this is spam, meet me at 9am", "en-US"
        )
        self.assertEqual(result, "I am happy this is spam, meet me at 9 A M")

    def test_time_9am(self):
        result = _normalize_dates_and_times("the train leaves at 9am", "en-US")
        self.assertEqual(result, "the train leaves at 9 A M")

    def test_time_with_space(self):
        result = _normalize_dates_and_times("lunch is at 10 pm", "en-US")
        self.assertEqual(result, "lunch is at 10 P M")

    def test_time_uppercase(self):
        result = _normalize_dates_and_times("wake up at 7AM sharp", "en-US")
        self.assertEqual(result, "wake up at 7 A M sharp")

    def test_time_dotted_pm(self):
        result = _normalize_dates_and_times("arrive by 12p.m.", "en-US")
        self.assertTrue(result.startswith("arrive by 12 P M"))

    def test_multiple_times_in_one_sentence(self):
        result = _normalize_dates_and_times(
            "open from 9am to 5pm daily", "en-US"
        )
        self.assertEqual(result, "open from 9 A M to 5 P M daily")

    def test_non_english_language_untouched(self):
        # am/pm normalization is English-only; other languages must be
        # unaffected even when the substrings "am"/"pm" appear.
        text = "Je suis am spam et le programme est fantastique"
        result = _normalize_dates_and_times(text, "fr-FR")
        self.assertEqual(result, text)


class TestMultiDateExpansion(unittest.TestCase):
    """Regression tests for expanding every date in the text.

    Previously date_pattern.search(text) only located and expanded the
    FIRST date, leaving any subsequent dates in the text unspoken.
    """

    def test_single_date_expanded(self):
        result = _normalize_dates_and_times(
            "The event is on 12/25/2024", "en-US", "MDY"
        )
        self.assertNotIn("12/25/2024", result)
        self.assertIn("december", result.lower())

    def test_two_dates_both_expanded(self):
        result = _normalize_dates_and_times(
            "From 01/02/2025 until 03/04/2025", "en-US", "MDY"
        )
        self.assertNotIn("01/02/2025", result)
        self.assertNotIn("03/04/2025", result)
        self.assertIn("january", result.lower())
        self.assertIn("march", result.lower())

    def test_three_dates_all_expanded(self):
        result = _normalize_dates_and_times(
            "Dates: 01/02/2025, 03/04/2025 and 05/06/2025", "en-US", "MDY"
        )
        for raw_date in ("01/02/2025", "03/04/2025", "05/06/2025"):
            self.assertNotIn(raw_date, result)

    def test_identical_dates_both_expanded(self):
        # Regression: the old implementation used text.replace(match, ...)
        # which would replace ALL occurrences of an identical date string
        # from a single match, but relied on search() to only ever find
        # the first one. Confirm repeated identical dates are handled.
        result = _normalize_dates_and_times(
            "Reminder on 01/02/2025 and again 01/02/2025", "en-US", "MDY"
        )
        self.assertNotIn("01/02/2025", result)
        self.assertEqual(result.lower().count("january"), 2)


if __name__ == "__main__":
    unittest.main()
