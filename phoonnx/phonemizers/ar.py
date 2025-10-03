from phoonnx.config import Alphabet
from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.thirdparty.bw2ipa import translate as bw2ipa
from phoonnx.thirdparty.mantoq import g2p as mantoq


class MantoqPhonemizer(BasePhonemizer):

    def __init__(self, alphabet=Alphabet.BUCKWALTER):
        if alphabet not in [Alphabet.IPA, Alphabet.BUCKWALTER]:
            raise ValueError("unsupported alphabet")
        super().__init__(alphabet)

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        """
        Validates and returns the closest supported language code.

        Args:
            target_lang (str): The language code to validate.

        Returns:
            str: The validated language code.

        Raises:
            ValueError: If the language code is unsupported.
        """
        # this check is here only to throw an exception if invalid language is provided
        return cls.match_lang(target_lang, ["ar"])

    def phonemize_string(self, text: str, lang: str = "ar") -> str:
        """
        Phonemizes an Arabic string using the Mantoq G2P tool.
        If the alphabet is set to IPA, it then converts the result using bw2ipa.
        """
        lang = self.get_lang(lang)
        # The mantoq function returns a tuple of (normalized_text, phonemes)
        normalized_text, phonemes = mantoq(text)

        # The phonemes are a list of characters, we join them into a string
        # and replace the word separator token with a space.
        phonemes = "".join(phonemes).replace("_+_", " ")

        if self.alphabet == Alphabet.IPA:
            # If the alphabet is IPA, we use the bw2ipa function to translate
            # the Buckwalter-like phonemes into IPA.
            return bw2ipa(phonemes)

        # Otherwise, we return the phonemes in the default Mantoq alphabet.
        return phonemes


if __name__ == "__main__":
    from phoonnx.phonemizers.mul import EspeakPhonemizer

    espeak = EspeakPhonemizer()

    # Initialize phonemizers for both MANTOQ and IPA alphabets
    pho_mantoq = MantoqPhonemizer(alphabet=Alphabet.IPA)


    def compare(text):
        print(f"Original Text: {text}")
        print(f"  Mantoq: {pho_mantoq.phonemize_string(text, 'ar')}")
        print(f"  Espeak: {espeak.phonemize_string(text, 'ar')}")

        ts = pho_mantoq.add_diacritics(text, 'ar')
        print(f"Tashkeel Text: {ts}")
        print(f"  Mantoq: {pho_mantoq.phonemize_string(ts, 'ar')}")
        print(f"  Espeak: {espeak.phonemize_string(ts, 'ar')}")
        print("\n#########################")


    text = "مرحبا بالعالم"
    compare(text)

    text = "ذهب الطالب إلى المكتبة لقراءة كتاب عن تاريخ الأندلس."
    compare(text)

    # 1. Test for gemination of a sun letter (e.g., ash-shams)
    text = "الشمس"
    compare(text)

    # 2. Test for long vowels (e.g., 'fil' - elephant)
    text = "فيل"
    compare(text)

    # 3. Test for glide (e.g., 'yawm' - day)
    text = "يوم"
    compare(text)

    # 4. Test for long vowels (e.g., 'suwr' - wall)
    text = "سور"
    compare(text)

    # 5. Test for glide (e.g., 'law' - if)
    text = "لو"
    compare(text)
