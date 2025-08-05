from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.thirdparty.mantoq import g2p as mantoq
from phoonnx.config import Alphabet
from phoonnx.thirdparty.bw2ipa import translate as bw2ipa


class MantoqPhonemizer(BasePhonemizer):

    def __init__(self, alphabet=Alphabet.MANTOQ):
        if alphabet not in [Alphabet.IPA, Alphabet.MANTOQ]:
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
        phonemes =  "".join(phonemes).replace("_+_", " ")

        if self.alphabet == Alphabet.IPA:
            # If the alphabet is IPA, we use the bw2ipa function to translate
            # the Buckwalter-like phonemes into IPA.
            return bw2ipa(phonemes)

        # Otherwise, we return the phonemes in the default Mantoq alphabet.
        return phonemes


if __name__ == "__main__":
    # Initialize phonemizers for both MANTOQ and IPA alphabets
    pho_mantoq = MantoqPhonemizer()
    pho_ipa = MantoqPhonemizer(alphabet=Alphabet.IPA)

    text1 = "مرحبا بالعالم"
    print(f"Original Text: {text1}")
    print(f"  Mantoq Phonemizer: {pho_mantoq.phonemize_string(text1, 'ar')}")
    print(f"  IPA Phonemizer:    {pho_ipa.phonemize_string(text1, 'ar')}")
    print("-" * 20)

    text2 = "ذهب الطالب إلى المكتبة لقراءة كتاب عن تاريخ الأندلس."
    print(f"Original Text: {text2}")
    print(f"  Mantoq Phonemizer: {pho_mantoq.phonemize_string(text2, 'ar')}")
    print(f"  IPA Phonemizer:    {pho_ipa.phonemize_string(text2, 'ar')}")
    print("-" * 20)

    # --- New Test Cases to check bw2ipa logic ---
    print("--- New Test Cases for bw2ipa logic ---")

    # 1. Test for gemination of a sun letter (e.g., ash-shams)
    text3 = "الشمس"
    print(f"Original Text: '{text3}'")
    print(f"  Mantoq Phonemizer: {pho_mantoq.phonemize_string(text3, 'ar')}")
    print(f"  IPA Phonemizer:    {pho_ipa.phonemize_string(text3, 'ar')}")
    print("-" * 20)

    # 2. Test for long vowels (e.g., 'fil' - elephant)
    text4 = "فيل"
    print(f"Original Text: '{text4}'")
    print(f"  Mantoq Phonemizer: {pho_mantoq.phonemize_string(text4, 'ar')}")
    print(f"  IPA Phonemizer:    {pho_ipa.phonemize_string(text4, 'ar')}")
    print("-" * 20)

    # 3. Test for glide (e.g., 'yawm' - day)
    text5 = "يوم"
    print(f"Original Text: '{text5}'")
    print(f"  Mantoq Phonemizer: {pho_mantoq.phonemize_string(text5, 'ar')}")
    print(f"  IPA Phonemizer:    {pho_ipa.phonemize_string(text5, 'ar')}")
    print("-" * 20)

    # 4. Test for long vowels (e.g., 'suwr' - wall)
    text6 = "سور"
    print(f"Original Text: '{text6}'")
    print(f"  Mantoq Phonemizer: {pho_mantoq.phonemize_string(text6, 'ar')}")
    print(f"  IPA Phonemizer:    {pho_ipa.phonemize_string(text6, 'ar')}")
    print("-" * 20)

    # 5. Test for glide (e.g., 'law' - if)
    text7 = "لو"
    print(f"Original Text: '{text7}'")
    print(f"  Mantoq Phonemizer: {pho_mantoq.phonemize_string(text7, 'ar')}")
    print(f"  IPA Phonemizer:    {pho_ipa.phonemize_string(text7, 'ar')}")
    print("-" * 20)
