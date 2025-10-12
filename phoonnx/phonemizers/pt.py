from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.config import Alphabet


class TugaphonePhonemizer(BasePhonemizer):

    def __init__(self):
        from tugaphone import TugaPhonemizer
        self.tuga = TugaPhonemizer()
        super().__init__(Alphabet.IPA)

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
        return cls.match_lang(target_lang, ["pt-PT", "pt-BR", "pt-AO", "pt-MZ", "pt-TL"])

    def phonemize_string(self, text: str, lang: str) -> str:
        lang = self.get_lang(lang)
        # TODO - support regional dialects
        return self.tuga.phonemize_sentence(text, lang)



if __name__ == "__main__":

    pho = TugaphonePhonemizer()

    sentences = [
        "O gato dorme.",
        "Tu falas português muito bem.",
        "O comboio chegou à estação.",
        "A menina comeu o pão todo.",
        "Vou pôr a manteiga no frigorífico.",
        "Ele está a trabalhar no escritório.",
        "Choveu muito ontem à noite.",
        "A rapariga comprou um telemóvel novo.",
        "Vamos tomar um pequeno-almoço.",
        "O carro ficou sem gasolina."
    ]

    for s in sentences:
        print(s)
        for code in ["pt-PT", "pt-BR", "pt-AO", "pt-MZ", "pt-TL"]:
            print(f"{code} → {pho.phonemize_string(s, code)}")
        print("######")
