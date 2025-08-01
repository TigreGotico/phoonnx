import abc
from typing import List


from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.thirdparty.zh_num import num2str


class JiebaPhonemizer(BasePhonemizer):
    """
    A non-phonemizing class that simply uses Jieba to segment Chinese text
    into words with spaces for token separation.
    """

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
        return cls.match_lang(target_lang, ["zh"])

    def phonemize_string(self, text: str, lang: str = "zh") -> str:
        """
        Segments the input Chinese text using Jieba.

        Args:
            text (str): The input sentence.
            lang (str): Language code (must be "zh").

        Returns:
            str: Tokenized text with words separated by spaces.
        """
        import jieba
        lang = self.get_lang(lang)
        seg_list = jieba.cut(text, cut_all=False)
        seg_list = [num2str(w) if w.isdigit() else w for w in seg_list]
        return " ".join(seg_list)


class BaseChinesePinyinPhonemizer(BasePhonemizer):
    """
    Base class for Chinese phonemizers using different pinyin G2P libraries.
    Supports optional IPA conversion and segmentation via Jieba.
    """

    def __init__(self, ipa: bool = False, jieba: bool = True, retone=True):
        """
        Initializes the phonemizer.

        Args:
            ipa (bool): Whether to convert pinyin to IPA.
            jieba (bool): Whether to segment text using Jieba before phonemization.
        """
        self.ipa = ipa
        self.jieba = jieba
        self.retone = retone
        from pinyin_to_ipa import pinyin_to_ipa
        self.pinyin_to_ipa = pinyin_to_ipa

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
        return cls.match_lang(target_lang, ["zh"])

    @staticmethod
    def _retone(p):
        p = p.replace('˧˩˧', '↓')  # third tone
        p = p.replace('˧˥', '↗')  # second tone
        p = p.replace('˥˩', '↘')  # fourth tone
        p = p.replace('˥', '→')  # first tone
        p = p.replace(chr(635) + chr(809), 'ɨ').replace(chr(633) + chr(809), 'ɨ')
        assert chr(809) not in p, p
        return p

    def to_ipa(self, phones: List[str]) -> List[str]:
        """
        Converts a list of pinyin syllables to IPA. Falls back to the original syllable if conversion fails.

        Args:
            phones (List[str]): List of pinyin syllables or phrases.

        Returns:
            List[str]: Corresponding IPA or original syllables.
        """
        ipa_phones: List[str] = []
        for p in phones:
            if p == " ":
                ipa_phones.append(" ")
                continue
            pho_str = ""
            for sp in p.split():  # G2P might return phrases with multiple syllables
                try:
                    pho = self.pinyin_to_ipa(sp.strip())[0][0]
                    if self.retone:
                        pho = self._retone(pho)
                    pho_str += pho
                except Exception:
                    pass
            ipa_phones.append(pho_str)
        return ipa_phones

    def phonemize_string(self, text: str, lang: str = "zh") -> str:
        """
        Converts input text to a space-separated phoneme string.

        Args:
            text (str): The input sentence.
            lang (str): Language code (must be "zh").

        Returns:
            str: Space-separated phoneme string.
        """
        phones: List[str] = []
        lang = self.get_lang(lang)
        if self.jieba:
            import jieba
            for chunk in jieba.cut(text, cut_all=False):
                if chunk.isdigit():
                    chunk = num2str(chunk)
                phones += self.get_pinyin(chunk)
                phones += [" "]  # keep jieba whitespace
        else:
            phones = self.get_pinyin(text)
        if self.ipa:
            phones = self.to_ipa(phones)
        return "".join(phones)

    @abc.abstractmethod
    def get_pinyin(self, text: str) -> List[str]:
        """
        Abstract method to be implemented by subclasses for converting text to pinyin.

        Args:
            text (str): Input Chinese text.

        Returns:
            List[str]: List of pinyin tokens.
        """
        return NotImplemented


class G2pCPhonemizer(BaseChinesePinyinPhonemizer):
    """
    Phonemizer using g2pc (CRF-based Grapheme-to-Phoneme converter).
    https://github.com/Kyubyong/g2pC
    """

    def __init__(self, ipa: bool = False, jieba: bool = True):
        from g2pc import G2pC
        self.g2p = G2pC()
        super().__init__(ipa, jieba)

    def get_pinyin(self, text: str) -> List[str]:
        """
        Returns a list of pinyin syllables from g2pc.

        Args:
            text (str): Input Chinese text.

        Returns:
            List[str]: Pinyin tokens.
        """
        print(self.g2p(text))
        return [a[3] for a in self.g2p(text)]


class G2pMPhonemizer(BaseChinesePinyinPhonemizer):
    """
    Phonemizer using g2pM - A Neural Grapheme-to-Phoneme Conversion Package for Mandarin Chinese
    https://github.com/kakaobrain/g2pm
    """

    def __init__(self, tone: bool = True, char_split: bool = False, ipa: bool = False, jieba: bool = True):
        from g2pM import G2pM
        self.g2p = G2pM()
        self.tone = tone
        self.char_split = char_split
        super().__init__(ipa, jieba)

    def get_pinyin(self, text: str) -> List[str]:
        """
        Returns a list of pinyin tokens from g2pM.

        Args:
            text (str): Input Chinese text.

        Returns:
            List[str]: Pinyin tokens.
        """
        return self.g2p(text, tone=self.tone, char_split=self.char_split)


class XpinyinPhonemizer(BaseChinesePinyinPhonemizer):
    """
    Phonemizer using xpinyin (basic pinyin generator with optional tone marks).
    """

    def __init__(self, tone_marks: str = "numbers", ipa: bool = False, jieba: bool = True):
        from xpinyin import Pinyin
        self.g2p = Pinyin()
        self.tone_marks = tone_marks
        super().__init__(ipa, jieba)

    def get_pinyin(self, text: str) -> List[str]:
        """
        Returns a list of pinyin tokens from xpinyin.

        Args:
            text (str): Input Chinese text.

        Returns:
            List[str]: Pinyin tokens.
        """
        return self.g2p.get_pinyin(text, tone_marks=self.tone_marks).split("-")


class PypinyinPhonemizer(BaseChinesePinyinPhonemizer):
    """
    Phonemizer using pypinyin (comprehensive and accurate pinyin library).
    """

    def __init__(self, ipa: bool = False, jieba: bool = True):
        from pypinyin import pinyin
        self.g2p = pinyin
        super().__init__(ipa, jieba)

    def get_pinyin(self, text: str) -> List[str]:
        """
        Returns a list of pinyin tokens from pypinyin.

        Args:
            text (str): Input Chinese text.

        Returns:
            List[str]: Pinyin tokens.
        """
        return [p[0] for p in self.g2p(text)]


if __name__ == "__main__":
    lang = "zh"
    text = "然而，他红了20年以后，他竟退出了大家的视线。"

    pho = JiebaPhonemizer()
    pho1 = G2pCPhonemizer(ipa=True)
    pho2 = G2pMPhonemizer(ipa=True)
    pho3 = XpinyinPhonemizer(ipa=True)
    pho4 = PypinyinPhonemizer(ipa=True)

    from phoonnx.phonemizers.mul import MisakiPhonemizer

    pho5 = MisakiPhonemizer()

    print(f"\n--- Getting phonemes for '{text}' ---")

    phones = pho5.phonemize(text, lang)
    print(f" Misaki: {phones}")

    phones = pho1.phonemize(text, lang)
    print(f" G2pC: {phones}")

    phones = pho2.phonemize(text, lang)
    print(f" G2pM: {phones}")

    phones = pho3.phonemize(text, lang)
    print(f" Xpinyin: {phones}")

    phones = pho4.phonemize(text, lang)
    print(f" Pypinyin: {phones}")

    phones = pho.phonemize(text, lang)
    print(f" Jieba: {phones}")

    #
    exit()
    #   Phonemes: [('ran2 er2 ， ta1 hong2 le5 2 0 nian2 yi3 hou4 ， ta1 jing4 tui4 chu1 le5 da4 jia1 de5 shi4 xian4 。', '.', True)]
    #   Phonemes: [('ran2 er2 ， ta1 hong2 le5 20 nian2 yi3 hou4 ， ta1 jing4 tui4 chu1 le5 da4 jia1 de5 shi4 xian4 。', '.', True)]
    #   Phonemes: [('ran2 er2 ， ta1 hong2 le5 20 nian2 yi3 hou4 ， ta1 jing4 tui4 chu1 le5 da4 jia1 de5 shi4 xian4 。', '.', True)]
    #   Phonemes: [('rán ér ， tā hóng le 20 nián yǐ hòu ， tā jìng tuì chū le dà jiā de shì xiàn 。', '.', True)]
