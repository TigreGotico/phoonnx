import csv
import math
import os.path
from base64 import b64decode
from pathlib import Path
from typing import Union, List, Dict

import regex as re


# ----------------------------
# Classes and Helper Functions
# ----------------------------

class ConversionTable:
    def __init__(self, name: str, tables_dir: Path):
        self.name = name
        # Open the CSV file located in the 'tables' folder
        table_path = tables_dir / f'{self.name}.csv'
        if not table_path.exists():
            raise FileNotFoundError(f"无法找到转换表文件: {table_path}")
        with open(table_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=',')
            # Iterate over each row in the file
            for row in reader:
                # For each header, set it as an attribute if it's not already set
                for header, value in row.items():
                    # Add the value to a list associated with the header
                    if not hasattr(self, header):
                        setattr(self, header, [])
                    getattr(self, header).append(value)
        # Convert lists to tuples because the contents should be immutable
        for header in reader.fieldnames:
            setattr(self, header, tuple(getattr(self, header)))

    def apply(self, text: str, find_in: str = '_from') -> str:
        # for a single phoneme, find it among _from (or any attribute name find_in)
        # and convert it to _to
        try:
            from_tuple = getattr(self, find_in)
            ind = from_tuple.index(text)
            return self._to[ind]
        except (AttributeError, ValueError):
            return text

    def sub(self, text: str, find_in: str = '_from') -> str:
        from_tuple = getattr(self, find_in)
        for index, item in enumerate(from_tuple):
            text = text.replace(item, self._to[index])
        return text

    def safe_index(self, attribute: str, element: str) -> int:
        target_tuple = getattr(self, attribute)
        try:
            return target_tuple.index(element)
        except ValueError:
            return -1

    def __str__(self):
        return str(f'ConversionTable {self.name}')


class Word:
    def __init__(self, hangul: str, tables_dir: Path):
        # word to convert
        self.hangul = hangul
        self.tables_dir = tables_dir
        self._jamo = self.to_jamo(hangul)
        self._cv = self.mark_CV(self.jamo)

    @property
    def jamo(self) -> str:
        return self._jamo

    @jamo.setter
    def jamo(self, value: str):
        self._jamo = value
        self._cv = self.mark_CV(self._jamo)

    @property
    def cv(self) -> str:
        return self._cv

    def mark_CV(self, jamo: str, convention: ConversionTable = None) -> str:
        # identify each element in jamo as either consonant or vowel
        r = ''

        if convention is None:
            convention = ConversionTable('ipa', self.tables_dir)

        consonants = convention.C
        vowels = convention.V

        for j in jamo:
            if j in vowels:
                r += 'V'
            elif j in consonants:
                r += 'C'
        return r

    def to_jamo(self, hangul: str, no_empty_onset: bool = True, sboundary: bool = False) -> str:
        # Convert Hangul forms to jamo, remove empty onset ㅇ
        # e.g., input "안녕" output "ㅏㄴㄴㅕㅇ"
        not_hangul = r'[^가-힣ㄱ-ㅎㅏ-ㅣ]'
        cleaned_hangul = re.sub(not_hangul, '', hangul)  # hangul without special characters
        jamo_forms = hangul_to_jamos(cleaned_hangul)

        jamo_forms = self.separate_double_coda(jamo_forms)  # divide double coda (e.g., "ㄳ" -> "ㄱㅅ")

        if no_empty_onset:  # remove soundless syllable initial ㅇ
            jamo_forms = self.remove_empty_onset(jamo_forms)

        if sboundary:
            # not implemented
            pass

        return ''.join(jamo_forms)

    def remove_empty_onset(self, syllables: List[str]) -> List[str]:
        r = []
        for syllable in syllables:
            to_append = syllable[1:] if syllable[0] == 'ㅇ' else syllable
            r.append(to_append)
        return r

    def separate_double_coda(self, syllables: List[str]) -> List[str]:
        r = []
        CT_double_codas = ConversionTable('double_coda', self.tables_dir)
        for syllable in syllables:
            if len(syllable) < 3:
                r.append(syllable)
                continue
            coda = syllable[2]
            try:
                separated_coda = CT_double_codas._separated[CT_double_codas._double.index(coda)]
                r.append(syllable[:2] + separated_coda)
                continue
            except ValueError:
                r.append(syllable)
                continue
        return r

    def __str__(self):
        return self.hangul


# ----------------------------
# Hangul Tools
# ----------------------------

GA_CODE = 44032  # The unicode representation of the Korean syllabic orthography starts with GA_CODE
G_CODE = 12593  # The unicode representation of the Korean phonetic (jamo) orthography starts with G_CODE
ONSET = 588
CODA = 28

# ONSET LIST. 00 -- 18
ONSET_LIST = ('ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ')

# VOWEL LIST. 00 -- 20
VOWEL_LIST = ('ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ',
              'ㅡ', 'ㅢ', 'ㅣ')

# CODA LIST. 00 -- 27 + 1 (0 for open syllable)
CODA_LIST = ('', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ',
             'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ')


def hangul_to_jamos(hangul: str) -> List[str]:
    # convert hangul strings to jamos
    # hangul: str. multiple korean letters like 가나다라마바사
    syllables = list(hangul)
    r = []

    for letter in syllables:
        if bool(re.match(r'^[가-힣]+$', letter)):  # if letter is a hangul character
            chr_code = ord(letter) - GA_CODE
            onset = math.floor(chr_code / ONSET)
            vowel = math.floor((chr_code - (ONSET * onset)) / CODA)
            coda = math.floor((chr_code - (ONSET * onset) - (CODA * vowel)))

            syllable = f'{ONSET_LIST[onset]}{VOWEL_LIST[vowel]}{CODA_LIST[coda]}'
        else:  # if letter is NOT a hangul character
            syllable = letter
        r.append(syllable)

    return r


def jamo_to_hangul(syllable: str) -> str:
    # only accept one syllable length of jamos and convert it to one hangul character
    if len(syllable) > 1:
        jamos = list(syllable)
        onset = ONSET_LIST.index(jamos[0])
        vowel = VOWEL_LIST.index(jamos[1])
        coda = CODA_LIST.index(jamos[2]) if len(syllable) == 3 else 0

        utf_pointer = (((onset * 21) + vowel) * 28) + coda + GA_CODE
        syllable = chr(utf_pointer)
    return syllable


# ----------------------------
# Hanja Tools
# ----------------------------

HIGHV_DIPHTHONGS = ("ㅑ", "ㅕ", "ㅖ", "ㅛ", "ㅠ", "ㅣ")


def realize_hanja(raw: str) -> str:
    # convert the Unicode code point (e.g., U+349A) into actual hanja 㒚
    stripped_raw = raw.strip('U+')  # 'U+' part is meaningless so strip
    r = chr(int(stripped_raw, 16))  # hexadecimal part into int and then into character
    return r


def load_jajeon(tables_dir: Path) -> Dict[str, str]:
    # import a 漢字 - 한글 conversion table
    jajeon = {}
    jajeon_path = tables_dir / 'hanja.tsv'
    if not jajeon_path.exists():
        raise FileNotFoundError(f"无法找到汉字转换表文件: {jajeon_path}")
    with open(jajeon_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 2:
                continue  # 跳过不完整的行
            # the original file uses the Unicode code point (e.g., U+349A), so need to convert this to the actual hanja
            key = realize_hanja(row[0])
            value = row[1]
            jajeon[key] = value
    return jajeon


def hanja_to_hangul(jajeon: Dict[str, str], char: str) -> str:
    try:
        r = jajeon[char]
    except KeyError:
        r = char
    return r


def initial_rule(char: str, jajeon: Dict[str, str]) -> str:
    # apply the 'initial rule' (두음규칙) where 'l' becomes 'n' and 'n' gets deleted word-initially
    # char: hangul character
    changed_flag = False
    jamos = hangul_to_jamos(char)
    jamos = ''.join(jamos)
    onset = jamos[0]
    nucleus = jamos[1]
    if onset == 'ㄹ':
        onset = 'ㄴ'
        changed_flag = True
    if onset == 'ㄴ' and nucleus in HIGHV_DIPHTHONGS:
        onset = 'ㅇ'
        changed_flag = True

    if changed_flag:
        jamo_list = list(jamos)
        jamo_list[0], jamo_list[1] = onset, nucleus
        jamos = ''.join(jamo_list)

    return jamo_to_hangul(jamos)


def hanja_cleaner(word: str, hanja_loc: List[int], tables_dir: Path) -> str:
    jajeon = load_jajeon(tables_dir)
    chars = list(word)

    for i in hanja_loc:
        if chars[i] in ["不", "不"] and (i < len(chars) - 1):  # if 不 appears in a non-final syllable
            if chars[i + 1] == "實":
                # special case: 不實 = 부실
                chars[i] = "부"
                chars[i + 1] = "실"
                continue
            else:
                # special case: 不 is pronounced as 부[pu] before an alveolar ㄷㅈ
                chars[i + 1] = hanja_to_hangul(jajeon, chars[i + 1])
                next_syllable = hangul_to_jamos(chars[i + 1])
                if len(next_syllable) == 0:
                    following_onset = ''
                else:
                    following_onset = ''.join(next_syllable)[0]
                chars[i] = "부" if following_onset in ["ㄷ", "ㅈ"] else "불"
                continue

        chars[i] = hanja_to_hangul(jajeon, chars[i])

        if i == 0:  # apply the 'initial rule' (두음법칙)
            chars[i] = initial_rule(chars[i], jajeon)

    return ''.join(chars)


# ----------------------------
# Phonological Rules
# ----------------------------

CT_double_codas = None
CT_neutral = None
CT_tensification = None
CT_assimilation = None
CT_aspiration = None
CT_convention = None

CONSONANTS = ()
VOWELS = ()
C_SONORANTS = ('ㄴ', 'ㄹ', 'ㅇ', 'ㅁ')
OBSTRUENTS = ()
SONORANTS = ()


def initialize_conversion_tables(tables_dir: Path):
    global CT_double_codas, CT_neutral, CT_tensification, CT_assimilation, CT_aspiration, CT_convention
    CT_double_codas = ConversionTable('double_coda', tables_dir)
    CT_neutral = ConversionTable('neutralization', tables_dir)
    CT_tensification = ConversionTable('tensification', tables_dir)
    CT_assimilation = ConversionTable('assimilation', tables_dir)
    CT_aspiration = ConversionTable('aspiration', tables_dir)
    CT_convention = ConversionTable('ipa', tables_dir)

    global CONSONANTS, VOWELS, OBSTRUENTS, SONORANTS
    CONSONANTS = tuple(
        list(CT_convention.C)[:-2])  # from the C column of the IPA table, remove special characters # and $
    VOWELS = tuple(list(CT_convention.V))  # from the V column of the IPA table
    OBSTRUENTS = tuple(set(CONSONANTS) - set(C_SONORANTS))
    SONORANTS = VOWELS + C_SONORANTS


def get_substring_ind(string: str, pattern: str) -> List[int]:
    return [match.start() for match in re.finditer(f'(?={pattern})', string)]


def transcribe(jamos: str, convention: ConversionTable = None, str_return: bool = False) -> Union[List[str], str]:
    if convention is None:
        convention = CT_convention
    transcribed = []
    for jamo in jamos:
        is_C = convention.safe_index('C', jamo)
        is_V = convention.safe_index('V', jamo)
        if is_V >= 0:
            transcribed.append(convention.VSymbol[is_V])
        elif is_C >= 0:
            transcribed.append(convention.CSymbol[is_C])

    if str_return:
        return ''.join(transcribed)
    return transcribed


def palatalize(word: Word) -> str:
    palatalization_table = {
        'ㄷ': 'ㅈ',
        'ㅌ': 'ㅊ'
    }
    hangul_syllables = list(word.hangul)
    to_jamo_bound = word.to_jamo
    syllables_in_jamo = [to_jamo_bound(syl) for syl in hangul_syllables]
    for i, syllable in enumerate(syllables_in_jamo):
        try:
            next_syllable = syllables_in_jamo[i + 1]
            if next_syllable[0] == 'ㅣ':
                new_coda = palatalization_table.get(syllable[-1], syllable[-1])
                syllables_in_jamo[i] = ''.join(list(syllables_in_jamo[i])[:-1] + [new_coda])
        except IndexError:
            continue
    new_jamo = ''.join(syllables_in_jamo)
    return new_jamo


def aspirate(word: Word) -> str:
    return CT_aspiration.sub(word.jamo)


def assimilate(word: Word) -> str:
    return CT_assimilation.sub(word.jamo)


def pot(word: Word) -> str:
    return CT_tensification.sub(word.jamo)


def neutralize(word: Word) -> str:
    new_jamos = list(word.jamo)
    for i, jamo in enumerate(new_jamos):
        if i == len(new_jamos) - 1 or word.cv[i + 1] == 'C':
            new_jamos[i] = CT_neutral.apply(jamo)
    return ''.join(new_jamos)


def delete_h(word: Word) -> str:
    h_locations = get_substring_ind(string=word.jamo, pattern='ㅎ')

    for h_location in reversed(h_locations):
        if h_location == 0 or h_location == len(word.jamo) - 1:
            # a word-initial h cannot undergo deletion
            continue
        preceding = word.jamo[h_location - 1]
        succeeding = word.jamo[h_location + 1]
        if preceding in SONORANTS and succeeding in SONORANTS:
            word.jamo = word.jamo[:h_location] + word.jamo[h_location + 1:]
    return word.jamo


def simplify_coda(input_word: Word, word_final: bool = False) -> Word:
    def simplify(jamo: str, loc: int) -> str:
        # coda cluster simplification

        list_jamo = list(jamo)
        before = ''.join(list_jamo[:loc + 1])
        double_coda = ''.join(list_jamo[loc + 1:loc + 3])
        after = ''.join(list_jamo[loc + 3:])

        converted = CT_double_codas.apply(text=double_coda, find_in='_separated')
        return before + converted + after

    while True:
        double_coda_loc = get_substring_ind(input_word.cv, 'VCCC')  # get all CCC location
        if len(double_coda_loc) == 0:
            break  # if no, exit while-loop

        cc = double_coda_loc[0]  # work on the leftest CCC
        new_jamo = simplify(input_word.jamo, cc)
        input_word.jamo = new_jamo

    # additionally, simplify word-final consonant cluster
    final_CC = get_substring_ind(input_word.cv, 'CC$')
    if len(final_CC) > 0:
        cc = final_CC[0] - 1
        new_jamo = simplify(input_word.jamo, cc)
        input_word.jamo = new_jamo
    return input_word


def non_coronalize(input_word: Word) -> str:
    velars = list('ㄱㅋㄲ')
    bilabials = list('ㅂㅍㅃㅁ')
    non_velar_nasals = list('ㅁㄴ')

    res = list(input_word.jamo)
    for i, jamo in enumerate(input_word.jamo[:-1]):
        if i == 0 or jamo not in non_velar_nasals:
            continue
        succeeding = input_word.jamo[i + 1]
        if succeeding in velars:
            res[i] = 'ㅇ'
        elif succeeding in bilabials:
            res[i] = 'ㅁ'
    return ''.join(res)


def inter_v(symbols: List[str]) -> List[str]:
    voicing_table = {
        'p': 'b',
        't': 'd',
        'k': 'ɡ',
        'tɕ': 'dʑ'
    }
    ipa_sonorants = [transcribe(s, str_return=True) for s in SONORANTS]

    res = list(symbols)

    for index, symbol in enumerate(symbols[:-1]):
        if index == 0 or symbol not in voicing_table.keys():
            continue
        preceding = symbols[index - 1]
        succeeding = symbols[index + 1]

        if preceding in ipa_sonorants:
            if succeeding in ipa_sonorants:
                res[index] = voicing_table.get(symbol, symbol)
            elif succeeding == 'ɕ':
                res[index] = voicing_table.get(symbol, symbol)
                res[index + 1] = 'ʑ'

    return res


def alternate_lr(symbols: List[str]) -> List[str]:
    ipa_vowels = [transcribe(v, str_return=True) for v in VOWELS]

    res = list(symbols)

    l_locs = [index for index, value in enumerate(symbols) if value == 'l']

    for l_loc in reversed(l_locs):
        if l_loc == 0 or l_loc == (len(symbols) - 1):
            continue

        preceding = symbols[l_loc - 1]
        succeeding = symbols[l_loc + 1]
        if preceding in ipa_vowels and succeeding in ipa_vowels:
            res[l_loc] = 'ɾ'

    return res


def apply_rules(word: Word, rules_to_apply: str = 'pastcnhovr') -> Word:
    # 规则的种类和顺序
    # (P)alatalization: 구개음화 (맏이 -> 마지)
    # (A)spiration: 격음화 (북한 -> 부칸)
    # a(S)similation: 음운동화
    # (T)ensification: 표준발음법 제23항(예외없는 경음화) 적용
    # (C)omplex coda simplification: 자음군단순화 (닭도 -> 닭도, 닭 -> 닭)
    # coda (N)eutralization: 음절말 장애음 중화 (빛/빚/빗 -> 빝)
    # intersonorant (H)-deletion: 공명음 사이 'ㅎ' 삭제
    # intersonorant Obstruent (V)oicing: 공명음 사이 장애음 유성음화

    # apply palatalization
    if 'p' in rules_to_apply and ('ㄷㅣ' in word.jamo or 'ㅌㅣ' in word.jamo):
        word.jamo = palatalize(word)

    # apply aspiration
    if 'a' in rules_to_apply and 'ㅎ' in word.jamo:
        word.jamo = aspirate(word)

    # apply place assimilation
    if 's' in rules_to_apply:
        word.jamo = assimilate(word)

    # apply post-obstruent tensification
    if 't' in rules_to_apply and any(jm in word.jamo for jm in OBSTRUENTS):
        word.jamo = pot(word)

    # apply complex coda simplification
    if 'c' in rules_to_apply:
        word = simplify_coda(word)

    # apply coda neutralization
    if 'n' in rules_to_apply:
        word.jamo = neutralize(word)

    # apply intersonorant H-deletion
    if 'h' in rules_to_apply and 'ㅎ' in word.jamo[1:-1]:
        word.jamo = delete_h(word)

    # apply (optional) non-coronalization
    if 'o' in rules_to_apply:
        word.jamo = non_coronalize(word)

    return word


def apply_phonetics(ipa_symbols: List[str], rules_to_apply: str) -> List[str]:
    if 'v' in rules_to_apply:
        ipa_symbols = inter_v(ipa_symbols)
    if 'r' in rules_to_apply and 'l' in ipa_symbols:
        ipa_symbols = alternate_lr(ipa_symbols)
    return ipa_symbols


# ----------------------------
# IPA to Pinyin Conversion
# ----------------------------

def ipa_to_pinyin(ipa: str) -> str:
    ipa_to_pinyin_dict = {
        # Consonants
        'p': 'b', 'pʰ': 'p', 'm': 'm', 'f': 'f',
        't': 'd', 'tʰ': 't', 'n': 'n', 'l': 'l',
        'k': 'g', 'kʰ': 'k', 'x': 'h', 'h': 'h', 'ɣ': 'e', 'χ': 'h', 'ʁ': 'ʁ', 'ħ': 'haʰoʰ', 'ʕ': 'haʰo', 'ɦ': 'aʰ',
        'tɕ': 'j', 'tɕʰ': 'q', 'ɕ': 'x', 't͡ɕ': 'j', 't͡ɕʰ': 'q',
        'tʂ': 'zh', 'tʂʰ': 'ch', 'ʂ': 'sh', 'ɻ': 'r', 'ʐ': 'r', 't͡s': 'z', 't͡sʰ': 'c', 'ʈ͡ʂ': 'zh', 'ʈ͡ʂʰ': 'ch',
        'ts': 'z', 'tsʰ': 'c', 's': 's', 'd͡z': 'zi', 'dz': 'zi',
        'ŋ': 'ng', 'ɲ': 'ni', 'ɲ̟': 'ni',
        'ʔ': 'ʔ',
        'ɉ': 'i',
        'w': 'u', 'ɥ': 'ü',
        'j': 'i', 'ç': 'xi', 'd͡ʑ': 'ji', 'dʑ': 'ji',

        # Syllabic Consonants
        'm̩': 'm', 'm̥': 'hm',
        'n̩': 'n', 'ŋ̍': 'ng', 'ŋ̊': 'hng',
        'ɹ̩': 'i', 'ɻ̩': 'ri',

        # Vowels
        'i': 'i', 'u': 'u', 'y': 'ü', 'u˞': 'ur',
        'ai': 'a', 'ä': 'a', 'ɑ': 'ao', 'e̞': 'ie', 'ə': 'en', 'a̠': 'a',
        'o': 'o', 'ɔ': 'ao', 'o̞': 'o', 'o̞˞': 'or',
        'ɤ': 'e', 'ɛ': 'i', 'e': 'ie', 'œ': 'ue', 'o̜': 'o',
        'ɵ': 'ou', 'ʊ': 'ong', 'ʊ̃˞': 'ongr', 'ɤ˞': 'e', 'ɤ̞˞': 'eng', 'ɤ˞˞': 'er',
        'ɚ': 'r', 'ɐ': 'i', 'ɚ̃': 'ngr', 'ʌ̹': 'ao',
        'i̞': 'ie',

        # Diphthongs and Triphthongs
        'ja': 'ia', 'wa': 'ua',
        'jo': 'io', 'wo': 'uo',
        'jɛ': 'ie', 'ɥɛ': 'üe',
        'aɪ': 'ai', 'waɪ': 'uai', 'ai̯': 'ai',
        'eɪ': 'ei', 'weɪ': 'ui', 'ei̯': 'ei',
        'ɑʊ': 'ao', 'jɑʊ': 'iao', 'ɑu̯': 'ao', 'ɑu̯˞': 'aor',
        'oʊ': 'ou', 'joʊ': 'iu', 'ou̯': 'iu', 'ou̯˞': 'our',

        # R-colored vowels and combinations
        'äɚ̯': 'r', 'ä̃ɚ̯̃': 'angr', 'ɐɚ̯': 'yanr',

        'an': 'an', 'jɛn': 'ian', 'wan': 'uan', 'ɥæn': 'üan',
        'ən': 'en', 'in': 'in', 'wən': 'un', 'yn': 'ün',
        'ɑŋ': 'ang', 'jɑŋ': 'iang', 'wɑŋ': 'uang',
        'ɤŋ': 'eng', 'iŋ': 'ing', 'wɤŋ': 'ueng',
        'ʊŋ': 'ong', 'jʊŋ': 'iong',
        'ɚ̃': 'a',

        # Tones
        '˥˥': '55', '˧˥': '35', '˨˩˦': '214', '˨˩˩': '211',
        '˩˦': '14', '˥˩': '51', '˥˧': '53',
        '˨˩': '21', '˧˩': '31', '˦˩': '41', '˩˩': '11', '˨˥': '25',
        '˧': '33', '˩˧': '13', '˨˧': '23', '˨': '22',

        # Neutral Tone
        'k˥': '5', 'k˧': '3', 'k˨': '2', '˥': '55',
    }

    # Sort the keys by length in descending order to match longer patterns first
    sorted_ipa_symbols = sorted(ipa_to_pinyin_dict.keys(), key=lambda x: len(x), reverse=True)
    # Create a regex pattern to match any of the IPA symbols
    pattern = '|'.join(re.escape(symbol) for symbol in sorted_ipa_symbols)
    ipa_regex = re.compile(pattern)

    def replace_match(match):
        return ipa_to_pinyin_dict.get(match.group(0), match.group(0))

    return ipa_regex.sub(replace_match, ipa)


# ----------------------------
# Worker Functions
# ----------------------------

def transcription_convention(convention: str, tables_dir: Path) -> ConversionTable:
    # supported transcription conventions: ipa, yale, park
    convention = convention.lower()
    if convention not in ['ipa', 'yale', 'park']:
        raise ValueError(f"您的输入 {convention} 不被支持。")
    return ConversionTable(convention, tables_dir)


def sanitize(word: str, tables_dir: Path) -> str:
    """
    converts all hanja 漢字 letters to hangul
    and also remove any space in the middle of the word
    """
    if len(word) < 1:  # if empty input, no sanitize
        return word

    word = word.replace(' ', '')

    hanja_idx = [match.start() for match in re.finditer(r'\p{Han}', word)]
    if len(hanja_idx) == 0:  # if no hanja, no sanitize
        return word

    r = hanja_cleaner(word, hanja_idx, tables_dir)
    return r


def convert(hangul: str,
            rules_to_apply: str = 'pastcnhovr',
            convention: str = 'ipa',
            sep: str = '',
            tables_dir: Path = Path('tables')) -> Dict[str, str]:
    # the main function for IPA and Pinyin conversion

    if len(hangul) < 1:  # if no content, then return no content
        return {"ipa": "", "pinyin": ""}

    # prepare
    rules_to_apply = rules_to_apply.lower()
    CT_convention = transcription_convention(convention, tables_dir)
    hangul = sanitize(hangul, tables_dir)

    word = Word(hangul=hangul, tables_dir=tables_dir)

    # resolve word-final consonant clusters right off the bat
    simplify_coda(word)

    # apply rules
    word = apply_rules(word, rules_to_apply)

    # high mid/back vowel merger after bilabial (only for the Yale convention)
    if CT_convention.name == 'yale' and 'u' in rules_to_apply:
        bilabials = list("ㅂㅃㅍㅁ")
        applied = list(word.jamo)
        for i, jamo in enumerate(word.jamo[:-1]):
            if jamo in bilabials and word.jamo[i + 1] == "ㅜ":
                applied[i + 1] = "ㅡ"
        word.jamo = ''.join(applied)

    # convert to IPA or Yale
    transcribed = transcribe(word.jamo, CT_convention)

    # apply phonetic rules
    if CT_convention.name == 'ipa':
        transcribed = apply_phonetics(transcribed, rules_to_apply)

    ipa_result = sep.join(transcribed)

    # Convert IPA to Pinyin
    pinyin_result = ipa_to_pinyin(ipa_result)

    return {"ipa": ipa_result, "pinyin": pinyin_result}


def convert_many(long_content: str,
                 rules_to_apply: str = 'pastcnhovr',
                 convention: str = 'ipa',
                 sep: str = '',
                 tables_dir: Path = Path('tables'),
                 output_mode: str = 'both') -> Union[int, str]:
    """
    Convert many words from the decoded content.
    output_mode: 'ipa', 'pinyin', or 'both'
    """
    # decode uploaded file and create a wordlist to pass to convert()
    decoded = b64decode(long_content).decode('utf-8')
    decoded = decoded.replace('\r\n', '\n').replace('\r', '\n')  # normalize line endings
    decoded = decoded.replace('\n\n', '\n')  # keep single newlines

    res = []
    # Split the decoded content into lines
    lines = decoded.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        tokens = re.findall(r'\w+|[^\w\s]', line, re.UNICODE)
        annotated_line = []
        for token in tokens:
            if re.match(r'\w+', token, re.UNICODE):
                # It's a word, convert it
                converted_r = convert(hangul=token,
                                      rules_to_apply=rules_to_apply,
                                      convention=convention,
                                      sep=sep,
                                      tables_dir=tables_dir)
                if output_mode == 'ipa':
                    anno = f'\\anno{{{token}}}{{{converted_r["ipa"]}}}'
                    annotated_line.append(anno)
                elif output_mode == 'pinyin':
                    anno = f'\\anno{{{token}}}{{{converted_r["pinyin"]}}}'
                    annotated_line.append(anno)
                elif output_mode == 'both':
                    anno_ipa = f'\\anno{{{token}}}{{{converted_r["ipa"]}}}'
                    anno_pinyin = f'\\anno{{{token}}}{{{converted_r["pinyin"]}}}'
                    annotated_line.append(anno_ipa)
                    annotated_line.append(anno_pinyin)
                else:
                    # Default to both
                    anno_ipa = f'\\anno{{{token}}}{{{converted_r["ipa"]}}}'
                    anno_pinyin = f'\\anno{{{token}}}{{{converted_r["pinyin"]}}}'
                    annotated_line.append(anno_ipa)
                    annotated_line.append(anno_pinyin)
            else:
                # It's punctuation or space, retain as-is
                annotated_line.append(token)
        # Join the annotated tokens into a line
        res.append(' '.join(annotated_line))
    # Replace newline with double backslash
    final_output = '\\\\\n'.join(res)
    return final_output


# ----------------------------
# Initialization
# ----------------------------


def hangul2ipa(input_text: str) -> str:
    tables = Path(f'{os.path.dirname(os.path.dirname(__file__))}/thirdparty/ko_tables')
    initialize_conversion_tables(tables)
    ipa = []
    for h in input_text.split():
        converted_results = convert(hangul=h,
                                    rules_to_apply='pastcnhovr',
                                    convention="ipa",
                                    sep='',
                                    tables_dir=tables)
        ipa.append(converted_results["ipa"])
    return " ".join(ipa)


if __name__ == "__main__":
    input_text = "안녕하세요"
    print(hangul2ipa(input_text))
