#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text helpers vendored from ``k2-fsa/OmniVoice`` (``omnivoice/models/omnivoice.py``
and ``omnivoice/utils/text.py``), with the torch dependency removed.

The prompt text of a cloning call is the reference transcription and the target text
joined into **one** string before tokenization, so the model reads the target as a
continuation of what the reference clip already said. Getting this join wrong (or
dropping the reference transcription, as the community ONNX reference script does)
breaks in-context cloning, so the upstream rules are reproduced verbatim here.
"""
import re
from typing import Callable, List, Optional

END_PUNCTUATION = ".?!,;:。？！，；："

NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)
"""Non-verbal tags OmniVoice understands inline, e.g. ``"I know [laughter] really"``."""


def combine_text(text: str, ref_text: Optional[str] = None) -> str:
    """Join a cloning reference transcription and the target text, then normalize.

    Mirrors upstream ``_combine_text``: newlines go, full-width parentheses become
    ASCII ones, runs of spaces collapse, and spaces touching a Chinese character are
    removed (Chinese is written without them, and a stray space changes the reading).
    """
    if ref_text:
        full_text = ref_text.strip() + " " + text.strip()
    else:
        full_text = text.strip()

    full_text = re.sub(r"[\r\n]+", "", full_text)
    full_text = full_text.replace("（", "(").replace("）", ")")
    full_text = re.sub(r"[ \t]+", " ", full_text)
    chinese_range = r"[一-鿿]"
    pattern = rf"(?<={chinese_range})\s+|\s+(?={chinese_range})"
    full_text = re.sub(pattern, "", full_text)
    return full_text


def tokenize_with_nonverbal_tags(text: str, encode: Callable[[str], List[int]]) -> List[int]:
    """Tokenize ``text``, encoding every non-verbal tag on its own.

    A tag such as ``[laughter]`` must land on the same ids whatever surrounds it. A
    subword tokenizer merges across the bracket when the neighbouring text is Chinese
    but not when it is English, so the tag is cut out and encoded alone.

    ``encode`` maps a string to token ids **without** special tokens.
    """
    parts: List[List[int]] = []
    last_end = 0
    for m in NONVERBAL_PATTERN.finditer(text):
        if m.start() > last_end:
            ids = encode(text[last_end:m.start()])
            if ids:
                parts.append(ids)
        tag_ids = encode(m.group())
        if tag_ids:
            parts.append(tag_ids)
        last_end = m.end()
    if last_end < len(text):
        ids = encode(text[last_end:])
        if ids:
            parts.append(ids)

    if not parts:
        return encode(text)
    combined: List[int] = []
    for p in parts:
        combined.extend(p)
    return combined


def add_punctuation(text: str) -> str:
    """End a cloning reference transcription with punctuation if it has none."""
    text = text.strip()
    if not text:
        return text
    if text[-1] not in END_PUNCTUATION:
        is_chinese = any("一" <= char <= "鿿" for char in text)
        text += "。" if is_chinese else "."
    return text
