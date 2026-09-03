#!/usr/bin/env python3
"""Prove the double-BOS documented in ``phoonnx.engines.orpheus``'s module docstring.

    python probe_prompt.py --tokenizer ./tokenizer.json

Needs only the checkpoint's tokenizer — no weights, no network beyond the one small
file. It builds the served prompt two ways and prints the token-id diff:

1. **Literal source reading.** ``OrpheusModel._format_prompt`` (Canopy Labs' own vLLM
   integration) does::

       input_ids = tokenizer(f"{voice}: {text}", return_tensors="pt").input_ids
       all_input_ids = [start_token, *input_ids[0], end_tokens...]

   The HF tokenizer prepends ``<|begin_of_text|>`` (128000) by default, so
   ``all_input_ids`` already carries one BOS before ``start_token`` (128259,
   ``<custom_token_3>``) is prepended.

2. **What the server actually sends.** ``_format_prompt`` then does
   ``prompt_string = tokenizer.decode(all_input_ids)`` and hands that *string* to
   vLLM's ``LLM.generate``, which re-tokenizes it through its own tokenizer call with
   ``add_special_tokens=True`` — the default. That re-tokenization prepends a
   **second** BOS, because the string still starts with the literal
   ``<|begin_of_text|>`` text and a special-token-aware tokenizer adds its own BOS on
   top of a fresh encode regardless of what's already in the string.

The two id sequences differ by exactly one leading 128000. That is the defect a
straightforward "read the source, drop the tokenizer's own BOS" port would produce:
it would serve the model a prompt it was never trained to see. This adapter's
:meth:`OrpheusAdapter.build_prompt_ids` reproduces sequence 2, matching what the
checkpoint is actually served in production, not what a literal reading of
``_format_prompt`` alone implies.
"""
from __future__ import annotations

import argparse

BOS = 128000                 # <|begin_of_text|>
START_OF_HUMAN = 128259      # <custom_token_3>
END_OF_HUMAN = 128260        # <custom_token_4>
START_OF_AI = 128261         # <custom_token_5>
START_OF_SPEECH = 128257     # <custom_token_1>
EOT = 128009                 # <|eot_id|>


def literal_all_input_ids(tokenizer, voice: str, text: str) -> list[int]:
    """``OrpheusModel._format_prompt``, read straight off the source.

    The HF tokenizer call implicitly adds a leading BOS (``add_special_tokens`` is
    on by default for a plain ``tokenizer(text)`` call); ``start_token`` is then
    prepended on top of that.
    """
    body_ids = tokenizer.encode(f"{voice}: {text}", add_special_tokens=True).ids
    return [START_OF_HUMAN, *body_ids, EOT, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]


def served_ids(tokenizer, all_input_ids: list[int]) -> list[int]:
    """What vLLM actually receives: decode ``all_input_ids`` back to a string, then
    re-tokenize it with ``add_special_tokens=True`` — vLLM's default for a raw-string
    ``generate`` call.

    ``skip_special_tokens=False`` matters here and is not a detail to drop: upstream's
    ``_format_prompt`` calls the HF slow-tokenizer ``.decode()``, which by default keeps
    special-token text in the string (``skip_special_tokens`` defaults to ``False`` on
    the plain call upstream uses). Decoding with the opposite default silently drops the
    literal ``<|begin_of_text|>`` text and the double BOS never reappears on re-encode —
    which is exactly the wrong-by-one-flag mistake that would hide this bug.
    """
    prompt_string = tokenizer.decode(all_input_ids, skip_special_tokens=False)
    return tokenizer.encode(prompt_string, add_special_tokens=True).ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="./tokenizer.json",
                     help="the checkpoint's tokenizer.json (mirrored at "
                          "OpenVoiceOS/phoonnx-orpheus/orpheus-3b-en-onnx/tokenizer.json)")
    ap.add_argument("--voice", default="tara")
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)

    literal = literal_all_input_ids(tok, args.voice, args.text)
    served = served_ids(tok, literal)

    print("literal all_input_ids          :", literal[:8], "...")
    print("vLLM re-encode(add_special=T)  :", served[:8], "...")
    print("literal[0], served[0], served[1]:", literal[0], served[0], served[1])
    print("literal starts with single BOS :", literal[0] == START_OF_HUMAN)
    print("served starts with double BOS  :", served[0] == BOS and served[1] == START_OF_HUMAN
          and served[2] == BOS)
    print("MATCHES?", literal == served)

    # this adapter's own construction, straight from phoonnx.engines.orpheus
    adapter_prompt = [BOS, START_OF_HUMAN, *literal[1:-4], EOT, END_OF_HUMAN,
                       START_OF_AI, START_OF_SPEECH]
    print("adapter build_prompt_ids matches served form:", adapter_prompt == served)

    if served[0] != BOS or served[1] != START_OF_HUMAN or served[2] != BOS:
        raise SystemExit("double-BOS finding did NOT reproduce against this tokenizer")
    if literal == served:
        raise SystemExit("literal and served sequences unexpectedly match — "
                          "the double-BOS claim did not reproduce")
    print("\ndouble-BOS finding: CONFIRMED")


if __name__ == "__main__":
    main()
