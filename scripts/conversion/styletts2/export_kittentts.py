"""KittenML KittenTTS  ->  phoonnx StyleTTS2 adapter config + voice styles.

KittenTTS ships the exact StyleTTS2Adapter single-graph contract already:

    input_ids(int64) + style(1,256) + speed(1)  ->  waveform @ 24kHz

No new adapter is needed -- this script only builds the phoonnx-canonical
``config.json`` (vocab + inference defaults) and splits ``voices.npz`` into
one ``.bin`` style file per voice (the hitz/bsc precedent), so the existing
StyleTTS2Adapter can load them straight off engine_params.style_path.

Vocab: KittenTTS's ``kittentts/__init__.py`` builds its char->id map inline as
``{symbol: i for i, symbol in enumerate(list(SYMBOLS))}`` where ``SYMBOLS`` is a
literal string of punctuation + ASCII + IPA + suprasegmentals. This script
reproduces that *exact* literal (copied verbatim from the installed
``kittentts==0.1.3`` package source, not retyped) and builds the id map the
same way -- correct-by-construction, not hand-copied from a rendered table.

Usage:
    python export_kittentts.py <tier> <model.onnx> <voices.npz> <out_dir>
    # tier one of: nano-0.1, nano-0.2, mini-0.1
"""
import json
import sys
from pathlib import Path

import numpy as np

# Verbatim from kittentts==0.1.3 kittentts/__init__.py (KittenTTS.__init__):
#   self._word_index_dictionary = {symbol: i for i, symbol in enumerate(list(
#       '$;:,.!?¡¿—…"«»"" ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
#       'ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢ'
#       'ǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘\'̩\'ᵻ'))}
_KITTENTTS_SYMBOLS = (
    '$;:,.!?¡¿—…"«»"" ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    'ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢ'
    'ǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘\'̩\'ᵻ'
)

TIERS = {
    "nano-0.1": {"model_file": "kitten_tts_nano_v0_1.onnx", "repo": "KittenML/kitten-tts-nano-0.1"},
    "nano-0.2": {"model_file": "kitten_tts_nano_v0_2.onnx", "repo": "KittenML/kitten-tts-nano-0.2"},
    "mini-0.1": {"model_file": "kitten_tts_mini_v0_1.onnx", "repo": "KittenML/kitten-tts-mini-0.1"},
}

# 5000-sample trailing trim: NOT part of upstream kittentts==0.1.3 inference
# (kittentts/__init__.py's KittenTTS.generate() returns the raw graph output
# with no post-trim). Left here as a documented non-finding rather than a
# hardcoded no-op: if a future tier needs a trim, wire it through
# engine_params (e.g. ``trim_trailing_samples``) rather than hardcoding it in
# the adapter.
TRAILING_TRIM_SAMPLES = 0


def build_phoneme_id_map() -> dict:
    return {symbol: i for i, symbol in enumerate(list(_KITTENTTS_SYMBOLS))}


def build_config(tier: str) -> dict:
    phoneme_id_map = build_phoneme_id_map()
    return {
        "phoonnx_version": "1.0",
        "engine": "styletts2",
        "phoneme_type": "espeak",
        "alphabet": "ipa",
        "lang_code": "en-US",
        "audio": {"sample_rate": 24000},
        "num_symbols": len(phoneme_id_map),
        "num_speakers": 1,
        "num_langs": 1,
        "speaker_id_map": {},
        "lang_id_map": {},
        "phonemizer_model": None,
        "add_diacritics": False,
        "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
        "phoneme_id_map": phoneme_id_map,
        "kittentts_tier": tier,
        "engine_params": {"trim_trailing_samples": TRAILING_TRIM_SAMPLES},
    }


def split_voices(voices_npz_path: str, out_dir: Path) -> list:
    voices = np.load(voices_npz_path)
    names = []
    for name in voices.files:
        style = np.asarray(voices[name], dtype=np.float32).reshape(-1)
        assert style.shape[0] == 256, f"unexpected style dim for {name}: {style.shape}"
        style.tofile(out_dir / f"{name}.bin")
        names.append(name)
    return names


def main():
    tier, model_path, voices_path, out_dir = sys.argv[1:5]
    assert tier in TIERS, f"unknown tier {tier!r}, expected one of {list(TIERS)}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import shutil
    shutil.copy(model_path, out / "model.onnx")

    cfg = build_config(tier)
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

    voice_names = split_voices(voices_path, out)
    print(f"[{tier}] wrote model.onnx, config.json, {len(voice_names)} style .bin files -> {out}")
    print("voices:", voice_names)


if __name__ == "__main__":
    main()
