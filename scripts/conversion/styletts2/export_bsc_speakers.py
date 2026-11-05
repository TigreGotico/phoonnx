"""BSC-LT multispeaker StyleTTS2  ->  per-speaker style assets.

``export_bsc.py`` exports the two BSC-LT multispeaker checkpoints
(``BSC-LT/styletts2-spanish-multispeaker``, ``BSC-LT/styletts2-catalan-multispeaker``)
as **zero-shot cloning** voices: ``model.onnx`` conditions on a 256-d style vector and
``style_encoder.onnx`` produces one from a reference clip. That makes them usable only
when the caller has a reference clip on hand.

This script closes that gap. It runs ``style_encoder.onnx`` over reference audio drawn
from each model's own **training corpus** and writes one ``<speaker>.bin`` per speaker,
so the checkpoints also ship as ordinary named preset voices. Same asset shape as the
HiTZ Basque voices: a flat ``float32`` blob that reshapes to ``[1, 256]``, referenced
from the voice index as ``style_url`` and loaded by the StyleTTS2 adapter through
``engine_params["style_path"]``.

The 256 values are ``ref_p`` ++ ``ref_s`` -- the acoustic style (decoder) followed by
the prosodic style (predictor); the adapter splits them again at 128.

Speakers and reference corpora
------------------------------
* **ca** -- ``projecte-aina/festcat_trimmed_denoised``, 11 named speakers
  (``bet eli eva jan mar ona pau pep pol teo uri``). The second Catalan training
  corpus (``openslr-slr69-ca``) has anonymous ids only, so it contributes no voices.
* **es** -- ``ylacombe/cml-tts`` (spanish), the 6 speakers that carry essentially the
  whole corpus. CML-TTS speaker ids are numeric, so the voices are named after them.

A style vector is the **mean over N reference clips** of that speaker. Averaging cancels
the per-utterance prosody of any single clip and leaves the speaker identity, which is
what a preset voice wants; ``--per-clip`` writes the individual clip vectors instead so
the choice can be re-measured.

Usage::

    python export_bsc_speakers.py ca /path/to/style_encoder.onnx /refs/ca /out/bsc-ca-styletts2
    python export_bsc_speakers.py es /path/to/style_encoder.onnx /refs/es /out/bsc-es-styletts2

``refs/<lang>/<speaker>_<n>.wav`` is the expected reference layout (see
``fetch_bsc_speaker_refs.py``).
"""
import argparse
import os
import re
from collections import defaultdict

import numpy as np

SR = 24000
STYLE_DIM = 128

# speakers we publish, per language, in the order they appear in the voice index
SPEAKERS = {
    "ca": ["bet", "eli", "eva", "jan", "mar", "ona", "pau", "pep", "pol", "teo", "uri"],
    "es": ["3946", "8882", "9972", "10246", "11797", "12367"],
}


def load_wav(path: str) -> np.ndarray:
    """Mono float32 at 24 kHz -- the sample rate the style encoder's mel front-end bakes in."""
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        # polyphase resampling; the mel front-end is sensitive to the sample rate,
        # a mismatch silently shifts every filterbank centre.
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(sr, SR)
        y = resample_poly(y, SR // g, sr // g).astype(np.float32)
    return y.astype(np.float32)


def encode(session, wav: np.ndarray) -> np.ndarray:
    """reference waveform -> 256-d style (ref_p ++ ref_s)."""
    ref_p, ref_s = session.run(None, {"waveform": wav[None, :].astype(np.float32)})
    return np.concatenate([np.asarray(ref_p).reshape(-1), np.asarray(ref_s).reshape(-1)])


def group_refs(ref_dir: str):
    """``<speaker>_<n>.wav`` -> {speaker: [path, ...]}."""
    out = defaultdict(list)
    for name in sorted(os.listdir(ref_dir)):
        m = re.fullmatch(r"(.+)_(\d+)\.wav", name)
        if m:
            out[m.group(1)].append(os.path.join(ref_dir, name))
    return out


def export(lang: str, encoder: str, ref_dir: str, out: str, per_clip: bool = False) -> None:
    import onnxruntime as ort

    os.makedirs(out, exist_ok=True)
    sess = ort.InferenceSession(encoder, providers=["CPUExecutionProvider"])
    refs = group_refs(ref_dir)

    missing = [s for s in SPEAKERS[lang] if s not in refs]
    assert not missing, f"no reference clips for {missing} under {ref_dir}"

    for speaker in SPEAKERS[lang]:
        vecs = np.stack([encode(sess, load_wav(p)) for p in refs[speaker]])
        if per_clip:
            for i, v in enumerate(vecs):
                v.astype(np.float32).tofile(f"{out}/{speaker}.clip{i}.bin")
        style = vecs.mean(axis=0).astype(np.float32)
        assert style.shape == (2 * STYLE_DIM,), style.shape
        assert np.isfinite(style).all(), f"{speaker}: non-finite style"
        style.tofile(f"{out}/{speaker}.bin")
        # spread across the reference clips: high spread means the clips disagree
        # about who this speaker is (bad reference selection), not just prosody.
        spread = float(np.mean(np.std(vecs, axis=0)) / (np.mean(np.abs(style)) + 1e-9))
        print(f"{lang}/{speaker}: {len(vecs)} clips  |style|={np.abs(style).mean():.4f} "
              f"spread={spread:.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lang", choices=sorted(SPEAKERS))
    ap.add_argument("encoder", help="style_encoder.onnx from the exported voice")
    ap.add_argument("ref_dir", help="directory of <speaker>_<n>.wav reference clips")
    ap.add_argument("out")
    ap.add_argument("--per-clip", action="store_true",
                    help="also write the per-clip vectors, for re-measuring the averaging")
    a = ap.parse_args()
    export(a.lang, a.encoder, a.ref_dir, a.out, a.per_clip)
