import wave, struct, math
import scriptconv.diacritics as scd
from phoonnx.voice import TTSVoice
from phoonnx.config import SynthesisConfig

# spy on diacritize to PROVE the diacritization step runs inside the synth path
_orig = scd.diacritize
_calls = []
def _spy(text, lang="und", *a, **k):
    out = _orig(text, lang, *a, **k)
    _calls.append((lang, text, out))
    return out
scd.diacritize = _spy

TEXT = "ذهب محمد إلى المدرسة في الصباح"   # undiacritized Arabic

def wav_metrics(path):
    w = wave.open(path, "rb"); n=w.getnframes(); sr=w.getframerate()
    data = struct.unpack("<%dh" % n, w.readframes(n)); w.close()
    peak = max(abs(x) for x in data) if data else 0
    rms = int(math.sqrt(sum(x*x for x in data)/len(data))) if data else 0
    return sr, n, round(n/sr,3), peak, rms

for name, base in [("miro","arvoices/miro_ar"), ("dii","arvoices/dii_ar")]:
    _calls.clear()
    v = TTSVoice.load(base+".onnx", config_path=base+".json")
    out = f"arvoices/{name}_out.wav"
    with wave.open(out,"wb") as wf:
        v.synthesize_wav(TEXT, wf, syn_config=SynthesisConfig(add_diacritics=True))
    sr,n,dur,peak,rms = wav_metrics(out)
    diac = _calls[0] if _calls else None
    print(f"\n=== {name} ({base}) ===")
    print(f"  lang={v.config.lang_code} phoneme_type={v.config.phoneme_type} add_diacritics(cfg)={v.config.add_diacritics}")
    print(f"  diacritize called: {len(_calls)}x")
    if diac: print(f"    in : {diac[1]}\n    out: {diac[2]}\n    changed: {diac[1]!=diac[2]}")
    print(f"  WAV: sr={sr} samples={n} dur={dur}s peak={peak} rms={rms}  real_audio={dur>0.3 and peak>500}")
