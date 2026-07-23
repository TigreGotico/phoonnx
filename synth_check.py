import glob, io, math, os, struct, traceback, wave
from phoonnx.voice import TTSVoice
from phoonnx.config import SynthesisConfig

V = "/home/miro/.cache/phoonnx/voices"
TEXT = {"gl":"Ola, isto é unha proba de voz.","ar":"ذهب محمد إلى المدرسة.",
        "en":"Hello, this is a voice test.","ca":"Hola, això és una prova de veu.",
        "eu":"Kaixo, hau ahots proba bat da."}
CASES = [
    (f"{V}/proxectonos/celtia",                      "gl", "coqui/graphemes"),
    (f"{V}/piper/ar_JO-kareem-low",                  "ar", "piper/espeak-ipa (ar)"),
    (f"{V}/OpenVoiceOS/pipertts_en-US_miro",         "en", "piper/espeak-ipa (en)"),
    (f"{V}/OpenVoiceOS/phoonnx_eu-ES_dii_espeak",    "eu", "phoonnx/espeak-ipa (eu)"),
    (f"{V}/facebook/mms-tts-eng-English",            "en", "transformers/graphemes"),
    (f"{V}/OpenVoiceOS/matxa-cat-multispeaker-wavenext","ca","matcha (acoustic+vocoder)"),
    (f"{V}/mimic3/en_UK/apope_low",                  "en", "mimic3"),
]
def cfg_for(d):
    for pat in ("config.json","model.json","*.piper.json","*.json"):
        g = sorted(glob.glob(os.path.join(d,pat)))
        if g: return g[0]
    return None
def metrics(buf):
    w=wave.open(io.BytesIO(buf),"rb"); n=w.getnframes(); sr=w.getframerate()
    d=struct.unpack("<%dh"%n, w.readframes(n)); w.close()
    peak=max(abs(x) for x in d) if d else 0
    rms=int(math.sqrt(sum(x*x for x in d)/len(d))) if d else 0
    return sr,n,round(n/sr,2),peak,rms
for d,lang,label in CASES:
    name=d.replace(V+"/","")
    onnx=os.path.join(d,"model.onnx"); c=cfg_for(d)
    if not os.path.isfile(onnx): print(f"SKIP  {label:32s} {name}: no model.onnx"); continue
    try:
        v=TTSVoice.load(onnx, config_path=c)
        b=io.BytesIO()
        with wave.open(b,"wb") as wf:
            v.synthesize_wav(TEXT[lang], wf, syn_config=SynthesisConfig(normalize_audio=False))
        raw=b.getvalue()
        sr,n,dur,peak,rms=metrics(raw)
        ok = dur>0.3 and peak>500
        open(f"/tmp/synthout/{name.replace('/','_')}.wav","wb").write(raw)
        print(f"{'OK  ' if ok else 'BAD '} {label:32s} {name:45s} sr={sr} dur={dur}s peak={peak} rms={rms}")
    except Exception as e:
        print(f"FAIL {label:32s} {name:45s} {type(e).__name__}: {str(e)[:110]}")
