import sys, glob, json5
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from types import SimpleNamespace
from forward_tts import ForwardTTS, ForwardTTSArgs

class FPWrap(torch.nn.Module):
    def __init__(self, m, multispk=False): super().__init__(); self.m = m; self.multispk = multispk
    def forward(self, token_ids, speaker=None):
        aux = {"d_vectors": None, "speaker_ids": speaker if self.multispk else None}
        return self.m.inference(token_ids, aux_input=aux)["model_outputs"].transpose(1, 2)  # [B,80,T]

cfgp = sys.argv[1]; ck = sys.argv[2]; out = sys.argv[3]
cfg = json5.load(open(cfgp, encoding="utf-8"))
sd = torch.load(ck, map_location="cpu", weights_only=False)["model"]
ma = cfg.get("model_args", {}); emb = [k for k in sd if k.endswith("emb.weight")][0]; ma["num_chars"] = sd[emb].shape[0]
d = ForwardTTSArgs().__dict__; args = ForwardTTSArgs(**{k: v for k, v in ma.items() if k in d})
nspk = sd["emb_g.weight"].shape[0] if "emb_g.weight" in sd else 1
multispk = nspk > 1
spk_mgr = SimpleNamespace(num_speakers=nspk) if multispk else None
conf = SimpleNamespace(model_args=args, num_chars=args.num_chars, num_speakers=nspk,
                       use_speaker_embedding=multispk, use_d_vector_file=False, d_vector_dim=0)
model = ForwardTTS(conf, None, None, spk_mgr); model.load_state_dict(sd, strict=False); model.eval()
w = FPWrap(model, multispk)
x = torch.randint(1, ma["num_chars"] - 1, (1, 30))
if multispk:
    sid = torch.zeros(1, dtype=torch.int32)
    eargs = (x, sid); names = ["token_ids", "speaker"]
    dyn = {"token_ids": {0: "b", 1: "t"}, "speaker": {0: "b"}, "mel_spec": {0: "b", 2: "frame"}}
else:
    eargs = (x,); names = ["token_ids"]
    dyn = {"token_ids": {0: "b", 1: "t"}, "mel_spec": {0: "b", 2: "frame"}}
with torch.no_grad(): print("mel", tuple(w(*eargs).shape), "| speakers:", nspk)
torch.onnx.export(w, eargs, out, input_names=names, output_names=["mel_spec"],
                  dynamic_axes=dyn, opset_version=14, dynamo=False)
print("exported", out)
