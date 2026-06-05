import sys, glob, json5
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from types import SimpleNamespace
from forward_tts import ForwardTTS, ForwardTTSArgs

class FPWrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, token_ids):
        return self.m.inference(token_ids)["model_outputs"].transpose(1, 2)  # [B,80,T]

cfgp = sys.argv[1]; ck = sys.argv[2]; out = sys.argv[3]
cfg = json5.load(open(cfgp, encoding="utf-8"))
sd = torch.load(ck, map_location="cpu", weights_only=False)["model"]
ma = cfg.get("model_args", {}); emb = [k for k in sd if k.endswith("emb.weight")][0]; ma["num_chars"] = sd[emb].shape[0]
d = ForwardTTSArgs().__dict__; args = ForwardTTSArgs(**{k: v for k, v in ma.items() if k in d})
conf = SimpleNamespace(model_args=args, num_chars=args.num_chars, num_speakers=1,
                       use_speaker_embedding=False, use_d_vector_file=False, d_vector_dim=0)
model = ForwardTTS(conf); model.load_state_dict(sd, strict=False); model.eval()
w = FPWrap(model)
x = torch.randint(1, ma["num_chars"] - 1, (1, 30))
with torch.no_grad(): print("mel", tuple(w(x).shape))
torch.onnx.export(w, (x,), out, input_names=["token_ids"], output_names=["mel_spec"],
                  dynamic_axes={"token_ids": {0: "b", 1: "t"}, "mel_spec": {0: "b", 2: "frame"}},
                  opset_version=14, dynamo=False)
print("exported", out)
