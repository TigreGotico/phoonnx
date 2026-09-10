import sys, torch, numpy as np
from _paths import checkpoint_path, out_dir
THRESHOLD=1e-2  # audio samples; looser than the logit-space graphs above
from nemo.collections.tts.models.magpietts import MagpieTTSModel
P=checkpoint_path()
m=MagpieTTSModel.restore_from(P, map_location='cpu'); m.eval()
codec=m._codec_model; codec.eval()

class CodecDec(torch.nn.Module):
    def __init__(self, c): super().__init__(); self.c=c
    def forward(self, tokens):
        n=torch.full((tokens.shape[0],), tokens.shape[2], dtype=torch.long)
        a,al=self.c.decode(tokens=tokens, tokens_len=n)
        return a

w=CodecDec(codec).eval()
T=40
toks=torch.randint(0,2000,(1,8,T)).long()
with torch.no_grad(): ref=w(toks)
print('ref audio', ref.shape)
out=out_dir()+'/codec_decoder.onnx'
torch.onnx.export(w, (toks,), out, input_names=['codes'], output_names=['audio'],
    dynamic_axes={'codes':{0:'B',2:'T'},'audio':{0:'B',1:'S'}}, opset_version=18, do_constant_folding=True)
print('exported')
import onnxruntime as ort
s=ort.InferenceSession(out, providers=['CPUExecutionProvider'])
o=s.run(None,{'codes':toks.numpy()})[0]
d=np.abs(o-ref.numpy()); d1=d.max(); print('MAXDIFF', d1, 'shape', o.shape)
T2=57; t2=torch.randint(0,2000,(1,8,T2)).long()
with torch.no_grad(): r2=w(t2)
o2=s.run(None,{'codes':t2.numpy()})[0]
d2=np.abs(o2-r2.numpy()).max()
print('MAXDIFF_dynT', d2, o2.shape)

if max(d1,d2)>THRESHOLD:
    print(f'GATE FAIL: codec maxdiff {max(d1,d2)} > {THRESHOLD}', file=sys.stderr); sys.exit(1)
print('GATE PASS: codec')
