import sys, torch, numpy as np
from _paths import checkpoint_path, out_dir
THRESHOLD=1e-3
from nemo.collections.tts.models.magpietts import MagpieTTSModel
P=checkpoint_path()
m=MagpieTTSModel.restore_from(P, map_location='cpu'); m.eval()

class Enc(torch.nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s, text, text_mask):
        te=s.m.text_embedding(text).transpose(1,2) if False else s.m.embed_text(text, text_mask)
        return s.m.encoder(te, text_mask)['output']

import inspect
print(inspect.getsource(m.embed_text))
w=Enc(m).eval()
T=23
text=torch.randint(0,50,(1,T)).long()
mask=torch.ones(1,T)
with torch.no_grad(): ref=w(text,mask)
print('enc out', ref.shape)
out=out_dir()+'/text_encoder.onnx'
torch.onnx.export(w,(text,mask),out,input_names=['text','text_mask'],output_names=['encoder_out'],
  dynamic_axes={'text':{0:'B',1:'T'},'text_mask':{0:'B',1:'T'},'encoder_out':{0:'B',1:'T'}},opset_version=18)
import onnxruntime as ort
s=ort.InferenceSession(out,providers=['CPUExecutionProvider'])
o=s.run(None,{'text':text.numpy(),'text_mask':mask.numpy()})[0]
d1=np.abs(o-ref.numpy()).max(); print('MAXDIFF',d1)
T2=41; t2=torch.randint(0,50,(1,T2)).long(); m2=torch.ones(1,T2)
with torch.no_grad(): r2=w(t2,m2)
o2=s.run(None,{'text':t2.numpy(),'text_mask':m2.numpy()})[0]
d2=np.abs(o2-r2.numpy()).max(); print('MAXDIFF_dynT',d2)
if max(d1,d2)>THRESHOLD:
    print(f'GATE FAIL: encoder maxdiff {max(d1,d2)} > {THRESHOLD}', file=sys.stderr); sys.exit(1)
print('GATE PASS: encoder')
