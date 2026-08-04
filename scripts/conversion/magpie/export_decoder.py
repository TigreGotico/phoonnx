import os, sys
import torch, numpy as np
from _paths import checkpoint_path, out_dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dec_module import DecoderStep, CrossKV
from nemo.collections.tts.models.magpietts import MagpieTTSModel
P=checkpoint_path()
m=MagpieTTSModel.restore_from(P, map_location='cpu'); m.eval()
O=out_dir()+'/'
torch.manual_seed(0)
B,Tt=2,37
cond=torch.randn(B,Tt,768)*0.1; cond_mask=torch.ones(B,Tt)
ck=CrossKV(m).eval()
with torch.no_grad(): rk,rv=ck(cond)
torch.onnx.export(ck,(cond,),O+'cross_kv.onnx',input_names=['cond'],output_names=['cross_k','cross_v'],
  dynamic_axes={'cond':{0:'B',1:'Tt'},'cross_k':{1:'B',2:'Tt'},'cross_v':{1:'B',2:'Tt'}},opset_version=18)
import onnxruntime as ort
s=ort.InferenceSession(O+'cross_kv.onnx',providers=['CPUExecutionProvider'])
ok,ov=s.run(None,{'cond':cond.numpy()})
print('CROSSKV maxdiff',np.abs(ok-rk.numpy()).max(),np.abs(ov-rv.numpy()).max())

ds=DecoderStep(m).eval()
T=5; Tp=3
x=torch.randn(B,T,768)*0.1
pos=torch.arange(Tp,Tp+T)
sk=torch.randn(12,B,Tp,12,64)*0.1; sv=torch.randn(12,B,Tp,12,64)*0.1
prior=torch.rand(2,1,Tt)
args=(x,pos,sk,sv,rk,rv,cond_mask,prior)
with torch.no_grad(): ref=ds(*args)
names=['x','pos','self_k','self_v','cross_k','cross_v','cond_mask','attn_prior']
outs=['logits','dec_out','new_self_k','new_self_v','cross_attn_probs']
dyn={'x':{0:'B',1:'T'},'pos':{0:'T'},'self_k':{1:'B',2:'Tp'},'self_v':{1:'B',2:'Tp'},
 'cross_k':{1:'B',2:'Tt'},'cross_v':{1:'B',2:'Tt'},'cond_mask':{0:'B',1:'Tt'},'attn_prior':{0:'Bp',2:'Tt'},
 'logits':{0:'B',1:'T'},'dec_out':{0:'B',1:'T'},'new_self_k':{1:'B',2:'Tn'},'new_self_v':{1:'B',2:'Tn'},
 'cross_attn_probs':{1:'B',3:'T',4:'Tt'}}
torch.onnx.export(ds,args,O+'decoder_step.onnx',input_names=names,output_names=outs,dynamic_axes=dyn,opset_version=18)
s2=ort.InferenceSession(O+'decoder_step.onnx',providers=['CPUExecutionProvider'])
feed=dict(zip(names,[a.numpy() for a in args]))
o=s2.run(None,feed)
for n,a,b in zip(outs,o,ref): print('DEC',n,'maxdiff',np.abs(a-b.numpy()).max())
# dynamic shapes recheck: prefill-like T=218 Tp=0
T2=218; Tt2=41
cond2=torch.randn(B,Tt2,768)*0.1; cm2=torch.ones(B,Tt2)
with torch.no_grad(): k2,v2=ck(cond2)
a2=(torch.randn(B,T2,768)*0.1, torch.arange(T2), torch.zeros(12,B,0,12,64), torch.zeros(12,B,0,12,64), k2,v2,cm2, torch.ones(2,1,Tt2))
with torch.no_grad(): r2=ds(*a2)
o2=s2.run(None,dict(zip(names,[a.numpy() for a in a2])))
print('DEC dyn logits maxdiff',np.abs(o2[0]-r2[0].numpy()).max(), o2[0].shape)
