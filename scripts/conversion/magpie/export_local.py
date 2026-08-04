import torch, numpy as np
import torch.nn.functional as F
from _paths import checkpoint_path, out_dir
from nemo.collections.tts.models.magpietts import MagpieTTSModel
P=checkpoint_path()
m=MagpieTTSModel.restore_from(P, map_location='cpu'); m.eval()
O=out_dir()+'/'
import onnxruntime as ort
NCB=m.num_audio_codebooks*m.frame_stacking_factor
print('NCB',NCB,'ntok',m.num_all_tokens_per_codebook)

class AudioEmbed(torch.nn.Module):
    '''codes (B,NCB) -> decoder input embedding (B,1,768)'''
    def __init__(s,m):
        super().__init__()
        s.register_buffer('W', torch.stack([e.weight for e in m.audio_embeddings]))  # (NCB,V,D)
        s.n=NCB
    def forward(s, codes):
        idx=codes.unsqueeze(-1).expand(-1,-1,s.W.shape[-1])       # (B,NCB,D)
        e=torch.gather(s.W.unsqueeze(0).expand(codes.shape[0],-1,-1,-1), 2, idx.unsqueeze(2)).squeeze(2)
        return (e.sum(1)/s.n).unsqueeze(1)

ae=AudioEmbed(m).eval()
codes=torch.randint(0,2000,(2,NCB))
with torch.no_grad():
    ref=ae(codes)
    r2,_=m.embed_audio_tokens(codes.reshape(2,m.frame_stacking_factor,m.num_audio_codebooks).permute(0,2,1), torch.full((2,),m.frame_stacking_factor))
print('AUDIOEMBED vs nemo maxdiff',(ref-r2).abs().max().item())
torch.onnx.export(ae,(codes,),O+'audio_embed.onnx',input_names=['codes'],output_names=['emb'],
  dynamic_axes={'codes':{0:'B'},'emb':{0:'B'}},opset_version=18)
s=ort.InferenceSession(O+'audio_embed.onnx',providers=['CPUExecutionProvider'])
print('AUDIOEMBED onnx maxdiff',np.abs(s.run(None,{'codes':codes.numpy()})[0]-ref.numpy()).max())

class LocalStep(torch.nn.Module):
    def __init__(s,m):
        super().__init__(); s.lt=m.local_transformer
        s.register_buffer('PW', torch.stack([p.weight for p in m.local_transformer_out_projections]))
        s.register_buffer('PB', torch.stack([p.bias for p in m.local_transformer_out_projections]))
    def forward(s, h, pos, cache_k, cache_v, cb):
        B,T,D=h.shape
        x=h+s.lt.position_embeddings(pos).unsqueeze(0)
        nk,nv=[],[]
        # T is always 1 for the local step: every cached key is valid, no causal mask needed
        for i,layer in enumerate(s.lt.layers):
            sa=layer.self_attention
            qkv=sa.qkv_net(layer.norm_self(x)).reshape(B,T,3,sa.n_heads,sa.d_head)
            q,k,v=[t.squeeze(2) for t in qkv.chunk(3,dim=2)]
            k=torch.cat([cache_k[i],k],1); v=torch.cat([cache_v[i],v],1); nk.append(k); nv.append(v)
            sc=torch.matmul(q.transpose(1,2),k.transpose(1,2).transpose(2,3))*sa.scale
            y=torch.matmul(F.softmax(sc,-1),v.transpose(1,2)).transpose(1,2).contiguous().view(B,T,-1)
            x=x+sa.o_net(y)
            x=x+layer.pos_ff(layer.norm_pos_ff(x),torch.ones(B,T))
        x=s.lt.norm_out(x).squeeze(1)
        logits=torch.matmul(x, s.PW[cb].t())+s.PB[cb]
        return logits, torch.stack(nk), torch.stack(nv)

ls=LocalStep(m).eval()
h=torch.randn(2,1,768)*0.1
ck=torch.randn(2,2,3,12,64)*0.1; cv=torch.randn(2,2,3,12,64)*0.1
args=(h,torch.arange(3,4),ck,cv,torch.tensor(3))
with torch.no_grad(): ref=ls(*args)
torch.onnx.export(ls,args,O+'local_step.onnx',
  input_names=['h','pos','cache_k','cache_v','cb'],output_names=['logits','new_k','new_v'],
  dynamic_axes={'h':{0:'B'},'pos':{},'cache_k':{1:'B',2:'Tp'},'cache_v':{1:'B',2:'Tp'},'cb':{}},opset_version=18)
s2=ort.InferenceSession(O+'local_step.onnx',providers=['CPUExecutionProvider'])
o=s2.run(None,{'h':h.numpy(),'pos':np.arange(3,4),'cache_k':ck.numpy(),'cache_v':cv.numpy(),'cb':np.array(3)})
print('LOCALSTEP onnx maxdiff',np.abs(o[0]-ref[0].numpy()).max())

class LTEmbed(torch.nn.Module):
    def __init__(s,m):
        super().__init__()
        s.register_buffer('W', torch.stack([e.weight for e in m.audio_embeddings]))
        s.aip=m.audio_in_projection; s.lip=m.local_transformer_in_projection
    def forward(s, tok, cb):
        e=s.W[cb][tok].unsqueeze(1)
        return s.lip(s.aip(e))
lte=LTEmbed(m).eval()
tok=torch.randint(0,2000,(2,))
with torch.no_grad(): ref3=lte(tok,torch.tensor(3))
torch.onnx.export(lte,(tok,torch.tensor(3)),O+'lt_embed.onnx',input_names=['tok','cb'],output_names=['emb'],
  dynamic_axes={'tok':{0:'B'},'emb':{0:'B'}},opset_version=18)
s3=ort.InferenceSession(O+'lt_embed.onnx',providers=['CPUExecutionProvider'])
print('LTEMBED onnx maxdiff',np.abs(s3.run(None,{'tok':tok.numpy(),'cb':np.array(3)})[0]-ref3.numpy()).max())
