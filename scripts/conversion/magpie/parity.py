import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import checkpoint_path, out_dir
from nemo.collections.tts.models.magpietts import MagpieTTSModel
from nemo.collections.tts.parts.utils.tts_dataset_utils import chunk_text_for_inference
P=checkpoint_path()
m=MagpieTTSModel.restore_from(P, map_location='cpu'); m.eval()
KV=len(sys.argv)>3 and sys.argv[3]=='kv'
m.use_kv_cache_for_inference=KV
m.inference_parameters.temperature=0.01
m.inference_parameters.topk=1
TXT=sys.argv[1] if len(sys.argv)>1 else 'Hello, this is a test of the Magpie text to speech system.'
LANG=sys.argv[2] if len(sys.argv)>2 else 'en'
tok_name=m.cfg.language_to_tokenizer_mapping[LANG][0]
nt=m._get_normalized_text(TXT, LANG)
ct,cl,_=chunk_text_for_inference(text=nt,language=LANG,tokenizer_name=tok_name,text_tokenizer=m.tokenizer,eos_token_id=m.eos_id)
print('chunks',len(ct),'len',cl,'KV',KV)
tokens=ct[0].unsqueeze(0); tl=int(cl[0])
batch={'text':tokens,'text_lens':torch.tensor([tl]),'speaker_indices':0}
torch.manual_seed(1234)
with torch.no_grad():
    out=m.infer_batch(batch,use_cfg=True,use_local_transformer_for_inference=True)
tc=out.predicted_codes[0,:,:out.predicted_codes_lens[0]]
print('TORCH codes',tuple(tc.shape))
from onnx_infer import MagpieOnnx
mo=MagpieOnnx(m)
torch.manual_seed(1234)
import time; t0=time.time()
audio,oc=mo.generate(tokens[0].numpy(), tl, speaker_index=0, use_cfg=True, temperature=0.01, topk=1, use_kv_cache=KV)
el=time.time()-t0
oc=oc[0]
print('ONNX codes',tuple(oc.shape),'wall',round(el,2),'RTF',round(el/(audio.shape[-1]/22050),3))
L=min(tc.shape[1],oc.shape[1])
agree=(tc[:,:L]==oc[:,:L]).float().mean().item()
print('GREEDY AGREEMENT all codebooks:',round(agree*100,3),'%  len torch',tc.shape[1],'onnx',oc.shape[1])
for c in range(8):
    print(' cb',c,round((tc[c,:L]==oc[c,:L]).float().mean().item()*100,2))
import soundfile as sf
__import__('os').makedirs(out_dir()+'/samples',exist_ok=True); sf.write(out_dir()+'/samples/onnx_%s.wav'%LANG, audio[0], 22050)
