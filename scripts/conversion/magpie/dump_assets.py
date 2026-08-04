import json, os, shutil
import torch, numpy as np
from _paths import checkpoint_path, out_dir, extract_dir
from nemo.collections.tts.models.magpietts import MagpieTTSModel
P=checkpoint_path()
m=MagpieTTSModel.restore_from(P, map_location='cpu'); m.eval()
D=out_dir()+'/'
os.makedirs(D+'assets',exist_ok=True)
ctx=[]
for i in range(m.num_baked_speakers):
    e,l=m.get_baked_context_embeddings_batch(batch_size=1, speaker_indices=i)
    ctx.append(e[0].detach().numpy())
ctx=np.stack(ctx).astype(np.float32)
np.save(D+'assets/context_embeddings.npy', ctx)
print('context_embeddings', ctx.shape)
spk=json.load(open(__import__('glob').glob(extract_dir()+'/*speakers.json')[0]))
json.dump(spk, open(D+'assets/speakers.json','w'), indent=2)
print('speakers', spk)
ip=m.inference_parameters
cfgd={
 'model':'magpie_tts_multilingual_357m','source':'nvidia/magpie_tts_multilingual_357m',
 'sample_rate':int(m._codec_model.sample_rate),
 'num_audio_codebooks':int(m.num_audio_codebooks),
 'num_all_tokens_per_codebook':int(m.num_all_tokens_per_codebook),
 'codebook_size':int(m.codebook_size),
 'frame_stacking_factor':int(m.frame_stacking_factor),
 'audio_bos_id':int(m.audio_bos_id),'audio_eos_id':int(m.audio_eos_id),
 'decoder_n_layers':12,'decoder_d_model':768,'sa_n_heads':12,'sa_d_head':64,
 'xa_n_heads':1,'xa_d_head':128,
 'local_transformer_n_layers':2,'context_length':int(ctx.shape[1]),
 'apply_prior_to_layers':list(ip.apply_prior_to_layers),
 'estimate_alignment_from_layers':list(ip.estimate_alignment_from_layers),
 'transcript_decoder_layers':list(m.transcript_decoder_layers),
 'inference':{k:(list(v) if isinstance(v,(list,tuple)) else v) for k,v in vars(ip).items() if isinstance(v,(int,float,str,bool,list,tuple))},
 'language_to_tokenizer':{k:list(v) for k,v in m.cfg.language_to_tokenizer_mapping.items()},
}
json.dump(cfgd, open(D+'assets/magpie_config.json','w'), indent=2)
print('config keys',len(cfgd))
# tokenizer artifacts (dicts/heteronyms) shipped inside the .nemo
import glob
os.makedirs(D+'assets/tokenizer',exist_ok=True)
for f in glob.glob(extract_dir()+'/*'):
    b=os.path.basename(f)
    if b in ('model_config.yaml','model_weights.ckpt'): continue
    shutil.copy(f, D+'assets/tokenizer/'+b)
shutil.copy(extract_dir()+'/model_config.yaml', D+'assets/model_config.yaml')
# per-language token vocabularies
vocab={}
for name,tk in m.tokenizer.items() if isinstance(m.tokenizer,dict) else []:
    try: vocab[name]=getattr(tk,'tokens',None) and list(tk.tokens)
    except Exception as e: vocab[name]=None
json.dump({k:v for k,v in vocab.items() if v}, open(D+'assets/tokenizer_vocabs.json','w'), ensure_ascii=False, indent=1)
print('vocabs', {k:(len(v) if v else 0) for k,v in vocab.items()})
