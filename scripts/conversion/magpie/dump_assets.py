import json, os, shutil
import torch, numpy as np
from _paths import REPO, checkpoint_path, out_dir, extract_dir
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
spk_files=__import__('glob').glob(extract_dir()+'/*speakers.json')
# a fine-tune may ship baked speakers without the name map: name them by index
spk=json.load(open(spk_files[0])) if spk_files else {str(i):i for i in range(m.num_baked_speakers)}
json.dump(spk, open(D+'assets/speakers.json','w'), indent=2)
print('speakers', spk)
ip=m.inference_parameters
cfgd={
 'model':'magpie_tts_multilingual_357m','source':REPO,
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
 'language_to_tokenizer':{k:list(v) for k,v in m.cfg.get('language_to_tokenizer_mapping',{}).items()},
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

# the aggregated tokenizer, with the encoder kind of each sub-tokenizer
from transformers import ByT5Tokenizer
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import (
    ArabicCharsTokenizer, BaseCharsTokenizer)
from nemo.collections.common.tokenizers.text_to_speech.tokenizer_utils import (
    any_locale_text_preprocessing)
def preprocessing_of(func):
    if func is any_locale_text_preprocessing: return 'any_locale'
    if func('Ab’ É') == 'Ab’ É': return 'none'
    return None
def kind_of(tk):
    # anything phoonnx cannot encode exactly is recorded by class name, and refused
    if isinstance(tk, ByT5Tokenizer):
        return {'kind':'byt5'}
    pre=getattr(tk, 'text_preprocessing_func', None)
    if type(tk) in (BaseCharsTokenizer, ArabicCharsTokenizer) and preprocessing_of(pre):
        return {'kind':'chars',
                'accept':'any' if isinstance(tk, ArabicCharsTokenizer) else 'alnum',
                'punct':list(tk.PUNCT_LIST) if tk.punct else [],
                'pad_with_space':bool(tk.pad_with_space),
                'preprocessing':preprocessing_of(pre)}
    return {'kind':type(tk).__name__}
agg=m.tokenizer
tok={'vocab_size':agg.vocab_size,'tokenizer_names':list(agg.tokenizer_names),
     'tokenizer_offsets':agg.tokenizer_offsets,'num_tokens_per_tokenizer':agg.num_tokens_per_tokenizer,
     'tokenizer_pad_ids':agg.tokenizer_pad_ids,'tokens':list(agg.tokens),'eos_id':int(m.eos_id),
     'tokenizer_kinds':{n:kind_of(t) for n,t in agg.tokenizers.items()}}
json.dump(tok, open(D+'assets/tokenizer.json','w'), ensure_ascii=False, indent=1)
print('tokenizer kinds', {n:k['kind'] for n,k in tok['tokenizer_kinds'].items()})
