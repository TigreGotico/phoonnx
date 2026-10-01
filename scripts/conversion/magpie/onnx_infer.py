'''ONNX-runtime inference driver for Magpie-TTS. Neural compute in ONNX; the
attention-prior / EOS control logic is NeMo's own, reused verbatim.'''
import numpy as np, torch, onnxruntime as ort
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths

O=out_dir()+'/'
def _sess(n):
    so=ort.SessionOptions(); so.log_severity_level=3
    return ort.InferenceSession(O+n, so, providers=['CPUExecutionProvider'])

class MagpieOnnx:
    def __init__(self, model):
        self.m=model
        self.enc=_sess('text_encoder.onnx'); self.xkv=_sess('cross_kv.onnx')
        self.dec=_sess('decoder_step.onnx'); self.lt=_sess('local_step.onnx')
        self.ae=_sess('audio_embed.onnx'); self.lte=_sess('lt_embed.onnx')
        self.cod=_sess('codec_decoder.onnx')
        self.NCB=model.num_audio_codebooks*model.frame_stacking_factor

    def _lt_sample(self, dec_out, temperature, topk, unfinished, finished, cfg_scale, forbid_eos):
        m=self.m; B=dec_out.shape[0]
        h=dec_out[:,None,:]
        ck=np.zeros((2,B,0,12,64),np.float32); cv=np.zeros((2,B,0,12,64),np.float32)
        preds=[]
        for cb in range(self.NCB):
            lg,ck,cv=self.lt.run(None,{'h':h,'pos':np.array([cb],np.int64),'cache_k':ck,'cache_v':cv,'cb':np.array(cb,np.int64)})
            lg=torch.from_numpy(lg.copy())
            half=B//2
            lg[:half]=cfg_scale*lg[:half]+(1.0-cfg_scale)*lg[half:]
            for i in unfinished: lg[i, m.audio_eos_id]=float('-inf')
            for i in finished: lg[i,:]=float('-inf'); lg[i,m.audio_eos_id]=0.0
            from nemo.collections.tts.modules.magpietts_modules import clear_forbidden_logits
            lg=clear_forbidden_logits(lg.unsqueeze(1), m.codebook_size, forbid_audio_eos=forbid_eos).squeeze(1)
            tk=torch.topk(lg,topk,dim=-1)[0]
            lg=lg.masked_fill(lg<tk[:,-1:], float('-inf'))
            p=torch.softmax(lg/temperature,dim=-1)
            tok=torch.multinomial(p,1)
            tok[half:]=tok[:half]
            preds.append(tok)
            h=self.lte.run(None,{'tok':tok.squeeze(-1).numpy().astype(np.int64),'cb':np.array(cb,np.int64)})[0]
        allp=torch.cat(preds,dim=1)
        return allp.reshape(-1,m.frame_stacking_factor,m.num_audio_codebooks).permute(0,2,1)

    def generate(self, text_ids, text_len, speaker_index=0, use_cfg=True, temperature=None, topk=None, max_steps=None, use_kv_cache=False):
        m=self.m; ip=m.inference_parameters
        temperature = ip.temperature if temperature is None else temperature
        topk = ip.topk if topk is None else topk
        cfg_scale = ip.cfg_scale
        from nemo.collections.tts.models.magpietts import EOSDetectionMethod
        eos_method = EOSDetectionMethod(ip.eos_detection_method)
        text=torch.as_tensor(text_ids).long().view(1,-1)
        text_lens=torch.tensor([text_len]).long()
        text_mask=get_mask_from_lengths(text_lens).float()
        cond=self.enc.run(None,{'text':text.numpy(),'text_mask':text_mask.numpy()})[0]
        Tt=cond.shape[1]
        if use_cfg:
            cond2=np.concatenate([cond, np.zeros_like(cond)],0)
            cm=np.zeros((2,Tt),np.float32); cm[0,:text_len]=1; cm[1,0]=1
        else:
            cond2=cond; cm=text_mask.numpy()
        B=cond2.shape[0]
        ck,cv=self.xkv.run(None,{'cond':cond2})
        ctx,ctx_lens=m.get_baked_context_embeddings_batch(batch_size=1, speaker_indices=speaker_index)
        ctx=ctx.detach().numpy().astype(np.float32)
        Tc=ctx.shape[1]
        ctx2=np.concatenate([ctx, np.zeros_like(ctx)],0) if use_cfg else ctx
        codes=np.full((B,self.NCB), m.audio_bos_id, np.int64)
        emb=self.ae.run(None,{'codes':codes})[0]
        x=np.concatenate([ctx2, emb],1)
        sk=np.zeros((12,B,0,12,64),np.float32); sv=np.zeros((12,B,0,12,64),np.float32)
        prior=np.ones((B,1,Tt),np.float32)
        pos=np.arange(x.shape[1],dtype=np.int64)
        all_pred=[]; end_indices={}; last_att=[[1]]; att_counter=[{}]
        unfin={}; fin={}; unfinished_texts={}; finished_counter={}
        maxs=(max_steps or ip.max_decoder_steps)//m.frame_stacking_factor
        for idx in range(maxs):
            logits,dec_out,sk,sv,xprobs=self.dec.run(None,{'x':x,'pos':pos,'self_k':sk,'self_v':sv,
                'cross_k':ck,'cross_v':cv,'cond_mask':cm,'attn_prior':prior})
            lt=torch.from_numpy(logits[:, -1, :].copy())
            if use_cfg:
                lt=cfg_scale*lt[:1]+(1-cfg_scale)*lt[1:]
                lt=torch.cat([lt,lt],0)
            ap=[{'cross_attn_probabilities':[torch.from_numpy(xprobs[i][:, :, -1:, :].copy())]} for i in range(12)]
            cas,_=m.get_cross_attention_scores(ap)
            alg,_=m.get_cross_attention_scores(ap, filter_layers=ip.estimate_alignment_from_layers)
            if ip.apply_attention_prior and idx>=ip.start_prior_after_n_audio_steps:
                tstep,att_counter=m.get_most_attended_text_timestep(alignment_attention_scores=alg,
                    last_attended_timesteps=last_att, text_lens=text_lens,
                    lookahead_window_size=ip.attention_prior_lookahead_window,
                    attended_timestep_counter=att_counter, batch_size=1)
                last_att.append(tstep)
                np_,unfinished_texts,finished_counter=m.construct_inference_prior(
                    prior_epsilon=ip.attention_prior_epsilon, cross_attention_scores=cas,
                    text_lens=text_lens, text_time_step_attended=tstep,
                    attended_timestep_counter=att_counter, unfinished_texts=unfinished_texts,
                    finished_texts_counter=finished_counter, end_indices=end_indices,
                    lookahead_window_size=ip.attention_prior_lookahead_window, batch_size=1)
                prior=np_.detach().numpy().astype(np.float32)
            if ip.ignore_finished_sentence_tracking: fin={}; unfin={}
            else:
                fin={k:v for k,v in finished_counter.items() if v>=20}
                unfin={k:v for k,v in unfinished_texts.items() if v}
            forbid=idx*m.frame_stacking_factor < ip.min_generated_frames
            nxt=self._lt_sample(dec_out[:, -1, :], temperature, topk, unfin, fin, cfg_scale, forbid)
            argm=m.sample_codes_from_logits(lt.clone(), temperature=0.01, topk=1,
                unfinished_items=unfin, finished_items=fin, forbid_audio_eos=forbid)
            ef=m.detect_eos(nxt[0], argm[0], eos_method)
            if 0 not in end_indices and ef!=float('inf'):
                end_indices[0]=idx*m.frame_stacking_factor+ef
            all_pred.append(nxt[:1])
            codes=nxt[:1].permute(0,2,1).reshape(1,-1).numpy().astype(np.int64)
            codes=np.repeat(codes,B,0)
            nxt_emb=self.ae.run(None,{'codes':codes})[0]
            if use_kv_cache:
                x=nxt_emb; pos=np.array([sk.shape[2]],np.int64)
            else:
                # NeMo reference semantics: use_kv_cache_for_inference defaults to False, and the
                # latest attention prior is re-applied across the whole history, so the residual
                # stream must be recomputed from scratch every step to stay bit-comparable.
                x=np.concatenate([x, nxt_emb],1)
                sk=np.zeros((12,B,0,12,64),np.float32); sv=np.zeros((12,B,0,12,64),np.float32)
                pos=np.arange(x.shape[1],dtype=np.int64)
            if 0 in end_indices and len(all_pred)>=4: break
        pc=torch.cat(all_pred,dim=-1)
        L=int(end_indices.get(0, ip.max_decoder_steps))
        pc=pc[:, :, :L]
        audio=self.cod.run(None,{'codes':pc.numpy().astype(np.int64)})[0]
        return audio, pc
