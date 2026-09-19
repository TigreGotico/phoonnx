import torch, torch.nn.functional as F

APPLY_PRIOR_LAYERS = [2,3,4,5,6,7,8,9,10]

class DecoderStep(torch.nn.Module):
    '''External-KV-cache reimplementation of the Magpie decoder stack.
    Handles both prefill (T>1, empty past) and decode (T=1) in one graph.'''
    def __init__(self, model):
        super().__init__()
        self.dec = model.decoder
        self.final_proj = model.final_proj
        self.n_layers = len(self.dec.layers)

    def forward(self, x, pos, self_k, self_v, cross_k, cross_v, cond_mask, attn_prior):
        # x: (B,T,D) already-embedded decoder input
        # pos: (T,) int64 absolute positions
        # self_k/self_v: (L,B,Tp,nh,dh)
        # cross_k/cross_v: (L,B,Tt,xnh,xdh)
        # cond_mask: (B,Tt) float, attn_prior: (Bp,T,Tt) float
        B, T, D = x.shape
        Tp = self_k.shape[2]
        x = x + self.dec.position_embeddings(pos).unsqueeze(0)
        new_k, new_v, xprobs = [], [], []
        # causal mask over (T, Tp+T)
        tot = Tp + T
        ar = torch.arange(tot, device=x.device).unsqueeze(0)
        qi = (torch.arange(T, device=x.device) + Tp).unsqueeze(1)
        causal = (ar <= qi).to(x.dtype).view(1,1,T,tot)
        cmask = cond_mask[:, None, None, :]
        for i, layer in enumerate(self.dec.layers):
            sa = layer.self_attention
            h = layer.norm_self(x)
            qkv = sa.qkv_net(h).reshape(B, T, 3, sa.n_heads, sa.d_head)
            q, k, v = [t.squeeze(2) for t in qkv.chunk(3, dim=2)]
            k = torch.cat([self_k[i], k], dim=1)
            v = torch.cat([self_v[i], v], dim=1)
            new_k.append(k); new_v.append(v)
            qt, kt, vt = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
            sc = torch.matmul(qt, kt.transpose(2,3)) * sa.scale
            sc = sc.masked_fill(causal == 0, float('-inf'))
            p = F.softmax(sc, dim=-1)
            y = torch.matmul(p, vt).transpose(1,2).contiguous().view(B, T, -1)
            x = x + sa.o_net(y)
            # cross attention
            ca = layer.cross_attention
            hq = layer.norm_xattn_query(x)
            q2 = ca.q_net(hq).reshape(B, T, ca.n_heads, ca.d_head).transpose(1,2)
            k2 = cross_k[i].transpose(1,2); v2 = cross_v[i].transpose(1,2)
            sc2 = torch.matmul(q2, k2.transpose(2,3)) * ca.scale
            sc2 = sc2.masked_fill(cmask == 0, float('-inf'))
            pr2 = F.softmax(sc2, dim=-1)
            if i in APPLY_PRIOR_LAYERS:
                ap = attn_prior[:, None] + 1e-30
                pr2 = pr2 * ap
                pr2 = pr2 / pr2.sum(dim=-1, keepdim=True)
            pr2 = pr2.masked_fill(cmask == 0, 0.0)
            xprobs.append(pr2)
            y2 = torch.matmul(pr2, v2).transpose(1,2).contiguous().view(B, T, -1)
            x = x + ca.o_net(y2)
            ones = torch.ones(B, T, device=x.device, dtype=x.dtype)
            x = x + layer.pos_ff(layer.norm_pos_ff(x), ones)
        x = self.dec.norm_out(x)
        logits = self.final_proj(x)
        return logits, x, torch.stack(new_k), torch.stack(new_v), torch.stack(xprobs)


class CrossKV(torch.nn.Module):
    '''Precompute per-layer cross-attention K/V from encoder output.'''
    def __init__(self, model):
        super().__init__(); self.dec = model.decoder
    def forward(self, cond):
        ks, vs = [], []
        for layer in self.dec.layers:
            mem = layer.norm_xattn_memory(cond)
            ca = layer.cross_attention
            B, Tt, _ = mem.shape
            kv = ca.kv_net(mem).reshape(B, Tt, 2, ca.n_heads, ca.d_head)
            k, v = [t.squeeze(2) for t in kv.chunk(2, dim=2)]
            ks.append(k); vs.append(v)
        return torch.stack(ks), torch.stack(vs)
