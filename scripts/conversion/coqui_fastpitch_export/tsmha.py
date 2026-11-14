import torch, torch.nn.functional as F
from torch import nn

class TracerSafeMHA(nn.Module):
    """Drop-in for nn.MultiheadAttention (self-attention, batch_first=False).

    Same parameter names (in_proj_weight/in_proj_bias/out_proj) so a pretrained
    nn.MultiheadAttention state_dict loads unchanged. Reshapes use -1 + live
    tensor sizes so the legacy ONNX tracer keeps the sequence length dynamic.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.modules.linear.NonDynamicallyQuantizableLinear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=True):
        # query/key/value: [T, B, C] (self-attention -> all the same tensor)
        H, hd = self.num_heads, self.head_dim
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)                       # each [T, B, C]
        q = q * (hd ** -0.5)
        # [T, B, C] -> [B*H, T, hd]   (-1 = B*H, stays symbolic)
        q = q.reshape(q.shape[0], -1, hd).transpose(0, 1)
        k = k.reshape(k.shape[0], -1, hd).transpose(0, 1)
        v = v.reshape(v.shape[0], -1, hd).transpose(0, 1)
        attn = torch.bmm(q, k.transpose(1, 2))               # [B*H, Tq, Tk]
        if key_padding_mask is not None:
            attn = attn.reshape(-1, H, q.shape[1], k.shape[1])
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            attn = attn.reshape(-1, q.shape[1], k.shape[1])
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)                             # [B*H, Tq, hd]
        # [B*H, Tq, hd] -> [Tq, B, C]   (-1 = B)
        out = out.transpose(0, 1).reshape(query.shape[0], -1, self.embed_dim)
        out = self.out_proj(out)
        # head-averaged weights, like nn.MultiheadAttention(need_weights=True)
        attn_w = attn.reshape(-1, self.num_heads, q.shape[1], k.shape[1]).mean(dim=1)
        return out, attn_w
