# input a 12 dimensional vector, output a 12 dimensional vector
# the output vector is a function of the input vector
# the function is a flow matching function
# the flow matching function is a function that matches the input vector to the output vector
# the flow matching function is a function that is a function of the input vector
# imitating the architecture of the FMT block in the paper FlashLips
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in=1024, d_out=1024, n_head=8, dropout=0.1, qkv_bias=False, proj_bias=False):
        super(MultiHeadAttention, self).__init__()
        self.head_dim  = d_out // n_head
        self.n_head = n_head
        self.query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, bias = proj_bias)
        self.dropout = nn.Dropout(dropout)
        self.d_out = d_out

    def forward(self, x):
        batch, seq_len, d_in = x.shape
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        queries = query.view(batch, seq_len, self.n_head, self.head_dim).transpose(1,2)
        keys = key.view(batch, seq_len, self.n_head, self.head_dim).transpose(1,2)
        values = value.view(batch, seq_len, self.n_head, self.head_dim).transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)
        d_keys = key.shape[-1]
        normalized_scores = attn_scores / d_keys ** 0.5
        attn_weights = F.softmax(normalized_scores, dim = -1)
        context_vectors = attn_weights @ values
        context_vectors = context_vectors.transpose(1,2).contiguous().view(batch, seq_len, self.d_out)
        output = self.out_proj(context_vectors)
        output = self.dropout(output)
        return output

class MLP(nn.Module):
    def __init__(self, d_in=1024, d_out=1024, dropout=0.1, bias=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_in, 4*d_in, bias=bias)
        self.fc2 = nn.Linear(4*d_in, d_out, bias=bias)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class FMTBlock(nn.Module):
    def __init__(self, d_in=1024, d_out=1024, dh=1024, n_head=8, dropout=0.1, qkv_bias=False, proj_bias=False):
        super(FMTBlock, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_in, d_out, n_head, dropout, qkv_bias, proj_bias)
        self.mlp = MLP(dh, dh, dropout)
        self.norm1 = nn.LayerNorm(dh, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dh, elementwise_affine=False, eps=1e-6)
        self.ada_ln_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dh, 6*dh),
        )
    
    def forward(self, x, conditioning_vector):
        mod_params = self.ada_ln_mlp(conditioning_vector).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [mod_params[i].unsqueeze(1) for i in range(6)]
        res1 = self.norm1(x)
        res1 = res1 * (1 + scale_msa) + shift_msa
        attn_out = self.multihead_attention(res1)
        x = x + gate_msa * attn_out
        res2 = self.norm2(x)
        res2 = res2 * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(res2)
        x = x + gate_mlp * mlp_out
        return x

if __name__ == "__main__":
    dh = 1024
    T = 60 # 60 frames as per architecture notes
    time_steps = 60
    block = FMTBlock(d_in=1024, d_out=1024, dh=1024, n_head=8, dropout=0.1, qkv_bias=False, proj_bias=False)
    motion_seq = torch.randn(2, T, dh) # [Batch, Time, Dim]
    condition = torch.randn(2, dh)    # [Batch, Dim]
    # initial velocity
    pred_vt = torch.zeros(2, T, dh)
    for t in range(time_steps):
        pass
    # Mock inputs matching your specs
    
    
    output = block(motion_seq, condition)
    
    print(f"Architecture Verification:")
    print(f"Input Motion Shape: {motion_seq.shape}")
    print(f"Condition Latent Shape: {condition.shape}")
    print(f"FMT Block Output Shape: {output.shape}")
    
    # parameter count for the FMT block
    total_params = sum(p.numel() for p in block.parameters())
    print(f"Total Parameters: {total_params}")