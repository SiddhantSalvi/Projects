import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, vocab_size=50265, context_length=256, attn_blocks=12, embedding_size=768, num_heads=12, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(context_length, embedding_size)
        self.norm = nn.LayerNorm()
        self.attention = MultiheadAttention(num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLP(embedding_size, dropout)
    
    def forward(self, x):
        token_embedding = self.token_embedding(x)
        position_embedding = self.position_embedding(torch.arange(x.shape[1], device=x.device))
        x = token_embedding + position_embedding
        residuals = x.clone()
        x = self.norm(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + residuals
        residuals = x.clone()
        x = self.norm(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residuals
        return x

class MLP(nn.Module):
    def __init__(self, embedding_size, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embedding_size, embedding_size * 4)
        self.fc2 = nn.Linear(embedding_size * 4, embedding_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads = 12, dropout = 0.1, qkv_bias = False, proj_bias = False):
        super(MultiHeadAttention, self).__init__()
        self.query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, bias = proj_bias)
        self.dropout = nn.Dropout(dropout)
        self.head_dim = d_out // num_heads
        self.num_heads = num_heads
    
    def forward(self, x):
        batch, seq_len, d_in = x.shape
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # transpose to [batch, num_heads, seq_len, head_dim]
        queries = query.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = key.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = value.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = queries @ keys.transpose(2,3)
        d_key = keys.shape[-1]
        normalized_scores = attn_scores / d_key ** 0.5
        attn_weights = F.softmax(normalized_scores, dim=-1) # [batch, num_heads, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)
        context_vectors = attn_weights @ values # [batch, num_heads, seq_len, head_dim]
        context_vectors = context_vectors.transpose(1, 2).contiguous().view(batch, seq_len, d_out)
        context_vectors = self.out_proj(context_vectors)
        return context_vectors


if __name__ == "__main__":
    # test the MultiHeadAttention module
    batch_size = 1
    seq_len = 5
    d_in = 128
    d_out = 128
    num_heads = 4
    dropout = 0.1
    qkv_bias = False
    proj_bias = False
    x = torch.randn(batch_size, seq_len, d_in)
    multihead_attention = MultiHeadAttention(d_in, d_out, num_heads, dropout, qkv_bias, proj_bias)
    output = multihead_attention(x)
    print(output.shape)