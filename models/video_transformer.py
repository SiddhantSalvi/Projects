import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
import math
import time
import random
import argparse
import logging
import warnings
import importlib
import collections

class ImageTransformer(nn.Module):
    def __init__(self, num_heads, num_layers, embedding_size, dropout=0.1):
        super(ImageTransformer, self).__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_size=embedding_size,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for block in self.transformer_blocks:
            x = block(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, embedding_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.d_out = d_out
        self.d_in = d_in
        self.head_dim = d_out // num_heads
        self.query = nn.Linear(d_in, d_out, bias=False)
        self.key = nn.Linear(d_in, d_out, bias=False)
        self.value = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch, seq_len, d_in = x.shape
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        queries = query.view(batch, seq_len, num_heads, head_dim)
        queries = queries.transpose(1, 2)
        keys = key.view(batch, seq_len, num_heads, head_dim)
        keys = keys.transpose(1, 2)
        values = value.view(batch, seq_len, num_heads, head_dim)
        values = values.transpose(1, 2)
        attn_scores = queries @ keys.transpose(2,3)
        d_key = keys.shape[-1]
        normalized_scores = attn_scores / d_key ** 0.5
        attn_weights = F.softmax(normalized_scores, dim=-1)
        # mask the attention weights
        attn_weights = attn_weights.masked_fill(~mask, -float('inf'))
        attn_weights = self.dropout(attn_weights)
        context_vectors = attn_weights @ values
        context_vectors = context_vectors.transpose(1, 2).contiguous().view(batch, seq_len, d_out)
        context_vectors = self.out_proj(context_vectors)
        return context_vectors
 

class MLP(nn.Module):
    def __init__(self, embedding_size, dropout=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embedding_size, embedding_size * 4, bias=False)
        self.fc2 = nn.Linear(embedding_size * 4, embedding_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_in, d_out, embedding_size, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention = MultiHeadAttention(d_in, d_out, embedding_size, num_heads, dropout)
        self.mlp = MLP(embedding_size, dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
    def forward(self, x):
        residuals = x.clone()
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + residuals
        residuals = x.clone()
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residuals
        return x

class ImageTransformer(nn.Module):
    def __init__(self, d_in, d_out, num_heads, num_layers, embedding_size, dropout=0.1):
        super(ImageTransformer, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_in=d_in,
                d_out=d_out,
                embedding_size=embedding_size,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for block in self.transformer_blocks:
            x = block(x)
        return x

class VQVAE(nn.Module):
    def __init__(self, frame_size)