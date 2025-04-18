import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model, head_size, dropout, context_size):
        super().__init__()
        self.key = nn.Linear(d_model, head_size)
        self.query = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)
        self.dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(context_size, context_size)).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)
        v = self.value(x)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(C)
        scores = scores.masked_fill(~self.mask[:T, :T], float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, head_size, dropout, context_size):
        super().__init__()
        self.heads = nn.ModuleList([
            SelfAttention(d_model, head_size, dropout, context_size)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(num_heads * head_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1)]
        return x


class Block(nn.Module):
    def __init__(self, d_model, num_heads, head_size, dropout, context_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, head_size, dropout, context_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x