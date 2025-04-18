import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import PositionalEncoding, Block


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        head_size: int,
        num_layers: int,
        dropout: float,
        context_size: int,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=context_size)
        self.layers = nn.Sequential(
            *[
                Block(d_model, num_heads, head_size, dropout, context_size)
                for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, y=None):
        x = self.token_emb(x)
        x = self.pos_enc(x)
        x = self.layers(x)
        x = self.ln(x)
        logits = self.fc_out(x)
        if y is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            return logits, loss
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.pos_enc.pe.size(0) :]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx