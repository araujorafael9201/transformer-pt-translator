import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim=128, n_attn_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, n_attn_heads)
        self.ln = nn.LayerNorm(embed_dim)
        self.ff = nn.Linear(embed_dim, embed_dim)
        self.ln_proj = nn.LayerNorm(embed_dim)

    def forward(self, X):
        out, w = self.mha(X, X, X)
        X += out
        X = self.ln(X)
        X += self.ff(X)
        X = self.ln_proj(X)

        return X

class Translator(nn.Module):
    def __init__(self, n_layer_enc=6, emb_dim=128, vocab_size=512, seq_len=64):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.ModuleDict(dict(
            w_emb = nn.Embedding(vocab_size, emb_dim),
            w_pos = nn.Embedding(seq_len, emb_dim),
            h = nn.ModuleList([EncoderBlock(emb_dim) for _ in range(n_layer_enc)])
        ))

    def forward(self, X):
        emb = self.encoder.w_emb(X)
        pos_emb = self.encoder.w_pos(torch.arange(self.seq_len))

        res = emb + pos_emb
        for b in self.encoder.h:
            outputs = b(res)

        return outputs

import torch
embed_size = 128
seq_len = 8
batch_size = 64

enc = Translator(emb_dim=embed_size, seq_len=seq_len)
X = torch.randint(0, 256, (batch_size, seq_len))

print(enc(X).shape)