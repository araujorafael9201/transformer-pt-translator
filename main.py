import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim=128, n_attn_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, n_attn_heads, batch_first=True)
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
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim=128, n_heads=8):
        super().__init__()
        self.masked_mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

        self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.l = nn.LayerNorm(embed_dim)

        self.ff = nn.Linear(embed_dim, embed_dim)
        self.ln_proj = nn.LayerNorm(embed_dim)

    
    def forward(self, X, encoder_out):
        out, w = self.masked_mha(X, X, X, attn_mask=torch.tril(torch.ones(seq_len, seq_len)))
        X += out
        X = self.ln(X)
        out, w = self.mha(X, encoder_out, encoder_out)
        X += out
        X = self.l(X)

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
        self.decoder = nn.ModuleDict(dict(
            w_emb = nn.Embedding(vocab_size, emb_dim),
            w_pos = nn.Embedding(seq_len, emb_dim),
            h = nn.ModuleList([DecoderBlock(emb_dim) for _ in range(n_layer_enc)])
        ))
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, X, y):
        positions = torch.arange(self.seq_len)

        emb = self.encoder.w_emb(X)
        pos_emb = self.encoder.w_pos(positions)

        enc_output = emb + pos_emb
        for b in self.encoder.h:
            enc_output = b(enc_output) 

        dec_out = self.decoder.w_emb(y)
        dec_pos_emb = self.decoder.w_pos(positions)
        dec_out += dec_pos_emb

        for b in self.decoder.h:
            dec_out = b(dec_out, enc_output)

        output = self.lm_head(dec_out)
        return output

import torch
embed_size = 128
seq_len = 8
batch_size = 64
vocab_size=256

enc = Translator(vocab_size=vocab_size, emb_dim=embed_size, seq_len=seq_len)
X = torch.randint(0, vocab_size, (batch_size, seq_len))
y = torch.zeros((batch_size, seq_len), dtype=torch.long)

print(enc(X, y).shape)