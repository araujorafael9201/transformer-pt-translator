import torch.nn as nn
import torch

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim=128, n_attn_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, n_attn_heads, batch_first=True, dropout=0.2)
        self.ln = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ln_proj = nn.LayerNorm(embed_dim)

    def forward(self, X, key_padding_mask=None):
        out, _ = self.mha(X, X, X, key_padding_mask=key_padding_mask)
        X = self.ln(X + out)
        X = self.ln_proj(X + self.ff(X))
        return X
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim=128, n_heads=8):
        super().__init__()
        self.masked_mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.2)
        self.ln = nn.LayerNorm(embed_dim)

        self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.2)
        self.l = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ln_proj = nn.LayerNorm(embed_dim)

    
    def forward(self, X, enc_out, tgt_mask=None, enc_key_padding_mask=None, tgt_key_padding_mask=None):
        # Masked self-attention
        out, _ = self.masked_mha(X, X, X, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        X = self.ln(X + out)

        # Cross-attention
        out, _ = self.mha(X, enc_out, enc_out, key_padding_mask=enc_key_padding_mask)
        X = self.l(X + out)

        # Feed-forward
        X = self.ln_proj(X + self.ff(X))
        return X

class Translator(nn.Module):
    def __init__(self, n_layer_enc=6, emb_dim=128, vocab_size=512, seq_len=128, eos_token=793, bos_token=200019, pad_token=0):
        super().__init__()
        self.vocab_size = vocab_size # already counting the special bos token
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.pad_token = pad_token
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
        device = X.device
        T_src = X.size(1)
        T_tgt = y.size(1)

        # Encoder
        src_emb = self.encoder.w_emb(X)
        src_pos = self.encoder.w_pos(torch.arange(T_src, device=device)).unsqueeze(0)
        enc_out = src_emb + src_pos

        src_pad_mask = (X == self.pad_token)
        for block in self.encoder.h:
            enc_out = block(enc_out, key_padding_mask=src_pad_mask)

        # Decoder
        tgt_emb = self.decoder.w_emb(y)
        tgt_pos = self.decoder.w_pos(torch.arange(T_tgt, device=device)).unsqueeze(0)
        dec_out = tgt_emb + tgt_pos

        tgt_pad_mask = (y == self.pad_token)
        # causal mask for decoder self-attention
        causal_mask = torch.triu(torch.ones(T_tgt, T_tgt, device=device, dtype=torch.bool), diagonal=1)

        for block in self.decoder.h:
            dec_out = block(dec_out, enc_out, tgt_mask=causal_mask,
                            enc_key_padding_mask=src_pad_mask,
                            tgt_key_padding_mask=tgt_pad_mask)

        logits = self.lm_head(dec_out)
        return logits

    @torch.no_grad
    def translate(self, X):
        device = X.device
        y = torch.tensor([self.bos_token], device=device).unsqueeze(0)

        while y[0, -1] != self.eos_token and y.size(1) != self.seq_len:
            T_src = X.size(1)
            T_tgt = y.size(1)

            # Encoder
            src_emb = self.encoder.w_emb(X)
            src_pos = self.encoder.w_pos(torch.arange(T_src, device=device)).unsqueeze(0)
            enc_out = src_emb + src_pos

            src_pad_mask = (X == self.pad_token)
            for block in self.encoder.h:
                enc_out = block(enc_out, key_padding_mask=src_pad_mask)

            # Decoder
            tgt_emb = self.decoder.w_emb(y)
            tgt_pos = self.decoder.w_pos(torch.arange(T_tgt, device=device)).unsqueeze(0)
            dec_out = tgt_emb + tgt_pos

            tgt_pad_mask = (y == self.pad_token)
            # causal mask for decoder self-attention
            causal_mask = torch.triu(torch.ones(T_tgt, T_tgt, device=device, dtype=torch.bool), diagonal=1)

            for block in self.decoder.h:
                dec_out = block(dec_out, enc_out, tgt_mask=causal_mask,
                                enc_key_padding_mask=src_pad_mask,
                                tgt_key_padding_mask=tgt_pad_mask)

            logits = self.lm_head(dec_out)

            # reduce repeated token probability
            for token in y[0][:-5]:
                logits[0, -1, token] /= 1.2

            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.argmax(probs, dim=1)

            y = torch.cat((y, next_token.unsqueeze(0)), dim=1)

        return y[0][1:] # remove BOS token
