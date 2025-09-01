import torch.nn as nn
import torch
import tiktoken
import argparse
import os

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
    def __init__(self, n_layer_enc=6, emb_dim=128, vocab_size=512, seq_len=128):
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
        device = X.device
        B, T_src = X.size()
        T_tgt = y.size(1)

        # Encoder
        src_emb = self.encoder.w_emb(X)
        src_pos = self.encoder.w_pos(torch.arange(T_src, device=device)).unsqueeze(0)
        enc_out = src_emb + src_pos

        src_pad_mask = (X == 0)
        for block in self.encoder.h:
            enc_out = block(enc_out, key_padding_mask=src_pad_mask)

        # Decoder
        tgt_emb = self.decoder.w_emb(y)
        tgt_pos = self.decoder.w_pos(torch.arange(T_tgt, device=device)).unsqueeze(0)
        dec_out = tgt_emb + tgt_pos

        tgt_pad_mask = (y == 0)
        # causal mask for decoder self-attention
        causal_mask = torch.triu(torch.ones(T_tgt, T_tgt, device=device, dtype=torch.bool), diagonal=1)

        for block in self.decoder.h:
            dec_out = block(dec_out, enc_out, tgt_mask=causal_mask,
                            enc_key_padding_mask=src_pad_mask,
                            tgt_key_padding_mask=tgt_pad_mask)

        logits = self.lm_head(dec_out)
        return logits
    
    
class DataLoader:
    def __init__(self, en_file_name, pt_file_name, batch_size, max_seq_len, max_dataset_size, enc=tiktoken.get_encoding("o200k_base")):
        self.current_pos = 0
        self.batch_size = batch_size
        with open(en_file_name, "r") as en:
            self.en = [enc.encode(s)[:max_seq_len] for s in en.readlines()[:max_dataset_size]]
        with open(pt_file_name, "r") as pt:
            self.pt = [enc.encode(s)[:max_seq_len] for s in pt.readlines()[:max_dataset_size]]

        assert len(self.pt) == len(self.en), "original and translated datasets have different sizes"
        self.size = len(self.pt)

    def get_max_seq_len(self):
        return max([len(s) for s in self.en + self.pt])

    def get_next_batch(self):
        positions = [(self.current_pos + i) % self.size for i in range(self.batch_size)]

        self.current_pos += self.batch_size
        self.current_pos %= self.size

        X = [torch.tensor(self.en[p], dtype=torch.long) for p in positions]
        y = [torch.tensor(self.pt[p], dtype=torch.long) for p in positions]

        X = nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)
        y = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)

        return X, y

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"using device {device}")

    torch.set_float32_matmul_precision('high')

    # tokenizazion
    print(f"preparing dataset")
    enc = tiktoken.get_encoding("o200k_base")
    vocab_size = enc.max_token_value + 1
    print(f"vocab_size: {vocab_size}")

    dl = DataLoader(args.en_file, args.pt_file, args.batch_size, enc=enc, max_seq_len=args.max_seq_len, max_dataset_size=args.max_dataset_size)
    print(f"max_seq_len: {args.max_seq_len}")

    # init model
    model = Translator(emb_dim=args.embed_size, vocab_size=vocab_size, seq_len=args.max_seq_len)
    model.to(device)
    if os.path.exists(args.checkpoint_path):
        print(f"using checkpoint in {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, weights_only=True, map_location=torch.device(device)))
    
    if args.compile_model:
        compiled_model = torch.compile(model)
    else:
        compiled_model = model

    # training setup
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=args.learning_rate)
    grad_accum_steps = args.batch_size // 32 if args.batch_size > 32 else 1 # considering gpu supports batch of size 32
    print(f"grad_accum_steps: {grad_accum_steps}")

    # training
    print(f"starting training")
    num_batches = dl.size // args.batch_size
    for i in range(args.epochs):
        epoch_loss = 0.0
        model.train()

        for _ in range(num_batches):
            optimizer.zero_grad()
            batch_loss = 0.0

            for _ in range(grad_accum_steps):
                X, y = dl.get_next_batch()
                X, y = X.to(device), y.to(device)

                logits = compiled_model(X, y[:, :-1])
                B, T, C = logits.shape
                loss = criterion(logits.view(B * T, C), y[:, 1:].reshape(-1))
                loss /= grad_accum_steps
                loss.backward()

                batch_loss += loss.item()

            optimizer.step()
            epoch_loss += batch_loss

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {i+1}/{args.epochs} finished. Average Loss: {avg_epoch_loss:.4f}")

        if i % 10 == 0:
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f"checkpoint saved to {args.checkpoint_path}")

    
    # save model
    torch.save(model.state_dict(), args.model_path)
    print(f"model saved to {args.model_path}")

def main():
    parser = argparse.ArgumentParser(description="A script to train and run a Transformer-based translator.")
    
    # Shared arguments
    parser.add_argument('--embed_size', type=int, default=128, help='Embedding dimension size.')
    parser.add_argument('--compile_model', type=bool, default=True, help='Pre compile the model with torch.compile')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--max_seq_len', type=int, default=128, help='Maximum length of input data.')
    parser.add_argument('--max_dataset_size', type=int, default=10000000, help='Optionally limit dataset size for faster training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--en_file', type=str, default='data/en.txt', help='Path to the English source file.')
    parser.add_argument('--pt_file', type=str, default='data/pt.txt', help='Path to the Portuguese target file.')
    parser.add_argument('--model_path', type=str, default='translator_model.pth', help='Path to save the trained model.')
    parser.add_argument('--checkpoint_path', type=str, default='translator_model_checkpoint.pth', help='Path to save the checkpoints during training.')

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
