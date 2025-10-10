import argparse
import os
import tiktoken
import torch
import torch.nn as nn

from model import Translator
from utils.dataloader import DataLoader

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"using device {device}")

    torch.set_float32_matmul_precision('high')

    # tokenizazion
    print(f"preparing dataset")
    enc = tiktoken.get_encoding("o200k_base")
    vocab_size = enc.max_token_value + 1 + 1 # +1 for special bos token, +1 to get count (token values go from 0 to max_token_value). \n is considered eos
    print(f"vocab_size: {vocab_size}")

    gpu_batch_size = min(args.gpu_batch_size, args.batch_size)
    dl = DataLoader(args.en_file, args.pt_file, batch_size=gpu_batch_size, enc=enc, max_seq_len=args.max_seq_len, max_dataset_size=args.max_dataset_size)
    print(f"max_seq_len: {args.max_seq_len}")

    # init model
    model = Translator(emb_dim=args.embed_size, vocab_size=vocab_size, seq_len=args.max_seq_len, eos_token=793, bos_token=vocab_size-1, pad_token=0)
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

    grad_accum_steps = max(1, args.batch_size // args.gpu_batch_size)
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
    parser.add_argument('--compile_model', action='store_true', help='Pre compile the model with torch.compile')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--max_seq_len', type=int, default=128, help='Maximum length of input data.')
    parser.add_argument('--max_dataset_size', type=int, default=10000000, help='Optionally limit dataset size for faster training')
    parser.add_argument('--batch_size', type=int, default=256, help='Effective batch size for training after gradient accumulation')
    parser.add_argument('--gpu_batch_size', type=int, default=64, help='Batch size supported by GPU')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--en_file', type=str, default='data/en.txt', help='Path to the English source file.')
    parser.add_argument('--pt_file', type=str, default='data/pt.txt', help='Path to the Portuguese target file.')
    parser.add_argument('--model_path', type=str, default='translator_model.pth', help='Path to save the trained model.')
    parser.add_argument('--checkpoint_path', type=str, default='translator_model_checkpoint.pth', help='Path to save the checkpoints during training.')

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
