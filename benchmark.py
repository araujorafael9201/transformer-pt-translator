import argparse
import torch
import tiktoken
import sacrebleu
from tqdm import tqdm

from model import Translator
from utils.dataloader import create_dataloader


def benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device {device}")

    # tokenizer
    enc = tiktoken.get_encoding("o200k_base")
    vocab_size = enc.max_token_value + 2

    # dataloader (validation split)
    dl = create_dataloader(
        args.en_file,
        args.pt_file,
        batch_size=1,
        max_seq_len=args.max_seq_len,
        split="val",
        shuffle=False,
    )

    # init model
    model = Translator(
        emb_dim=args.embed_size,
        vocab_size=vocab_size,
        seq_len=args.max_seq_len,
        eos_token=793,
        bos_token=vocab_size - 1,
        pad_token=0,
    )
    model.to(device)

    print(f"loading model from {args.model_path}")
    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True)
    )
    model.eval()

    targets = []
    predictions = []

    limit = args.limit if args.limit > 0 else len(dl)
    print(f"starting benchmark on {limit} samples")

    for i, (X, y) in enumerate(tqdm(dl, total=limit)):
        if i >= limit:
            break

        X = X.to(device)

        # generate translation
        with torch.no_grad():
            output_tokens = model.translate(
                X, method="greedy", repetition_penalty=args.repetition_penalty
            )

        # decode
        pred_text = enc.decode(output_tokens.tolist())

        # target text (strip BOS and pad)
        target_tokens = y[0].tolist()
        if vocab_size - 1 in target_tokens:  # remove BOS
            target_tokens.remove(vocab_size - 1)
        if 0 in target_tokens:  # remove PAD
            target_tokens = [t for t in target_tokens if t != 0]

        target_text = enc.decode(target_tokens)

        # cleanup pred_text if EOS was reached
        if "\n" in pred_text:
            pred_text = pred_text.split("\n")[0]
        if "\n" in target_text:
            target_text = target_text.split("\n")[0]

        predictions.append(pred_text.strip())
        targets.append([target_text.strip()])

        if args.show_samples and i < 5:
            en_text = enc.decode(X[0].tolist()).strip()
            print(f"\nSample {i + 1}:")
            print(f"  EN: {en_text}")
            print(f"  TARGET: {target_text.strip()}")
            print(f"  PRED: {pred_text.strip()}")

    # calculate BLEU
    bleu = sacrebleu.corpus_bleu(predictions, targets)
    print(f"\nBLEU score: {bleu.score:.2f}")
    print(bleu)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the translator model using BLEU score."
    )
    parser.add_argument(
        "--en_file", type=str, default="data/en.txt", help="Path to English source file"
    )
    parser.add_argument(
        "--pt_file",
        type=str,
        default="data/pt.txt",
        help="Path to Portuguese target file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="translator_model.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--embed_size", type=int, default=128, help="Embedding dimension size"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=128, help="Maximum sequence length"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optionally limit number of samples for benchmark",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="Optional repetition penalty",
    )
    parser.add_argument(
        "--show_samples", action="store_true", help="Show first 5 translation samples"
    )

    args = parser.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
