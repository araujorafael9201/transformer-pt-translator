import argparse
import torch
import tiktoken

from model import Translator

def translate(args):

    enc = tiktoken.get_encoding("o200k_base")
    vocab_size = enc.max_token_value + 2

    model = Translator(seq_len=args.max_seq_len, vocab_size=vocab_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu'), weights_only=True))
    else:
        model.load_state_dict(torch.load(args.model_path, weights_only=True))

    tokens = model.translate(torch.tensor(enc.encode(args.input_text)).unsqueeze(0))
    return enc.decode(tokens.tolist())

def main():
    parser = argparse.ArgumentParser(description="A script to train and run a Transformer-based translator.")
    
    # Shared arguments
    parser.add_argument('input_text', type=str, help='Input text to be translated')
    parser.add_argument('--max_seq_len', type=int, default=128, help='Maximum length of input data.')
    parser.add_argument('--model_path', type=str, default='translator_model.pth', help='Model file path')

    args = parser.parse_args()
    ret = translate(args)

    print(ret)

if __name__ == "__main__":
    main()

