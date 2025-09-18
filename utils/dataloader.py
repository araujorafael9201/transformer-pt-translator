import tiktoken
import torch
import torch.nn as nn

class DataLoader:
    def __init__(self, en_file_name, pt_file_name, batch_size, max_seq_len, max_dataset_size, enc=tiktoken.get_encoding("o200k_base")):
        self.bos_token = enc.max_token_value + 1
        self.current_pos = 0
        self.batch_size = batch_size
        with open(en_file_name, "r") as en:
            self.en = [enc.encode(s)[:max_seq_len] for s in en.readlines()[:max_dataset_size]]
        with open(pt_file_name, "r") as pt:
            self.pt = [[self.bos_token] + enc.encode(s)[:max_seq_len-1] for s in pt.readlines()[:max_dataset_size]]

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