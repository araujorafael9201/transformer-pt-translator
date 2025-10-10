import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class TranslationDataset(Dataset):
    def __init__(self, en_file_name, pt_file_name, max_seq_len, max_dataset_size=None):
        self.enc = tiktoken.get_encoding("o200k_base")
        self.bos_token = self.enc.max_token_value + 1
        self.max_seq_len = max_seq_len

        with open(en_file_name, "r") as en_file:
            self.en_lines = en_file.readlines()
        with open(pt_file_name, "r") as pt_file:
            self.pt_lines = pt_file.readlines()

        if max_dataset_size is not None:
            self.en_lines = self.en_lines[:max_dataset_size]
            self.pt_lines = self.pt_lines[:max_dataset_size]

        assert len(self.pt_lines) == len(self.en_lines), "Original and translated datasets have different sizes"

    def __len__(self):
        return len(self.en_lines)

    def __getitem__(self, idx):
        en_text = self.en_lines[idx]
        pt_text = self.pt_lines[idx]

        en_tokens = self.enc.encode(en_text)[:self.max_seq_len]
        pt_tokens = [self.bos_token] + self.enc.encode(pt_text)[:self.max_seq_len-1]

        return torch.tensor(en_tokens, dtype=torch.long), torch.tensor(pt_tokens, dtype=torch.long)

def create_dataloader(en_file_name, pt_file_name, batch_size, max_seq_len, max_dataset_size=None, shuffle=True, num_workers=0):
    dataset = TranslationDataset(en_file_name, pt_file_name, max_seq_len, max_dataset_size)

    def collate_fn(batch):
        en_batch, pt_batch = zip(*batch)
        en_batch_padded = torch.nn.utils.rnn.pad_sequence(en_batch, batch_first=True, padding_value=0)
        pt_batch_padded = torch.nn.utils.rnn.pad_sequence(pt_batch, batch_first=True, padding_value=0)
        return en_batch_padded, pt_batch_padded

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader
