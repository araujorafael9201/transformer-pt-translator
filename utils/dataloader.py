import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import random


class TranslationDataset(Dataset):
    def __init__(
        self,
        en_file_name,
        pt_file_name,
        max_seq_len,
        max_dataset_size=None,
        split="all",
        split_ratio=0.9,
        seed=42,
    ):
        self.enc = tiktoken.get_encoding("o200k_base")
        self.bos_token = self.enc.max_token_value + 1
        self.max_seq_len = max_seq_len

        with open(en_file_name, "r") as en_file:
            en_lines = en_file.readlines()
        with open(pt_file_name, "r") as pt_file:
            pt_lines = pt_file.readlines()

        if max_dataset_size is not None:
            en_lines = en_lines[:max_dataset_size]
            pt_lines = pt_lines[:max_dataset_size]

        assert len(pt_lines) == len(en_lines), (
            "Original and translated datasets have different sizes"
        )

        # Split logic
        indices = list(range(len(en_lines)))
        if split != "all":
            random.Random(seed).shuffle(indices)
            split_idx = int(len(indices) * split_ratio)
            if split == "train":
                indices = indices[:split_idx]
            elif split == "val":
                indices = indices[split_idx:]

        self.en_lines = [en_lines[i] for i in indices]
        self.pt_lines = [pt_lines[i] for i in indices]

    def __len__(self):
        return len(self.en_lines)

    def __getitem__(self, idx):
        en_text = self.en_lines[idx]
        pt_text = self.pt_lines[idx]

        en_tokens = self.enc.encode(en_text)[: self.max_seq_len]
        pt_tokens = [self.bos_token] + self.enc.encode(pt_text)[: self.max_seq_len - 1]

        return torch.tensor(en_tokens, dtype=torch.long), torch.tensor(
            pt_tokens, dtype=torch.long
        )


def create_dataloader(
    en_file_name,
    pt_file_name,
    batch_size,
    max_seq_len,
    max_dataset_size=None,
    shuffle=True,
    num_workers=0,
    split="all",
    split_ratio=0.9,
    seed=42,
):
    dataset = TranslationDataset(
        en_file_name,
        pt_file_name,
        max_seq_len,
        max_dataset_size,
        split=split,
        split_ratio=split_ratio,
        seed=seed,
    )

    def collate_fn(batch):
        en_batch, pt_batch = zip(*batch)
        en_batch_padded = torch.nn.utils.rnn.pad_sequence(
            en_batch, batch_first=True, padding_value=0
        )
        pt_batch_padded = torch.nn.utils.rnn.pad_sequence(
            pt_batch, batch_first=True, padding_value=0
        )
        return en_batch_padded, pt_batch_padded

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return dataloader
