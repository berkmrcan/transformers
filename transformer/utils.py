import math
import torch
from torch.utils.data import DataLoader, Dataset


def prep(lines):
    """
    Clean raw text: remove unwanted chars, lowercase, append end token '#'.
    """
    # remove non-alpha/punctuation
    chars = sorted(set("".join(lines)))
    keep = set([c for c in chars if c.isalpha()] + list(" .:,?!;-*+"))
    cleaned = []
    for line in lines:
        for c in line:
            if c not in keep:
                line = line.replace(c, '')
        cleaned.append(line.lower() + '#')
    return cleaned


def build_vocab(lines):
    """Build character-level vocab from cleaned lines."""
    chars = sorted(set("".join(lines)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return chars, stoi, itos


def encode_string(s, stoi):
    """Encode string to list of indices."""
    return [stoi[c] for c in s]


def decode_indices(indices, itos):
    """Decode list of indices to string."""
    return ''.join(itos[i] for i in indices)


class CharDataset(Dataset):
    """Char-level dataset for next-token prediction."""
    def __init__(self, data, context_size):
        # data: list of integer token IDs
        self.data = data
        self.context_size = context_size

    def __len__(self):
        return len(self.data) - self.context_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx : idx + self.context_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1 : idx + 1 + self.context_size], dtype=torch.long)
        return x, y


def get_dataloaders(lines, stoi, context_size, batch_size, split_ratio=0.9):
    """
    Create train/val DataLoaders from cleaned lines.
    """
    # flatten all lines into one sequence
    data = []
    for line in lines:
        data.extend(encode_string(line, stoi))
    n = len(data)
    split = int(n * split_ratio)
    train_data = data[:split]
    val_data = data[split:]

    train_ds = CharDataset(train_data, context_size)
    val_ds = CharDataset(val_data, context_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader