"""
transformer-from-scratch/data.py

Character-level dataset from tiny Shakespeare.
Downloads ~1MB text on first run, caches locally.

CHARACTER-LEVEL TOKENISATION:
  Vocabulary = set of unique chars in corpus.
  Token = single character → integer index.
  Why char-level? Zero external dependencies. No BPE/SentencePiece needed.
  Downside: longer sequences for same information content vs. BPE.

  Tiny Shakespeare vocab ≈ 65 chars (letters + punctuation + whitespace).
  GPT-2 BPE vocab = 50,257. We use 65. Same attention math; smaller V.
"""

import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def download_shakespeare(path: str = "data/shakespeare.txt") -> str:
    """Download tiny Shakespeare if not already cached."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading tiny Shakespeare → {path}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
        print("Done.")
    return path


class CharDataset(Dataset):
    """
    Sliding window dataset over a character sequence.

    NEXT-TOKEN PREDICTION SETUP:
      Given context x = text[i : i+block_size]
      Predict target y = text[i+1 : i+block_size+1]

      Each position j in the context predicts position j+1.
      This gives T training examples per sample (teacher forcing).

    INDEXING:
      stoi: char → int   (string-to-index)
      itos: int → char   (index-to-string)
    """
    def __init__(self, text: str, block_size: int):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.block_size = block_size

        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

        # Encode full text as integer tensor
        data = [self.stoi[c] for c in text]
        self.data = torch.tensor(data, dtype=torch.long)

        print(f"Dataset: {len(text):,} chars | vocab: {self.vocab_size} | "
              f"block_size: {block_size} | samples: {len(self):,}")

    def __len__(self):
        # Each starting position i from 0 to N-block_size-1 is one sample
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        x = self.data[idx     : idx + self.block_size]      # context
        y = self.data[idx + 1 : idx + self.block_size + 1]  # targets (shifted by 1)
        return x, y

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        return "".join(self.itos[t.item()] for t in tokens)


def get_dataloaders(
    block_size: int = 128,
    batch_size: int = 32,
    train_split: float = 0.9,
    data_path: str = "data/shakespeare.txt",
):
    path = download_shakespeare(data_path)
    text = open(path).read()

    # Train / val split
    n = int(len(text) * train_split)
    train_text = text[:n]
    val_text   = text[n:]

    # Build datasets (vocab built from full text so val has same stoi)
    full_ds    = CharDataset(text, block_size)
    train_ds   = CharDataset(train_text, block_size)
    val_ds     = CharDataset(val_text, block_size)

    # Make val_ds share the same vocab as train_ds
    val_ds.stoi      = train_ds.stoi
    val_ds.itos      = train_ds.itos
    val_ds.vocab_size = train_ds.vocab_size

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, train_ds