# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class ContrastiveDataset(Dataset):
    """Wraps a DataFrame into (embedding, label) tensors."""

    def __init__(self, frame: pd.DataFrame):
        vecs = np.vstack(frame["embedding"].values).astype("float32")
        self.vecs = torch.tensor(vecs)
        self.labels = torch.tensor(frame["class"].values.astype("int64"))

    def __getitem__(self, idx):
        return self.vecs[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels) 