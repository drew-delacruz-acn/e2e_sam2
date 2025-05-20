# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class ContrastiveDataset(Dataset):
    """Wraps a DataFrame into (embedding, label) tensors."""

    def __init__(self, frame: pd.DataFrame):
        vecs = np.vstack(frame["finetuned_embedding"].values).astype("float32")
        self.vecs = torch.tensor(vecs)
        
        # Create a mapping from string classes to integer indices
        unique_classes = frame["class"].unique()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Convert classes to integer indices
        integer_labels = [self.class_to_idx[cls] for cls in frame["class"].values]
        self.labels = torch.tensor(integer_labels)
        
        # Store the original classes for reference
        self.original_classes = frame["class"].values

    def __getitem__(self, idx):
        return self.vecs[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels) 