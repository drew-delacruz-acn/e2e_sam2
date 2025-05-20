# training.py
import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import SupConLoss

from .model import ProjectionHead

def train_supcon(
    dataset,
    in_dim: int,
    epochs: int = 1,
    batch_size: int = 128,
    temperature: float = 0.1,
    lr: float = 1e-3,
):
    """
    Train a projection head using supervised contrastive learning.
    
    Parameters
    ----------
    dataset : Dataset
        A PyTorch dataset that returns (embedding, label) pairs.
    in_dim : int
        Input dimension of the embeddings.
    epochs : int, default=1
        Number of training epochs.
    batch_size : int, default=128
        Batch size for training.
    temperature : float, default=0.1
        Temperature parameter for the contrastive loss.
    lr : float, default=1e-3
        Learning rate for the optimizer.
        
    Returns
    -------
    ProjectionHead
        Trained projection head model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProjectionHead(in_dim).to(device)
    loss_fn = SupConLoss(temperature=temperature)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z = model(x)
            loss = loss_fn(z, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
    return model 