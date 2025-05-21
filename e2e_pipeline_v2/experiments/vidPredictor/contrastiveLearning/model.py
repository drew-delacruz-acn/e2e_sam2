# model.py
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    """2-layer MLP → 128-dim, then ℓ2-normalise (SupCon style)."""

    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=1) 