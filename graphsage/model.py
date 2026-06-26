import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # edge_index shape = [2, E]
        src, dst = edge_index

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])

        deg = torch.bincount(dst, minlength=x.size(0)).float().unsqueeze(1)
        deg = deg.clamp(min=1.0)
        agg = agg / deg

        out = self.lin_self(x) + self.lin_neigh(agg)
        return out


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 100, dropout: float = 0.2):
        super().__init__()
        self.conv1 = MeanSAGELayer(in_dim, hidden_dim)
        self.conv2 = MeanSAGELayer(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.normalize(x, p=2, dim=1)
        return x