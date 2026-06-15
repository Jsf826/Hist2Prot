import torch
import torch.nn as nn


class CellEncoderCNN(nn.Module):
    def __init__(self, in_ch: int = 3, hidden: int = 128, drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).flatten(1)
        return self.fc(x)


class GraphConvolution(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        norm_adj = adj / degree
        return self.linear(torch.bmm(norm_adj, x))


class TopologyEncoderGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, drop: float = 0.3):
        super().__init__()
        self.gcn1 = GraphConvolution(in_dim, hidden)
        self.gcn2 = GraphConvolution(hidden, hidden)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.gcn1(x, adj)))
        return self.act(self.gcn2(x, adj))


class AttentionFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        w = self.attn(torch.cat([a, b], dim=-1))
        return w[..., 0:1] * a + w[..., 1:2] * b


class Hist2Prot(nn.Module):
    def __init__(
        self,
        topo_dim: int = 4,
        protein_dim: int = 18,
        num_neighbor_types: int = 8,
        num_cell_types: int = 8,
        num_tissue_types: int = 4,
        hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden = hidden
        self.cell_enc = CellEncoderCNN(hidden=hidden, drop=dropout)
        self.topo_enc = TopologyEncoderGCN(topo_dim, hidden, drop=dropout)
        self.fusion = AttentionFusion(hidden)

        self.protein = nn.Linear(hidden, protein_dim)
        self.neigh = nn.Linear(hidden, num_neighbor_types)
        self.cell = nn.Linear(hidden, num_cell_types)
        self.tissue = nn.Linear(hidden, num_tissue_types)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        cell_imgs: torch.Tensor,
        topo_feat: torch.Tensor,
        adjacency: torch.Tensor = None,
    ):
        squeeze_cells = False
        if cell_imgs.dim() == 4:
            cell_imgs = cell_imgs.unsqueeze(1)
            topo_feat = topo_feat.unsqueeze(1)
            if adjacency is not None and adjacency.dim() == 2:
                adjacency = adjacency.unsqueeze(0)
            squeeze_cells = True

        bsz, n_cells = cell_imgs.shape[:2]
        if adjacency is None:
            adjacency = torch.eye(n_cells, device=cell_imgs.device).unsqueeze(0).repeat(bsz, 1, 1)

        flat_imgs = cell_imgs.reshape(bsz * n_cells, *cell_imgs.shape[2:])

        hc = self.cell_enc(flat_imgs)
        ht = self.topo_enc(topo_feat, adjacency).reshape(bsz * n_cells, self.hidden)
        z = self.fusion(hc, ht).reshape(bsz, n_cells, self.hidden)

        out = {
            "protein": self.protein(z),
            "neighbor_logits": self.neigh(z),
            "cell_logits": self.cell(z),
            "tissue_logits": self.tissue(z),
            "embedding": z,
        }
        if squeeze_cells:
            out = {k: v.squeeze(1) for k, v in out.items()}
        return out
