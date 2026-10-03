import torch
import torch.nn as nn


class DensityAwareMorphologyEncoder(nn.Module):
    """Encode masked cell morphology together with local cell density."""

    def __init__(self, in_ch: int = 3, hidden: int = 128, drop: float = 0.2):
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
        self.morphology_projection = nn.Sequential(
            nn.Linear(128, hidden),
            nn.ReLU(inplace=True),
        )
        self.density_projection = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        morphology = self.morphology_projection(self.net(x).flatten(1))
        density = self.density_projection(torch.log1p(density.clamp_min(0.0)))
        return self.fusion(torch.cat((morphology, density), dim=-1))


# Backward-compatible import name. New checkpoints use the explicit module name
# below and are not expected to be compatible with the previous architecture.
CellEncoderCNN = DensityAwareMorphologyEncoder


class GraphConvolution(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        norm_adj = adj / degree
        return self.linear(torch.bmm(norm_adj, x))


class TopologyEncoderGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, drop: float = 0.2):
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
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden = hidden
        self.density_morphology_encoder = DensityAwareMorphologyEncoder(
            hidden=hidden, drop=dropout
        )
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
        valid_mask: torch.Tensor = None,
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
        if valid_mask is None:
            valid_mask = torch.ones((bsz, n_cells), dtype=torch.bool, device=cell_imgs.device)

        flat_imgs = cell_imgs.reshape(bsz * n_cells, *cell_imgs.shape[2:])
        flat_topology = topo_feat.reshape(bsz * n_cells, -1)
        flat_valid = valid_mask.reshape(-1)
        hc = flat_imgs.new_zeros((bsz * n_cells, self.hidden))
        if flat_valid.any():
            hc[flat_valid] = self.density_morphology_encoder(
                flat_imgs[flat_valid], flat_topology[flat_valid, 3:4]
            )
        ht = self.topo_enc(topo_feat, adjacency).reshape(bsz * n_cells, self.hidden)
        ht = ht * flat_valid.unsqueeze(-1).to(ht.dtype)
        z = self.fusion(hc, ht).reshape(bsz, n_cells, self.hidden)

        valid_pairs = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
        neighbor_adjacency = (adjacency > 0) & valid_pairs
        eye = torch.eye(n_cells, dtype=torch.bool, device=z.device).unsqueeze(0)
        neighbor_adjacency = neighbor_adjacency & ~eye
        neighbor_weights = neighbor_adjacency.to(z.dtype)
        neighbor_count = neighbor_weights.sum(dim=-1, keepdim=True)
        neighborhood_embedding = torch.bmm(neighbor_weights, z) / neighbor_count.clamp_min(1.0)
        neighbor_valid_mask = valid_mask & (neighbor_count.squeeze(-1) > 0)

        out = {
            "protein": self.protein(z),
            "neighbor_logits": self.neigh(neighborhood_embedding),
            "cell_logits": self.cell(z),
            "tissue_logits": self.tissue(z),
            "embedding": z,
            "neighborhood_embedding": neighborhood_embedding,
            "neighbor_valid_mask": neighbor_valid_mask,
        }
        if squeeze_cells:
            out = {k: v.squeeze(1) for k, v in out.items()}
        return out
