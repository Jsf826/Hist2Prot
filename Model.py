import torch
import torch.nn as nn


class CellEncoderCNN(nn.Module):
    def __init__(self, in_ch=3, hidden=128, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, hidden),
            nn.ReLU(),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = self.net(x).squeeze(-1).squeeze(-1)
        return self.fc(x)


class TopologyEncoderMLP(nn.Module):


    def __init__(self, in_dim: int, hidden: int = 128, drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.Tanh(),
            nn.Linear(dim, 2),
            nn.Softmax(1)
        )

    def forward(self, a, b):
        w = self.attn(torch.cat([a,b],1))
        return w[:,0:1]*a + w[:,1:2]*b


class Hist2Prot(nn.Module):
    def __init__(
        self,
        topo_dim: int = 4,
        protein_dim: int = 15,
        num_neighbor_types: int = 8,
        num_cell_types: int = 8,
        num_tissue_types: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.cell_enc = CellEncoderCNN(drop=dropout)
        self.topo_enc = TopologyEncoderMLP(topo_dim, 128, drop=dropout)
        self.fusion = AttentionFusion(128)

        self.protein = nn.Linear(128, protein_dim)
        self.neigh = nn.Linear(128, num_neighbor_types)
        self.cell = nn.Linear(128, num_cell_types)
        self.tissue = nn.Linear(128, num_tissue_types)

    def forward(self, cell_imgs: torch.Tensor, topo_feat: torch.Tensor):
        """
        Args:
            cell_imgs: [B,3,H,W]
            topo_feat: [B,topo_dim]
        Returns:
            dict with keys: protein, neighbor_logits, cell_logits, tissue_logits
        """
        hc = self.cell_enc(cell_imgs)
        ht = self.topo_enc(topo_feat)
        z = self.fusion(hc, ht)

        return {
            "protein": self.protein(z),
            "neighbor_logits": self.neigh(z),
            "cell_logits": self.cell(z),
            "tissue_logits": self.tissue(z),
        }
