# Model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch_geometric.nn import GCNConv


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


class TopologyGCN(nn.Module):
    def __init__(self, in_dim, hidden, drop=0.3):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.drop = drop

    def forward(self, x, edge):
        x = F.relu(self.gcn1(x, edge))
        x = F.dropout(x, self.drop, self.training)
        return self.gcn2(x, edge)


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


class Hist2Prot(L.LightningModule):
    def __init__(self, topo_dim, protein_dim,
                 n_neigh, n_cell, n_tissue, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.cell_enc = CellEncoderCNN()
        self.topo_enc = TopologyGCN(topo_dim, 128)
        self.fusion = AttentionFusion(128)

        self.protein = nn.Linear(128, protein_dim)
        self.neigh = nn.Linear(128, n_neigh)
        self.cell = nn.Linear(128, n_cell)
        self.tissue = nn.Linear(128, n_tissue)

        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x_cell, x_topo, edge):
        hc = self.cell_enc(x_cell)
        ht = self.topo_enc(x_topo, edge)
        z = self.fusion(hc, ht)
        return z

    def training_step(self, b, _):
        z = self(b["X_cell"], b["X_topo"], b["edge_index"])

        loss = (
            self.mse(self.protein(z), b["y_protein"]) +
            self.ce(self.neigh(z), b["y_neighbor"]) +
            self.ce(self.cell(z), b["y_cell_type"]) +
            self.ce(self.tissue(z), b["y_tissue_type"])
        )
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
