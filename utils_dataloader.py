# DataLoader.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Hist2ProtDataset(Dataset):
    def __init__(self, samples, out_folder):
        self.samples = samples
        self.out = out_folder
        self.data = []

        for s in samples:
            df = pd.read_csv(f"{out_folder}/Process/csv/{s}.csv", index_col=0)
            topo = np.load(f"{out_folder}/Process/topology_features/{s}_topo.npy", allow_pickle=True).item()
            edge = np.load(f"{out_folder}/Process/topology_features/{s}_edge.npy")

            for cid in df.index:
                self.data.append((s, cid, edge))

        self.df_cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, cid, edge = self.data[idx]

        if s not in self.df_cache:
            self.df_cache[s] = pd.read_csv(f"{self.out}/Process/csv/{s}.csv", index_col=0)

        df = self.df_cache[s]

        x_cell = np.load(f"{self.out}/Process/image_features/{s}/{cid}.npy")
        x_topo = np.load(f"{self.out}/Process/topology_features/{s}_topo.npy", allow_pickle=True).item()[cid]

        return {
            "X_cell": torch.tensor(x_cell, dtype=torch.float32),
            "X_topo": torch.tensor(x_topo, dtype=torch.float32),
            "edge_index": torch.tensor(edge, dtype=torch.long),
            "y_protein": torch.tensor(df.filter(like="protein").loc[cid].values, dtype=torch.float32),
            "y_neighbor": torch.tensor(df.loc[cid,"neighbor_label"], dtype=torch.long),
            "y_cell_type": torch.tensor(df.loc[cid,"cell_type_id"], dtype=torch.long),
            "y_tissue_type": torch.tensor(df.loc[cid,"region_type_id"], dtype=torch.long),
        }
