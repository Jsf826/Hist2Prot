import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _read_split_file(path: str) -> List[str]:

    samples: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            samples.append(line)
    return samples


class Hist2ProtDataset(Dataset):

    def __init__(
        self,
        data_root: Optional[str] = None,
        split_file: Optional[str] = None,
        samples: Optional[List[str]] = None,
        out_folder: Optional[str] = None,
        require_labels: bool = True,
    ):

        if data_root is None:
            data_root = out_folder
        if data_root is None:
            raise ValueError("Please provide `data_root` (or legacy `out_folder`).")

        self.data_root = data_root
        self.process_dir = os.path.join(self.data_root, "Process")
        self.require_labels = require_labels

        if samples is None:
            if split_file is None:
                raise ValueError("Please provide `split_file` or `samples`.")
            samples = _read_split_file(split_file)

        # (sample_id, cell_id)
        self.data: List[Tuple[str, str]] = []
        for s in samples:
            df = pd.read_csv(os.path.join(self.process_dir, "csv", f"{s}.csv"), index_col=0)
            for cid in df.index.astype(str):
                self.data.append((s, cid))

        self.data.sort(key=lambda x: (x[0], x[1]))

        # cache
        self.df_cache: Dict[str, pd.DataFrame] = {}
        self.topo_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.data)

    def _get_df(self, sample_id: str) -> pd.DataFrame:
        if sample_id not in self.df_cache:
            self.df_cache[sample_id] = pd.read_csv(
                os.path.join(self.process_dir, "csv", f"{sample_id}.csv"),
                index_col=0,

            self.df_cache[sample_id].index = self.df_cache[sample_id].index.astype(str)
        return self.df_cache[sample_id]

    def _get_topo(self, sample_id: str) -> Dict[str, np.ndarray]:
        if sample_id not in self.topo_cache:
            topo = np.load(
                os.path.join(self.process_dir, "topology_features", f"{sample_id}_topo.npy"),
                allow_pickle=True,
            ).item()

            self.topo_cache[sample_id] = {str(k): v for k, v in topo.items()}
        return self.topo_cache[sample_id]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_id, cell_id = self.data[idx]
        topo = self._get_topo(sample_id)

        x_cell = np.load(os.path.join(self.process_dir, "image_features", sample_id, f"{cell_id}.npy"))
        x_topo = topo[cell_id]

        item: Dict[str, torch.Tensor] = {
            "sample_id": sample_id,
            "cell_id": cell_id,
            "cell_imgs": torch.tensor(x_cell, dtype=torch.float32),
            "topo_feat": torch.tensor(x_topo, dtype=torch.float32),
        }

        if self.require_labels:
            df = self._get_df(sample_id)
            item.update(
                {
                    "protein_gt": torch.tensor(
                        df.filter(like="protein").loc[cell_id].values, dtype=torch.float32
                    ),
                    "neighbor_label": torch.tensor(df.loc[cell_id, "neighbor_label"], dtype=torch.long),
                    "cell_type": torch.tensor(df.loc[cell_id, "cell_type_id"], dtype=torch.long),
                    "tissue_type": torch.tensor(df.loc[cell_id, "region_type_id"], dtype=torch.long),
                }
            )
        return item
