import torch
import os
import numpy as np
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils_dataloader import Hist2ProtDataset
from Model import Hist2Prot


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--test_list", type=str, default="test_samples.txt")
    p.add_argument("--model_path", type=str, default=os.path.join("out", "best_model.pth"))
    p.add_argument("--save_dir", type=str, default="inference")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--topo_dim", type=int, default=4)
    p.add_argument("--protein_dim", type=int, default=15)
    p.add_argument("--num_cell_types", type=int, default=8)
    p.add_argument("--num_tissue_types", type=int, default=4)
    p.add_argument("--num_neighbor_types", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    return p.parse_args()


args = parse_args()

data_root = args.data_root
test_list = os.path.join(data_root, args.test_list)
model_path = os.path.join(data_root, args.model_path)
save_dir = os.path.join(data_root, args.save_dir)
os.makedirs(save_dir, exist_ok=True)

protein_dim = args.protein_dim
num_cell_types = args.num_cell_types
num_tissue_types = args.num_tissue_types
num_neighbor_types = args.num_neighbor_types

# =========================
# Dataset
# =========================
test_dataset = Hist2ProtDataset(
    data_root=data_root,
    split_file=test_list,
    require_labels=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_workers
)

# =========================
# Model
# =========================
model = Hist2Prot(
    topo_dim=args.topo_dim,
    protein_dim=protein_dim,
    num_cell_types=num_cell_types,
    num_tissue_types=num_tissue_types,
    num_neighbor_types=num_neighbor_types,
    dropout=args.dropout
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# =========================
# Inference
# =========================
cur_sample = None
buf = {"cell_id": [], "protein": [], "cell_type": [], "neighbor_type": [], "tissue_type": []}

def flush(sample_id: str):
    if sample_id is None or len(buf["cell_id"]) == 0:
        return
    result = {
        "cell_id": np.array(buf["cell_id"], dtype=object),
        "protein": np.stack(buf["protein"], axis=0),
        "cell_type": np.array(buf["cell_type"], dtype=np.int64),
        "neighbor_type": np.array(buf["neighbor_type"], dtype=np.int64),
        "tissue_type": np.array(buf["tissue_type"], dtype=np.int64),
    }
    np.savez(os.path.join(save_dir, f"{sample_id}_pred.npz"), **result)

with torch.no_grad():
    for batch in tqdm(test_loader):
        sample_id = batch["sample_id"][0]
        cell_id = batch["cell_id"][0]

        if cur_sample is None:
            cur_sample = sample_id
        if sample_id != cur_sample:
            flush(cur_sample)
            buf = {"cell_id": [], "protein": [], "cell_type": [], "neighbor_type": [], "tissue_type": []}
            cur_sample = sample_id

        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
            if k not in ("sample_id", "cell_id")
        }

        out = model(
            cell_imgs=batch["cell_imgs"],
            topo_feat=batch["topo_feat"]
        )

        buf["cell_id"].append(cell_id)
        buf["protein"].append(out["protein"].cpu().numpy()[0])
        buf["cell_type"].append(int(out["cell_logits"].argmax(1).cpu().numpy()[0]))
        buf["neighbor_type"].append(int(out["neighbor_logits"].argmax(1).cpu().numpy()[0]))
        buf["tissue_type"].append(int(out["tissue_logits"].argmax(1).cpu().numpy()[0]))

flush(cur_sample)
