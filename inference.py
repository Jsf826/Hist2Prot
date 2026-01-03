import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils_dataloader import Hist2ProtDataset
from Model import Hist2Prot


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_root = ""
test_list = os.path.join(data_root, "test_samples.txt")
model_path = os.path.join(data_root, "out/best_model.pth")
save_dir = os.path.join(data_root, "inference")
os.makedirs(save_dir, exist_ok=True)

protein_dim = 15
num_cell_types = 8
num_tissue_types = 4
num_neighbor_types = 8

# =========================
# Dataset
# =========================
test_dataset = Hist2ProtDataset(
    data_root=data_root,
    split_file=test_list
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

# =========================
# Model
# =========================
model = Hist2Prot(
    protein_dim=protein_dim,
    num_cell_types=num_cell_types,
    num_tissue_types=num_tissue_types,
    num_neighbor_types=num_neighbor_types
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# =========================
# Inference
# =========================
with torch.no_grad():
    for batch in tqdm(test_loader):
        sample_id = batch["sample_id"][0]

        batch = {k: v.to(device) for k, v in batch.items() if k != "sample_id"}

        out = model(
            cell_imgs=batch["cell_imgs"],
            adj=batch["adj"],
            cell_type=batch["cell_type"],
            tissue_type=batch["tissue_type"]
        )

        result = {
            "protein": out["protein"].cpu().numpy(),
            "cell_type": out["cell_logits"].argmax(1).cpu().numpy(),
            "neighbor_type": out["neighbor_logits"].argmax(1).cpu().numpy(),
            "tissue_type": out["tissue_logits"].argmax(1).cpu().numpy()
        }

        np.savez(
            f"{save_dir}/{sample_id}_pred.npz",
            **result
        )
