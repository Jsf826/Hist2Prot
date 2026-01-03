import os
import torch
import random
import yaml
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, CrossEntropyLoss
from tqdm import tqdm

from utils_dataloader import Hist2ProtDataset
from Model import Hist2Prot


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


data_root = ""
train_list = os.path.join(data_root, "train_samples.txt")
val_list = os.path.join(data_root, "val_samples.txt")
out_dir = os.path.join(data_root, "out")
os.makedirs(out_dir, exist_ok=True)


batch_size = 1024
lr = 1e-4
epochs = 200
patience = 15

lambda_cell = 0.3
lambda_tissue = 0.2
lambda_neighbor = 0.2

protein_dim = 15
num_cell_types = 8
num_tissue_types = 4
num_neighbor_types = 8

# =========================
# Dataset & DataLoader
# =========================
train_dataset = Hist2ProtDataset(
    data_root=data_root,
    split_file=train_list
)

val_dataset = Hist2ProtDataset(
    data_root=data_root,
    split_file=val_list
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
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
    num_neighbor_types=num_neighbor_types,
    dropout=0.3
).to(device)

# =========================
# Optimizer & Loss
# =========================
optimizer = Adam(model.parameters(), lr=lr)

loss_reg = MSELoss()
loss_ce = CrossEntropyLoss()

# =========================
# Training Loop
# =========================
best_val = 1e9
early_stop = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(
            cell_imgs=batch["cell_imgs"],
            adj=batch["adj"],
            cell_type=batch["cell_type"],
            tissue_type=batch["tissue_type"]
        )

        loss_main = loss_reg(out["protein"], batch["protein_gt"])
        loss_cell = loss_ce(out["cell_logits"], batch["cell_type"])
        loss_tissue = loss_ce(out["tissue_logits"], batch["tissue_type"])
        loss_neighbor = loss_ce(out["neighbor_logits"], batch["neighbor_label"])

        loss = (
            loss_main
            + lambda_cell * loss_cell
            + lambda_tissue * loss_tissue
            + lambda_neighbor * loss_neighbor
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # =========================
    # Validation
    # =========================
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(
                cell_imgs=batch["cell_imgs"],
                adj=batch["adj"],
                cell_type=batch["cell_type"],
                tissue_type=batch["tissue_type"]
            )

            loss = loss_reg(out["protein"], batch["protein_gt"])
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"[Epoch {epoch}] Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # =========================
    # Early Stopping
    # =========================
    if val_loss < best_val:
        best_val = val_loss
        early_stop = 0
        torch.save(model.state_dict(), f"{out_dir}/best_model.pth")
    else:
        early_stop += 1
        if early_stop >= patience:
            print("Early stopping triggered.")
            break


hparam = {
    "lr": lr,
    "batch_size": batch_size,
    "protein_dim": protein_dim,
    "lambda_cell": lambda_cell,
    "lambda_tissue": lambda_tissue,
    "lambda_neighbor": lambda_neighbor
}

with open(f"{out_dir}/hparam.yaml", "w") as f:
    yaml.dump(hparam, f)
