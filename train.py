import os
import torch
import random
import yaml
import numpy as np
import argparse
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--train_list", type=str, default="train_samples.txt")
    p.add_argument("--val_list", type=str, default="val_samples.txt")
    p.add_argument("--out_dir", type=str, default="out")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--topo_dim", type=int, default=4)
    p.add_argument("--protein_dim", type=int, default=15)
    p.add_argument("--num_cell_types", type=int, default=8)
    p.add_argument("--num_tissue_types", type=int, default=4)
    p.add_argument("--num_neighbor_types", type=int, default=8)
    p.add_argument("--lambda_cell", type=float, default=0.3)
    p.add_argument("--lambda_tissue", type=float, default=0.2)
    p.add_argument("--lambda_neighbor", type=float, default=0.2)
    return p.parse_args()


args = parse_args()

data_root = args.data_root
train_list = os.path.join(data_root, args.train_list)
val_list = os.path.join(data_root, args.val_list)
out_dir = os.path.join(data_root, args.out_dir)
os.makedirs(out_dir, exist_ok=True)


batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
patience = args.patience

lambda_cell = args.lambda_cell
lambda_tissue = args.lambda_tissue
lambda_neighbor = args.lambda_neighbor

topo_dim = args.topo_dim
protein_dim = args.protein_dim
num_cell_types = args.num_cell_types
num_tissue_types = args.num_tissue_types
num_neighbor_types = args.num_neighbor_types

# =========================
# Dataset & DataLoader
# =========================
train_dataset = Hist2ProtDataset(
    data_root=data_root,
    split_file=train_list,
    require_labels=True,
)

val_dataset = Hist2ProtDataset(
    data_root=data_root,
    split_file=val_list,
    require_labels=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=args.num_workers
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
    topo_dim=topo_dim,
    protein_dim=protein_dim,
    num_neighbor_types=num_neighbor_types,
    num_cell_types=num_cell_types,
    num_tissue_types=num_tissue_types,
    dropout=args.dropout
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
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        out = model(
            cell_imgs=batch["cell_imgs"],
            topo_feat=batch["topo_feat"]
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
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            out = model(
                cell_imgs=batch["cell_imgs"],
                topo_feat=batch["topo_feat"]
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
        torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
    else:
        early_stop += 1
        if early_stop >= patience:
            print("Early stopping triggered.")
            break


hparam = {
    "lr": lr,
    "batch_size": batch_size,
    "topo_dim": topo_dim,
    "protein_dim": protein_dim,
    "lambda_cell": lambda_cell,
    "lambda_tissue": lambda_tissue,
    "lambda_neighbor": lambda_neighbor
}

with open(os.path.join(out_dir, "hparam.yaml"), "w", encoding="utf-8") as f:
    yaml.dump(hparam, f)
