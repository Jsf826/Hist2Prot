import argparse
import json
import os
import random

import numpy as np
import torch
import yaml
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import MetricAverager, protein_pcc, protein_ssim
from Model import Hist2Prot
from utils_dataloader import Hist2ProtPatchDataset


device = torch.device("cpu")


def fix_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="demo_data")
    p.add_argument("--out_dir", type=str, default="out2")
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--topo_dim", type=int, default=4)
    p.add_argument("--protein_dim", type=int, default=18)
    p.add_argument("--max_cells", type=int, default=256)
    p.add_argument("--cell_size", type=int, default=32)
    p.add_argument("--neighbor_radius", type=float, default=50.0)
    p.add_argument("--metric_grid_size", type=int, default=64)
    p.add_argument("--stain_norm", type=str, default="none", choices=["none", "macenko"])
    p.add_argument("--cell_image_mode", type=str, default="mask", choices=["crop", "mask"])
    p.add_argument("--aux_label_dir", type=str, default=None)
    p.add_argument("--aux_label_suffix", type=str, default="_aux")
    p.add_argument("--num_cell_types", type=int, default=8)
    p.add_argument("--num_tissue_types", type=int, default=4)
    p.add_argument("--num_neighbor_types", type=int, default=8)
    p.add_argument("--lambda_cell", type=float, default=0.3)
    p.add_argument("--lambda_tissue", type=float, default=0.2)
    p.add_argument("--lambda_neighbor", type=float, default=0.2)
    p.add_argument("--use_aux_tasks", dest="use_aux_tasks", action="store_true", default=False)
    p.add_argument("--no_aux_tasks", dest="use_aux_tasks", action="store_false")
    p.add_argument("--use_cell_task", dest="use_cell_task", action="store_true", default=True)
    p.add_argument("--no_cell_task", dest="use_cell_task", action="store_false")
    p.add_argument("--use_tissue_task", dest="use_tissue_task", action="store_true", default=True)
    p.add_argument("--no_tissue_task", dest="use_tissue_task", action="store_false")
    p.add_argument("--use_neighbor_task", dest="use_neighbor_task", action="store_true", default=True)
    p.add_argument("--no_neighbor_task", dest="use_neighbor_task", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default='0', help="Legacy single GPU id. Use -1 to force CPU.")
    p.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU ids for DataParallel, e.g. 0,1,2. Use -1 to force CPU.")
    return p.parse_args()


def parse_gpu_ids(gpus, gpu):
    if gpus:
        ids = []
        for raw in gpus.split(","):
            raw = raw.strip()
            if raw:
                ids.append(int(raw))
        return ids
    if gpu is not None:
        return [gpu]
    return [0]


def valid_gpu_ids(requested_ids):
    if not torch.cuda.is_available():
        return [], "CUDA is not available; falling back to CPU"
    device_count = torch.cuda.device_count()
    valid = [idx for idx in requested_ids if 0 <= idx < device_count]
    invalid = [idx for idx in requested_ids if idx not in valid]
    note = None
    if invalid:
        note = (
            f"Ignoring unavailable GPU id(s) {invalid}; "
            f"this machine exposes {device_count} CUDA device(s)."
        )
    if not valid:
        return [], note or "No valid GPU ids were requested; falling back to CPU"
    return valid, note


def is_main_process():
    return True


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def resolve_device(gpus, gpu):
    requested = parse_gpu_ids(gpus, gpu)
    if any(idx == -1 for idx in requested):
        return torch.device("cpu"), [], "CPU forced by --gpus/-1"
    if not torch.cuda.is_available():
        return torch.device("cpu"), [], "CUDA is not available; falling back to CPU"

    valid, note = valid_gpu_ids(requested)
    if not valid:
        return torch.device("cpu"), [], note
    return torch.device(f"cuda:{valid[0]}"), valid, note


def load_protein_dim(data_root: str, fallback: int = 18) -> int:
    meta_path = os.path.join(data_root, "Process", "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta["protein_dim"])
    return fallback


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_fn) -> torch.Tensor:
    valid = mask.unsqueeze(-1).expand_as(pred)
    if valid.sum() == 0:
        return pred.sum() * 0.0
    return loss_fn(pred[valid], target[valid])


def masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_fn) -> torch.Tensor:
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return loss_fn(logits[mask], target[mask])


def run_epoch(model, loader, optimizer, args, train: bool):
    model.train(train)
    loss_reg = MSELoss()
    loss_ce = CrossEntropyLoss()
    total = 0.0
    n_batches = 0
    metrics = MetricAverager()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train" if train else "val", disable=not is_main_process()):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch["cell_imgs"], batch["topo_feat"], batch["adjacency"])
            mask = batch["valid_mask"]

            loss_main = masked_mse(out["protein"], batch["protein_gt"], mask, loss_reg)
            loss = loss_main
            if args.use_aux_tasks and args.use_cell_task:
                loss_cell = masked_ce(out["cell_logits"], batch["cell_type"], mask, loss_ce)
                loss = loss + args.lambda_cell * loss_cell
            if args.use_aux_tasks and args.use_tissue_task:
                loss_tissue = masked_ce(out["tissue_logits"], batch["tissue_type"], mask, loss_ce)
                loss = loss + args.lambda_tissue * loss_tissue
            if args.use_aux_tasks and args.use_neighbor_task:
                loss_neighbor = masked_ce(out["neighbor_logits"], batch["neighbor_label"], mask, loss_ce)
                loss = loss + args.lambda_neighbor * loss_neighbor

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                pcc, _ = protein_pcc(out["protein"], batch["protein_gt"], mask)
                ssim, _ = protein_ssim(
                    out["protein"],
                    batch["protein_gt"],
                    batch["coords"],
                    batch["patch_box"],
                    mask,
                    grid_size=args.metric_grid_size,
                )
                metrics.update("pcc", pcc)
                metrics.update("ssim", ssim)

            total += float(loss.item())
            n_batches += 1

    result = {"loss": total / max(n_batches, 1)}
    if not train:
        result["pcc"] = metrics.mean("pcc")
        result["ssim"] = metrics.mean("ssim")
    return result


def main() -> None:
    global device
    args = parse_args()
    device, device_ids, device_note = resolve_device(args.gpus, args.gpu)
    if device_note:
        print(device_note)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        names = [torch.cuda.get_device_name(i) for i in device_ids]
        if len(device_ids) > 1:
            print(f"Running on GPUs: {device_ids} ({names}) with DataParallel")
        else:
            print(f"Running on GPU: {device} ({names[0]})")
    else:
        print("Running on CPU")
    fix_seed(args.seed)

    out_dir = os.path.join(args.data_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    protein_dim = args.protein_dim or load_protein_dim(args.data_root)
    use_cell_task = bool(args.use_aux_tasks and args.use_cell_task)
    use_tissue_task = bool(args.use_aux_tasks and args.use_tissue_task)
    use_neighbor_task = bool(args.use_aux_tasks and args.use_neighbor_task)
    print(
        "Aux tasks:",
        {
            "cell": use_cell_task,
            "tissue": use_tissue_task,
            "neighbor": use_neighbor_task,
        },
    )
    train_dataset = Hist2ProtPatchDataset(
        data_root=args.data_root,
        split="train",
        require_labels=True,
        max_cells=args.max_cells,
        cell_size=args.cell_size,
        neighbor_radius=args.neighbor_radius,
        random_sample_cells=True,
        use_cell_task=use_cell_task,
        use_tissue_task=use_tissue_task,
        use_neighbor_task=use_neighbor_task,
        aux_label_dir=args.aux_label_dir,
        aux_label_suffix=args.aux_label_suffix,
        stain_norm=args.stain_norm,
        cell_image_mode=args.cell_image_mode,
    )
    val_dataset = Hist2ProtPatchDataset(
        data_root=args.data_root,
        split="val",
        require_labels=True,
        max_cells=args.max_cells,
        cell_size=args.cell_size,
        neighbor_radius=args.neighbor_radius,
        random_sample_cells=False,
        use_cell_task=use_cell_task,
        use_tissue_task=use_tissue_task,
        use_neighbor_task=use_neighbor_task,
        aux_label_dir=args.aux_label_dir,
        aux_label_suffix=args.aux_label_suffix,
        stain_norm=args.stain_norm,
        cell_image_mode=args.cell_image_mode,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = Hist2Prot(
        topo_dim=args.topo_dim,
        protein_dim=protein_dim,
        num_neighbor_types=args.num_neighbor_types,
        num_cell_types=args.num_cell_types,
        num_tissue_types=args.num_tissue_types,
        dropout=args.dropout,
    ).to(device)
    if device.type == "cuda" and len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_pcc = -float("inf")
    best_val_loss = float("inf")
    best_ssim = -float("inf")
    best_epoch = -1
    early_stop = 0
    for epoch in range(args.epochs):
        train_result = run_epoch(model, train_loader, optimizer, args, train=True)
        val_result = run_epoch(model, val_loader, optimizer, args, train=False)
        train_loss = train_result["loss"]
        val_loss = val_result["loss"]
        print(
            f"[Epoch {epoch}] "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Val_PCC: {val_result['pcc']:.4f} | "
            f"Val_SSIM: {val_result['ssim']:.4f}"
        )

        if val_result["pcc"] > best_pcc:
            best_pcc = val_result["pcc"]
            best_val_loss = val_loss
            best_ssim = val_result["ssim"]
            best_epoch = epoch
            early_stop = 0
            torch.save(unwrap_model(model).state_dict(), os.path.join(out_dir, "best_model.pth"))
        else:
            early_stop += 1
            if early_stop >= args.patience:
                print("Early stopping triggered.")
                break

    print(
        "Best validation result "
        f"(selected by PCC): Epoch {best_epoch} | "
        f"Val_PCC: {best_pcc:.4f} | "
        f"Val_SSIM: {best_ssim:.4f} | "
        f"Val_Loss: {best_val_loss:.4f}"
    )

    hparam = vars(args).copy()
    hparam["protein_dim"] = protein_dim
    hparam["best_epoch"] = best_epoch
    hparam["best_val_pcc"] = best_pcc
    hparam["best_val_ssim"] = best_ssim
    hparam["best_val_loss"] = best_val_loss
    hparam["best_metric"] = "Val_PCC"
    with open(os.path.join(out_dir, "hparam.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(hparam, f, sort_keys=False)


if __name__ == "__main__":
    main()
