import argparse
import json
import os
import random
import shutil

import numpy as np
import torch
import yaml
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model import Hist2Prot
from config_utils import flattened_defaults, load_config
from precomputed_dataset import PrecomputedHist2ProtDataset
from utils_dataloader import Hist2ProtPatchDataset, hist2prot_collate


device = torch.device("cpu")


def fix_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config_file", default="configs/hist2prot.json")
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config_file)
    defaults = flattened_defaults(
        config, ("dataset", "data", "model", "training", "evaluation")
    )
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", default=config_args.config_file)
    p.add_argument("--data_root", type=str, default=defaults.get("data_root", "data"))
    p.add_argument("--out_dir", type=str, default=defaults.get("out_dir", "out2"))
    p.add_argument("--batch_size", type=int, default=int(defaults.get("batch_size", 32)))
    p.add_argument("--lr", type=float, default=float(defaults.get("lr", 1e-4)))
    p.add_argument("--epochs", type=int, default=int(defaults.get("epochs", 500)))
    p.add_argument("--save_every", type=int, default=int(defaults.get("save_every", 25)))
    p.add_argument("--num_workers", type=int, default=int(defaults.get("num_workers", 0)))
    p.add_argument("--dropout", type=float, default=float(defaults.get("dropout", 0.2)))
    p.add_argument("--hidden", type=int, default=int(defaults.get("hidden", 128)))
    p.add_argument("--topo_dim", type=int, default=int(defaults.get("topo_dim", 4)))
    p.add_argument("--protein_dim", type=int, default=defaults.get("protein_dim"))
    p.add_argument("--max_cells", type=int, default=int(defaults.get("max_cells", 256)))
    p.add_argument("--cell_size", type=int, default=int(defaults.get("cell_size", 32)))
    p.add_argument("--neighbor_radius", type=float, default=float(defaults.get("neighbor_radius", 50.0)))
    p.add_argument("--metric_grid_size", type=int, default=int(defaults.get("metric_grid_size", 256)))
    p.add_argument("--stain_norm", type=str, default=defaults.get("stain_norm", "none"), choices=["none", "macenko"])
    p.add_argument("--cell_image_mode", type=str, default=defaults.get("cell_image_mode", "mask"), choices=["crop", "mask"])
    p.add_argument("--aux_label_dir", type=str, default=None)
    p.add_argument("--aux_label_suffix", type=str, default="_aux")
    p.add_argument("--zarr_root", type=str, default=defaults.get("zarr_root"))
    p.add_argument(
        "--zarr_cache_size", type=int, default=int(defaults.get("zarr_cache_size", 4))
    )
    p.add_argument(
        "--use_precomputed_zarr", dest="use_precomputed_zarr", action="store_true",
        default=bool(defaults.get("use_precomputed_zarr", True)),
    )
    p.add_argument(
        "--no_precomputed_zarr", dest="use_precomputed_zarr", action="store_false"
    )
    p.add_argument("--num_cell_types", type=int, default=int(defaults.get("num_cell_types", 8)))
    p.add_argument("--num_tissue_types", type=int, default=int(defaults.get("num_tissue_types", 4)))
    p.add_argument("--num_neighbor_types", type=int, default=int(defaults.get("num_neighbor_types", 8)))
    p.add_argument("--lambda_cell", type=float, default=float(defaults.get("lambda_cell", 0.3)))
    p.add_argument("--lambda_tissue", type=float, default=float(defaults.get("lambda_tissue", 0.2)))
    p.add_argument("--lambda_neighbor", type=float, default=float(defaults.get("lambda_neighbor", 0.2)))
    p.add_argument(
        "--use_aux_tasks",
        dest="use_aux_tasks",
        action="store_true",
        default=bool(defaults.get("use_aux_tasks", True)),
    )
    p.add_argument("--no_aux_tasks", dest="use_aux_tasks", action="store_false")
    p.add_argument(
        "--use_cell_task", dest="use_cell_task", action="store_true",
        default=bool(defaults.get("use_cell_task", True)),
    )
    p.add_argument("--no_cell_task", dest="use_cell_task", action="store_false")
    p.add_argument(
        "--use_tissue_task", dest="use_tissue_task", action="store_true",
        default=bool(defaults.get("use_tissue_task", True)),
    )
    p.add_argument("--no_tissue_task", dest="use_tissue_task", action="store_false")
    p.add_argument(
        "--use_neighbor_task", dest="use_neighbor_task", action="store_true",
        default=bool(defaults.get("use_neighbor_task", True)),
    )
    p.add_argument("--no_neighbor_task", dest="use_neighbor_task", action="store_false")
    p.add_argument("--seed", type=int, default=int(defaults.get("seed", 42)))
    p.add_argument("--resume", type=str, default=None, help="Resume from a training checkpoint.")
    p.add_argument("--gpu", type=int, default=0, help="Legacy single GPU id. Use -1 to force CPU.")
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


def masked_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_fn,
    task_name: str,
) -> torch.Tensor:
    if mask.sum() == 0:
        return logits.sum() * 0.0
    selected = target[mask]
    if selected.min() < 0 or selected.max() >= logits.shape[-1]:
        raise ValueError(
            f"Preprocessed {task_name} label IDs must be in [0, {logits.shape[-1] - 1}], "
            f"but observed [{int(selected.min())}, {int(selected.max())}]."
        )
    return loss_fn(logits[mask], selected)


def run_epoch(model, loader, optimizer, args, train: bool):
    model.train(train)
    loss_reg = MSELoss()
    loss_ce = CrossEntropyLoss()
    total = 0.0
    n_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train", disable=not is_main_process()):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(
                batch["cell_imgs"], batch["topo_feat"], batch["adjacency"],
                valid_mask=batch["valid_mask"],
            )
            mask = batch["valid_mask"] & batch["target_mask"]

            loss_main = masked_mse(out["protein"], batch["protein_gt"], mask, loss_reg)
            loss = loss_main
            if args.use_aux_tasks and args.use_cell_task:
                loss_cell = masked_ce(
                    out["cell_logits"], batch["cell_type"], mask, loss_ce, "cell-task"
                )
                loss = loss + args.lambda_cell * loss_cell
            if args.use_aux_tasks and args.use_tissue_task:
                loss_tissue = masked_ce(
                    out["tissue_logits"], batch["tissue_type"], mask, loss_ce, "tissue-task"
                )
                loss = loss + args.lambda_tissue * loss_tissue
            if args.use_aux_tasks and args.use_neighbor_task:
                neighbor_mask = mask & out["neighbor_valid_mask"]
                loss_neighbor = masked_ce(
                    out["neighbor_logits"], batch["neighbor_label"], neighbor_mask, loss_ce,
                    "neighbor-task",
                )
                loss = loss + args.lambda_neighbor * loss_neighbor

            if train:
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss detected: {float(loss.detach().cpu())}. "
                        "Check protein values, auxiliary labels and learning rate."
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total += float(loss.item())
            n_batches += 1

    return {"loss": total / max(n_batches, 1)}


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
    if args.use_precomputed_zarr:
        train_dataset = PrecomputedHist2ProtDataset(
            data_root=args.data_root,
            split="train",
            zarr_root=args.zarr_root,
            cache_size=args.zarr_cache_size,
            require_labels=True,
            use_cell_task=use_cell_task,
            use_tissue_task=use_tissue_task,
            use_neighbor_task=use_neighbor_task,
            expected_cell_size=args.cell_size,
            expected_neighbor_radius=args.neighbor_radius,
            expected_cell_image_mode=args.cell_image_mode,
            expected_stain_norm=args.stain_norm,
            expected_protein_dim=protein_dim,
        )
        print(f"Training from lazy Zarr shards: {train_dataset.manifest_path}")
    else:
        train_dataset = Hist2ProtPatchDataset(
            data_root=args.data_root,
            split="train",
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
        print("Training from raw WSI/mask/CSV fallback (--no_precomputed_zarr).")

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=hist2prot_collate,
        worker_init_fn=seed_worker,
        generator=loader_generator,
        persistent_workers=args.num_workers > 0,
    )

    model = Hist2Prot(
        topo_dim=args.topo_dim,
        protein_dim=protein_dim,
        num_neighbor_types=args.num_neighbor_types,
        num_cell_types=args.num_cell_types,
        num_tissue_types=args.num_tissue_types,
        dropout=args.dropout,
        hidden=args.hidden,
    ).to(device)
    if device.type == "cuda" and len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    optimizer = Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    if args.resume:
        resume_path = args.resume
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(args.data_root, resume_path)
        checkpoint = torch.load(resume_path, map_location=device)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Resume checkpoint must contain model_state_dict and optimizer_state_dict.")
        unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        if start_epoch >= args.epochs:
            raise ValueError(
                f"Checkpoint epoch {start_epoch} is not below requested total epochs {args.epochs}."
            )
        print(f"Resuming from epoch {start_epoch}: {resume_path}")

    for epoch in range(start_epoch, args.epochs):
        train_result = run_epoch(model, train_loader, optimizer, args, train=True)
        train_loss = train_result["loss"]
        print(f"[Epoch {epoch + 1}/{args.epochs}] Train loss: {train_loss:.6f}")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": vars(args).copy(),
                    "protein_dim": protein_dim,
                },
                os.path.join(out_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
            )

    hparam = vars(args).copy()
    hparam["protein_dim"] = protein_dim
    hparam["completed_epochs"] = args.epochs
    hparam["checkpoint_selection"] = "fixed_final_epoch_no_validation"
    with open(os.path.join(out_dir, "hparam.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(hparam, f, sort_keys=False)
    final_checkpoint = {
        "epoch": args.epochs,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": hparam,
        "protein_dim": protein_dim,
    }
    torch.save(final_checkpoint, os.path.join(out_dir, "final_model.pth"))
    for filename in ("metadata.json", "sample_splits.csv", "protein_norm.json"):
        source = os.path.join(args.data_root, "Process", filename)
        if os.path.isfile(source):
            shutil.copyfile(source, os.path.join(out_dir, filename))
    if args.use_precomputed_zarr:
        shutil.copyfile(
            train_dataset.manifest_path,
            os.path.join(out_dir, "zarr_train_manifest.json"),
        )
    config_copy = os.path.join(out_dir, os.path.basename(args.config_file))
    if os.path.abspath(args.config_file) != os.path.abspath(config_copy):
        shutil.copyfile(args.config_file, config_copy)
    train_dataset.close()
    print(f"Saved fixed final-epoch model: {os.path.join(out_dir, 'final_model.pth')}")


if __name__ == "__main__":
    main()
