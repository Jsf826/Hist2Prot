import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import MetricAverager, protein_pcc, protein_ssim
from Model import Hist2Prot
from utils_dataloader import Hist2ProtPatchDataset


device = torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="demo_data")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--model_path", type=str, default=os.path.join("out", "best_model.pth"))
    p.add_argument("--save_dir", type=str, default="inference")
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--topo_dim", type=int, default=4)
    p.add_argument("--protein_dim", type=int, default=None)
    p.add_argument("--max_cells", type=int, default=256)
    p.add_argument("--cell_size", type=int, default=32)
    p.add_argument("--neighbor_radius", type=float, default=50.0)
    p.add_argument("--metric_grid_size", type=int, default=64)
    p.add_argument("--stain_norm", type=str, default="none", choices=["none", "macenko"])
    p.add_argument("--cell_image_mode", type=str, default="crop", choices=["crop", "mask"])
    p.add_argument("--num_cell_types", type=int, default=8)
    p.add_argument("--num_tissue_types", type=int, default=4)
    p.add_argument("--num_neighbor_types", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--eval", dest="eval", action="store_true", default=True)
    p.add_argument("--no_eval", dest="eval", action="store_false")
    p.add_argument("--gpu", type=int, default=None, help="Legacy single GPU id. Use -1 to force CPU.")
    p.add_argument("--gpus", type=str, default='0,1,2,3,4', help="Comma-separated GPU ids for DataParallel, e.g. 0,1,2. Use -1 to force CPU.")
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


def resolve_device(gpus, gpu):
    requested = parse_gpu_ids(gpus, gpu)
    if any(gpu_id == -1 for gpu_id in requested):
        return torch.device("cpu"), [], "CPU forced by --gpus/-1"
    if not torch.cuda.is_available():
        return torch.device("cpu"), [], "CUDA is not available; falling back to CPU"

    device_count = torch.cuda.device_count()
    valid = []
    invalid = []
    for gpu_id in requested:
        if 0 <= gpu_id < device_count:
            if gpu_id not in valid:
                valid.append(gpu_id)
        else:
            invalid.append(gpu_id)
    note = None
    if invalid:
        note = (
            f"Ignoring unavailable GPU id(s) {invalid}; "
            f"this machine exposes {device_count} CUDA device(s)."
        )
    if valid:
        return torch.device(f"cuda:{valid[0]}"), valid, note
    return (
        torch.device("cpu"),
        [],
        note or f"Requested GPU id(s) {requested}, but none are available; falling back to CPU",
    )


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def load_protein_dim(data_root: str, fallback: int = 18) -> int:
    meta_path = os.path.join(data_root, "Process", "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta["protein_dim"])
    return fallback


def load_protein_names(data_root: str):
    meta_path = os.path.join(data_root, "Process", "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return np.array(meta.get("protein_names", []), dtype=object)
    return np.array([], dtype=object)


def uncollate_string_cells(cell_ids, batch_size: int):
    if batch_size == 1 and cell_ids and isinstance(cell_ids[0], (list, tuple)):
        return [[str(x[0]) for x in cell_ids]]
    return [[str(cell_ids[i][b]) for i in range(len(cell_ids))] for b in range(batch_size)]


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
    model_path = os.path.join(args.data_root, args.model_path)
    save_dir = os.path.join(args.data_root, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    protein_dim = args.protein_dim or load_protein_dim(args.data_root)
    protein_names = load_protein_names(args.data_root)
    dataset = Hist2ProtPatchDataset(
        data_root=args.data_root,
        split=args.split,
        require_labels=args.eval,
        max_cells=args.max_cells,
        cell_size=args.cell_size,
        neighbor_radius=args.neighbor_radius,
        random_sample_cells=False,
        stain_norm=args.stain_norm,
        cell_image_mode=args.cell_image_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = Hist2Prot(
        topo_dim=args.topo_dim,
        protein_dim=protein_dim,
        num_cell_types=args.num_cell_types,
        num_tissue_types=args.num_tissue_types,
        num_neighbor_types=args.num_neighbor_types,
        dropout=args.dropout,
    ).to(device)
    if device.type == "cuda" and len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    unwrap_model(model).load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    buffers = defaultdict(lambda: {
        "cell_id": [],
        "patch_id": [],
        "coords": [],
        "protein": [],
        "cell_type": [],
        "neighbor_type": [],
        "tissue_type": [],
    })
    metrics = MetricAverager()

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"infer:{args.split}"):
            sample_ids = batch["sample_id"]
            patch_ids = batch["patch_id"]
            cell_ids = batch["cell_id"]
            valid_mask = batch["valid_mask"].to(device)
            coords = batch["coords"].cpu().numpy()

            model_in = {
                k: v.to(device)
                for k, v in batch.items()
                if torch.is_tensor(v) and k in ("cell_imgs", "topo_feat", "adjacency")
            }
            out = model(model_in["cell_imgs"], model_in["topo_feat"], model_in["adjacency"])

            if args.eval:
                pcc, _ = protein_pcc(out["protein"], batch["protein_gt"].to(device), valid_mask)
                ssim, _ = protein_ssim(
                    out["protein"],
                    batch["protein_gt"].to(device),
                    batch["coords"].to(device),
                    batch["patch_box"].to(device),
                    valid_mask,
                    grid_size=args.metric_grid_size,
                )
                metrics.update("pcc", pcc)
                metrics.update("ssim", ssim)

            protein = out["protein"].cpu().numpy()
            cell_type = out["cell_logits"].argmax(-1).cpu().numpy()
            neighbor = out["neighbor_logits"].argmax(-1).cpu().numpy()
            tissue = out["tissue_logits"].argmax(-1).cpu().numpy()
            mask = valid_mask.cpu().numpy()
            cell_ids_by_batch = uncollate_string_cells(cell_ids, len(sample_ids))

            for b, sample_id in enumerate(sample_ids):
                sample_id = str(sample_id)
                n_valid = int(mask[b].sum())
                if n_valid == 0:
                    continue
                buf = buffers[sample_id]
                buf["cell_id"].extend(cell_ids_by_batch[b][:n_valid])
                buf["patch_id"].extend([str(patch_ids[b])] * n_valid)
                buf["coords"].append(coords[b, :n_valid])
                buf["protein"].append(protein[b, :n_valid])
                buf["cell_type"].append(cell_type[b, :n_valid])
                buf["neighbor_type"].append(neighbor[b, :n_valid])
                buf["tissue_type"].append(tissue[b, :n_valid])

    for sample_id, buf in buffers.items():
        result = {
            "cell_id": np.array(buf["cell_id"], dtype=object),
            "patch_id": np.array(buf["patch_id"], dtype=object),
            "coords": np.concatenate(buf["coords"], axis=0),
            "protein_names": protein_names,
            "protein": np.concatenate(buf["protein"], axis=0),
            "cell_type": np.concatenate(buf["cell_type"], axis=0),
            "neighbor_type": np.concatenate(buf["neighbor_type"], axis=0),
            "tissue_type": np.concatenate(buf["tissue_type"], axis=0),
        }
        protein_names = load_protein_names(args.data_root)
        if protein_names.size:
            result["protein_names"] = protein_names
        np.savez(os.path.join(save_dir, f"{sample_id}_{args.split}_pred.npz"), **result)

    if args.eval:
        metrics_result = {
            "split": args.split,
            "pcc": metrics.mean("pcc"),
            "ssim": metrics.mean("ssim"),
        }
        np.savez(os.path.join(save_dir, f"{args.split}_metrics.npz"), **metrics_result)
        print(
            f"[{args.split}] PCC: {metrics_result['pcc']:.4f} | "
            f"SSIM: {metrics_result['ssim']:.4f}"
        )


if __name__ == "__main__":
    main()
