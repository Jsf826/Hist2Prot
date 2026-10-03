import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_utils import flattened_defaults, load_config
from metrics import patient_marker_metrics
from Model import Hist2Prot
from precomputed_dataset import PrecomputedHist2ProtDataset
from utils_dataloader import Hist2ProtPatchDataset, hist2prot_collate


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config_file", default="configs/hist2prot.json")
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config_file)
    defaults = flattened_defaults(config, "dataset", "model", "data", "evaluation")

    p = argparse.ArgumentParser()
    p.add_argument("--config_file", default=config_args.config_file)
    p.add_argument("--data_root", default=defaults.get("data_root", "data"))
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument(
        "--patient_id",
        default=None,
        help="Run exactly one patient/sample. Recommended for the held-out test set.",
    )
    p.add_argument("--model_path", default=os.path.join("out2", "final_model.pth"))
    p.add_argument("--save_dir", default="inference")
    p.add_argument("--batch_size", type=int, default=int(defaults.get("batch_size", 32)))
    p.add_argument("--num_workers", type=int, default=int(defaults.get("num_workers", 0)))
    p.add_argument("--zarr_root", default=defaults.get("zarr_root"))
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
    p.add_argument("--topo_dim", type=int, default=int(defaults.get("topo_dim", 4)))
    p.add_argument("--protein_dim", type=int, default=defaults.get("protein_dim"))
    p.add_argument("--hidden", type=int, default=int(defaults.get("hidden", 128)))
    p.add_argument("--max_cells", type=int, default=int(defaults.get("max_cells", 256)))
    p.add_argument("--cell_size", type=int, default=int(defaults.get("cell_size", 32)))
    p.add_argument(
        "--neighbor_radius", type=float, default=float(defaults.get("neighbor_radius", 50.0))
    )
    p.add_argument(
        "--metric_grid_size", type=int, default=int(defaults.get("metric_grid_size", 256))
    )
    p.add_argument(
        "--stain_norm", default=defaults.get("stain_norm", "none"), choices=["none", "macenko"]
    )
    p.add_argument(
        "--cell_image_mode", default=defaults.get("cell_image_mode", "mask"), choices=["crop", "mask"]
    )
    p.add_argument("--num_cell_types", type=int, default=int(defaults.get("num_cell_types", 8)))
    p.add_argument("--num_tissue_types", type=int, default=int(defaults.get("num_tissue_types", 4)))
    p.add_argument("--num_neighbor_types", type=int, default=int(defaults.get("num_neighbor_types", 8)))
    p.add_argument("--dropout", type=float, default=float(defaults.get("dropout", 0.2)))
    p.add_argument("--eval", dest="eval", action="store_true", default=True)
    p.add_argument("--no_eval", dest="eval", action="store_false")
    p.add_argument("--gpu", type=int, default=None, help="Legacy single GPU id. Use -1 for CPU.")
    p.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids for DataParallel. Use -1 to force CPU.",
    )
    return p.parse_args()


def parse_gpu_ids(gpus: str, gpu: int) -> List[int]:
    if gpus:
        return [int(value.strip()) for value in gpus.split(",") if value.strip()]
    if gpu is not None:
        return [gpu]
    return [0]


def resolve_device(gpus: str, gpu: int):
    requested = parse_gpu_ids(gpus, gpu)
    if any(gpu_id == -1 for gpu_id in requested):
        return torch.device("cpu"), [], "CPU forced by --gpus/-1"
    if not torch.cuda.is_available():
        return torch.device("cpu"), [], "CUDA is unavailable; falling back to CPU"
    count = torch.cuda.device_count()
    valid = list(dict.fromkeys(gpu_id for gpu_id in requested if 0 <= gpu_id < count))
    invalid = [gpu_id for gpu_id in requested if gpu_id not in valid]
    note = f"Ignoring unavailable GPU id(s) {invalid}; {count} device(s) detected." if invalid else None
    if valid:
        return torch.device(f"cuda:{valid[0]}"), valid, note
    return torch.device("cpu"), [], note or "No requested GPU is available; using CPU"


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def load_metadata(data_root: str) -> Dict:
    path = os.path.join(data_root, "Process", "metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_state(model, model_path: str, device: torch.device) -> None:
    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    unwrap_model(model).load_state_dict(state)


def average_duplicate_cells(buf: Dict[str, list], protein_dim: int) -> Dict[str, np.ndarray]:
    """Merge repeated context/core occurrences without weighting dense regions twice."""
    grouped = defaultdict(lambda: {"coords": [], "protein": [], "protein_gt": []})
    for cell_id, coord, prediction, target in zip(
        buf["cell_id"], buf["coords"], buf["protein"], buf["protein_gt"]
    ):
        item = grouped[str(cell_id)]
        item["coords"].append(coord)
        item["protein"].append(prediction)
        if target is not None:
            item["protein_gt"].append(target)

    cell_ids = sorted(grouped)
    result = {
        "cell_id": np.asarray(cell_ids, dtype=object),
        "coords": np.asarray([np.mean(grouped[c]["coords"], axis=0) for c in cell_ids]),
        "protein": np.asarray([np.mean(grouped[c]["protein"], axis=0) for c in cell_ids]),
    }
    if all(grouped[c]["protein_gt"] for c in cell_ids):
        result["protein_gt"] = np.asarray(
            [np.mean(grouped[c]["protein_gt"], axis=0) for c in cell_ids]
        )
    else:
        result["protein_gt"] = np.empty((0, protein_dim), dtype=np.float32)
    return result


def main() -> None:
    args = parse_args()
    device, device_ids, note = resolve_device(args.gpus, args.gpu)
    if note:
        print(note)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"Running on {device}; GPU ids={device_ids}")

    metadata = load_metadata(args.data_root)
    protein_dim = args.protein_dim or int(metadata.get("protein_dim", 18))
    protein_names = np.asarray(
        metadata.get("protein_names", [f"protein_{index}" for index in range(protein_dim)]),
        dtype=object,
    )
    if len(protein_names) != protein_dim:
        raise ValueError("metadata protein_names length does not match protein_dim")

    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(args.data_root, model_path)
    save_dir = args.save_dir
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(args.data_root, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if args.use_precomputed_zarr:
        dataset = PrecomputedHist2ProtDataset(
            data_root=args.data_root,
            split=args.split,
            sample_id=args.patient_id,
            zarr_root=args.zarr_root,
            cache_size=args.zarr_cache_size,
            require_labels=args.eval,
            expected_cell_size=args.cell_size,
            expected_neighbor_radius=args.neighbor_radius,
            expected_cell_image_mode=args.cell_image_mode,
            expected_stain_norm=args.stain_norm,
            expected_protein_dim=protein_dim,
        )
        print(f"Inference from lazy Zarr shards: {dataset.manifest_path}")
    else:
        dataset = Hist2ProtPatchDataset(
            data_root=args.data_root,
            split=args.split,
            sample_id=args.patient_id,
            require_labels=args.eval,
            max_cells=args.max_cells,
            cell_size=args.cell_size,
            neighbor_radius=args.neighbor_radius,
            random_sample_cells=False,
            stain_norm=args.stain_norm,
            cell_image_mode=args.cell_image_mode,
        )
    if args.patient_id is not None and len(dataset) == 0:
        raise ValueError(f"Patient {args.patient_id!r} is not present in the {args.split} split")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=hist2prot_collate,
        persistent_workers=args.num_workers > 0,
    )

    model = Hist2Prot(
        topo_dim=args.topo_dim,
        protein_dim=protein_dim,
        hidden=args.hidden,
        num_cell_types=args.num_cell_types,
        num_tissue_types=args.num_tissue_types,
        num_neighbor_types=args.num_neighbor_types,
        dropout=args.dropout,
    ).to(device)
    if device.type == "cuda" and len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    load_state(model, model_path, device)
    model.eval()

    buffers = defaultdict(lambda: {"cell_id": [], "coords": [], "protein": [], "protein_gt": []})
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"infer:{args.split}"):
            valid_mask = batch["valid_mask"].to(device)
            target_mask = batch["target_mask"].to(device)
            output = model(
                batch["cell_imgs"].to(device),
                batch["topo_feat"].to(device),
                batch["adjacency"].to(device),
                valid_mask=valid_mask,
            )
            prediction = output["protein"].cpu().numpy()
            coordinates = batch["coords"].cpu().numpy()
            targets = batch["protein_gt"].cpu().numpy() if args.eval else None
            selected = (valid_mask & target_mask).cpu().numpy()

            for batch_index, sample_id in enumerate(batch["sample_id"]):
                keep = np.flatnonzero(selected[batch_index])
                buf = buffers[str(sample_id)]
                ids = batch["cell_id"][batch_index]
                for index in keep:
                    buf["cell_id"].append(str(ids[index]))
                    buf["coords"].append(coordinates[batch_index, index])
                    buf["protein"].append(prediction[batch_index, index])
                    buf["protein_gt"].append(
                        targets[batch_index, index] if targets is not None else None
                    )

    summary_rows = []
    for sample_id, buf in sorted(buffers.items()):
        result = average_duplicate_cells(buf, protein_dim)
        result["protein_names"] = protein_names
        np.savez(os.path.join(save_dir, f"{sample_id}_{args.split}_pred.npz"), **result)

        if args.eval:
            marker_metrics = pd.DataFrame(
                patient_marker_metrics(
                    result["protein"],
                    result["protein_gt"],
                    result["coords"],
                    protein_names,
                    grid_size=args.metric_grid_size,
                )
            )
            marker_metrics.insert(0, "patient_id", sample_id)
            marker_path = os.path.join(save_dir, f"{sample_id}_patient_marker_metrics.csv")
            marker_metrics.to_csv(marker_path, index=False)
            summary = {"patient_id": sample_id, "n_cells": len(result["cell_id"])}
            for metric in ("pcc", "spearman", "ssim", "mse"):
                summary[metric] = float(marker_metrics[metric].mean(skipna=True))
            pd.DataFrame([summary]).to_csv(
                os.path.join(save_dir, f"{sample_id}_patient_summary_metrics.csv"), index=False
            )
            summary_rows.append(summary)
            print(
                f"[{sample_id}] cells={summary['n_cells']} | PCC={summary['pcc']:.4f} | "
                f"Spearman={summary['spearman']:.4f} | SSIM={summary['ssim']:.4f} | "
                f"MSE={summary['mse']:.4f}"
            )

    if args.eval and summary_rows:
        summary_paths = [
            path for path in glob.glob(os.path.join(save_dir, "*_patient_summary_metrics.csv"))
            if os.path.basename(path) not in {
                "train_patient_summary_metrics.csv", "test_patient_summary_metrics.csv"
            }
        ]
        cumulative = pd.concat([pd.read_csv(path) for path in sorted(summary_paths)], ignore_index=True)
        cumulative = cumulative.drop_duplicates("patient_id", keep="last").sort_values("patient_id")
        cumulative.to_csv(
            os.path.join(save_dir, f"{args.split}_patient_summary_metrics.csv"), index=False
        )
    dataset.close()


if __name__ == "__main__":
    main()
