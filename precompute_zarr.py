"""Precompute Hist2Prot patch tensors into sharded Zarr stores.

Auxiliary labels are copied verbatim after integer validation by the source
dataset. This script never constructs, clusters, infers, or remaps labels.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
from numcodecs import Blosc
from tqdm import tqdm

try:
    import zarr
except ImportError as error:  # pragma: no cover
    raise ImportError("Install zarr and numcodecs before running precompute_zarr.py") from error

from config_utils import flattened_defaults, load_config
from precomputed_dataset import default_zarr_root
from utils_dataloader import Hist2ProtPatchDataset


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config_file", default="configs/hist2prot.json")
    config_args, _ = config_parser.parse_known_args()
    defaults = flattened_defaults(
        load_config(config_args.config_file), "dataset", "data", "model", "training", "precompute"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=config_args.config_file)
    parser.add_argument("--data_root", default=defaults.get("data_root", "data"))
    parser.add_argument("--zarr_root", default=defaults.get("zarr_root"))
    parser.add_argument("--splits", default="train,test", help="Comma-separated: train,test")
    parser.add_argument(
        "--patches_per_store", type=int, default=int(defaults.get("patches_per_store", 32))
    )
    parser.add_argument("--cell_size", type=int, default=int(defaults.get("cell_size", 32)))
    parser.add_argument("--max_cells", type=int, default=int(defaults.get("max_cells", 256)))
    parser.add_argument(
        "--neighbor_radius", type=float, default=float(defaults.get("neighbor_radius", 50.0))
    )
    parser.add_argument(
        "--stain_norm", default=defaults.get("stain_norm", "none"), choices=["none", "macenko"]
    )
    parser.add_argument(
        "--cell_image_mode", default=defaults.get("cell_image_mode", "mask"),
        choices=["crop", "mask"],
    )
    parser.add_argument("--aux_label_dir", default=None)
    parser.add_argument("--aux_label_suffix", default="_aux")
    parser.add_argument(
        "--use_aux_tasks", dest="use_aux_tasks", action="store_true",
        default=bool(defaults.get("use_aux_tasks", True)),
    )
    parser.add_argument("--no_aux_tasks", dest="use_aux_tasks", action="store_false")
    for task in ("cell", "tissue", "neighbor"):
        parser.add_argument(
            f"--use_{task}_task", dest=f"use_{task}_task", action="store_true",
            default=bool(defaults.get(f"use_{task}_task", True)),
        )
        parser.add_argument(
            f"--no_{task}_task", dest=f"use_{task}_task", action="store_false"
        )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _unicode_array(values) -> np.ndarray:
    values = [str(value) for value in values]
    width = max(1, max((len(value) for value in values), default=1))
    return np.asarray(values, dtype=f"<U{width}")


def _create(group, name: str, data: np.ndarray, compressor, chunks=None) -> None:
    data = np.asarray(data)
    if chunks is None:
        first = max(1, min(int(data.shape[0]), 256)) if data.ndim else None
        chunks = (first,) + data.shape[1:] if data.ndim else None
    group.array(name, data=data, chunks=chunks, compressor=compressor, overwrite=True)


def write_shard(path: Path, items: List[Dict[str, object]]) -> Dict[str, object]:
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    group = zarr.open_group(str(path), mode="w")
    node_counts = [int(item["cell_imgs"].shape[0]) for item in items]
    node_offsets = np.concatenate(([0], np.cumsum(node_counts, dtype=np.int64)))

    edge_src, edge_dst, edge_counts = [], [], []
    for item in items:
        source, target = np.nonzero(item["adjacency"].numpy() > 0)
        edge_src.append(source.astype(np.int32))
        edge_dst.append(target.astype(np.int32))
        edge_counts.append(len(source))
    edge_offsets = np.concatenate(([0], np.cumsum(edge_counts, dtype=np.int64)))

    cell_images = np.concatenate([item["cell_imgs"].numpy() for item in items], axis=0)
    cell_images = np.rint(np.clip(cell_images, 0.0, 1.0) * 255.0).astype(np.uint8)
    arrays = {
        "node_offsets": node_offsets,
        "edge_offsets": edge_offsets,
        "edge_src": np.concatenate(edge_src).astype(np.int32),
        "edge_dst": np.concatenate(edge_dst).astype(np.int32),
        "sample_id": _unicode_array([item["sample_id"] for item in items]),
        "patch_id": _unicode_array([item["patch_id"] for item in items]),
        "patch_box": np.stack([item["patch_box"].numpy() for item in items]).astype(np.float32),
        "cell_id": _unicode_array(
            [cell_id for item in items for cell_id in item["cell_id"]]
        ),
        "coords": np.concatenate([item["coords"].numpy() for item in items]).astype(np.float32),
        "cell_imgs": cell_images,
        "topo_feat": np.concatenate([item["topo_feat"].numpy() for item in items]).astype(np.float32),
        "target_mask": np.concatenate([item["target_mask"].numpy() for item in items]).astype(np.bool_),
        "protein_gt": np.concatenate([item["protein_gt"].numpy() for item in items]).astype(np.float32),
        "cell_type": np.concatenate([item["cell_type"].numpy() for item in items]).astype(np.int64),
        "tissue_type": np.concatenate([item["tissue_type"].numpy() for item in items]).astype(np.int64),
        "neighbor_label": np.concatenate(
            [item["neighbor_label"].numpy() for item in items]
        ).astype(np.int64),
    }
    for name, array in arrays.items():
        if name == "cell_imgs":
            chunks = (min(len(array), 128),) + array.shape[1:]
        elif name in {"edge_src", "edge_dst"}:
            chunks = (max(1, min(len(array), 65536)),)
        else:
            chunks = None
        _create(group, name, array, compressor, chunks=chunks)
    group.attrs.update(
        {
            "schema_version": 1,
            "n_patches": len(items),
            "n_nodes": int(node_offsets[-1]),
            "n_edges": int(edge_offsets[-1]),
        }
    )
    return {
        "path": path.name,
        "n_patches": len(items),
        "n_nodes": int(node_offsets[-1]),
        "sample_ids": [str(item["sample_id"]) for item in items],
    }


def prepare_split(args: argparse.Namespace, split: str, output_root: Path) -> None:
    use_aux = bool(args.use_aux_tasks and split == "train")
    use_cell = bool(use_aux and args.use_cell_task)
    use_tissue = bool(use_aux and args.use_tissue_task)
    use_neighbor = bool(use_aux and args.use_neighbor_task)
    dataset = Hist2ProtPatchDataset(
        data_root=args.data_root,
        split=split,
        require_labels=True,
        max_cells=args.max_cells,
        cell_size=args.cell_size,
        neighbor_radius=args.neighbor_radius,
        random_sample_cells=False,
        use_cell_task=use_cell,
        use_tissue_task=use_tissue,
        use_neighbor_task=use_neighbor,
        aux_label_dir=args.aux_label_dir,
        aux_label_suffix=args.aux_label_suffix,
        stain_norm=args.stain_norm,
        cell_image_mode=args.cell_image_mode,
    )
    split_dir = output_root / split
    if split_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Precomputed split already exists: {split_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True)

    stores, buffer = [], []
    for index in tqdm(range(len(dataset)), desc=f"precompute:{split}"):
        buffer.append(dataset[index])
        if len(buffer) == args.patches_per_store or index + 1 == len(dataset):
            store_path = split_dir / f"patches_{len(stores):05d}.zarr"
            stores.append(write_shard(store_path, buffer))
            buffer.clear()

    metadata_path = Path(args.data_root) / "Process" / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as handle:
        source_metadata = json.load(handle)
    manifest = {
        "schema_version": 1,
        "split": split,
        "n_patches": len(dataset),
        "protein_names": source_metadata.get("protein_names", []),
        "protein_dim": source_metadata.get("protein_dim"),
        "cell_size": args.cell_size,
        "neighbor_radius": args.neighbor_radius,
        "cell_image_mode": args.cell_image_mode,
        "stain_norm": args.stain_norm,
        "image_storage": "uint8_scaled_0_255",
        "auxiliary_labels": {
            "cell": use_cell,
            "tissue": use_tissue,
            "neighbor": use_neighbor,
            "policy": "copied_preprocessed_integer_ids_without_remapping",
        },
        "stores": stores,
    }
    manifest_tmp = split_dir / "manifest.json.tmp"
    with open(manifest_tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    manifest_tmp.replace(split_dir / "manifest.json")
    dataset.close()
    print(f"Precomputed {len(dataset)} {split} patches into {len(stores)} Zarr shard(s).")


def main() -> None:
    args = parse_args()
    if args.patches_per_store <= 0:
        raise ValueError("patches_per_store must be positive.")
    splits = [value.strip().lower() for value in args.splits.split(",") if value.strip()]
    if not splits or any(split not in {"train", "test"} for split in splits):
        raise ValueError("--splits must contain train and/or test.")
    output_root = Path(args.zarr_root) if args.zarr_root else default_zarr_root(args.data_root)
    if not output_root.is_absolute():
        output_root = Path(args.data_root) / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    for split in splits:
        prepare_split(args, split, output_root)


if __name__ == "__main__":
    main()
