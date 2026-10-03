"""Lazy reader for sharded, precomputed Hist2Prot Zarr patches."""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import zarr
except Exception as error:  # pragma: no cover - reported at runtime
    zarr = None
    _ZARR_IMPORT_ERROR = error
else:
    _ZARR_IMPORT_ERROR = None


def default_zarr_root(data_root: str) -> Path:
    return Path(data_root) / "Process" / "zarr"


class PrecomputedHist2ProtDataset(Dataset):
    """Build a light global index and lazily open only recently used shards."""

    def __init__(
        self,
        data_root: str,
        split: str,
        sample_id: Optional[str] = None,
        zarr_root: Optional[str] = None,
        cache_size: int = 4,
        require_labels: bool = True,
        use_cell_task: bool = False,
        use_tissue_task: bool = False,
        use_neighbor_task: bool = False,
        expected_cell_size: Optional[int] = None,
        expected_neighbor_radius: Optional[float] = None,
        expected_cell_image_mode: Optional[str] = None,
        expected_stain_norm: Optional[str] = None,
        expected_protein_dim: Optional[int] = None,
    ):
        if zarr is None:
            raise RuntimeError(
                "zarr is required for precomputed Hist2Prot datasets. "
                f"Original import error: {_ZARR_IMPORT_ERROR}"
            )
        root = Path(zarr_root) if zarr_root else default_zarr_root(data_root)
        if zarr_root and not root.is_absolute():
            root = Path(data_root) / root
        self.split_dir = root / str(split).lower()
        self.manifest_path = self.split_dir / "manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing precomputed Zarr manifest: {self.manifest_path}. "
                "Run precompute_zarr.py before training or use --no_precomputed_zarr."
            )
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if int(self.manifest.get("schema_version", 0)) != 1:
            raise ValueError(f"Unsupported Hist2Prot Zarr schema: {self.manifest_path}")
        expected = {
            "cell_size": expected_cell_size,
            "cell_image_mode": expected_cell_image_mode,
            "stain_norm": expected_stain_norm,
            "protein_dim": expected_protein_dim,
        }
        mismatches = []
        for key, expected_value in expected.items():
            if expected_value is not None and self.manifest.get(key) != expected_value:
                mismatches.append(
                    f"{key}: stored={self.manifest.get(key)!r}, requested={expected_value!r}"
                )
        if expected_neighbor_radius is not None and not np.isclose(
            float(self.manifest.get("neighbor_radius")), float(expected_neighbor_radius)
        ):
            mismatches.append(
                "neighbor_radius: "
                f"stored={self.manifest.get('neighbor_radius')!r}, "
                f"requested={expected_neighbor_radius!r}"
            )
        if mismatches:
            raise ValueError(
                "Precomputed Zarr settings do not match the current configuration. "
                "Re-run precompute_zarr.py or restore the matching config. " + "; ".join(mismatches)
            )

        self.require_labels = bool(require_labels)
        requested_tasks = {
            "cell": bool(use_cell_task),
            "tissue": bool(use_tissue_task),
            "neighbor": bool(use_neighbor_task),
        }
        stored_tasks = self.manifest.get("auxiliary_labels", {})
        missing = [name for name, enabled in requested_tasks.items() if enabled and not stored_tasks.get(name)]
        if missing:
            raise ValueError(
                f"Precomputed split {split!r} does not contain requested auxiliary labels: {missing}. "
                "Re-run precompute_zarr.py with the same multi-task configuration."
            )

        self._cache_size = max(1, int(cache_size))
        self._store_cache: OrderedDict[int, object] = OrderedDict()
        self.stores = []
        self.index = []
        selected_sample = None if sample_id is None else str(sample_id)
        available_samples = set()
        for store_index, info in enumerate(self.manifest.get("stores", [])):
            store_path = self.split_dir / info["path"]
            if not store_path.is_dir():
                raise FileNotFoundError(f"Missing Zarr shard listed in manifest: {store_path}")
            self.stores.append(store_path)
            sample_ids = [str(value) for value in info.get("sample_ids", [])]
            available_samples.update(sample_ids)
            for local_index, patch_sample in enumerate(sample_ids):
                if selected_sample is None or patch_sample == selected_sample:
                    self.index.append((store_index, local_index))
        if selected_sample is not None and selected_sample not in available_samples:
            raise ValueError(f"Patient {selected_sample!r} is not present in precomputed {split!r} data.")
        if not self.index:
            raise ValueError(f"No precomputed patches found for split {split!r}.")

    def __len__(self) -> int:
        return len(self.index)

    def _group(self, store_index: int):
        if store_index in self._store_cache:
            group = self._store_cache.pop(store_index)
            self._store_cache[store_index] = group
            return group
        group = zarr.open_group(str(self.stores[store_index]), mode="r")
        self._store_cache[store_index] = group
        while len(self._store_cache) > self._cache_size:
            self._store_cache.popitem(last=False)
        return group

    def close(self) -> None:
        self._store_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, index: int) -> Dict[str, object]:
        store_index, local_index = self.index[index]
        group = self._group(store_index)
        node_offsets = group["node_offsets"]
        node_start = int(node_offsets[local_index])
        node_end = int(node_offsets[local_index + 1])
        n_nodes = node_end - node_start
        edge_offsets = group["edge_offsets"]
        edge_start = int(edge_offsets[local_index])
        edge_end = int(edge_offsets[local_index + 1])

        adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        if edge_end > edge_start:
            edge_src = np.asarray(group["edge_src"][edge_start:edge_end], dtype=np.int64)
            edge_dst = np.asarray(group["edge_dst"][edge_start:edge_end], dtype=np.int64)
            adjacency[edge_src, edge_dst] = 1.0

        cell_images = np.asarray(group["cell_imgs"][node_start:node_end], dtype=np.float32) / 255.0
        item: Dict[str, object] = {
            "sample_id": str(group["sample_id"][local_index]),
            "patch_id": str(group["patch_id"][local_index]),
            "cell_id": [str(value) for value in group["cell_id"][node_start:node_end]],
            "coords": torch.from_numpy(np.asarray(group["coords"][node_start:node_end], dtype=np.float32)),
            "patch_box": torch.from_numpy(np.asarray(group["patch_box"][local_index], dtype=np.float32)),
            "cell_imgs": torch.from_numpy(cell_images),
            "topo_feat": torch.from_numpy(
                np.asarray(group["topo_feat"][node_start:node_end], dtype=np.float32)
            ),
            "adjacency": torch.from_numpy(adjacency),
            "valid_mask": torch.ones(n_nodes, dtype=torch.bool),
            "target_mask": torch.from_numpy(
                np.asarray(group["target_mask"][node_start:node_end], dtype=np.bool_)
            ),
        }
        if self.require_labels:
            item.update(
                {
                    "protein_gt": torch.from_numpy(
                        np.asarray(group["protein_gt"][node_start:node_end], dtype=np.float32)
                    ),
                    "cell_type": torch.from_numpy(
                        np.asarray(group["cell_type"][node_start:node_end], dtype=np.int64)
                    ),
                    "tissue_type": torch.from_numpy(
                        np.asarray(group["tissue_type"][node_start:node_end], dtype=np.int64)
                    ),
                    "neighbor_label": torch.from_numpy(
                        np.asarray(group["neighbor_label"][node_start:node_end], dtype=np.int64)
                    ),
                }
            )
        return item
