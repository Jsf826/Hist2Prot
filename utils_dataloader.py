import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


MACENKO_TARGET_STAIN_MATRIX = np.array(
    [
        [0.5626, 0.2159],
        [0.7201, 0.8012],
        [0.4062, 0.5581],
    ],
    dtype=np.float32,
)
MACENKO_TARGET_CONCENTRATION = np.array([1.9705, 1.0308], dtype=np.float32)


def _read_patch_ids(path: str) -> List[str]:
    ids: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    return ids


def _ensure_hwc_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    elif img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)
    if img.ndim == 3 and img.shape[-1] > 3:
        img = img[..., :3]
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected an RGB-like image array, got shape {img.shape}")
    return img


def macenko_stain_normalize(
    img: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.15,
    io: float = 240.0,
) -> np.ndarray:
    img = _ensure_hwc_rgb(img)
    img = img.astype(np.float32)
    if img.max() <= 1.0:
        img = img * 255.0

    flat = img.reshape((-1, 3))
    od = -np.log((flat + 1.0) / io)
    od_hat = od[(od > beta).all(axis=1)]
    if od_hat.shape[0] < 10:
        return img.astype(np.uint8)

    _, _, vh = np.linalg.svd(od_hat, full_matrices=False)
    stain_matrix = vh[:2].T
    if stain_matrix[0, 0] < 0:
        stain_matrix[:, 0] *= -1
    if stain_matrix[0, 1] < 0:
        stain_matrix[:, 1] *= -1

    projected = np.dot(od_hat, stain_matrix)
    phi = np.arctan2(projected[:, 1], projected[:, 0])
    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100.0 - alpha)
    v1 = np.dot(stain_matrix, np.array([np.cos(min_phi), np.sin(min_phi)]))
    v2 = np.dot(stain_matrix, np.array([np.cos(max_phi), np.sin(max_phi)]))
    he = np.array([v1, v2]).T
    he = he / np.linalg.norm(he, axis=0, keepdims=True).clip(min=1e-8)
    if he[0, 0] < he[0, 1]:
        he = he[:, [1, 0]]

    concentrations, _, _, _ = np.linalg.lstsq(he, od.T, rcond=None)
    max_concentration = np.percentile(concentrations, 99, axis=1).clip(min=1e-8)
    concentrations = concentrations / max_concentration[:, None]
    concentrations = concentrations * MACENKO_TARGET_CONCENTRATION[:, None]

    normalized_od = np.dot(MACENKO_TARGET_STAIN_MATRIX, concentrations)
    normalized = io * np.exp(-normalized_od)
    normalized = normalized.T.reshape(img.shape)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def crop_cell_from_patch(
    patch_img: np.ndarray,
    x: float,
    y: float,
    x0: int,
    y0: int,
    cell_size: int,
) -> np.ndarray:
    half = cell_size // 2
    lx = int(round(x - x0))
    ly = int(round(y - y0))
    h, w = patch_img.shape[:2]

    sx0, sx1 = max(0, lx - half), min(w, lx + half)
    sy0, sy1 = max(0, ly - half), min(h, ly + half)
    crop = patch_img[sy0:sy1, sx0:sx1]

    out = np.zeros((cell_size, cell_size, 3), dtype=np.float32)
    dy = max(0, half - ly)
    dx = max(0, half - lx)
    out[dy:dy + crop.shape[0], dx:dx + crop.shape[1]] = crop.astype(np.float32)
    out /= 255.0 if out.max() > 1.0 else 1.0
    return np.transpose(out, (2, 0, 1))


def crop_cell_with_mask_from_patch(
    patch_img: np.ndarray,
    patch_mask: np.ndarray,
    cell_id: str,
    x: float,
    y: float,
    x0: int,
    y0: int,
    cell_size: int,
) -> np.ndarray:
    half = cell_size // 2
    lx = int(round(x - x0))
    ly = int(round(y - y0))
    h, w = patch_img.shape[:2]

    sx0, sx1 = max(0, lx - half), min(w, lx + half)
    sy0, sy1 = max(0, ly - half), min(h, ly + half)
    crop = patch_img[sy0:sy1, sx0:sx1]
    mask_crop = patch_mask[sy0:sy1, sx0:sx1]

    out = np.zeros((cell_size, cell_size, 3), dtype=np.float32)
    out_mask = np.zeros((cell_size, cell_size), dtype=bool)
    dy = max(0, half - ly)
    dx = max(0, half - lx)
    out[dy:dy + crop.shape[0], dx:dx + crop.shape[1]] = crop.astype(np.float32)

    try:
        numeric_cell_id = int(float(cell_id))
        local_mask = mask_crop == numeric_cell_id
    except ValueError:
        local_mask = mask_crop.astype(str) == str(cell_id)

    if not local_mask.any() and 0 <= ly < patch_mask.shape[0] and 0 <= lx < patch_mask.shape[1]:
        center_label = patch_mask[ly, lx]
        if center_label != 0:
            local_mask = mask_crop == center_label

    out_mask[dy:dy + mask_crop.shape[0], dx:dx + mask_crop.shape[1]] = local_mask

    if out_mask.any():
        out[~out_mask] = 0.0
    out /= 255.0 if out.max() > 1.0 else 1.0
    return np.transpose(out, (2, 0, 1))


def build_topology(coords: np.ndarray, radius: float) -> np.ndarray:
    n = coords.shape[0]
    topo = np.zeros((n, 4), dtype=np.float32)
    if n == 0:
        return topo

    for i in range(n):
        dist = np.sqrt(((coords - coords[i]) ** 2).sum(axis=1))
        neigh = np.where((dist > 0) & (dist <= radius))[0]
        neigh_dist = dist[neigh]
        degree = float(len(neigh))
        mean_dist = float(neigh_dist.mean()) if degree > 0 else 0.0
        min_dist = float(neigh_dist.min()) if degree > 0 else 0.0
        density = degree / (np.pi * radius * radius)
        topo[i] = np.array(
            [
                degree,
                mean_dist,
                min_dist,
                density,
            ],
            dtype=np.float32,
        )
    return topo


def build_adjacency(coords: np.ndarray, radius: float, max_cells: int) -> np.ndarray:
    adj = np.zeros((max_cells, max_cells), dtype=np.float32)
    n = coords.shape[0]
    if n == 0:
        return adj
    for i in range(n):
        dist = np.sqrt(((coords - coords[i]) ** 2).sum(axis=1))
        neigh = np.where((dist > 0) & (dist <= radius))[0]
        adj[i, neigh] = 1.0
    adj[:n, :n] = np.maximum(adj[:n, :n], adj[:n, :n].T)
    adj[:n, :n] += np.eye(n, dtype=np.float32)
    return adj


class WsiPatchReader:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self._array = None

    def _load(self) -> np.ndarray:
        if self._array is None:
            try:
                import tifffile

                self._array = tifffile.memmap(self.image_path)
            except Exception:
                from skimage import io

                self._array = io.imread(self.image_path)
        return self._array

    def read_region(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        arr = self._load()
        patch = np.asarray(arr[y0:y1, x0:x1])
        return _ensure_hwc_rgb(patch)

    def close(self) -> None:
        array = self._array
        if array is not None:
            mmap_handle = getattr(array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        self._array = None


class Hist2ProtPatchDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        split_file: Optional[str] = None,
        require_labels: bool = True,
        max_cells: int = 256,
        cell_size: int = 32,
        neighbor_radius: float = 50.0,
        random_sample_cells: bool = False,
        use_cell_task: bool = False,
        use_tissue_task: bool = False,
        use_neighbor_task: bool = False,
        aux_label_dir: Optional[str] = None,
        aux_label_suffix: str = "_aux",
        stain_norm: str = "none",
        cell_image_mode: str = "crop",
        sample_id: Optional[str] = None,
    ):
        self.data_root = data_root
        self.process_dir = os.path.join(data_root, "Process")
        self.require_labels = require_labels
        self.max_cells = max_cells
        self.cell_size = cell_size
        self.neighbor_radius = neighbor_radius
        self.random_sample_cells = random_sample_cells
        self.use_cell_task = use_cell_task
        self.use_tissue_task = use_tissue_task
        self.use_neighbor_task = use_neighbor_task
        self.aux_label_dir = aux_label_dir
        self.aux_label_suffix = aux_label_suffix
        self.stain_norm = stain_norm.lower()
        if self.stain_norm not in ("none", "macenko"):
            raise ValueError("stain_norm must be one of: none, macenko")
        self.cell_image_mode = cell_image_mode.lower()
        if self.cell_image_mode not in ("crop", "mask"):
            raise ValueError("cell_image_mode must be one of: crop, mask")

        self.samples = pd.read_csv(os.path.join(self.process_dir, "samples.csv"))
        self.samples["sample_id"] = self.samples["sample_id"].astype(str)
        self.samples = self.samples.set_index("sample_id")

        patch_csv = os.path.join(self.process_dir, "patches", f"{split}_patches.csv")
        self.patches = pd.read_csv(patch_csv)
        self.patches["sample_id"] = self.patches["sample_id"].astype(str)
        self.patches["patch_id"] = self.patches["patch_id"].astype(str)

        if split_file is not None:
            patch_ids = set(_read_patch_ids(split_file))
            self.patches = self.patches[self.patches["patch_id"].isin(patch_ids)].copy()
        if sample_id is not None:
            sample_id = str(sample_id)
            if sample_id not in self.samples.index:
                raise ValueError(f"Unknown sample_id {sample_id!r}.")
            split_samples = set(self.patches["sample_id"].astype(str))
            if sample_id not in split_samples:
                raise ValueError(f"Sample {sample_id!r} is not assigned to split {split!r}.")
            self.patches = self.patches[self.patches["sample_id"] == sample_id].copy()

        self.patches = self.patches.reset_index(drop=True)
        self.df_cache: Dict[str, pd.DataFrame] = {}
        self.spatial_cache: Dict[str, tuple] = {}
        self.aux_cache: Dict[str, pd.DataFrame] = {}
        self.mask_cache: Dict[str, np.ndarray] = {}
        self.reader_cache: Dict[str, WsiPatchReader] = {}
        self.protein_cols = self._load_protein_cols()

        if len(self.patches) == 0:
            raise ValueError(f"No patches found for split '{split}'.")

    def __len__(self) -> int:
        return len(self.patches)

    def close(self) -> None:
        for reader in self.reader_cache.values():
            reader.close()
        self.reader_cache.clear()
        for mask in self.mask_cache.values():
            mmap_handle = getattr(mask, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        self.mask_cache.clear()
        self.df_cache.clear()
        self.spatial_cache.clear()
        self.aux_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _load_protein_cols(self) -> List[str]:
        meta_path = os.path.join(self.process_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "protein_names" in meta:
                return [str(c) for c in meta["protein_names"]]
        first_sample = str(self.samples.index[0])
        csv_path = self.samples.loc[first_sample, "csv_path"]
        columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
        protein_cols = [c for c in columns if c.startswith("protein_")]
        if protein_cols:
            return sorted(protein_cols, key=lambda c: int(c.split("_")[1]))
        return [c for c in columns if c.endswith("_intensity_mean")]

    def _get_df(self, sample_id: str) -> pd.DataFrame:
        if sample_id not in self.df_cache:
            csv_path = self.samples.loc[sample_id, "csv_path"]
            df = pd.read_csv(csv_path, index_col=0)
            df.index = df.index.astype(str)
            self.df_cache[sample_id] = df
            x_values = df["x"].to_numpy(dtype=np.float64, copy=False)
            order = np.argsort(x_values, kind="stable")
            self.spatial_cache[sample_id] = (order, x_values[order])
        return self.df_cache[sample_id]

    def _select_region(
        self, sample_id: str, x0: float, y0: float, x1: float, y1: float
    ) -> pd.DataFrame:
        """Select a rectangular region without scanning a million-row table."""
        df = self._get_df(sample_id)
        order, sorted_x = self.spatial_cache[sample_id]
        left = int(np.searchsorted(sorted_x, x0, side="left"))
        right = int(np.searchsorted(sorted_x, x1, side="left"))
        candidates = order[left:right]
        candidate_y = df["y"].to_numpy(dtype=np.float64, copy=False)[candidates]
        candidates = candidates[(candidate_y >= y0) & (candidate_y < y1)]
        # Restore source CSV order for deterministic graph/node ordering.
        candidates.sort()
        return df.iloc[candidates].copy()

    def _has_active_aux_task(self) -> bool:
        return self.require_labels and (
            self.use_cell_task or self.use_tissue_task or self.use_neighbor_task
        )

    def _aux_path(self, sample_id: str) -> str:
        if self.aux_label_dir is not None:
            base_dir = self.aux_label_dir
        else:
            base_dir = os.path.join(self.process_dir, "csv")
        return os.path.join(base_dir, f"{sample_id}{self.aux_label_suffix}.csv")

    @staticmethod
    def _index_aux_df(df: pd.DataFrame) -> pd.DataFrame:
        if "cell_id" in df.columns:
            df = df.set_index("cell_id")
        df.index = df.index.astype(str)
        return df

    def _get_aux_df(self, sample_id: str) -> pd.DataFrame:
        if sample_id in self.aux_cache:
            return self.aux_cache[sample_id]

        path = self._aux_path(sample_id)
        if not os.path.exists(path):
            if self._has_active_aux_task():
                raise FileNotFoundError(
                    f"Auxiliary task labels were requested, but label file was not found: {path}. "
                    "Expected one CSV per WSI in the same directory as the protein CSV by default, "
                    "for example Process/csv/A02_aux.csv, with a cell_id column."
                )
            df = pd.DataFrame()
        else:
            df = self._index_aux_df(pd.read_csv(path))

        self.aux_cache[sample_id] = df
        return df

    @staticmethod
    def _preprocessed_integer_labels(series: pd.Series, task_name: str) -> np.ndarray:
        """Read precomputed class IDs without constructing or remapping labels."""
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any():
            invalid = series.loc[numeric.isna()].astype(str).unique().tolist()
            raise ValueError(
                f"{task_name} labels must be preprocessed numeric class IDs; "
                f"found non-numeric values: {invalid[:10]}"
            )
        values = numeric.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{task_name} labels contain non-finite values.")
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{task_name} labels must be integer class IDs.")
        if (values < 0).any():
            raise ValueError(f"{task_name} labels must be non-negative class IDs.")
        return values.astype(np.int64)

    @staticmethod
    def _resolve_column(df: pd.DataFrame, candidates: List[str], task_name: str) -> str:
        for col in candidates:
            if col in df.columns:
                return col
        raise KeyError(
            f"Missing required label column for {task_name}. "
            f"Expected one of {candidates}, available columns: {df.columns.tolist()}"
        )

    def _get_reader(self, sample_id: str) -> WsiPatchReader:
        if sample_id not in self.reader_cache:
            image_path = self.samples.loc[sample_id, "image_path"]
            self.reader_cache[sample_id] = WsiPatchReader(image_path)
        return self.reader_cache[sample_id]

    def _get_cell_mask(self, sample_id: str) -> np.ndarray:
        if sample_id not in self.mask_cache:
            mask_path = self.samples.loc[sample_id, "cell_mask_path"]
            self.mask_cache[sample_id] = np.load(mask_path, mmap_mode="r")
        return self.mask_cache[sample_id]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.patches.iloc[idx]
        sample_id = str(row["sample_id"])
        patch_id = str(row["patch_id"])
        x0, y0, x1, y1 = (int(row[c]) for c in ("x0", "y0", "x1", "y1"))
        cx0 = int(row.get("context_x0", x0))
        cy0 = int(row.get("context_y0", y0))
        cx1 = int(row.get("context_x1", x1))
        cy1 = int(row.get("context_y1", y1))

        patch_df = self._select_region(sample_id, cx0, cy0, cx1, cy1)
        target_np = (
            (patch_df["x"].to_numpy() >= x0)
            & (patch_df["x"].to_numpy() < x1)
            & (patch_df["y"].to_numpy() >= y0)
            & (patch_df["y"].to_numpy() < y1)
        )
        target_count = int(target_np.sum())
        if target_count > self.max_cells:
            raise RuntimeError(
                f"Dense patch {patch_id} has {target_count} target cells, exceeding "
                f"max_cells={self.max_cells}. No cells were dropped. Re-run Data_Process.py "
                "with the same --max_target_cells so dense regions are adaptively divided."
            )

        n = len(patch_df)
        aux_patch = None
        if self._has_active_aux_task():
            aux_df = self._get_aux_df(sample_id)
            aux_patch = aux_df.reindex(patch_df.index)
            if aux_patch.isna().all(axis=None):
                raise KeyError(
                    f"No matching auxiliary labels found for patch {patch_id}. "
                    "Check that the aux CSV cell_id values match the protein CSV index/cell_id."
                )

        protein_cols = self.protein_cols
        missing_protein_cols = [c for c in protein_cols if c not in patch_df.columns]
        if missing_protein_cols:
            numbered_cols = [c for c in patch_df.columns if c.startswith("protein_")]
            numbered_cols = sorted(numbered_cols, key=lambda c: int(c.split("_")[1]))
            if len(numbered_cols) == len(protein_cols):
                raise KeyError(
                    "CSV contains legacy protein_0/protein_1 columns, but metadata expects "
                    "real protein names. Re-run preprocessing into the same data_root used by "
                    "training, and use the same --data_root for every stage. Missing columns: "
                    f"{missing_protein_cols}"
                )
            raise KeyError(
                "CSV protein columns do not match Process/metadata.json. Missing columns: "
                f"{missing_protein_cols}"
            )

        patch_img = self._get_reader(sample_id).read_region(cx0, cy0, cx1, cy1)
        if self.stain_norm == "macenko":
            patch_img = macenko_stain_normalize(patch_img)
        patch_mask = None
        if self.cell_image_mode == "mask":
            patch_mask = np.asarray(self._get_cell_mask(sample_id)[cy0:cy1, cx0:cx1])

        cell_imgs = np.zeros((n, 3, self.cell_size, self.cell_size), dtype=np.float32)
        topo_feat = np.zeros((n, 4), dtype=np.float32)
        adjacency = np.zeros((n, n), dtype=np.float32)
        protein_gt = np.zeros((n, len(protein_cols)), dtype=np.float32)
        cell_type = np.zeros((n,), dtype=np.int64)
        tissue_type = np.zeros((n,), dtype=np.int64)
        neighbor_label = np.zeros((n,), dtype=np.int64)
        valid_mask = np.ones((n,), dtype=np.bool_)
        target_mask = target_np.astype(np.bool_, copy=False)

        cell_ids: List[str] = patch_df.index.astype(str).tolist()
        coords_out = np.zeros((n, 2), dtype=np.float32)

        if n > 0:
            coords = patch_df[["x", "y"]].to_numpy(dtype=np.float32)

            topo = build_topology(coords, self.neighbor_radius)
            adjacency = build_adjacency(coords, self.neighbor_radius, n)

            for i, (cell_id, cell) in enumerate(patch_df.iterrows()):
                if self.cell_image_mode == "mask":
                    cell_imgs[i] = crop_cell_with_mask_from_patch(
                        patch_img,
                        patch_mask,
                        str(cell_id),
                        cell["x"],
                        cell["y"],
                        cx0,
                        cy0,
                        self.cell_size,
                    )
                else:
                    cell_imgs[i] = crop_cell_from_patch(
                        patch_img, cell["x"], cell["y"], cx0, cy0, self.cell_size
                    )
            topo_feat[:n] = topo
            if protein_cols:
                protein_gt[:n] = patch_df[protein_cols].to_numpy(dtype=np.float32)
            if aux_patch is not None and self.use_cell_task:
                col = self._resolve_column(aux_patch, ["cell_type_id", "cell_type"], "cell")
                if aux_patch.loc[target_mask, col].isna().any():
                    raise ValueError(f"Missing cell-task labels for patch {patch_id}.")
                cell_type[target_mask] = self._preprocessed_integer_labels(
                    aux_patch.loc[target_mask, col], "cell-task"
                )
            if aux_patch is not None and self.use_tissue_task:
                col = self._resolve_column(
                    aux_patch,
                    ["region_type_id", "region_type", "tissue_type_id", "tissue_type"],
                    "tissue",
                )
                if aux_patch.loc[target_mask, col].isna().any():
                    raise ValueError(f"Missing tissue-task labels for patch {patch_id}.")
                tissue_type[target_mask] = self._preprocessed_integer_labels(
                    aux_patch.loc[target_mask, col], "tissue-task"
                )
            if aux_patch is not None and self.use_neighbor_task:
                col = self._resolve_column(
                    aux_patch,
                    ["neighbor_label", "neighbor_type_id", "neighbor_type", "neighborhood_label"],
                    "neighbor",
                )
                if aux_patch.loc[target_mask, col].isna().any():
                    raise ValueError(f"Missing neighbor-task labels for patch {patch_id}.")
                neighbor_label[target_mask] = self._preprocessed_integer_labels(
                    aux_patch.loc[target_mask, col], "neighbor-task"
                )
            coords_out[:n] = coords

        item: Dict[str, object] = {
            "sample_id": sample_id,
            "patch_id": patch_id,
            "cell_id": cell_ids,
            "coords": torch.tensor(coords_out, dtype=torch.float32),
            "patch_box": torch.tensor([x0, y0, x1, y1], dtype=torch.float32),
            "cell_imgs": torch.tensor(cell_imgs, dtype=torch.float32),
            "topo_feat": torch.tensor(topo_feat, dtype=torch.float32),
            "adjacency": torch.tensor(adjacency, dtype=torch.float32),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),
            "target_mask": torch.tensor(target_mask, dtype=torch.bool),
        }

        if self.require_labels:
            item.update(
                {
                    "protein_gt": torch.tensor(protein_gt, dtype=torch.float32),
                    "neighbor_label": torch.tensor(neighbor_label, dtype=torch.long),
                    "cell_type": torch.tensor(cell_type, dtype=torch.long),
                    "tissue_type": torch.tensor(tissue_type, dtype=torch.long),
                }
            )
        return item


Hist2ProtDataset = Hist2ProtPatchDataset


def hist2prot_collate(batch):
    """Dynamically pad a batch while preserving variable-size context graphs."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    max_nodes = max(int(item["cell_imgs"].shape[0]) for item in batch)
    tensor_keys = {
        key for item in batch for key, value in item.items()
        if torch.is_tensor(value)
    }
    result: Dict[str, object] = {
        "sample_id": [str(item["sample_id"]) for item in batch],
        "patch_id": [str(item["patch_id"]) for item in batch],
        "cell_id": [list(item["cell_id"]) + [""] * (max_nodes - len(item["cell_id"])) for item in batch],
    }
    for key in sorted(tensor_keys):
        values = [item[key] for item in batch]
        if key == "patch_box":
            result[key] = torch.stack(values, dim=0)
            continue
        if key == "adjacency":
            padded = values[0].new_zeros((len(values), max_nodes, max_nodes))
            for index, value in enumerate(values):
                n_nodes = value.shape[0]
                padded[index, :n_nodes, :n_nodes] = value
            result[key] = padded
            continue
        shape = (len(values), max_nodes) + tuple(values[0].shape[1:])
        padded = values[0].new_zeros(shape)
        for index, value in enumerate(values):
            padded[index, : value.shape[0]] = value
        result[key] = padded
    return result
