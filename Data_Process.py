import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config_utils import flattened_defaults, load_config

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


EXCLUDED_INTENSITY_CHANNELS = {
    "MsIgG1_intensity_mean",
    "MsIgG2a_intensity_mean",
    "cytoplasmicstain_intensity_mean",
    "nuclearstain_intensity_mean",
}


def fix_seed(seed: int = 2024) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sample_id_from_h5(path: Path) -> str:
    name = path.stem
    if name.startswith("adata_"):
        name = name[len("adata_"):]
    for suffix in ("_feature_matrix", "_filtered_feature_bc_matrix"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def image_shape_from_mask(mask_path: Path) -> Tuple[int, int]:
    mask = np.load(mask_path, mmap_mode="r")
    return int(mask.shape[0]), int(mask.shape[1])


def read_h5_obs(h5_path: Path) -> pd.DataFrame:
    adata = None
    try:
        import anndata as ad
        adata = ad.read_h5ad(str(h5_path), backed="r")
    except Exception:
        try:
            import scanpy as sc
            adata = sc.read_h5ad(str(h5_path), backed="r")
        except ImportError as exc:
            raise ImportError(
                "scanpy or anndata is required to read feature_matrix h5/h5ad files. "
                "Install project requirements first: pip install -r requirements.txt"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to read {h5_path} as h5ad: {exc}") from exc
    try:
        obs = adata.obs.copy()
        return obs.reset_index(drop=False)
    finally:
        file_manager = getattr(adata, "file", None)
        if file_manager is not None:
            file_manager.close()


def build_cell_csv(h5_path: Path, out_csv: Path) -> Tuple[pd.DataFrame, List[str]]:
    obs = read_h5_obs(h5_path)
    required = {"cell_id", "cell_x", "cell_y"}
    missing = required - set(obs.columns)
    if missing:
        raise ValueError(f"{h5_path} obs is missing required columns: {sorted(missing)}")

    protein_cols = [
        c for c in obs.columns
        if c.endswith("_intensity_mean") and c not in EXCLUDED_INTENSITY_CHANNELS
    ]
    if not protein_cols:
        raise ValueError(f"No protein columns ending with '_intensity_mean' found in {h5_path}")

    df = obs[["cell_id", "cell_x", "cell_y"] + protein_cols].copy()
    df = df.rename(columns={"cell_x": "x", "cell_y": "y"})

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"]).copy()
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    df.index = df["cell_id"].astype(str)
    if df.index.duplicated().any():
        duplicated = sorted(df.index[df.index.duplicated(keep=False)].unique())
        raise ValueError(f"{h5_path} contains duplicated cell_id values: {duplicated[:10]}")
    for column in protein_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    non_finite = ~np.isfinite(df[protein_cols].to_numpy(dtype=float))
    if non_finite.any():
        raise ValueError(
            f"{h5_path} contains {int(non_finite.sum())} missing or non-finite protein values."
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)
    return df, protein_cols


def find_raw_files(raw_root: Path, sample_id: str) -> Dict[str, Path]:
    # Supported layouts:
    #   <root>/Lung/<sample>/<sample>.tiff + <root>/Mask/<sample>/*.npy
    #   <root>/HE_mask/<sample>/<sample>_HE.ome.tiff + mask/*.npy
    layouts = [
        (
            raw_root / "Lung" / sample_id / f"{sample_id}.tiff",
            raw_root / "Mask" / sample_id / "nuclei.npy",
            raw_root / "Mask" / sample_id / "nuclei_exp.npy",
        ),
        (
            raw_root / "HE_mask" / sample_id / f"{sample_id}_HE.ome.tiff",
            raw_root / "HE_mask" / sample_id / "mask" / "nuclei.npy",
            raw_root / "HE_mask" / sample_id / "mask" / "nuclei_exp.npy",
        ),
    ]
    for image_path, nucleus_mask, cell_mask in layouts:
        if all(path.exists() for path in (image_path, nucleus_mask, cell_mask)):
            break
    else:
        attempted = [str(path) for layout in layouts for path in layout]
        raise FileNotFoundError(
            f"Missing raw files for {sample_id}; checked: " + ", ".join(attempted)
        )

    return {
        "image_path": image_path.resolve(),
        "nucleus_mask_path": nucleus_mask.resolve(),
        "cell_mask_path": cell_mask.resolve(),
    }


def make_patch_rows(
    sample_id: str,
    df: pd.DataFrame,
    height: int,
    width: int,
    patch_size: int,
    stride: int,
    min_cells: int,
    max_target_cells: int,
    context_radius: float,
) -> List[Dict[str, object]]:
    """Create non-overlapping target regions without dropping dense-patch cells.

    Dense regions are recursively divided into deterministic core boxes.  Each
    core is expanded by ``context_radius`` for data loading so every target cell
    retains its complete radius-neighbourhood.  Cells in the halo are context
    nodes only and are never counted twice as prediction targets.
    """
    rows: List[Dict[str, object]] = []
    patch_idx = 0
    xs = df["x"].to_numpy(dtype=np.float64, copy=False)
    ys = df["y"].to_numpy(dtype=np.float64, copy=False)

    if max_target_cells <= 0:
        raise ValueError("max_target_cells must be positive.")
    if context_radius < 0:
        raise ValueError("context_radius must be non-negative.")

    def add_or_split(
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        local_xs: np.ndarray,
        local_ys: np.ndarray,
        *,
        dense_child: bool = False,
    ) -> None:
        nonlocal patch_idx
        n_cells = int(len(local_xs))
        if n_cells == 0 or (not dense_child and n_cells < min_cells):
            return
        if n_cells > max_target_cells:
            box_width, box_height = x1 - x0, y1 - y0
            if box_width <= 1 and box_height <= 1:
                raise RuntimeError(
                    f"{sample_id}: {n_cells} cells share an indivisible region "
                    f"({x0}, {y0}, {x1}, {y1}); increase max_target_cells."
                )
            if box_width >= box_height and box_width > 1:
                midpoint = x0 + box_width // 2
                left = local_xs < midpoint
                add_or_split(
                    x0, y0, midpoint, y1, local_xs[left], local_ys[left], dense_child=True
                )
                add_or_split(
                    midpoint, y0, x1, y1, local_xs[~left], local_ys[~left], dense_child=True
                )
            else:
                midpoint = y0 + box_height // 2
                top = local_ys < midpoint
                add_or_split(
                    x0, y0, x1, midpoint, local_xs[top], local_ys[top], dense_child=True
                )
                add_or_split(
                    x0, midpoint, x1, y1, local_xs[~top], local_ys[~top], dense_child=True
                )
            return

        halo = int(np.ceil(context_radius))
        rows.append(
            {
                "sample_id": sample_id,
                "patch_id": f"{sample_id}_{patch_idx:08d}",
                "x0": int(x0),
                "y0": int(y0),
                "x1": int(x1),
                "y1": int(y1),
                "context_x0": max(0, int(x0) - halo),
                "context_y0": max(0, int(y0) - halo),
                "context_x1": min(width, int(x1) + halo),
                "context_y1": min(height, int(y1) + halo),
                "n_cells": n_cells,
            }
        )
        patch_idx += 1

    # Bin every cell once, then recurse only inside occupied tiles. The former
    # implementation scanned every cell for every tile (O(cells * patches)),
    # which is infeasible for million-cell WSIs.
    n_x_tiles = int(np.ceil(width / stride))
    tile_x = np.floor_divide(xs, stride).astype(np.int64)
    tile_y = np.floor_divide(ys, stride).astype(np.int64)
    tile_keys = tile_y * n_x_tiles + tile_x
    order = np.argsort(tile_keys, kind="stable")
    sorted_keys = tile_keys[order]
    unique_keys, starts, counts = np.unique(
        sorted_keys, return_index=True, return_counts=True
    )
    for key, start, count in zip(unique_keys, starts, counts):
        indices = order[start:start + count]
        tile_y_index, tile_x_index = divmod(int(key), n_x_tiles)
        x0, y0 = tile_x_index * stride, tile_y_index * stride
        x1, y1 = min(x0 + patch_size, width), min(y0 + patch_size, height)
        add_or_split(x0, y0, x1, y1, xs[indices], ys[indices])
    return rows


def load_sample_splits(
    split_csv: Optional[str],
    sample_ids: List[str],
    seed: int = 2024,
    train_fraction: float = 0.7,
) -> Dict[str, str]:
    """Load or deterministically create a patient-level 70:30 train/test split.

    If a CSV contains ``cancer_type`` but no ``split`` column, the split is
    generated within cancer type.  Without a CSV it is generated across all
    samples.  A supplied split is validated and never rewritten.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1).")

    if split_csv:
        split_path = Path(split_csv)
        if not split_path.exists():
            raise FileNotFoundError(f"Sample split CSV does not exist: {split_path}")
        split_df = pd.read_csv(split_path, dtype=str)
        if "sample_id" not in split_df.columns:
            raise ValueError("Sample split CSV must contain a sample_id column.")
    else:
        split_df = pd.DataFrame({"sample_id": list(sample_ids)})

    if "split" not in split_df.columns:
        rng = random.Random(seed)
        split_df["split"] = ""
        group_column = "cancer_type" if "cancer_type" in split_df.columns else None
        grouped = split_df.groupby(group_column, sort=True) if group_column else [("all", split_df)]
        for _, group in grouped:
            indices = list(group.index)
            rng.shuffle(indices)
            if len(indices) < 2:
                raise ValueError("Each split stratum must contain at least two patients.")
            test_count = min(len(indices) - 1, max(1, int(round(len(indices) * (1 - train_fraction)))))
            split_df.loc[indices[:test_count], "split"] = "test"
            split_df.loc[indices[test_count:], "split"] = "train"

    split_df = split_df[["sample_id", "split"]].copy()
    split_df["sample_id"] = split_df["sample_id"].str.strip()
    split_df["split"] = split_df["split"].str.strip().str.lower()

    if (
        split_df[["sample_id", "split"]].isna().any(axis=None)
        or split_df["sample_id"].eq("").any()
        or split_df["split"].eq("").any()
    ):
        raise ValueError("Sample split CSV contains empty sample_id or split values.")
    duplicated = split_df.loc[split_df["sample_id"].duplicated(keep=False), "sample_id"].unique()
    if len(duplicated) > 0:
        raise ValueError(
            "Each sample_id must appear exactly once in the sample split CSV. "
            f"Duplicated sample_id values: {duplicated.tolist()}"
        )

    valid_splits = {"train", "test"}
    invalid_splits = sorted(set(split_df["split"]) - valid_splits)
    if invalid_splits:
        raise ValueError(
            f"Invalid split values: {invalid_splits}. Expected train or test."
        )

    discovered = set(sample_ids)
    configured = set(split_df["sample_id"])
    missing_samples = sorted(discovered - configured)
    unknown_samples = sorted(configured - discovered)
    if missing_samples:
        raise ValueError(f"Samples missing from the sample split CSV: {missing_samples}")
    if unknown_samples:
        raise ValueError(f"Unknown samples in the sample split CSV: {unknown_samples}")

    split_map = dict(zip(split_df["sample_id"], split_df["split"]))
    empty_splits = sorted(valid_splits - set(split_map.values()))
    if empty_splits:
        raise ValueError(
            "The patient split must contain at least one sample for each split. "
            f"Missing split(s): {empty_splits}"
        )
    return split_map


def split_patches_by_sample(
    patch_rows: List[Dict[str, object]],
    sample_splits: Dict[str, str],
) -> Dict[str, List[Dict[str, object]]]:
    splits = {"train": [], "test": []}
    for row in patch_rows:
        sample_id = str(row["sample_id"])
        splits[sample_splits[sample_id]].append(row)

    for rows in splits.values():
        rows.sort(key=lambda r: (str(r["sample_id"]), str(r["patch_id"])))
    empty_splits = [split for split, rows in splits.items() if not rows]
    if empty_splits:
        raise ValueError(
            "No patches remain for the following split(s): "
            f"{empty_splits}. Check --min_cells and the assigned samples."
        )
    return splits


def collect_split_cell_ids(
    rows: List[Dict[str, object]],
    csv_dir: Path,
) -> Dict[str, set]:
    ids_by_sample: Dict[str, set] = {}
    df_cache: Dict[str, pd.DataFrame] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id not in df_cache:
            df_cache[sample_id] = pd.read_csv(csv_dir / f"{sample_id}.csv", index_col=0)
            df_cache[sample_id].index = df_cache[sample_id].index.astype(str)
        df = df_cache[sample_id]
        patch_df = df[
            (df["x"] >= float(row["x0"]))
            & (df["x"] < float(row["x1"]))
            & (df["y"] >= float(row["y0"]))
            & (df["y"] < float(row["y1"]))
        ]
        ids_by_sample.setdefault(sample_id, set()).update(patch_df.index.astype(str))
    return ids_by_sample


def normalize_protein_csvs(
    source_csv_dir: Path,
    output_csv_dir: Path,
    sample_ids: List[str],
    train_rows: List[Dict[str, object]],
    protein_names: List[str],
    lower: float,
    upper: float,
    log1p: bool,
    filter_outlier_cells: bool,
    out_path: Path,
) -> Dict[str, object]:
    train_sample_ids = {str(row["sample_id"]) for row in train_rows}
    train_values = []
    for sample_id in sample_ids:
        if sample_id not in train_sample_ids:
            continue
        df = pd.read_csv(source_csv_dir / f"{sample_id}.csv", index_col=0)
        train_values.append(df[protein_names].astype(float))

    if not train_values:
        raise ValueError("No train cells found for protein normalization.")

    train_matrix = pd.concat(train_values, axis=0)
    if log1p:
        train_matrix = np.log1p(train_matrix.clip(lower=0.0))

    q_low = train_matrix.quantile(lower / 100.0)
    q_high = train_matrix.quantile(upper / 100.0)
    scale = (q_high - q_low).replace(0, np.nan).fillna(1.0)

    filter_summary: Dict[str, Dict[str, int]] = {}
    output_csv_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in sample_ids:
        source_path = source_csv_dir / f"{sample_id}.csv"
        output_path = output_csv_dir / f"{sample_id}.csv"
        df = pd.read_csv(source_path, index_col=0)
        values = df[protein_names].astype(float)
        if log1p:
            values = np.log1p(values.clip(lower=0.0))

        if filter_outlier_cells:
            low_outlier = values.lt(q_low, axis=1)
            high_outlier = values.gt(q_high, axis=1)
            outlier_cells = (low_outlier | high_outlier).any(axis=1)
            filter_summary[sample_id] = {
                "before": int(len(df)),
                "removed": int(outlier_cells.sum()),
                "kept": int((~outlier_cells).sum()),
            }
            df = df.loc[~outlier_cells].copy()
            values = values.loc[~outlier_cells].copy()
        else:
            filter_summary[sample_id] = {
                "before": int(len(df)),
                "removed": 0,
                "kept": int(len(df)),
            }

        values = values.clip(lower=q_low, upper=q_high, axis=1)
        values = (values - q_low) / scale
        values = values.clip(lower=0.0, upper=1.0)
        df[protein_names] = values
        df.to_csv(output_path)

    norm = {
        "enabled": True,
        "method": "log1p_percentile_minmax" if log1p else "percentile_minmax",
        "fit_split": "train",
        "lower_percentile": lower,
        "upper_percentile": upper,
        "log1p": log1p,
        "filter_outlier_cells": filter_outlier_cells,
        "filter_rule": "remove cell if any protein is outside percentile range after log1p"
        if filter_outlier_cells
        else "keep cells and clip values",
        "filter_summary": filter_summary,
        "protein_names": protein_names,
        "low": {k: float(v) for k, v in q_low.items()},
        "high": {k: float(v) for k, v in q_high.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(norm, f, indent=2)
    return norm


def run_preprocess(
    raw_root: str,
    out_folder: str,
    split_csv: Optional[str] = None,
    patch_size: int = 256,
    stride: int = 256,
    min_cells: int = 1,
    max_target_cells: int = 256,
    neighbor_radius: float = 50.0,
    train_fraction: float = 0.7,
    seed: int = 2024,
    normalize_protein: bool = True,
    norm_lower: float = 2.5,
    norm_upper: float = 97.5,
    norm_log1p: bool = True,
    filter_outlier_cells: bool = False,
    sample_ids: Optional[List[str]] = None,
) -> None:
    if stride != patch_size:
        raise ValueError(
            "Hist2Prot target regions must be non-overlapping: set stride equal to patch_size. "
            "Overlapping or gapped target regions would duplicate or omit supervised cells."
        )
    fix_seed(seed)
    raw_root_path = Path(raw_root)
    out_path = Path(out_folder)
    proc_dir = out_path / "Process"
    raw_csv_dir = proc_dir / "raw_csv"
    csv_dir = proc_dir / "csv"
    patch_dir = proc_dir / "patches"
    raw_csv_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)

    h5_dir = raw_root_path / "H5_files"
    if not h5_dir.exists():
        h5_dir = raw_root_path / "h5_files"
    discovered_h5 = sorted(h5_dir.glob("*.h5*"))
    # Canonicalize duplicate adata_<sample>.h5ad and <sample>.h5ad names,
    # preferring the exact sample filename.
    selected_by_sample: Dict[str, Path] = {}
    for candidate in discovered_h5:
        canonical_id = sample_id_from_h5(candidate)
        existing = selected_by_sample.get(canonical_id)
        if existing is None or candidate.stem == canonical_id:
            selected_by_sample[canonical_id] = candidate
    if sample_ids:
        requested = list(dict.fromkeys(str(value).strip() for value in sample_ids if str(value).strip()))
        missing = [value for value in requested if value not in selected_by_sample]
        if missing:
            raise FileNotFoundError(f"Requested sample H5 files were not found: {missing}")
        h5_files = [selected_by_sample[value] for value in requested]
    else:
        h5_files = [selected_by_sample[key] for key in sorted(selected_by_sample)]
    if not h5_files:
        raise FileNotFoundError(f"No .h5/.h5ad files found under {h5_dir}")

    sample_rows = []
    all_patch_rows: List[Dict[str, object]] = []
    protein_names: List[str] = []
    sample_ids: List[str] = []

    for h5_path in tqdm(h5_files, desc="Preparing WSI metadata"):
        sample_id = sample_id_from_h5(h5_path)
        sample_ids.append(sample_id)
        files = find_raw_files(raw_root_path, sample_id)

        raw_cell_csv = raw_csv_dir / f"{sample_id}.csv"
        df, sample_proteins = build_cell_csv(h5_path, raw_cell_csv)
        if not protein_names:
            protein_names = sample_proteins
        elif protein_names != sample_proteins:
            raise ValueError(
                f"Protein columns differ for {sample_id}. Keep the same protein panel across WSIs."
            )

        height, width = image_shape_from_mask(files["cell_mask_path"])
        out_of_bounds = (
            (df["x"] < 0) | (df["x"] >= width) | (df["y"] < 0) | (df["y"] >= height)
        )
        if out_of_bounds.any():
            raise ValueError(
                f"{sample_id} contains {int(out_of_bounds.sum())} cell coordinates outside "
                f"the WSI bounds width={width}, height={height}."
            )
        sample_rows.append(
            {
                "sample_id": sample_id,
                "image_path": str(files["image_path"]),
                "cell_mask_path": str(files["cell_mask_path"]),
                "nucleus_mask_path": str(files["nucleus_mask_path"]),
                "raw_csv_path": str(raw_cell_csv.resolve()),
                "csv_path": str((csv_dir / f"{sample_id}.csv").resolve()),
                "height": height,
                "width": width,
            }
        )

        all_patch_rows.extend(
            make_patch_rows(
                sample_id, df, height, width, patch_size, stride, min_cells,
                max_target_cells, neighbor_radius,
            )
        )

    sample_splits = load_sample_splits(
        split_csv, sample_ids, seed=seed, train_fraction=train_fraction
    )
    for sample_row in sample_rows:
        sample_row["split"] = sample_splits[str(sample_row["sample_id"])]
    splits = split_patches_by_sample(all_patch_rows, sample_splits)

    protein_norm = {"enabled": False}
    if normalize_protein:
        protein_norm = normalize_protein_csvs(
            source_csv_dir=raw_csv_dir,
            output_csv_dir=csv_dir,
            sample_ids=sample_ids,
            train_rows=splits["train"],
            protein_names=protein_names,
            lower=norm_lower,
            upper=norm_upper,
            log1p=norm_log1p,
            filter_outlier_cells=filter_outlier_cells,
            out_path=proc_dir / "protein_norm.json",
        )
        if filter_outlier_cells:
            all_patch_rows = []
            dims_by_sample = {
                str(row["sample_id"]): (int(row["height"]), int(row["width"]))
                for row in sample_rows
            }
            for sample_id in sample_ids:
                filtered_df = pd.read_csv(csv_dir / f"{sample_id}.csv", index_col=0)
                height, width = dims_by_sample[sample_id]
                all_patch_rows.extend(
                    make_patch_rows(
                        sample_id,
                        filtered_df,
                        height,
                        width,
                        patch_size,
                        stride,
                        min_cells,
                        max_target_cells,
                        neighbor_radius,
                    )
                )
            splits = split_patches_by_sample(all_patch_rows, sample_splits)
    else:
        for sample_id in sample_ids:
            shutil.copyfile(
                raw_csv_dir / f"{sample_id}.csv",
                csv_dir / f"{sample_id}.csv",
            )

    pd.DataFrame(sample_rows).to_csv(proc_dir / "samples.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "split": [sample_splits[sample_id] for sample_id in sample_ids],
        }
    ).to_csv(proc_dir / "sample_splits.csv", index=False)
    patch_columns = [
        "patch_id", "sample_id", "x0", "y0", "x1", "y1",
        "context_x0", "context_y0", "context_x1", "context_y1", "n_cells",
    ]
    pd.DataFrame(all_patch_rows, columns=patch_columns).to_csv(
        patch_dir / "all_patches.csv", index=False
    )
    for split, rows in splits.items():
        pd.DataFrame(rows, columns=patch_columns).to_csv(
            patch_dir / f"{split}_patches.csv", index=False
        )
        with open(out_path / f"{split}_samples.txt", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(f"{row['patch_id']}\n")

    metadata = {
        "raw_root": str(raw_root_path.resolve()),
        "patch_size": patch_size,
        "stride": stride,
        "min_cells": min_cells,
        "max_target_cells": max_target_cells,
        "neighbor_radius": neighbor_radius,
        "train_fraction": train_fraction,
        "split_seed": seed,
        "sample_split_csv": str(Path(split_csv).resolve()) if split_csv else None,
        "sample_splits": sample_splits,
        "sample_order": sample_ids,
        "protein_names": protein_names,
        "protein_dim": len(protein_names),
        "protein_normalization": protein_norm,
    }
    with open(proc_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Data processing done. Processed {len(sample_ids)} WSI(s), {len(all_patch_rows)} patch(es).")
    split_sample_counts = {
        split: sum(assigned_split == split for assigned_split in sample_splits.values())
        for split in ("train", "test")
    }
    split_patch_counts = {split: len(rows) for split, rows in splits.items()}
    print(f"Sample splits: {split_sample_counts}")
    print(f"Patch splits: {split_patch_counts}")
    print(f"Protein dim: {len(protein_names)}")
    if protein_norm.get("enabled"):
        print(
            "Protein normalization: "
            f"{protein_norm['method']} fitted on train cells "
            f"({norm_lower}-{norm_upper} percentiles)."
        )
        if protein_norm.get("filter_outlier_cells"):
            removed = sum(v["removed"] for v in protein_norm["filter_summary"].values())
            kept = sum(v["kept"] for v in protein_norm["filter_summary"].values())
            print(f"Outlier cell filtering: removed {removed} cell(s), kept {kept} cell(s).")


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config_file", default="configs/hist2prot.json")
    config_args, _ = config_parser.parse_known_args()
    defaults = flattened_defaults(
        load_config(config_args.config_file), "dataset", "data", "preprocessing"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=config_args.config_file)
    parser.add_argument("--raw_root", type=str, default=defaults.get("raw_root", "raw_data"))
    parser.add_argument(
        "--out_folder", type=str,
        default=defaults.get("out_folder", defaults.get("data_root", "./data")),
    )
    parser.add_argument(
        "--split_csv",
        type=str,
        default=defaults.get("split_csv"),
        help=(
            "Optional patient CSV. Use sample_id and split (train/test) for a fixed split, "
            "or sample_id and cancer_type to generate a stratified 70:30 split."
        ),
    )
    parser.add_argument(
        "--sample_ids",
        type=str,
        default=None,
        help="Optional comma-separated sample IDs; all other H5 files are ignored.",
    )
    parser.add_argument("--patch_size", type=int, default=int(defaults.get("patch_size", 256)))
    parser.add_argument("--stride", type=int, default=int(defaults.get("stride", 256)))
    parser.add_argument("--min_cells", type=int, default=int(defaults.get("min_cells", 1)))
    parser.add_argument(
        "--max_target_cells", type=int, default=int(defaults.get("max_target_cells", 256))
    )
    parser.add_argument(
        "--neighbor_radius", type=float, default=float(defaults.get("neighbor_radius", 50.0))
    )
    parser.add_argument(
        "--train_fraction", type=float, default=float(defaults.get("train_fraction", 0.7))
    )
    parser.add_argument("--seed", type=int, default=int(defaults.get("split_seed", 2024)))
    parser.add_argument(
        "--normalize_protein", dest="normalize_protein", action="store_true",
        default=bool(defaults.get("normalize_protein", True)),
    )
    parser.add_argument("--no_normalize_protein", dest="normalize_protein", action="store_false")
    parser.add_argument("--norm_lower", type=float, default=float(defaults.get("norm_lower", 2.5)))
    parser.add_argument("--norm_upper", type=float, default=float(defaults.get("norm_upper", 97.5)))
    parser.add_argument(
        "--norm_log1p", dest="norm_log1p", action="store_true",
        default=bool(defaults.get("norm_log1p", True)),
    )
    parser.add_argument("--no_norm_log1p", dest="norm_log1p", action="store_false")
    parser.add_argument(
        "--filter_outlier_cells", dest="filter_outlier_cells", action="store_true",
        default=bool(defaults.get("filter_outlier_cells", False)),
    )
    parser.add_argument("--keep_outlier_cells", dest="filter_outlier_cells", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preprocess(
        raw_root=args.raw_root,
        out_folder=args.out_folder,
        split_csv=args.split_csv,
        patch_size=args.patch_size,
        stride=args.stride,
        min_cells=args.min_cells,
        max_target_cells=args.max_target_cells,
        neighbor_radius=args.neighbor_radius,
        train_fraction=args.train_fraction,
        seed=args.seed,
        normalize_protein=args.normalize_protein,
        norm_lower=args.norm_lower,
        norm_upper=args.norm_upper,
        norm_log1p=args.norm_log1p,
        filter_outlier_cells=args.filter_outlier_cells,
        sample_ids=[value.strip() for value in args.sample_ids.split(",") if value.strip()]
        if args.sample_ids
        else None,
    )
