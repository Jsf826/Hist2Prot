import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

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
    for suffix in ("_feature_matrix", "_filtered_feature_bc_matrix"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name.split("_")[0]


def image_shape_from_mask(mask_path: Path) -> Tuple[int, int]:
    mask = np.load(mask_path, mmap_mode="r")
    return int(mask.shape[0]), int(mask.shape[1])


def read_h5_obs(h5_path: Path) -> pd.DataFrame:
    try:
        import anndata as ad
        adata = ad.read_h5ad(str(h5_path))
    except Exception:
        try:
            import scanpy as sc
            adata = sc.read_h5ad(str(h5_path))
        except ImportError as exc:
            raise ImportError(
                "scanpy or anndata is required to read feature_matrix h5/h5ad files. "
                "Install project requirements first: pip install -r requirements.txt"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to read {h5_path} as h5ad: {exc}") from exc
    obs = adata.obs.copy()
    obs = obs.reset_index(drop=False)
    return obs


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

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)
    return df, protein_cols


def find_raw_files(raw_root: Path, sample_id: str) -> Dict[str, Path]:
    sample_dir = raw_root / "HE_mask" / sample_id
    image_path = sample_dir / f"{sample_id}_HE.ome.tiff"
    mask_dir = sample_dir / "mask"
    nucleus_mask = mask_dir / "nuclei.npy"
    cell_mask = mask_dir / "nuclei_exp.npy"

    missing = [p for p in (image_path, nucleus_mask, cell_mask) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw files for {sample_id}: " + ", ".join(str(p) for p in missing)
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
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    patch_idx = 0
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()

    for y0 in range(0, height, stride):
        y1 = min(y0 + patch_size, height)
        if y1 - y0 <= 0:
            continue
        for x0 in range(0, width, stride):
            x1 = min(x0 + patch_size, width)
            if x1 - x0 <= 0:
                continue

            inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
            n_cells = int(inside.sum())
            if n_cells < min_cells:
                continue

            rows.append(
                {
                    "sample_id": sample_id,
                    "patch_id": f"{sample_id}_{patch_idx:08d}",
                    "x0": int(x0),
                    "y0": int(y0),
                    "x1": int(x1),
                    "y1": int(y1),
                    "n_cells": n_cells,
                }
            )
            patch_idx += 1
    return rows


def split_first_wsi_train_val_second_test(
    patch_rows: List[Dict[str, object]],
    sample_ids: List[str],
    train_fraction: float,
    seed: int,
) -> Dict[str, List[Dict[str, object]]]:
    splits = {"train": [], "val": [], "test": []}
    if not sample_ids:
        return splits

    rng = random.Random(seed)
    by_sample: Dict[str, List[Dict[str, object]]] = {s: [] for s in sample_ids}
    for row in patch_rows:
        by_sample[str(row["sample_id"])].append(row)

    first = sample_ids[0]
    first_rows = by_sample.get(first, [])
    rng.shuffle(first_rows)
    n_train = int(round(len(first_rows) * train_fraction))
    splits["train"].extend(first_rows[:n_train])
    splits["val"].extend(first_rows[n_train:])

    for sample_id in sample_ids[1:]:
        splits["test"].extend(by_sample.get(sample_id, []))

    for rows in splits.values():
        rows.sort(key=lambda r: (str(r["sample_id"]), str(r["patch_id"])))
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
    csv_dir: Path,
    sample_ids: List[str],
    train_rows: List[Dict[str, object]],
    protein_names: List[str],
    lower: float,
    upper: float,
    log1p: bool,
    filter_outlier_cells: bool,
    out_path: Path,
) -> Dict[str, object]:
    train_ids = collect_split_cell_ids(train_rows, csv_dir)
    train_values = []
    for sample_id in sample_ids:
        ids = train_ids.get(sample_id, set())
        if not ids:
            continue
        df = pd.read_csv(csv_dir / f"{sample_id}.csv", index_col=0)
        df.index = df.index.astype(str)
        matched = df.index.intersection(list(ids))
        if len(matched) > 0:
            train_values.append(df.loc[matched, protein_names].astype(float))

    if not train_values:
        raise ValueError("No train cells found for protein normalization.")

    train_matrix = pd.concat(train_values, axis=0)
    if log1p:
        train_matrix = np.log1p(train_matrix.clip(lower=0.0))

    q_low = train_matrix.quantile(lower / 100.0)
    q_high = train_matrix.quantile(upper / 100.0)
    scale = (q_high - q_low).replace(0, np.nan).fillna(1.0)

    filter_summary: Dict[str, Dict[str, int]] = {}
    for sample_id in sample_ids:
        csv_path = csv_dir / f"{sample_id}.csv"
        df = pd.read_csv(csv_path, index_col=0)
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
        df.to_csv(csv_path)

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
    patch_size: int = 256,
    stride: int = 256,
    min_cells: int = 1,
    train_fraction: float = 0.7,
    seed: int = 2024,
    normalize_protein: bool = True,
    norm_lower: float = 2.5,
    norm_upper: float = 97.5,
    norm_log1p: bool = True,
    filter_outlier_cells: bool = True,
) -> None:
    fix_seed(seed)
    raw_root_path = Path(raw_root)
    out_path = Path(out_folder)
    proc_dir = out_path / "Process"
    csv_dir = proc_dir / "csv"
    patch_dir = proc_dir / "patches"
    csv_dir.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted((raw_root_path / "h5_files").glob("*.h5*"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5/.h5ad files found under {raw_root_path / 'h5_files'}")

    sample_rows = []
    all_patch_rows: List[Dict[str, object]] = []
    protein_names: List[str] = []
    sample_ids: List[str] = []

    for h5_path in tqdm(h5_files, desc="Preparing WSI metadata"):
        sample_id = sample_id_from_h5(h5_path)
        sample_ids.append(sample_id)
        files = find_raw_files(raw_root_path, sample_id)

        cell_csv = csv_dir / f"{sample_id}.csv"
        df, sample_proteins = build_cell_csv(h5_path, cell_csv)
        if not protein_names:
            protein_names = sample_proteins
        elif protein_names != sample_proteins:
            raise ValueError(
                f"Protein columns differ for {sample_id}. Keep the same protein panel across WSIs."
            )

        height, width = image_shape_from_mask(files["cell_mask_path"])
        sample_rows.append(
            {
                "sample_id": sample_id,
                "image_path": str(files["image_path"]),
                "cell_mask_path": str(files["cell_mask_path"]),
                "nucleus_mask_path": str(files["nucleus_mask_path"]),
                "csv_path": str(cell_csv.resolve()),
                "height": height,
                "width": width,
            }
        )

        all_patch_rows.extend(
            make_patch_rows(sample_id, df, height, width, patch_size, stride, min_cells)
        )

    splits = split_first_wsi_train_val_second_test(
        all_patch_rows, sample_ids, train_fraction=train_fraction, seed=seed
    )

    protein_norm = {"enabled": False}
    if normalize_protein:
        protein_norm = normalize_protein_csvs(
            csv_dir=csv_dir,
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
                    )
                )
            splits = split_first_wsi_train_val_second_test(
                all_patch_rows, sample_ids, train_fraction=train_fraction, seed=seed
            )

    pd.DataFrame(sample_rows).to_csv(proc_dir / "samples.csv", index=False)
    patch_columns = ["patch_id", "sample_id", "x0", "y0", "x1", "y1", "n_cells"]
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
        "train_fraction_first_wsi": train_fraction,
        "sample_order": sample_ids,
        "protein_names": protein_names,
        "protein_dim": len(protein_names),
        "protein_normalization": protein_norm,
    }
    with open(proc_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Data processing done. Processed {len(sample_ids)} WSI(s), {len(all_patch_rows)} patch(es).")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, default="Row data")
    parser.add_argument("--out_folder", type=str, default="./demo_data")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min_cells", type=int, default=40)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--normalize_protein", dest="normalize_protein", action="store_true", default=True)
    parser.add_argument("--no_normalize_protein", dest="normalize_protein", action="store_false")
    parser.add_argument("--norm_lower", type=float, default=2.5)
    parser.add_argument("--norm_upper", type=float, default=97.5)
    parser.add_argument("--norm_log1p", dest="norm_log1p", action="store_true", default=True)
    parser.add_argument("--no_norm_log1p", dest="norm_log1p", action="store_false")
    parser.add_argument("--filter_outlier_cells", dest="filter_outlier_cells", action="store_true", default=True)
    parser.add_argument("--keep_outlier_cells", dest="filter_outlier_cells", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preprocess(
        raw_root=args.raw_root,
        out_folder=args.out_folder,
        patch_size=args.patch_size,
        stride=args.stride,
        min_cells=args.min_cells,
        train_fraction=args.train_fraction,
        seed=args.seed,
        normalize_protein=args.normalize_protein,
        norm_lower=args.norm_lower,
        norm_upper=args.norm_upper,
        norm_log1p=args.norm_log1p,
        filter_outlier_cells=args.filter_outlier_cells,
    )
