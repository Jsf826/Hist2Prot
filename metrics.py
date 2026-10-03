from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

try:
    from scipy.stats import rankdata
except Exception:  # pragma: no cover
    rankdata = None


def protein_pcc(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[float, List[float]]:
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy().astype(bool)

    pccs: List[float] = []
    for protein_idx in range(pred_np.shape[-1]):
        y_pred = pred_np[..., protein_idx][mask_np]
        y_true = target_np[..., protein_idx][mask_np]
        if y_pred.size < 2:
            continue
        pred_std = y_pred.std()
        true_std = y_true.std()
        if pred_std < eps or true_std < eps:
            continue
        pccs.append(float(np.corrcoef(y_pred, y_true)[0, 1]))

    mean_pcc = float(np.mean(pccs)) if pccs else float("nan")
    return mean_pcc, pccs


def _rasterize_cells(
    values: np.ndarray,
    coords: np.ndarray,
    patch_box: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    x0, y0, x1, y1 = patch_box.astype(float)
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)

    grid_sum = np.zeros((grid_size, grid_size), dtype=np.float32)
    grid_count = np.zeros((grid_size, grid_size), dtype=np.float32)

    xs = np.clip(((coords[:, 0] - x0) / width * grid_size).astype(int), 0, grid_size - 1)
    ys = np.clip(((coords[:, 1] - y0) / height * grid_size).astype(int), 0, grid_size - 1)

    for x, y, value in zip(xs, ys, values):
        grid_sum[y, x] += float(value)
        grid_count[y, x] += 1.0

    occupied = grid_count > 0
    grid_sum[occupied] /= grid_count[occupied]
    return grid_sum


def _global_ssim(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    data_range = max(float(x.max()), float(y.max())) - min(float(x.min()), float(y.min()))
    data_range = max(data_range, 1.0)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mux = x.mean()
    muy = y.mean()
    vx = ((x - mux) ** 2).mean()
    vy = ((y - muy) ** 2).mean()
    cov = ((x - mux) * (y - muy)).mean()

    denom = (mux * mux + muy * muy + c1) * (vx + vy + c2)
    if abs(denom) < eps:
        return float("nan")
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / denom)


def protein_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    coords: torch.Tensor,
    patch_box: torch.Tensor,
    mask: torch.Tensor,
    grid_size: int = 64,
) -> Tuple[float, List[float]]:
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    coords_np = coords.detach().cpu().numpy()
    patch_box_np = patch_box.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy().astype(bool)

    scores: List[float] = []
    for batch_idx in range(pred_np.shape[0]):
        valid = mask_np[batch_idx]
        if valid.sum() < 2:
            continue
        valid_coords = coords_np[batch_idx, valid]
        for protein_idx in range(pred_np.shape[-1]):
            pred_map = _rasterize_cells(
                pred_np[batch_idx, valid, protein_idx],
                valid_coords,
                patch_box_np[batch_idx],
                grid_size,
            )
            true_map = _rasterize_cells(
                target_np[batch_idx, valid, protein_idx],
                valid_coords,
                patch_box_np[batch_idx],
                grid_size,
            )
            score = _global_ssim(pred_map, true_map)
            if not np.isnan(score):
                scores.append(score)

    mean_ssim = float(np.mean(scores)) if scores else float("nan")
    return mean_ssim, scores


class MetricAverager:
    def __init__(self) -> None:
        self.values: Dict[str, List[float]] = {}

    def update(self, name: str, value: float) -> None:
        if np.isnan(value):
            return
        self.values.setdefault(name, []).append(float(value))

    def mean(self, name: str) -> float:
        vals = self.values.get(name, [])
        return float(np.mean(vals)) if vals else float("nan")


def _safe_corr(x: np.ndarray, y: np.ndarray, spearman: bool = False, eps: float = 1e-8) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return float("nan")
    x, y = x[finite], y[finite]
    if spearman:
        if rankdata is not None:
            x, y = rankdata(x, method="average"), rankdata(y, method="average")
        else:
            x = np.argsort(np.argsort(x)).astype(np.float64)
            y = np.argsort(np.argsort(y)).astype(np.float64)
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denominator < eps or not np.isfinite(denominator):
        return float("nan")
    return float(np.sum(x * y) / denominator)


def patient_marker_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    coords: np.ndarray,
    protein_names: Sequence[str],
    grid_size: int = 256,
) -> List[Dict[str, float]]:
    """Compute one row per protein from all unique cells in one patient."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError(f"Expected matching [cells, proteins] arrays, got {pred.shape} and {target.shape}.")
    if pred.shape[0] != coords.shape[0] or coords.shape[1] != 2:
        raise ValueError("Coordinates must have shape [cells, 2] aligned to predictions.")
    if pred.shape[1] != len(protein_names):
        raise ValueError("protein_names length does not match prediction columns.")
    if pred.shape[0] == 0:
        return []
    patch_box = np.array(
        [coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max() + 1, coords[:, 1].max() + 1],
        dtype=np.float64,
    )
    rows: List[Dict[str, float]] = []
    for protein_index, protein_name in enumerate(protein_names):
        x = target[:, protein_index]
        y = pred[:, protein_index]
        finite = np.isfinite(x) & np.isfinite(y)
        n_cells = int(finite.sum())
        if n_cells:
            mse = float(np.mean((y[finite] - x[finite]) ** 2))
            pred_map = _rasterize_cells(y[finite], coords[finite], patch_box, grid_size)
            target_map = _rasterize_cells(x[finite], coords[finite], patch_box, grid_size)
            ssim = _global_ssim(pred_map, target_map)
        else:
            mse, ssim = float("nan"), float("nan")
        rows.append(
            {
                "protein": str(protein_name),
                "n_cells": n_cells,
                "pcc": _safe_corr(x, y, spearman=False),
                "spearman": _safe_corr(x, y, spearman=True),
                "ssim": ssim,
                "mse": mse,
            }
        )
    return rows
