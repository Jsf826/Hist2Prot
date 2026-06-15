from typing import Dict, List, Tuple

import numpy as np
import torch


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
