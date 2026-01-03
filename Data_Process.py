# DataProcess.py
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import random
from skimage import io


def fix_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def crop_tile(img, x, y, size):
    h, w = img.shape[:2]
    half = size // 2
    x1, x2 = max(0, x-half), min(w, x+half)
    y1, y2 = max(0, y-half), min(h, y+half)
    patch = img[y1:y2, x1:x2]

    if patch.shape[0] != size or patch.shape[1] != size:
        pad_h = size - patch.shape[0]
        pad_w = size - patch.shape[1]
        patch = np.pad(
            patch,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant"
        )
    return patch

def quality_control_tile(patch, thr=0.8):
    gray = patch.mean(axis=2)
    white_ratio = np.mean(gray > 240)
    return white_ratio < thr


fix_seed(2024)

OUT_FOLDER = "/path/to/data"
CELL_DIAMETER = 32
NEIGHBOR_RADIUS = 50

Path(f"{OUT_FOLDER}/Process/image_features").mkdir(parents=True, exist_ok=True)
Path(f"{OUT_FOLDER}/Process/topology_features").mkdir(parents=True, exist_ok=True)
Path(f"{OUT_FOLDER}/Process/inputX_protein").mkdir(parents=True, exist_ok=True)

SAMPLES = [f.replace(".tif", "") for f in os.listdir(f"{OUT_FOLDER}/Process/images") if f.endswith(".tif")]


def load_hovernet_seg(path):
    return np.load(path)

def extract_cell_features(img, seg):
    feats = {}
    coords = {}
    cell_ids = np.unique(seg)
    cell_ids = cell_ids[cell_ids != 0]

    for cid in cell_ids:
        mask = seg == cid
        y, x = np.argwhere(mask).mean(axis=0).astype(int)
        patch = crop_tile(img, x, y, CELL_DIAMETER)
        if not quality_control_tile(patch):
            continue

        # CNN input：[3, H, W]
        patch = patch.astype(np.float32) / 255.
        patch = np.transpose(patch, (2, 0, 1))
        feats[cid] = patch
        coords[cid] = (x, y)

    return feats, coords

def build_topology_features(coords, cell_types, radius=50):
    topo_feat = {}
    edge_index = []

    ids = list(coords.keys())
    for i, cid in enumerate(ids):
        x0, y0 = coords[cid]
        neighbors = []

        for j, oid in enumerate(ids):
            if cid == oid:
                continue
            x1, y1 = coords[oid]
            if np.sqrt((x0-x1)**2 + (y0-y1)**2) <= radius:
                neighbors.append(oid)
                edge_index.append([i, j])

        # topo feature = [degree, immune_ratio, stroma_ratio, epithelial_ratio]
        types = [cell_types[n] for n in neighbors]
        topo_feat[cid] = np.array([
            len(neighbors),
            types.count("immune"),
            types.count("stroma"),
            types.count("epithelial")
        ], dtype=np.float32)

    edge_index = np.array(edge_index).T if edge_index else np.zeros((2, 0), dtype=int)
    return topo_feat, edge_index

def build_neighbor_labels(df, coords, radius=50):
    labels = {}
    for cid in coords:
        x0, y0 = coords[cid]
        sub = df.copy()
        sub["dist"] = np.sqrt((sub.x-x0)**2 + (sub.y-y0)**2)
        neigh = sub[(sub.dist > 0) & (sub.dist <= radius)]

        if len(neigh) == 0:
            labels[cid] = -1
            continue

        maj = neigh.cell_type.value_counts().idxmax()
        cur = df.loc[cid, "cell_type"]

        mapping = {
            ("immune","immune"):0,
            ("immune","stroma"):1,
            ("immune","epithelial"):2,
            ("stroma","stroma"):3,
            ("stroma","epithelial"):4,
            ("epithelial","epithelial"):5,
        }
        labels[cid] = mapping.get((cur, maj), 6 if df.loc[cid,"region_type"]=="core" else 7)
    return labels

# =============================
# mian
# =============================

for sample in tqdm(SAMPLES):
    img = io.imread(f"{OUT_FOLDER}/Process/images/{sample}.tif")
    seg = load_hovernet_seg(f"{OUT_FOLDER}/Process/hovernet_seg/{sample}.npy")
    df = pd.read_csv(f"{OUT_FOLDER}/Process/csv/{sample}.csv", index_col=0)

    cell_feats, coords = extract_cell_features(img, seg)

    # 保存 cell image
    feat_dir = f"{OUT_FOLDER}/Process/image_features/{sample}"
    Path(feat_dir).mkdir(exist_ok=True)
    for cid, feat in cell_feats.items():
        np.save(f"{feat_dir}/{cid}.npy", feat)

    topo_feat, edge_index = build_topology_features(coords, df.cell_type.to_dict())
    np.save(f"{OUT_FOLDER}/Process/topology_features/{sample}_topo.npy", topo_feat)
    np.save(f"{OUT_FOLDER}/Process/topology_features/{sample}_edge.npy", edge_index)

    neighbor_label = build_neighbor_labels(df, coords)
    df["neighbor_label"] = df.index.map(neighbor_label)

    df.to_csv(f"{OUT_FOLDER}/Process/csv/{sample}.csv")
    df.filter(like="protein").to_pickle(
        f"{OUT_FOLDER}/Process/inputX_protein/{sample}.pkl"
    )

print("✅ Data processing done.")
