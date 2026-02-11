# DataProcess.py
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import random
from skimage import io


CELL_DIAMETER = 32
NEIGHBOR_RADIUS = 50


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


def run_preprocess(out_folder: str):
    """
    Input data:
        {out_folder}/Process/images/{sample}.tif
        {out_folder}/Process/hovernet_seg/{sample}.npy
        {out_folder}/Process/csv/{sample}.csv
    To:
        {out_folder}/Process/image_features/{sample}/{cell_id}.npy
        {out_folder}/Process/topology_features/{sample}_topo.npy
        {out_folder}/Process/topology_features/{sample}_edge.npy
    """
    fix_seed(2024)

    out_folder = str(out_folder)
    proc_dir = Path(out_folder) / "Process"

    (proc_dir / "image_features").mkdir(parents=True, exist_ok=True)
    (proc_dir / "topology_features").mkdir(parents=True, exist_ok=True)
    (proc_dir / "inputX_protein").mkdir(parents=True, exist_ok=True)

    samples = [
        f.replace(".tif", "")
        for f in os.listdir(proc_dir / "images")
        if f.endswith(".tif")
    ]

    for sample in tqdm(samples):
        img = io.imread(proc_dir / "images" / f"{sample}.tif")
        seg = load_hovernet_seg(proc_dir / "hovernet_seg" / f"{sample}.npy")
        df = pd.read_csv(proc_dir / "csv" / f"{sample}.csv", index_col=0)

        try:
            df.index = df.index.astype(int)
        except Exception:

            pass

        cell_feats, coords = extract_cell_features(img, seg)

        
        df_index_set = set(int(i) for i in df.index)
        common_ids = [cid for cid in coords.keys() if int(cid) in df_index_set]
        coords = {cid: coords[cid] for cid in common_ids}
        cell_feats = {cid: cell_feats[cid] for cid in common_ids}
        df = df.loc[common_ids]

        # save cell image
        feat_dir = proc_dir / "image_features" / sample
        feat_dir.mkdir(exist_ok=True)
        for cid, feat in cell_feats.items():
            np.save(feat_dir / f"{cid}.npy", feat)

        cell_types = df.cell_type.to_dict()
        topo_feat, edge_index = build_topology_features(coords, cell_types)
        np.save(proc_dir / "topology_features" / f"{sample}_topo.npy", topo_feat)
        np.save(proc_dir / "topology_features" / f"{sample}_edge.npy", edge_index)

        neighbor_label = build_neighbor_labels(df, coords)
        df["neighbor_label"] = df.index.map(neighbor_label)

        df.to_csv(proc_dir / "csv" / f"{sample}.csv")
        df.filter(like="protein").to_pickle(
            proc_dir / "inputX_protein" / f"{sample}.pkl"
        )

    print("✅ Data processing done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_folder",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    run_preprocess(args.out_folder)
