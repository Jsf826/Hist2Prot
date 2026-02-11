# AI-enabled Pan-cancer Spatial Protein Profiles of Single-cell Tumor Microenvironment from Histology

Hist2Prot is a deep learning framework for inferring single-cell–resolved spatial proteomic profiles from routine whole-slide histopathology (H&E) images. The method integrates cell-level morphology, cell–cell spatial topology, and multi-task learning to reconstruct protein expression landscapes across the tumor microenvironment (TME).

<p align="center">
  <img src="Figures/Figure1.svg" width="70%">
</p>

---

## Repository Structure

```
Hist2Prot/
├── Data_Process.py      # Preprocessing: H&E images + segmentation + CSV → image_features / topology_features
├── Model.py             # Model: CellEncoderCNN + TopologyEncoderMLP + AttentionFusion + multi-task heads
├── train.py             # Training script
├── inference.py         # Inference script
├── utils_dataloader.py  # Dataset and DataLoader
├── requirements.txt
└── README.md
```

---

## Data Preparation

### Expected Directory Layout

Before running preprocessing, prepare the following structure (`{data_root}` is your data root, e.g. `demo_data`):

```
{data_root}/
├── Process/
│   ├── images/              # H&E images, one .tif per sample
│   │   └── {sample_id}.tif
│   ├── hovernet_seg/        # Cell segmentation, one .npy per sample (same size as image)
│   │   └── {sample_id}.npy
│   └── csv/                 # Single-cell table: coordinates, cell type, protein expression, etc.
│       └── {sample_id}.csv
├── train_samples.txt        # Training sample IDs, one per line
├── val_samples.txt          # Validation sample IDs
└── test_samples.txt         # Test sample IDs
```

- **images**: Whole or cropped H&E images in `.tif` format.
- **hovernet_seg**: Instance segmentation arrays (same dimensions as images); background = 0, each cell = positive integer ID; `.npy` format.
- **csv**: Row index = cell ID (matching segmentation IDs). Required columns: `x`, `y`, `cell_type`, `region_type`, `cell_type_id`, `region_type_id`, and protein columns (names containing `protein`, e.g. `protein_0`, `protein_1`, ...). Preprocessing adds `neighbor_label`.

**Note**: Cell segmentation must be done externally (e.g. HoVerNet). This repository only reads segmentation results and tables.

### Preprocessing (Generate Model Inputs)

With the structure above in place, run preprocessing to generate per-cell image and topology features:

```bash
python Data_Process.py --out_folder {data_root}
```

This creates:

- `Process/image_features/{sample_id}/{cell_id}.npy`: image patch features per cell
- `Process/topology_features/{sample_id}_topo.npy`: topology features
- `Process/topology_features/{sample_id}_edge.npy`: edge indices
- Updated `Process/csv/{sample_id}.csv` (adds `neighbor_label`, etc.)

---

## Training

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Training

```bash
python train.py --data_root {data_root} [optional args]
```

**Common arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | required | Data root directory |
| `--protein_dim` | 18 | Number of proteins (must match `protein_*` columns in CSV) |
| `--epochs` | 200 | Max training epochs |
| `--batch_size` | 1024 | Batch size |
| `--patience` | 15 | Early-stopping patience |
| `--num_workers` | 0 | DataLoader workers (use 0 on Windows if needed) |

Example (18 proteins):

```bash
python train.py --data_root demo_data --protein_dim 18 --epochs 200 --batch_size 1024
```

Training saves the best model and hyperparameters to `{data_root}/out/best_model.pth` and `{data_root}/out/hparam.yaml`.

---

## Inference

After training, run inference on the test set:

```bash
python inference.py --data_root {data_root} [optional args]
```

**Common arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | required | Data root directory |
| `--protein_dim` | 18 | Must match training |
| `--model_path` | out/best_model.pth | Model path relative to data_root |
| `--save_dir` | inference | Output directory for predictions (relative to data_root) |

Example (18 proteins):

```bash
python inference.py --data_root demo_data --protein_dim 18
```

Predictions are saved as `{data_root}/inference/{sample_id}_pred.npz`, with keys such as `protein`, `cell_type`, `neighbor_type`, `tissue_type`, `cell_id`.

---

## Full Pipeline Example (Starting from an Existing Data Root)

Assuming the data root is `demo_data` and `Process/images`, `Process/hovernet_seg`, `Process/csv`, and the three split files are already prepared:

```bash
# 1. Preprocessing
python Data_Process.py --out_folder demo_data

# 2. Training (set --protein_dim to match your protein count)
python train.py --data_root demo_data --protein_dim 18

# 3. Inference
python inference.py --data_root demo_data --protein_dim 18
```

---

## Requirements

Key dependencies:

- PyTorch 2.x
- TorchVision
- NumPy / Pandas / SciPy
- scikit-image
- PyYAML / tqdm

See `requirements.txt` for the full list.

---

## Applications

- Spatial proteomics reconstruction
- Tumor microenvironment (TME) profiling
- Digital pathology–omics integration
- Biomarker discovery
- Retrospective analysis of archived H&E slides
