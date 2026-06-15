# AI-enabled Pan-cancer Spatial Protein Profiles of Single-cell Tumor Microenvironment from Histology

## Hist2Prot

**Hist2Prot is an AI-enabled framework that uses biologically informed
multi-task supervision to predict pan-cancer, single-cell-resolved spatial
protein profiles of the tumor microenvironment directly from routine H&E
histopathology slides.**

## Overview

The workflow contains four stages:

1. Read an H&E WSI, instance masks, cell coordinates, and protein measurements.
2. Divide each WSI into patches and collect the cells located in each patch.
3. Encode cell morphology using a CNN and geometry-only topology using a GCN.
4. Fuse both representations to predict protein expression for every cell.

```text
H&E WSI + cell mask + single-cell table
                  |
             WSI tiling
                  |
       cells grouped within patches
                  |
        +---------+---------+
        |                   |
  morphology CNN      topology GCN
        |                   |
        +---- attention ----+
                  |
    single-cell protein prediction
```

## Repository structure

```text
.
|-- Data_Process.py       # Data preprocessing and dataset splits
|-- utils_dataloader.py   # Patch reading and cell feature extraction
|-- Model.py              # Model architecture
|-- train.py              # Training and validation
|-- inference.py          # Inference and evaluation
|-- metrics.py            # PCC and SSIM metrics
|-- requirements.txt
|-- Row data/             # Example raw input
`-- demo_data/            # Processed data and outputs
```

## Installation

The code was developed for Python and PyTorch. Install the pinned dependencies
with:

```bash
conda create -n pytorch_ST python=3.9 -y
conda activate pytorch_ST
pip install -r requirements.txt
```


Each feature matrix is read with `anndata.read_h5ad` or
`scanpy.read_h5ad`. Its `obs` table must contain:

| Field | Description |
|---|---|
| `cell_id` | Unique cell identifier |
| `cell_x`, `cell_y` | Cell centroid coordinates in WSI pixel space |
| `*_intensity_mean` | Protein-expression measurements |

## Data preprocessing

Run preprocessing with:

```bash
conda run -n pytorch_ST python Data_Process.py \
  --raw_root "Row data" \
  --out_folder "demo_data" \
  --split_csv "sample_splits.csv" \
  --patch_size 256 \
  --stride 256 \
  --min_cells 40
```

The processed dataset is written to:

```text
demo_data/
|-- Process/
|   |-- samples.csv
|   |-- sample_splits.csv
|   |-- metadata.json
|   |-- protein_norm.json
|   |-- csv/
|   `-- patches/
|       |-- train_patches.csv
|       |-- val_patches.csv
|       `-- test_patches.csv
|-- train_samples.txt
|-- val_samples.txt
`-- test_samples.txt
```

### Protein normalization and outlier cells

Protein-specific thresholds are fitted using training cells only:

```text
non-negative clipping -> log1p -> training-set percentiles -> min-max [0, 1]
```

By default, a cell is removed if **any protein** falls below the 2.5th
percentile or above the 97.5th percentile. Filtered cells are removed from the
processed CSV files and therefore cannot enter training, validation, or test
inference. Patch lists are regenerated after filtering.

To retain all cells and only clip outlying expression values:

```bash
conda run -n pytorch_ST python Data_Process.py --keep_outlier_cells
```

To disable protein normalization entirely:

```bash
conda run -n pytorch_ST python Data_Process.py --no_normalize_protein
```

## Model

The morphology CNN and topology GCN produce cell-level representations. An
attention module fuses both branches before protein regression. Optional
classification heads predict cell type, tissue region, and neighborhood class.

Cell images can be extracted in two ways:

| Mode | Description |
|---|---|
| `crop` | Use a square RGB crop centered on each cell coordinate |
| `mask` | Use the same crop but retain only the cell instance from `nuclei_exp.npy` |

Macenko stain normalization can optionally be applied dynamically after
reading each WSI patch.

## Training

Run protein-regression training with:

```bash
conda run -n pytorch_ST python train.py \
  --data_root "demo_data" \
  --cell_image_mode mask \
  --batch_size 48 \
  --epochs 600 \
  --gpu 0
```

The best model is selected by the highest validation Pearson correlation
coefficient (PCC). Training also reports validation SSIM and loss. Outputs are
written under `demo_data/out2/` by default:

```text
demo_data/out2/
|-- best_model.pth
`-- hparam.yaml
```

### Device selection

```bash
# Single GPU
conda run -n pytorch_ST python train.py --gpu 0

# Selected GPUs using torch.nn.DataParallel
conda run -n pytorch_ST python train.py --gpus 0,1,2

# Force CPU execution
conda run -n pytorch_ST python train.py --gpu -1
```

### Optional stain normalization

```bash
conda run -n pytorch_ST python train.py --stain_norm macenko
```

### Optional auxiliary tasks

Auxiliary tasks are disabled by default. Enable all available tasks with:

```bash
conda run -n pytorch_ST python train.py --use_aux_tasks
```

Individual tasks can be disabled independently:

```bash
# Cell-type task only
conda run -n pytorch_ST python train.py \
  --use_aux_tasks \
  --no_tissue_task \
  --no_neighbor_task
```

Auxiliary labels are read from one CSV per WSI, using
`demo_data/Process/csv/{sample_id}_aux.csv` by default. Each file must contain
`cell_id` and the labels required by the enabled tasks:

| Task | Accepted label columns |
|---|---|
| Cell type | `cell_type_id` or `cell_type` |
| Tissue region | `region_type_id`, `region_type`, `tissue_type_id`, or `tissue_type` |
| Neighborhood | `neighbor_label`, `neighbor_type_id`, `neighbor_type`, or `neighborhood_label` |

## Inference

Run inference on the held-out test WSI:

```bash
conda run -n pytorch_ST python inference.py \
  --data_root "demo_data" \
  --model_path "out2/best_model.pth" \
  --split test \
  --cell_image_mode mask \
  --gpu 0
```
