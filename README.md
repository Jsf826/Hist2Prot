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

The topology branch uses only spatial geometry:

- number of neighboring cells;
- mean neighbor distance;
- nearest-neighbor distance;
- local cell density.

Cell-type and tissue-region labels are not used as topology inputs. They can
only be enabled as auxiliary supervision.

## Repository structure

```text
.
├── Data_Process.py       # WSI metadata, cell table, normalization, and splits
├── utils_dataloader.py   # Dynamic patch reading and cell feature extraction
├── Model.py              # CNN, GCN, attention fusion, and prediction heads
├── train.py              # Training and validation
├── inference.py          # Test-time prediction and evaluation
├── metrics.py            # PCC and SSIM metrics
├── requirements.txt
├── Row data/             # Example raw input
└── demo_data/            # Default processed-data and output root
```

## Installation

The code was developed for Python and PyTorch. Install the pinned dependencies
with:

```bash
conda create -n pytorch_ST python=3.9 -y
conda activate pytorch_ST
pip install -r requirements.txt
```

All examples below assume that commands are run from the repository root.

## Input data

The expected raw-data layout is:

```text
Row data/
├── h5_files/
│   ├── A02_feature_matrix.h5
│   └── B02_feature_matrix.h5
└── HE_mask/
    ├── A02/
    │   ├── A02_HE.ome.tiff
    │   └── mask/
    │       ├── nuclei.npy
    │       └── nuclei_exp.npy
    └── B02/
        ├── B02_HE.ome.tiff
        └── mask/
            ├── nuclei.npy
            └── nuclei_exp.npy
```

Each feature matrix is read with `anndata.read_h5ad` or
`scanpy.read_h5ad`. Its `obs` table must contain:

| Field | Description |
|---|---|
| `cell_id` | Unique cell identifier |
| `cell_x`, `cell_y` | Cell centroid coordinates in WSI pixel space |
| `*_intensity_mean` | Protein-expression measurements |

The following control/staining channels are excluded:

```text
MsIgG1_intensity_mean
MsIgG2a_intensity_mean
cytoplasmicstain_intensity_mean
nuclearstain_intensity_mean
```

`nuclei.npy` stores nuclear instances. `nuclei_exp.npy` stores expanded cell
instances and is used by the mask-based cell-image extraction mode.

## Data preprocessing

Run preprocessing with:

```bash
conda run -n pytorch_ST python Data_Process.py \
  --raw_root "Row data" \
  --out_folder "demo_data" \
  --patch_size 256 \
  --stride 256 \
  --min_cells 40
```

The processed dataset is written to:

```text
demo_data/
├── Process/
│   ├── samples.csv
│   ├── metadata.json
│   ├── protein_norm.json
│   ├── csv/
│   │   ├── A02.csv
│   │   └── B02.csv
│   └── patches/
│       ├── all_patches.csv
│       ├── train_patches.csv
│       ├── val_patches.csv
│       └── test_patches.csv
├── train_samples.txt
├── val_samples.txt
└── test_samples.txt
```

### Dataset partitioning

WSIs are ordered by their feature-matrix filenames.

- Patches from the first WSI are randomly divided into training and validation
  sets using a 70:30 split by default.
- All patches from subsequent WSI(s) are assigned to the test set.

For the provided example, `A02` supplies training and validation patches, while
`B02` is retained as an independent test WSI. No patch from the test WSI is
used for training or validation.

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

Normalization thresholds and filtering counts are recorded in
`demo_data/Process/protein_norm.json`.

## Model

For each patch, the model receives:

- cell-centered RGB image tensors;
- four geometry-derived topology features per cell;
- a fixed-radius cell adjacency matrix.

The morphology CNN and topology GCN produce cell-level representations. An
attention module fuses both branches before protein regression. Optional
classification heads predict cell type, tissue region, and neighborhood class.

Cell images can be extracted in two ways:

| Mode | Description |
|---|---|
| `crop` | Use a square RGB crop centered on each cell coordinate |
| `mask` | Use the same crop but retain only the cell instance from `nuclei_exp.npy` |

If a CSV `cell_id` does not match the instance-mask label, mask mode uses the
instance label at the cell centroid.

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
├── best_model.pth
└── hparam.yaml
```

The final training summary reports the epoch, PCC, SSIM, and loss associated
with the best validation PCC.

### Device selection

```bash
# Single GPU
conda run -n pytorch_ST python train.py --gpu 0

# Selected GPUs using torch.nn.DataParallel
conda run -n pytorch_ST python train.py --gpus 0,1,2

# Force CPU execution
conda run -n pytorch_ST python train.py --gpu -1
```

Invalid or unavailable GPU identifiers are ignored. If no requested GPU is
available, execution falls back to CPU and prints the selected device.

### Optional stain normalization

```bash
conda run -n pytorch_ST python train.py --stain_norm macenko
```

Stain normalization is performed dynamically by the Dataset and increases
data-loading time.

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

Use `--aux_label_dir` and `--aux_label_suffix` to change the lookup location
and filename suffix.

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

Use the same `cell_image_mode`, stain-normalization setting, model dimensions,
and auxiliary-head dimensions used during training.

Inference reports PCC and SSIM by default and writes:

```text
demo_data/inference/
├── {sample_id}_test_pred.npz
└── test_metrics.npz
```

Each prediction archive contains:

| Key | Description |
|---|---|
| `cell_id` | Cell identifiers |
| `patch_id` | Source patch identifiers |
| `coords` | WSI-space cell coordinates |
| `protein_names` | Ordered protein names |
| `protein` | Predicted protein-expression matrix |
| `cell_type` | Predicted auxiliary cell-type class |
| `neighbor_type` | Predicted auxiliary neighborhood class |
| `tissue_type` | Predicted auxiliary tissue class |

Disable evaluation when ground-truth proteins are unavailable:

```bash
conda run -n pytorch_ST python inference.py --no_eval
```

Multi-GPU and CPU inference use the same `--gpus` and `--gpu -1` options as
training.

## Evaluation

The implementation reports:

- **PCC**: Pearson correlation between predicted and measured protein
  expression, computed per protein over valid cells and then averaged.
- **SSIM**: structural similarity after rasterizing cell-level predictions and
  targets into patch-level spatial maps.

Model selection and early stopping use validation PCC as the primary metric.

## Reproducibility notes

- Protein names and ordering are stored in `Process/metadata.json`.
- Protein-normalization parameters are fitted on training cells only.
- Test WSIs are isolated from training and validation.
- Random seeds are configurable in preprocessing and training.
- WSI files are read from their original locations and are not copied into the
  processed dataset.
- Dynamic Macenko normalization and mask-based extraction can substantially
  increase data-loading time.
- The default outlier rule removes a cell when any measured protein is outside
  its training-derived percentile interval; this is intentionally strict and
  should be reported when used.

## Citation

Citation information will be added with the associated publication. Until
then, please cite the accompanying manuscript when using this implementation.

## License

No license file is currently included. Please contact the repository owner
before redistributing or using the code beyond research evaluation.
