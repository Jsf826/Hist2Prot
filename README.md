[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22991168.svg)](https://doi.org/10.5281/zenodo.22991168)

# Hist2Prot

Hist2Prot predicts single-cell spatial protein expression from H&E images.

## Installation

```bash
conda create -n pytorch_ST python=3.9 -y
conda activate pytorch_ST
pip install -r requirements.txt
```

Run all commands from the repository root. Shared defaults are defined in
`configs/hist2prot.json`.

## Input

Each sample requires:

- an H5AD feature table;
- an H&E whole-slide image;
- a nucleus instance mask;
- an expanded-cell instance mask.

The H5AD `obs` table must contain `cell_id`, `cell_x`, `cell_y`, and protein
columns ending in `_intensity_mean`.

For multi-task training, provide precomputed non-negative integer labels in one
auxiliary CSV per sample. 
## 1. Preprocessing

```bash
python Data_Process.py --config_file configs/hist2prot.json
```

By default, all samples under `raw_root` are processed and split 70:30 at the
patient level. To use a fixed split, provide a CSV containing `sample_id` and
`split`, where `split` is `train` or `test`:

```bash
python Data_Process.py \
  --config_file configs/hist2prot.json \
  --split_csv sample_splits.csv
```


## 2. Precompute Zarr

```bash
python precompute_zarr.py \
  --config_file configs/hist2prot.json \
  --splits train,test
```


## 3. Training

```bash
python train.py \
  --config_file configs/hist2prot.json \
  --gpu 0
```


Protein-only workflow when auxiliary labels are unavailable:

```bash
python precompute_zarr.py --splits train,test --no_aux_tasks
python train.py --no_aux_tasks --gpu 0
```

## 4. Test

Run each held-out patient separately:

```bash
python inference.py \
  --data_root data \
  --model_path out/final_model.pth \
  --split test \
  --patient_id PATIENT_ID \
  --gpu 0
```


## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-
NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/).
