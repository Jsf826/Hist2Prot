# Pan-cancer Virtual Spatial Proteomics of Single-cell Tumor Microenvironment from Histology

This repository provides the official implementation of **Hist2Prot**, an AI-enabled virtual spatial proteomics framework that computationally reconstructs **single-cell–resolved spatial protein expression profiles** directly from standard hematoxylin and eosin (H&E) histopathology slides.

Hist2Prot integrates **cell-level morphological representations**, **cell–cell spatial topology**, and **multi-task learning** to infer high-dimensional protein expression landscapes across the tumor microenvironment (TME).

---

## 🔬 Framework Overview

<p align="center">
  <img src="Figures/Figure1.svg" width="70%">
</p>

- Operates at the **single-cell level**
- Inputs:
  - H&E histopathology images
  - Precomputed cell segmentation masks
- Outputs:
  - Spatially resolved, cell-level protein expression profiles

---

## 📁 Repository Structure
`
Histo2Prot/
├── DataProcess.py        # Data preprocessing and feature construction
├── model.py              # Hist2Prot model architecture
├── train.py              # Model training pipeline
├── inference.py          # Inference on unseen H&E slides
├── requirements.txt      # Dependency list
└── README.md
`

## Dependencies:

**Hardware:**

* NVIDIA GPU (Recommended) with CUDA support for PyTorch Lightning acceleration.

**Software:**

* Python (3.8+), PyTorch (≥ 2.0), TorchVision


Step 1: Data Preparation and Preprocessing
Input Data Organization:

Whole-slide H&E images: Stained on the same tissue section as molecular profiling.

Segmentation results: Cell/nuclei segmentation results stored as .npy files (each file contains instance-level cell masks).

Single-cell protein expression matrix: Used as regression targets during training.

> Note: Cell segmentation is performed externally using HoVerNet. Histo2Prot does not include a segmentation inference module; segmentation results are directly consumed as input.

Quality Control & Preprocessing:

Run DataProcess.py to handle the data pipeline.

Image normalization: Color normalization applied to reduce staining variability.

Tissue processing: Automatic tissue detection on whole-slide images and tiling into non-overlapping 20× patches.

Tile filtering: Removal of background-dominated tiles and exclusion of low-information or artifact-prone regions.

Final dataset: Constructs paired H&E patches with corresponding single-cell protein expression profiles.

Step 2: Train Histo2Prot
Install Dependencies:

Bash

pip install -r requirements.txt
Run Training: Start the training pipeline. The model utilizes multi-task loss across protein targets and GPU acceleration via PyTorch Lightning.

Bash

python train.py
Optimization: Multi-task loss across protein targets.

Regularization: Early stopping to prevent overfitting.

Outputs: Trained model checkpoints, training loss curves, and hyperparameter configurations (YAML).

Step 3: Inference
Run Inference: Apply the trained model to unseen H&E slides.

Bash

python inference.py
Model loading: Automatically loads trained Histo2Prot weights.

Inputs: H&E image patches and corresponding segmentation masks.

Outputs: Cell-level protein expression predictions and spatial proteomic maps across tissue regions.

🎯 Applications
Virtual spatial proteomics reconstruction

Tumor microenvironment (TME) profiling

Digital pathology–omics integration

Spatial biomarker discovery

Retrospective analysis of archived H&E cohorts
