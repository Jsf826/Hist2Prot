# Pan-cancer Virtual Spatial Proteomics of Single-cell Tumor Microenvironment from Histology

This repository provides the official implementation of **Hist2Prot**, an AI-enabled virtual spatial proteomics framework that computationally reconstructs **single-cell–resolved spatial protein expression profiles** directly from standard hematoxylin and eosin (H&E) histopathology slides.

Hist2Prot integrates **cell-level morphological representations**, **cell–cell spatial topology**, and **multi-task learning** to infer high-dimensional protein expression landscapes across the tumor microenvironment (TME).

---

## 🔬 Framework Overview

<p align="center">
  <img src="Figures/Figure1.svg" width="80%">
</p>

- Operates at the **single-cell level**
- Inputs:
  - H&E histopathology images
  - Precomputed cell segmentation masks
- Outputs:
  - Spatially resolved, cell-level protein expression profiles

---

## 📁 Repository Structure

```text
Histo2Prot/
├── DataProcess.py        # Data preprocessing and feature construction
├── model.py              # Hist2Prot model architecture
├── train.py              # Model training pipeline
├── inference.py          # Inference on unseen H&E slides
├── requirements.txt      # Dependency list
└── README.md



##📦 Data Preparation
Input Data

Whole-slide H&E images

Cell / nuclei segmentation results

Precomputed externally

Stored as .npy files

Each file contains instance-level cell masks

Single-cell protein expression matrix

Used as regression targets during training

Note

Cell segmentation is performed using HoVerNet

Hist2Prot does not include a segmentation inference module

Segmentation results are directly consumed as input

##🧪 Quality Control and Preprocessing

H&E staining:

Performed on the same tissue section as molecular profiling

Image normalization:

Color normalization applied to reduce staining variability

Tissue processing:

Automatic tissue detection on whole-slide images

Tiling into non-overlapping 20× patches

Tile filtering:

Removal of background-dominated tiles

Exclusion of low-information or artifact-prone regions

Final dataset:

Paired H&E patches

Corresponding single-cell protein expression profiles

##🚀 Training
Step 1: Install Dependencies
pip install -r requirements.txt

Step 2: Run Training
python train.py


Optimization:

Multi-task loss across protein targets

Regularization:

Early stopping to prevent overfitting

Acceleration:

GPU support via PyTorch Lightning

Outputs:

Trained model checkpoints

Training loss curves

Hyperparameter configurations (YAML)

##🔍 Inference
python inference.py


Model loading:

Loads trained Hist2Prot weights

Inputs:

H&E image patches

Corresponding segmentation masks

Outputs:

Cell-level protein expression predictions

Spatial proteomic maps across tissue regions

##⚙️ Requirements

PyTorch ≥ 2.0

TorchVision

Scanpy

Squidpy

Scikit-learn

NumPy

Pandas

SciPy

See requirements.txt for the complete dependency list.

##🎯 Applications

Virtual spatial proteomics reconstruction

Tumor microenvironment (TME) profiling

Digital pathology–omics integration

Spatial biomarker discovery

Retrospective analysis of archived H&E cohorts
