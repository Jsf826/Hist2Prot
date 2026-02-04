#	AI-enabled Pan-cancer Spatial Protein Profiles of Single-cell Tumor Microenvironment from Histology

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
```
Histo2Prot/
├── DataProcess.py        # Data preprocessing and feature construction
├── model.py              # Hist2Prot model architecture
├── train.py              # Model training pipeline
├── inference.py          # Inference on unseen H&E slides
├── requirements.txt      # Dependency list
└── README.md
```

## Dependencies:

**Hardware:**

* NVIDIA A6000 GPU (*8) with CUDA support for PyTorch Lightning acceleration.

**Software:**

* Python (3.8+), PyTorch (≥ 2.0), TorchVision

## Step 1: WSI Segmentation

Segmenting the Whole Slide Images (WSIs) to obtain nuclei masks.

* **Tool:** We utilize **HoVerNet** for simultaneous nuclear segmentation and classification.
* **Source:** [HoVerNet GitHub Repository](https://github.com/vqdang/hover_net)
* **Procedure:**
    1.  Install HoVerNet following their official instructions.
    2.  Run inference on your raw H&E slides.
    3.  Save the output as **`.npy`** files.
* **Run:**
    ```bash
    # Run HoVerNet inference (refer to the official repository for specific arguments)
    python run_infer.py \
      --gpu='0' \
      --model_path=hovernet_fast_panoptic.tar \
      --nr_inference_workers=4 \
      --input_dir=/path/to/raw_wsis \
      --output_dir=/path/to/segmentation_results \
      --save_json=False \
      --save_mask=True
    ```


## Step 2: Data Preparation and Preprocessing

After obtaining the segmentation results (from Step 0), organize your data and run the preprocessing pipeline.

* **Input Data Organization:**
    * **Whole-slide H&E images:** Must be stained on the same tissue section used for molecular profiling.
    * **Segmentation results:** Cell/nuclei segmentation masks stored as **`.npy`** files (each file contains instance-level cell masks).
    * **Single-cell protein expression matrix:** Used as regression targets during training.

> **Note:** Cell segmentation is performed externally using **HoVerNet**. Histo2Prot does not include a segmentation inference module; segmentation results are directly consumed as input.

* **Quality Control & Preprocessing:**
    Run the main processing script to handle the pipeline:
    ```bash
    python DataProcess.py
    ```
    * **Image normalization:** Color normalization is applied to reduce staining variability.
    * **Tissue processing:** Performs automatic tissue detection on whole-slide images and tiles them into non-overlapping **20× patches**.
    * **Tile filtering:** Removes background-dominated tiles and excludes low-information or artifact-prone regions.
    * **Final dataset:** Constructs paired H&E patches with corresponding single-cell protein expression profiles, topology_features, neighbor_labels and tissue type.

## Step 3: Train Histo2Prot
* **Install Dependencies:**
    First, ensure all required libraries are installed:
    ```bash
    pip install -r requirements.txt
    ```

* **Run Training:**
    Start the training pipeline.
    ```bash
    python train.py
    ```
    * **Optimization:** Multi-task loss applied across protein targets.
    * **Regularization:** Implements early stopping to prevent overfitting.
    * **Outputs:** Trained model checkpoints.

## Step 4: Inference
* **Run Inference:**
    Apply the trained model to unseen H&E slides.
    ```bash
    python inference.py
    ```
    * **Model loading:** Automatically loads trained Histo2Prot weights.
    * **Inputs:** H&E image patches and corresponding segmentation masks.
    * **Outputs:** Cell-level protein expression predictions and spatial proteomic maps across tissue regions.

## 🎯 Applications
Virtual spatial proteomics reconstruction

Tumor microenvironment (TME) profiling

Digital pathology–omics integration

Spatial biomarker discovery

Retrospective analysis of archived H&E cohorts
