<div align="center">
  <img src="https://github.com/user-attachments/assets/3a32f78f-215c-45dd-ac52-b558364eb89f" width="900" />
</div>

<div align="center">
  <a href="https://www.python.org/downloads/release/python-3110/" target="_blank">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python 3.11"></a>
  <a href="https://github.com/Dalageo/vitdec-anomaly-detection/blob/prd/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/License-AGPL%20v3-800080" alt="License: AGPLv3"></a>
  <img src="https://img.shields.io/github/stars/Dalageo/vitdec-anomaly-detection?style=social" alt="GitHub stars">
</div> 

# ViT-Decoder Anomaly Detection

This repository contains the code and findings for the thesis project: **"Investigating the Performance of a Vision Transformer Model for Anomaly Detection in Laser Metal Deposition Imaging."** The research explores a [Vision Transformer (ViT)](https://arxiv.org/pdf/2010.11929) approach to identify anomalies within Laser Metal Deposition (LMD) images.

While Convolutional Neural Networks (CNNs) are the standard for finding defects in the computer vision sector, Transformer-based models remain largely underexplored in this domain. To bridge this gap, this project introduces a custom ViT-Decoder architecture designed to analyze melt pool images and cross-reference predictions with existing CNN models.

Evaluated on LMD melt pool images, the model achieved **99.78% accuracy** on the core dataset and successfully generalized to predict anomalies across 6,497 unseen frames with **97.83% accuracy**.

📌 *The melt pool image dataset used to achieve these results is not provided in this repository. For more information about the thesis, visit: [Investigating the Performance of a Vision Transformer Model for Anomaly Detection in Laser Metal Deposition Imaging](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1886506&dswid=7365)*

<br>

## 🛠️ Workflow Logic

The model combines a **Vision Transformer (ViT-Base/16)** encoder with a **Convolutional Transpose Decoder**. The ViT encoder produces two outputs in parallel: a **CLS token** for classification and **patch embeddings** for reconstruction. This dual-head design allows the model to simultaneously learn *what* is anomalous (classifier) and *how* normal images should look (reconstructor). 

The workflow below illustrates the solution that ultimately drove the best results: a hybrid Supervised-Reconstructive approach. By using Self-Supervised Learning to reconstruct features from base data, combined with Supervised Learning for classification between labels 0 and 1, this dual-path logic maximizes overall anomaly detection accuracy.

<div align="center">
  <img src="https://github.com/user-attachments/assets/ffaa58cb-fbfb-4286-8b7c-57d52770bc51" width="700" />
</div>

### 1️⃣ Feature Extraction
The process begins when an **Input Image** is fed into the **Vision Transformer** (using the ViT-Base/16 configuration initialized from `vit_base_patch16_384.npz` weights). This encoder extracts visual features and splits the output into two parallel streams:
* **Classification Path:** Utilizes the CLS token logits.
* **Reconstruction Path:** Utilizes the extracted patch embeddings.

### 2️⃣ The Reconstruction Check
In the reconstruction path, the **Extracted Features** are routed to a Convolutional Transpose **Decoder**, which attempts to rebuild the image. The system then evaluates the difference between the input and the output at the **Check Reconstruction Error Bound** stage:
* **Error Outside Bound:** If the reconstruction error is high, it means the model is struggling to process unfamiliar, anomalous patterns. The workflow immediately bypasses the classifier and flags the image as **Abnormal**.
* **Error Within Bound:** If the image passes this initial structural test, the decision is delegated to the classification path (indicated by the dashed line).

### 3️⃣ The Classification Head
If the image triggers the second stage, the **CLS logits output** from the encoder is processed through a **Softmax Activation** function. This classification head leverages the ViT's learned representations to catch more subtle abnormalities that passed the reconstruction test, outputting the final **Class** as either **Normal** or **Abnormal**.

**Summary:** The model catches distinct structural anomalies using the reconstruction error first, while relying on the classifier to capture more nuanced defects.

<br>

## 🎯 Training Strategy

### Dual-Objective Loss

The model optimizes two losses simultaneously, each applied selectively based on the sample's label:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{reconstruction}} + \mathcal{L}_{\text{classification}}$$

| Loss | Function | Applied To | Purpose |
|------|----------|------------|---------|
| $\mathcal{L}_{\text{recon}}$ | **L1 Loss** (Mean Absolute Error) | `base` (y=-1) and `normal` (y=0) samples | Teach the decoder to reconstruct normal melt pool morphology |
| $\mathcal{L}_{\text{cls}}$ | **Cross-Entropy Loss** | `normal` (y=0) and `anomaly` (y=1) samples | Teach the CLS token to discriminate between classes |

> Anomaly images are **never** used for reconstruction. This is intentional: the decoder should only learn the distribution of normal images, so that anomalous inputs produce high reconstruction error.

### Selective Gradient Flow

Within each training batch, samples are masked by label and routed through different model paths:

```
Batch: [base, normal, anomaly] samples
        │
        ├── recon_mask = (y == -1) | (y == 0)  ──► Encoder(no CLS) → Decoder → L1 Loss
        │
        └── cls_mask   = (y == 0) | (y == 1)   ──► Encoder(CLS)    → Head    → CE Loss
```

- The **ViT encoder** receives gradients from *both* objectives (it participates in both the reconstruction and classification forward passes).
- The **Decoder** receives gradients only from the reconstruction objective.
- The combined loss is backpropagated in a single pass using Automatic Mixed Precision (AMP) with `GradScaler`.

### Optimizer Configuration

Two separate Adam optimizers are used with distinct learning rates, reflecting the asymmetric roles of the encoder and decoder:

| Component | Optimizer | Learning Rate | Weight Decay | Role |
|-----------|-----------|:---:|:---:|------|
| ViT Encoder | Adam | 1e-5 | 1e-5 | Fine-tune pre-trained weights conservatively |
| Decoder | Adam | 1e-3 | 1e-5 | Train from scratch — higher LR needed |

Additional training controls:
- **ReduceLROnPlateau** scheduler on the encoder's optimizer (factor=0.1, patience=2)
- **Early Stopping** with patience=5 and δ=0.00001 on the combined validation loss
- **Automatic Mixed Precision (AMP)** enabled by default

<br>

## 📉 Threshold & Anomaly Detection Logic

The inference pipeline uses a **two-phase evaluation** strategy: first identify thresholds on the Bound set, then apply those thresholds to the unseen Test set.

### 1️⃣ Anomaly Score Calculation

For each image, the reconstruction error (anomaly score) is computed as the **pixel-wise MSE** between the original and reconstructed image, averaged across channels and spatial dimensions:

$$S_i = \frac{1}{C \cdot H \cdot W} \sum_{c,h,w} \left( x_{c,h,w}^{(i)} - \hat{x}_{c,h,w}^{(i)} \right)^2$$

where $x^{(i)}$ is the original image and $\hat{x}^{(i)}$ is the decoder's reconstruction.

> **Note:** While training uses L1 Loss for its gradient properties, the evaluation phase uses MSE to compute anomaly scores, as squared error amplifies larger deviations, making anomalies more separable.

### 2️⃣ Decision Boundary Estimation (Bound Phase)

Using the Bound subset (never seen during training), the system computes the score distributions for both classes and identifies three decision zones:

```
Score Axis ──────────────────────────────────────────────────────────►
│◄── Definite Zone A ──►│◄── Overlap Zone ──►│◄── Definite Zone B ──►│
│   (Reconstruction     │   (Ambiguous:      │   (Reconstruction     │
│    alone decides)     │    Classifier      │    alone decides)     │
│                       │    decides)        │                       │
                   overlap_start         overlap_end
```

### 3️⃣ Combined Inference (Test Phase)

At test time, each sample's anomaly score determines which decision-maker is used:

```
For each test image:
    1. Compute reconstruction score S
    2. Get CLS token class probabilities via Softmax
    3. Route to decision logic:

    if perfect_separation:
        → Use standalone reconstructor threshold only
    else:
        if S < overlap_start  OR  S > overlap_end:
            → Definite zone: Reconstructor decides (based on threshold rules)
        else:
            → Overlap zone: Classifier decides (argmax of CLS softmax)
```

<br>


## 🏗️ Dataset Structure 

### Dataset Configuration

To ensure compatibility with the custom data loaders and the hybrid training strategy, your dataset must be organized into the following directory structure, where each folder serves a specific role in teaching the Vision Transformer and Decoder:

| Folder | Label | Category | Training Type | Model Objective |
|--------|-------|----------|---------------|-----------------|
| `train/base/` | `-1` | `base` | Self-Supervised | **Reconstruction Only:** Learns general, healthy image features by reconstructing the inputs. No classification occurs here. |
| `train/normal/` | `0` | `normal` | Supervised (Multi-Task) | **Dual-Task:** Learns to both reconstruct images and classify them as "Normal" (0). |
| `train/anomaly/`| `1` | `anomaly` | Supervised | **Classification Only:** Learns to identify defects (1). Strictly excluded from the reconstruction task to prevent the model from learning how to reconstruct anomalies. |
| `test/normal/` | `0` | `test` | Evaluation | Verifies reconstruction accuracy and measures the false-positive classification rate. |
| `test/anomaly/` | `1` | `test` | Evaluation | Verifies defect detection capability and evaluates reconstruction error thresholds. |

This three-tier labeling scheme is the core of the hybrid approach: 
- **The Reconstruction Objective** learns exclusively from normal-looking samples (`base` + `normal`). Because it is never taught how to reconstruct a defect, the model naturally produces high reconstruction errors when fed an anomalous image later.
- **The Classification Objective** learns the explicit decision boundary between normal and anomaly.

### Data Splitting

The training and evaluation sets are split internally to optimize and properly evaluate the model using **stratified sampling** (preserving class ratios):

| Subset | Source | Ratio | Purpose |
|--------|--------|-------|---------|
| **Training** | `train/` | 85% of normal + base | Model parameter optimization |
| **Validation** | `train/` | 15% of normal + base | Early stopping & scheduler decisions |
| **Bound** | `test/` | 50% | Identify decision thresholds |
| **Test** | `test/` | 50% | Final evaluation |

> **Note:** The **Bound** subset is critical. It is used to calculate the reconstruction-score distributions for normal vs. anomalous samples and to define the decision boundaries *before* the model ever sees the final Test subset.

<br>

## 💻 Setup & Usage

### Prerequisites

- Python ≥ 3.11, < 3.13
- CUDA-compatible GPU (recommended)

### Installation

```bash
git clone https://github.com/Dalageo/vit-dec-anomaly-detection
cd vit-dec-anomaly-detection
pip install poetry
poetry install
```

### Training

```bash
python -m app.train
```

### Evaluation

```bash
python -m app.test
```

## 📁 Project Structure

```
├── app/
│   ├── config.py                 # Hyperparameters, paths, augmentation config
│   ├── train.py                  # Training entry point
│   ├── test.py                   # Evaluation entry point
│   ├── model/
│   │   ├── vit_decoder.py        # ViT encoder, Decoder, and ViTDecoder architecture
│   │   └── weights.py            # Pre-trained weight loading & decoder initialization
│   ├── utils/
│   │   ├── load_dataset.py       # Dataset class, transforms, and data splitting
│   │   ├── train_utils.py        # Trainer class with dual-objective training loop
│   │   ├── bound_utils.py        # Threshold calibration & boundary computation
│   │   ├── test_utils.py         # Combined inference with routing logic
│   │   ├── visualizer.py         # Plotting: reconstructions, loss curves, confusion matrices
│   │   └── log_utils.py          # Logger configuration & metric tracking
│   └── checkpoints/              # Saved model weights & training logs
├── dataset/                      # Melt pool image data (train/test splits)
├── model_weights/                # Pre-trained ViT weights (.npz)
├── scripts/
│   └── deploy_prd.sh             # Production deployment script
├── pyproject.toml                # Poetry dependency configuration
└── README.md
```

## Citation

If you use this code or find this research helpful in your work, please cite the associated Master's thesis:

```bibtex
@mastersthesis{Dalageorgos_Investigating_the_Performance_2024,
  author = {Dalageorgos, Konstantinos},
  month = may,
  title = {{Investigating the Performance of a Vision Transformer Model for Anomaly Detection in Laser Metal Deposition Imaging}},
  url = {https://www.diva-portal.org/smash/get/diva2:1886506/FULLTEXT01.pdf},
  year = {2024}
}
