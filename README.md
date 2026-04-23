<div align="center">
  <img src="https://github.com/user-attachments/assets/3a32f78f-215c-45dd-ac52-b558364eb89f" width="900" />
</div>

<div align="center">
  <a href="https://www.python.org/downloads/release/python-3110/" target="_blank">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python 3.11"></a>
  <a href="https://pytorch.org/get-started/locally/" target="_blank">
    <img src="https://img.shields.io/badge/PyTorch-2.9.1-orange.svg" alt="PyTorch 2.9.1"></a>
  <a href="https://developer.nvidia.com/cuda-12-8-0-download-archive" target="_blank">
  <img src="https://img.shields.io/badge/CUDA-12.8-brightgreen.svg" alt="CUDA 12.8"></a>
  <a href="https://github.com/Dalageo/vitdec-anomaly-detection/blob/prd/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/License-AGPL%20v3-800080" alt="License: AGPLv3"></a>
  <img src="https://img.shields.io/github/stars/Dalageo/vitdec-anomaly-detection?style=social" alt="GitHub stars">
</div> 

# ViT-Decoder for Anomaly Detection <img src="https://github.com/user-attachments/assets/47c5dbfb-4ff0-4102-83c7-1a8b60e02a5a" width="40">

This repository contains the code and findings for the thesis project: **"Investigating the Performance of a Vision Transformer Model for Anomaly Detection in Laser Metal Deposition Imaging."** The research explores a [Vision Transformer (ViT)](https://arxiv.org/pdf/2010.11929) approach to identify anomalies within Laser Metal Deposition (LMD) images.

While Convolutional Neural Networks (CNNs) are the standard for finding defects in the computer vision sector, Transformer-based models remain largely underexplored in this domain. To bridge this gap, this project introduces a custom ViT-Decoder architecture designed to analyze melt pool images and cross-reference predictions with existing CNN models.

Evaluated on LMD melt pool images, the model achieved **99.78% accuracy** on the core dataset and successfully generalized to predict anomalies across 6,497 unseen frames with **97.83% accuracy**.

📌 *The melt pool image dataset used to achieve these results is not provided in this repository. For more information about the thesis, visit: [Investigating the Performance of a Vision Transformer Model for Anomaly Detection in Laser Metal Deposition Imaging](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1886506&dswid=7365)*

<br>

## 🛠️ Workflow Logic

The model combines a **Vision Transformer (ViT-Base/16)** encoder with a **Convolutional Transpose Decoder**. The ViT encoder produces two outputs in parallel: a **CLS token** for classification and **patch embeddings** for reconstruction. This dual-head design allows the model to simultaneously learn *what* is anomalous (classifier) and *how* normal images should look (reconstructor). 

The workflow below illustrates the solution that ultimately drove the best results: a hybrid Supervised-Reconstructive approach. By using Self-Supervised Learning to reconstruct features from base data, combined with Supervised Learning for classification between labels 0 and 1, this dual-path logic maximizes overall anomaly detection accuracy.

<div align="center">
  <img src="https://github.com/user-attachments/assets/ffaa58cb-fbfb-4286-8b7c-57d52770bc51" width="600" />
</div>

### 1️⃣ Feature Extraction
The process begins when an **Input Image** is fed into the **Vision Transformer** (using the ViT-Base/16 configuration initialized from `vit_base_patch16_384.npz` weights). This encoder extracts visual features and splits the output into two parallel streams:
* **Classification Path:** Utilizes the CLS token logits.
* **Reconstruction Path:** Utilizes the extracted patch embeddings.

### 2️⃣ The Reconstruction Check
In the reconstruction path, the **Extracted Features** are routed to a Convolutional Transpose **Decoder**, which attempts to rebuild the image. The difference between the input and the output produces a reconstruction error at the **Check Reconstruction Error Bound** stage:
* **Error Outside Bound:** If the reconstruction error falls outside the predefined range, the model cannot properly reconstruct the image due to unfamiliar anomalies. The workflow immediately bypasses the classifier and flags the image as **Abnormal**.
* **Error Within Bound:** If the reconstruction error is within the predefined boundaries, the decision is delegated to the classification path (indicated by the dashed line).

### 3️⃣ The Classification Head
If the image triggers the second stage, the **CLS logits output** from the encoder is processed through a **Softmax Activation** function. This classification head leverages the ViT's learned representations to catch more subtle abnormalities that passed the reconstruction test, outputting the final class as either **Normal** or **Abnormal**.

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

> Anomaly images are **never** used for reconstruction during training. This is intentional: the decoder should only learn the distribution of normal images (self-supervised), so that anomalous inputs produce high reconstruction error.

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
| Decoder | Adam | 1e-3 | 1e-5 | Train from scratch - higher learning rate needed |

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

Using the Bound subset (which is never seen during training), the score distributions are computed for both classes, and three decision zones are identified:

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
    2. Route to decision logic:
    if perfect_separation:
        → Definite zone: Reconstructor decides (using standalone threshold)
    else:
        if S < overlap_start OR S > overlap_end:
            → Definite zone: Reconstructor decides (based on threshold rules)
        else:
            → Overlap zone (Ambiguous): 
              3. Get CLS token class probabilities via Softmax
              4. Classifier decides (argmax of CLS softmax)
```

<br>

## 🏗️ Dataset Structure 

### Dataset Configuration

To ensure compatibility with the custom data loaders and the hybrid training strategy, the dataset must be organized into the following directory structure, where each folder serves a specific role in training the Vision Transformer and Decoder:

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

> 💡 **Data Volume & Class Imbalance:** The self-supervised reconstruction path requires a proper volume of normal images to accurately map the "normal" manifold, so the `train/base/` directory should be heavily populated. This makes this hybrid architecture highly advantageous for real-world scenarios where there is **abundant normal data but very few anomalous samples**.

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
git clone https://github.com/Dalageo/vitdec-anomaly-detection
cd vitdec-anomaly-detection
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

<br>

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
├── scripts/
│   └── deploy_prd.sh             # Production deployment script
├── pyproject.toml                # Poetry dependency configuration
└── README.md
```

<br>

## ✨ Acknowledgments

This project was conducted as part of my Master's thesis. I would like to extend my deepest gratitude to the individuals and organizations that made this research possible:

* **GKN Aerospace Sweden AB:** To the entire team for providing access to the PTC facilities and fostering a welcoming environment. A special thanks to my industrial supervisor, **Erik Sanderson-Gull**, for giving me the opportunity to undertake this project and for his invaluable guidance.
* **University West:** To my academic supervisor, **Yongcui Mi**, for her expert assistance with the academic aspects of this research, and to the program coordinator, **Morgan Nilsen**, for his continuous support throughout the academic year.
* **The Open-Source Community:** Special thanks to the researchers behind [AnoViT](https://arxiv.org/pdf/2203.10808) as their open-source code served as a tremendous help and a valuable foundation for this work, as well as to **Google** for releasing the pre-trained [ViT-Base/16 weights](https://huggingface.co/google/vit-base-patch16-384), and **Hugging Face** for hosting these models and providing the essential libraries.

<br>
<div align="center">
  <a href="https://www.gknaerospace.com/">
    <img src="https://github.com/user-attachments/assets/56cb0142-ca4e-4912-8b34-0e62b60fcc6c" alt="GKN" width="200"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.hv.se/en/">
    <img src="https://github.com/user-attachments/assets/c5395123-7ce8-4f19-9d54-f0a9d75e31e8" alt="University West" width="200" ></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/">
    <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HuggingFace" width="100"/></a>
</div>

<br>

## ⚖️ License

This repository utilizes components with different licenses:

* **Code & Documentation:** Licensed under the **[AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.en.html)**.
  > The AGPL-3.0 license was chosen to promote open collaboration, ensure transparency, and require that any modifications or improvements must also be shared under the same license, with appropriate acknowledgment.

* **Adapted Components:** Portions of the underlying Vision Transformer architecture used in this project were adapted from [AnoViT](https://arxiv.org/abs/2203.10808), which is licensed under the **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)** license.

<br>
<div align="center">
  <a href="https://www.gnu.org/licenses/agpl-3.0.en.html">
    <img src="https://github.com/user-attachments/assets/f3c6face-aa86-45da-8d20-d8ae25e49e28" alt="AGPLv3-Logo" width="200""></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://creativecommons.org/licenses/by/4.0/">
    <img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by.png" alt="CC BY 4.0" width="200"></a>
</div>

<br>

## 📝 Citation

If you use this code or find this research helpful in your work, please cite the associated Master's thesis:

```bibtex
@mastersthesis{Dalageorgos_Investigating_the_Performance_2024,
  author = {Dalageorgos, Konstantinos},
  month = may,
  title = {{Investigating the Performance of a Vision Transformer Model for Anomaly Detection in Laser Metal Deposition Imaging}},
  url = {https://www.diva-portal.org/smash/get/diva2:1886506/FULLTEXT01.pdf},
  year = {2024}
}
```
