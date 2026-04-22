# ViT-Decoder Anomaly Detection

This repository contains the code and findings for the thesis project: "Investigating the Performance of a Vision Transformer Model for Anomaly Detection in Laser Metal Deposition Imaging." The research explores a Vision Transformer (ViT) approach to identify anomalies within Laser Metal Deposition (LMD) images.

While Convolutional Neural Networks (CNNs) are the standard for finding defects in the computer vision sector, Transformer-based models remain largely underexplored in this domain. To bridge this gap, this project introduces a custom ViT-Decoder architecture designed to analyze melt pool images and cross-reference predictions with existing CNN models.

Evaluated on LMD melt pool images, the model achieved **99.78% accuracy** on the core dataset and successfully generalized to predict anomalies across 6,497 unseen frames with **97.83% accuracy**.

📌 *The melt pool image dataset used to achieve these results is not provided in this repository. For more information about the thesis, visit: [Investigating the Performance of a Vision Transformer Model for Anomaly Detection in Laser Metal Deposition Imaging](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1886506&dswid=7365)*

## Dataset Structure & Pre-processing

### Directory Layout

The dataset follows a structured directory layout where the split between training and test data is defined at the folder level:

```
dataset/
├── train/
│   ├── base/       # Unlabeled (unsupervised) melt pool images
│   ├── normal/     # Labeled normal melt pool images
│   └── anomaly/    # Labeled anomalous melt pool images
└── test/
    ├── normal/     # Normal melt pool images for evaluation
    └── anomaly/    # Anomalous melt pool images for evaluation
```

### Label Semantics

Each image is assigned an integer label based on its source folder and intended role in training:

| Folder | Label | Category | Purpose |
|--------|-------|----------|---------|
| `train/base/` | `-1` | `base` | Unsupervised reconstruction — model learns to faithfully reconstruct normal melt pool morphology without class supervision |
| `train/normal/` | `0` | `normal` | Supervised — used for both reconstruction loss and classification loss |
| `train/anomaly/` | `1` | `anomaly` | Supervised — used for classification loss only (not reconstruction) |
| `test/normal/` | `0` | `test` | Evaluation |
| `test/anomaly/` | `1` | `test` | Evaluation |

This three-tier labeling scheme is central to the training strategy: the reconstruction objective learns exclusively from normal-looking samples (`base` + `normal`), while the classification objective learns the decision boundary between `normal` and `anomaly`.

### Image Transforms

All images are resized to **384 × 384** pixels using the LANCZOS resampling filter, matching the ViT-Base/16 input resolution.

| Transform Pipeline | Applied To | Operations |
|--------------------|------------|------------|
| **Supervised (Normal)** | `train/normal/` | Resize → RandomHorizontalFlip(p=0.5) → ToTensor |
| **Unsupervised (Base)** | `train/base/` | Resize → RandomHorizontalFlip(p=0.5) → ToTensor |
| **Supervised (Anomaly)** | `train/anomaly/` | Resize → ToTensor → Normalize(μ=0.5, σ=0.5) |
| **Test / Bound** | `test/*` | Resize → ToTensor |

Normalization with `μ = 0.5` and `σ = 0.5` per channel maps pixel values from `[0, 1]` to `[-1, 1]`, aligning with the Decoder's `Tanh` output activation.

### Data Splitting Strategy

The evaluation set (`test/`) is further split into two disjoint subsets using **stratified sampling** (preserving class ratios):

| Subset | Source | Ratio | Purpose |
|--------|--------|-------|---------|
| **Training** | `train/` | 85% of normal + base | Model parameter optimization |
| **Validation** | `train/` | 15% of normal + base | Early stopping & scheduler decisions |
| **Bound** | `test/` | 50% | Calibrate decision thresholds (overlap analysis) |
| **Test** | `test/` | 50% | Final held-out evaluation |

The **Bound** subset is critical: it is used to calculate the reconstruction-score distributions for normal vs. anomalous samples and to define the decision boundaries *before* the model ever sees the final Test subset. This ensures that threshold calibration does not leak information from the evaluation set.

---

## ViT-Decoder Architecture

### High-Level Overview

The model combines a **Vision Transformer (ViT-Base/16)** encoder with a **Convolutional Transpose Decoder**. The ViT encoder produces two outputs in parallel: a **CLS token** for classification and **patch embeddings** for reconstruction. This dual-head design allows the model to simultaneously learn *what* is anomalous (classifier) and *how* normal images should look (reconstructor).

```
Input Image (3 × 384 × 384)
        │
        ▼
┌─────────────────────┐
│   Patch Embedding    │  Conv2d(3, 768, kernel=16, stride=16)
│   576 patches        │  Each patch: 16×16 pixels → 768-dim vector
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  [CLS] + Patches     │  Prepend learnable CLS token → 577 tokens
│  + Position Embed    │  Add positional embeddings
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  12 Transformer      │  Multi-Head Self-Attention (12 heads)
│  Encoder Blocks      │  + MLP (768 → 3072 → 768) + LayerNorm
└─────────────────────┘
        │
        ├──── CLS token (index 0) ──► Linear(768, 2) ──► Class Logits
        │
        └──── Patch tokens (1..576) ──► Decoder ──► Reconstructed Image
```

### 1. Vision Transformer Encoder

The encoder follows the standard **ViT-Base/16** configuration, initialized from pre-trained weights (`vit_base_patch16_384.npz`):

| Parameter | Value |
|-----------|-------|
| Image Size | 384 × 384 |
| Patch Size | 16 × 16 |
| Number of Patches | 576 (24 × 24 grid) |
| Embedding Dimension | 768 |
| Transformer Depth | 12 blocks |
| Attention Heads | 12 |
| MLP Ratio | 4× (hidden dim = 3072) |
| Activation | GELU |

**Patch Embedding:** The input image is divided into a grid of 16×16-pixel patches. A convolutional projection (`Conv2d` with `kernel_size=16`, `stride=16`) maps each patch into a 768-dimensional embedding. For a 384×384 image, this produces **576 patch tokens**.

**CLS Token & Positional Encoding:** A learnable `[CLS]` token is prepended to the sequence, yielding 577 tokens total. Learnable positional embeddings are added to preserve spatial information.

**Self-Attention Blocks:** Each of the 12 Transformer blocks applies:
1. **Layer Normalization** (pre-norm)
2. **Multi-Head Self-Attention** — queries, keys, and values are computed jointly via a fused `QKV` linear layer, then split across 12 heads (head dim = 64). Attention scores are computed as:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

3. **Residual connection** + **Drop Path** (stochastic depth)
4. **Layer Normalization**
5. **Feed-Forward MLP** (768 → 3072 → 768 with GELU activation)
6. **Residual connection** + **Drop Path**

### 2. Dual-Head Output

After the final Transformer block, the 577-token output is split:

- **`features[:, 0]` → CLS Token:** The global classification representation. Passed through a linear head (`Linear(768, 2)`) to produce binary class logits (Normal vs. Anomaly). During training, these logits are supervised with `CrossEntropyLoss`.

- **`features[:, 1:]` → Patch Tokens:** The 576 spatially-encoded patch representations. These are passed to the Decoder for image reconstruction. The CLS token is explicitly **excluded** from reconstruction to preserve the separation of concerns.

### 3. Convolutional Decoder

The Decoder progressively upsamples the patch embeddings back to the original image resolution through **8 transposed convolution blocks**:

| Block | Operation | Input Shape | Output Shape |
|-------|-----------|-------------|--------------|
| Reshape | `(B, 576, 768)` → `(B, 768, 24, 24)` | — | `768 × 24 × 24` |
| Block 1 | `ConvTranspose2d(768, 384, 3×3)` + InstanceNorm + ReLU | `768 × 24 × 24` | `384 × 26 × 26` |
| Block 2 | `ConvTranspose2d(384, 192, 3×3)` + InstanceNorm + ReLU | `384 × 26 × 26` | `192 × 28 × 28` |
| Block 3 | `ConvTranspose2d(192, 96, 3×3)` + InstanceNorm + ReLU | `192 × 28 × 28` | `96 × 30 × 30` |
| Block 4 | `ConvTranspose2d(96, 48, 3×3)` + InstanceNorm + ReLU | `96 × 30 × 30` | `48 × 32 × 32` |
| Block 5 | `ConvTranspose2d(48, 24, 3×3)` + InstanceNorm + ReLU | `48 × 32 × 32` | `24 × 34 × 34` |
| Block 6 | `ConvTranspose2d(24, 12, 3×3, stride=2, pad=1)` + InstanceNorm + ReLU | `24 × 34 × 34` | `12 × 69 × 69` |
| Block 7 | `ConvTranspose2d(12, 6, 3×3, stride=2, pad=1)` + InstanceNorm + ReLU | `12 × 69 × 69` | `6 × 139 × 139` |
| Block 8 | `ConvTranspose2d(6, 3, 3×3, stride=2, pad=1)` + InstanceNorm + ReLU | `6 × 139 × 139` | `3 × 279 × 279` |
| Upsample | `UpsamplingBilinear2d(384, 384)` | `3 × 279 × 279` | `3 × 384 × 384` |
| Activation | `Tanh` | — | Output ∈ `[-1, 1]` |

**Key design choices:**
- **InstanceNorm2d** is used instead of BatchNorm to normalize each sample independently, which is more stable for reconstruction tasks with small batch sizes.
- **Kaiming initialization** (`fan_out`, ReLU) is applied to all transposed convolution weights.
- The final **Tanh** activation bounds the output to `[-1, 1]`, matching the normalized input range.

### 4. Composite Forward Pass

The `ViTDecoder` wrapper controls which outputs are produced via two boolean flags:

```python
cls_logits, reconstructed = model(x, return_logits=True, return_reconstruction=True)
```

| `return_logits` | `return_reconstruction` | Encoder Mode | Used During |
|:---:|:---:|---|---|
| `False` | `True` | Patch tokens only (no CLS) | Unsupervised reconstruction training |
| `True` | `False` | CLS token only (no decoder) | Classification training |
| `True` | `True` | Full dual-head | Inference / evaluation |

---

## Training Strategy

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

---

## Threshold & Anomaly Detection Logic

The inference pipeline uses a **two-phase evaluation** strategy: first calibrate thresholds on a held-out Bound set, then apply those thresholds to the unseen Test set.

### Step 1 — Anomaly Score Calculation

For each image, the reconstruction error (anomaly score) is computed as the **pixel-wise MSE** between the original and reconstructed image, averaged across channels and spatial dimensions:

$$S_i = \frac{1}{C \cdot H \cdot W} \sum_{c,h,w} \left( x_{c,h,w}^{(i)} - \hat{x}_{c,h,w}^{(i)} \right)^2$$

where $x^{(i)}$ is the original image and $\hat{x}^{(i)}$ is the decoder's reconstruction.

> **Note:** While training uses L1 Loss for its gradient properties, the evaluation phase uses MSE to compute anomaly scores, as squared error amplifies larger deviations — making anomalies more separable.

### Step 2 — Decision Boundary Estimation (Bound Phase)

Using the Bound subset (never seen during training), the system computes the score distributions for both classes and identifies three decision zones:

```
Score Axis ──────────────────────────────────────────────────►

│◄── Definite Zone A ──►│◄── Overlap Zone ──►│◄── Definite Zone B ──►│
│   (Reconstruction      │   (Ambiguous:      │   (Reconstruction      │
│    alone decides)       │    Classifier       │    alone decides)       │
│                         │    decides)         │                         │
                    overlap_start         overlap_end
```

**Boundary calculation logic:**

1. **Compute score ranges** for both classes on the Bound set:
   - Normal scores: $[\min_{\text{norm}},\ \max_{\text{norm}}]$
   - Anomalous scores: $[\min_{\text{anom}},\ \max_{\text{anom}}]$

2. **Identify the overlap region:**
   - $\text{overlap\_start} = \max(\min_{\text{norm}},\ \min_{\text{anom}})$
   - $\text{overlap\_end} = \min(\max_{\text{norm}},\ \max_{\text{anom}})$

3. **Assign threshold rules** based on which class extends beyond the overlap:
   - If anomalous scores extend *below* the overlap → scores below `overlap_start` are classified as **anomaly**
   - If anomalous scores extend *above* the overlap → scores above `overlap_end` are classified as **anomaly**
   - (Symmetrically for normal scores)

4. **Standalone threshold (fallback):** An optimal single threshold is computed by sweeping 100 candidate values across the normal score range and selecting the one that maximizes the **F1-score**:

$$\theta^* = \arg\max_{\theta} \ F_1\big(y,\ \mathbb{1}[S \geq \theta]\big)$$

### Step 3 — Combined Inference (Test Phase)

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

**Decision routing in detail:**

| Score Region | Decision Source | Logic |
|---|---|---|
| Below overlap start | **Reconstructor** | Label assigned by pre-computed threshold rule (e.g., if anomalies have lower scores → anomaly) |
| Within overlap range | **Classifier (CLS token)** | `argmax(softmax(logits))` — the ViT's classification head breaks the tie |
| Above overlap end | **Reconstructor** | Label assigned by pre-computed threshold rule (e.g., if anomalies have higher scores → anomaly) |
| Perfect separation | **Standalone Reconstructor** | F1-optimized threshold applied directly |

This combined approach leverages the strengths of both model heads: the reconstructor provides a robust, distribution-based anomaly signal for clear-cut cases, while the classifier's learned decision boundary resolves ambiguous cases where reconstruction scores overlap between normal and anomalous samples.

---

## Project Structure

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

## Setup & Usage

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

### Configuration

All hyperparameters are centralized in `app/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | 384 | Input resolution |
| `EPOCHS` | 50 | Maximum training epochs |
| `BATCH_SIZE` | 16 | Samples per batch |
| `LR_VIT` | 1e-5 | Encoder learning rate |
| `LR_DEC` | 1e-3 | Decoder learning rate |
| `AMP` | True | Automatic Mixed Precision |
| `VAL_RATIO` | 0.15 | Validation split ratio |
| `TEST_RATIO` | 0.5 | Bound/Test split ratio |