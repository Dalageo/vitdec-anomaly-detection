import os
import torch
from PIL import Image
from torchvision import transforms as T

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'app', 'checkpoints', 'training_curves.png')
LOG_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'app', 'checkpoints', 'training_log.txt')
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'app', 'checkpoints', 'checkpoint.pt')

DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset')
VIT_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'app', 'checkpoints', '24_12_run', 'checkpoint.pt')
VIT_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'model_weights', 'vit_base_patch16_384.npz')

# Auto-detect GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optimization settings
AMP = True           # Enable Automatic Mixed Precision (faster training, less memory)
BETA_1 = 0.85        # Adam optimizer first moment (momentum)
BETA_2 = 0.999       # Adam optimizer second moment

# Model Hyperparameters
LR_VIT = 1e-5        # Encoder's learning rate 
LR_DEC = 1e-3        # Decoder's learning rate 
WD_VIT = 1e-5        # Encoder's weight decay
WD_DEC = 1e-5        # Decoder's weight decay

# Training Loop
EPOCHS = 50
BATCH_SIZE = 16

# Dataset Splitting
VAL_RATIO = 0.15     # 15% for validation
TEST_RATIO = 0.5     # 50% of the non-training set to use for testing (0.5 --> 50% test, 50% bound)
SEED = 42            # Global random seed

# Image Preprocessing
IMG_SIZE = 384
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# Augmentation Config
AUGMENTATION_CONFIG = {  
        "basic_transforms": [
                    # Resizing using LANCZOS filter for high-quality downsampling
                    T.Resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS),
                    T.ToTensor(),
                    T.Normalize(MEAN, STD)
                ],
        
        "train_transforms": { 
            "supervised": { 
                "augmentation": True,
                "augmentation_transforms": [
                    T.Resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS),
                    # T.RandomApply([T.ColorJitter(brightness=(1.1, 1.3))], p=0.5),
                    T.RandomHorizontalFlip(p=0.5),
                    # T.RandomVerticalFlip(p=0.5),
                    T.ToTensor(),
                    ],
            },
            "unsupervised": { 
                "augmentation": True,
                "augmentation_transforms": [
                    T.Resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS),
                    # T.RandomApply([T.ColorJitter(brightness=(1.1, 1.3))], p=0.5),
                    T.RandomHorizontalFlip(p=0.5),
                    # T.RandomVerticalFlip(p=0.5),
                    T.ToTensor(),
                    ],
            },
        }
    }