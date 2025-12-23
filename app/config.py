
import torch
from PIL import Image
from torchvision import transforms as T

DATASET_PATH = "/home/dalageo/Github-Personal-Projects/vit-dec-anomaly-detection/dataset"
VIT_WEIGHTS_PATH = "/home/dalageo/Github-Personal-Projects/vit-dec-anomaly-detection/model_weights/vit_base_patch16_384.npz"
PLOT_OUTPUT_PATH = "app/checkpoints/training_curves.png"
LOG_OUTPUT_PATH = "app/checkpoints/training_log.txt"
CHECKPOINT_PATH = "app/checkpoints/checkpoint.pt"

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