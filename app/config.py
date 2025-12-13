
import torch
from PIL import Image
from torchvision import transforms as T

DATASET_PATH = "/home/dalageo/Github-Personal-Projects/vit-dec-anomaly-detection/dataset"


# Data & Training Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = True
BETA_1 = 0.85
BETA_2 = 0.999

# Vision Transformer
LR_VIT = 1e-5
LR_DEC = 1e-4
WD_VIT = 1e-5
WD_DEC = 1e-5


EPOCHS = 50
BATCH_SIZE = 16

# Data training configs
VAL_RATIO = 0.15
TEST_RATIO = 0.5
SEED = 42

IMG_RES = 384
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

AUGMENTATION_CONFIG = {  
        "basic_transforms": [
                    # Resizing using LANCZOS filter for high-quality downsampling
                    T.Resize((IMG_RES, IMG_RES), Image.LANCZOS),
                    T.ToTensor(),
                    T.Normalize(MEAN, STD)
                ],
        
        "train_transforms": { 
            "supervised": { 
                "augmentation": True,
                "augmentation_transforms": [
                    T.Resize((IMG_RES, IMG_RES), Image.LANCZOS),
                    # T.RandomApply([T.ColorJitter(brightness=(1.1, 1.3))], p=0.5),
                    T.RandomHorizontalFlip(p=0.5),
                    # T.RandomVerticalFlip(p=0.5),
                    T.ToTensor(),
                    ],
            },
            "unsupervised": { 
                "augmentation": True,
                "augmentation_transforms": [
                    T.Resize((IMG_RES, IMG_RES), Image.LANCZOS),
                    # T.RandomApply([T.ColorJitter(brightness=(1.1, 1.3))], p=0.5),
                    T.RandomHorizontalFlip(p=0.5),
                    # T.RandomVerticalFlip(p=0.5),
                    T.ToTensor(),
                    ],
            },
        }
    }