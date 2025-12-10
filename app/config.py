
from PIL import Image
from torchvision import transforms as T

DATASET_PATH = "/home/dalageo/Github-Personal-Projects/vit-dec-anomaly-detection/dataset"

# training configs
VAL_RATIO = 0.15
TEST_RATIO = 0.5
BATCH_SIZE = 16
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