
import os
import torch
import numpy as np
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
from torchvision import transforms as T, utils
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, SubsetRandomSampler
from config import AUGMENTATION_CONFIG, VAL_RATIO, SEED, BATCH_SIZE, TEST_RATIO, DATASET_PATH


# --------------------
# Custom Dataset Class
# --------------------
class MVTecDataset(Dataset):
    def __init__(self, data_path: str, is_train: bool, data_info: bool, transforms_dict: dict):
        super().__init__()
        self.data_path = data_path
        self.is_train = is_train
        self.data_info = data_info
        self.transforms_dict = transforms_dict
        
        self.phase = 'train' if self.is_train else 'test'
        self.x, self.y, self.category = self.load_dataset()

    def __getitem__(self, idx):
        x_path, y, category = self.x[idx], self.y[idx], self.category[idx]
        x = Image.open(x_path).convert('RGB')
        
        # Print initial pixel value range
        if self.data_info:
            img_np = np.array(x)
            print(f"Initial pixel range in {x_path}: min={img_np.min()}, max={img_np.max()}")

        # Apply the specific transformation based on the category
        if category in self.transforms_dict:
            x = self.transforms_dict[category](x)
            
        # Print pixel value range after transformation
        if self.data_info:
                x_tensor = torch.tensor(np.array(x)) / 255.0 
                print(f"Transformed pixel range for {x_path}: min={x_tensor.min().item()}, max={x_tensor.max().item()}")

        return x, y, category

    def __len__(self):
        return len(self.x)

    def load_dataset(self):
        x, y, category = [], [], []
        img_dir = os.path.join(self.data_path, self.phase)

        # Base/Normal images (class -1 --> unsupervised training)
        if self.is_train:
            base_img_path = os.path.join(img_dir, 'base')
            if os.path.isdir(base_img_path):
                base_img_paths = [os.path.join(base_img_path, img) for img in os.listdir(base_img_path)]
                x.extend(base_img_paths)
                # Assign label -1
                y.extend([-1] * len(base_img_paths))
                # Add unsupervised for category
                category.extend(['base'] * len(base_img_paths))

        # Normal images (class 0 --> supervised training)
        normal_img_path = os.path.join(img_dir, 'normal')
        if os.path.isdir(normal_img_path):
            normal_img_paths = [os.path.join(normal_img_path, img) for img in os.listdir(normal_img_path)]
            x.extend(normal_img_paths)
            # Assign label 0
            y.extend([0] * len(normal_img_paths))
            # Assign normal for category if its for training  while if its for testing asign test to them
            category.extend(['normal'] * len(normal_img_paths) if self.is_train else ['test'] * len(normal_img_paths))

        # Anomaly images (class 1 --> supervised training)
        anomaly_img_path = os.path.join(img_dir, 'anomaly')
        if os.path.isdir(anomaly_img_path):
            anomaly_img_paths = [os.path.join(anomaly_img_path, img) for img in os.listdir(anomaly_img_path)]
            x.extend(anomaly_img_paths)
            # Assign label 1
            y.extend([1] * len(anomaly_img_paths))
            # Assign anomaly for category if its for training (because 'anomaly' folder is also used for testing) while if its for testing asign test to them
            category.extend(['anomaly'] * len(anomaly_img_paths) if self.is_train else ['test'] * len(anomaly_img_paths))
            
        if self.data_info:
            print(f"Loaded {len(x)} images for {self.phase}.")
            print("Class distribution:", Counter(y))
            print("Categories:", Counter(category))
            
        return x, y, category
    

# ---------------------------
# Load and Transform the Data
# ---------------------------
class MVTecDataModule:
    def __init__(self):
        self.data_path = DATASET_PATH
        self.augmentation_dict = AUGMENTATION_CONFIG
        
        self.test_ratio = TEST_RATIO
        self.val_ratio = VAL_RATIO
        self.batch_size = BATCH_SIZE
        self.seed = SEED
        
        # Placeholders for datasets and indices
        self.train_dataset = None
        self.bound_dataset = None
        self.train_idx = None
        self.val_idx = None
        self.bound_idx = None
        self.test_idx = None
        
        # Prepare transforms & load/split data immediately
        self._prepare_transforms()
        self._setup_data()

    def _prepare_transforms(self):
        """Internal helper to setup transform dictionaries."""
        basic_transforms = self.augmentation_dict.get("basic_transforms")
        
        # Train Transforms
        train_cfg = self.augmentation_dict.get("train_transforms")
        
        # Supervised
        supervised_cfg = train_cfg.get("supervised")
        supervised_list = supervised_cfg.get("augmentation_transforms") if supervised_cfg.get("augmentation") else basic_transforms
        transform_supervised = T.Compose(supervised_list)
        
        # Unsupervised
        unsupervised_cfg = train_cfg.get("unsupervised")
        unsupervised_list = unsupervised_cfg.get("augmentation_transforms") if unsupervised_cfg.get("augmentation") else basic_transforms
        transform_unsupervised = T.Compose(unsupervised_list)
        
        # Anomaly (No augmentation)
        transform_supervised_anomaly = T.Compose(basic_transforms)

        self.train_transforms = {
            'base': transform_unsupervised,
            'normal': transform_supervised,
            'anomaly': transform_supervised_anomaly,
        }
        self.test_transforms = {'test': transform_unsupervised}


    def _setup_data(self, ):
        """Loads datasets once and calculates all split indices once."""
        print("Loading datasets and calculating splits...")
        
        # Load Datasets
        self.train_dataset = MVTecDataset(self.data_path, is_train=True, transforms_dict=self.train_transforms, data_info=None)
        self.bound_dataset = MVTecDataset(self.data_path, is_train=False, transforms_dict=self.test_transforms, data_info=None)

        # Calculate Train/Val Splits
        normal_indices = [i for i, (_, y, _) in enumerate(self.train_dataset) if y == 0]
        base_indices = [i for i, (_, y, _) in enumerate(self.train_dataset) if y == -1]
        anomaly_indices = [i for i, (_, y, _) in enumerate(self.train_dataset) if y == 1]

        # Normal Split
        np.random.shuffle(normal_indices)
        norm_split = int(self.val_ratio * len(normal_indices))
        val_norm_idx = normal_indices[:norm_split]
        train_norm_idx = normal_indices[norm_split:]

        # Base Split
        np.random.shuffle(base_indices)
        base_split = int(self.val_ratio * len(base_indices))
        val_base_idx = base_indices[:base_split]
        train_base_idx = base_indices[base_split:]

        # Combine
        self.train_idx = train_norm_idx + train_base_idx + anomaly_indices
        self.val_idx = val_norm_idx + val_base_idx
        
        # Shuffle final lists
        np.random.shuffle(self.train_idx)
        np.random.shuffle(self.val_idx)

        # Calculate Bound/Test Splits
        labels_bound = [y for _, y, _ in self.bound_dataset]
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=self.test_ratio, random_state=self.seed)
        self.bound_idx, self.test_idx = next(sss_test.split(np.zeros(len(labels_bound)), labels_bound))

        if self.data_info:
            print("\n---------- Dataset Distribution ----------")
            print("Number of training images:", len(self.train_idx))
            print("Number of validation images:", len(self.val_idx))
            print("Number of bound images:", len(self.bound_idx))
            print("Number of testing images:", len(self.test_idx))
            
    # Dataloaders
    @property
    def train_dataloader(self):
        return TorchDataLoader(self.train_dataset, batch_size=self.batch_size, sampler=SubsetRandomSampler(self.train_idx))

    @property
    def val_dataloader(self):
        return TorchDataLoader(self.train_dataset, batch_size=self.batch_size, sampler=SubsetRandomSampler(self.val_idx))

    @property
    def bound_dataloader(self):
        return TorchDataLoader(self.bound_dataset, batch_size=self.batch_size, sampler=SubsetRandomSampler(self.bound_idx))

    @property
    def test_dataloader(self):
        return TorchDataLoader(self.bound_dataset, batch_size=self.batch_size, sampler=SubsetRandomSampler(self.test_idx))
    
    
# ---------------------
# Visualize Image Class
# ---------------------
class VisualizeImages:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    # Denormalize a tensor image
    def denormalize(self, tensor):
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        return tensor * std + mean

    # Display an image on a given axes.
    def imshow(self, img, ax):
        img = self.denormalize(img)  # Denormalize the image
        img = img.clamp(0, 1)        # Ensure the values are in the range [0, 1]
        img = img.permute(1, 2, 0)   # Convert from CxHxW to HxWxC
        ax.imshow(img.numpy())       # Convert to numpy for imshow
        ax.axis('off')

    # Displays images from the data loader.
    def check_data(self, loader, img_batch_info):
        for i, (images, labels, categories) in enumerate(loader):
            if i >= img_batch_info:
                break
            print(f"Batch {i+1} labels:", labels)
            print(f"Batch {i+1} categories:", categories)

            # Denormalize the images in the batch before passing to make_grid
            images = torch.stack([self.denormalize(img) for img in images])
            
            # Create a grid with black padding (pad_value=0)
            grid_img = utils.make_grid(images, nrow=10, padding=15, pad_value=255)  # Pad with black (0)

            # Make sure the image tensor is in the right format
            plt.figure(figsize=(12, 8))
            plt.title(f"Batch {i+1}")  
            
            # Show the grid without denormalizing again
            ax = plt.gca()
            grid_img = grid_img.clamp(0, 1)         # Ensure the grid image is within [0, 1]
            grid_img = grid_img.permute(1, 2, 0)    # Convert to HxWxC for matplotlib
            ax.imshow(grid_img.numpy())             # Display using matplotlib
            ax.axis('off')  
            plt.tight_layout()  
            plt.show()