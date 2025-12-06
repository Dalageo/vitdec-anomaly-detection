
import os
import torch
import numpy as np
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
from torchvision import transforms as T, utils
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, SubsetRandomSampler


# --------------------
# Custom Dataset Class
# --------------------
class LoadDataset(Dataset):
    def __init__(self, data_path: str, is_train: bool, data_info: bool, transforms_dict: dict):
        super().__init__()
        self.data_path = data_path
        self.is_train = is_train
        self.data_info = data_info
        self.transforms_dict = transforms_dict
        
        self.phase = 'train' if self.is_train else 'test'
        self.x, self.y, self.category = self.load_dataset_folder()

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
                x_tensor = torch.tensor(np.array(x)) / 255.0  # Assuming x is an image after ToTensor (0-1 range)
                print(f"Transformed pixel range for {x_path}: min={x_tensor.min().item()}, max={x_tensor.max().item()}")

        return x, y, category

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        
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
    

IMG_RES = 384
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

augmentation_dict = {  
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
        
        
# ---------------------------
# Load and Transform the Data
# ---------------------------
class TransformDataset():
    def __init__(self, data_path: str, augmentation_dict: dict):
        self.data_path = data_path
        self.augmentation_dict = augmentation_dict
        
    def get_dataloader(self, is_train: bool): 
        
        basic_transforms_list = self.augmentation_dict.get("basic_transforms")
        basic_transforms = T.Compose(basic_transforms_list)
        
        if is_train:
            train_transforms = self.augmentation_dict.get("train_transforms")
            
            # --- Supervised Transforms (Normal/Anomaly) ---
            supervised_config = train_transforms.get("supervised")
            if supervised_config.get("augmentation"):
                transform_supervised_list = supervised_config.get("augmentation_transforms")
            else:
                transform_supervised_list = basic_transforms_list
            transform_supervised = T.Compose(transform_supervised_list)
                
            # --- Unsupervised Transforms (Base) ---
            unsupervised_config = train_transforms.get("unsupervised")
            if unsupervised_config.get("augmentation"):
                transform_unsupervised_list = unsupervised_config.get("augmentation_transforms")
            else:
                transform_unsupervised_list = basic_transforms_list
            transform_unsupervised = T.Compose(transform_unsupervised_list)
        
        
        # For anomaly, just transform + ToTensor + Normalize (no augmentation)
        transform_supervised_anomaly = T.Compose(basic_transforms)

        # Category dicts for transformation
        train_transform_dict = {
            'base': transform_unsupervised,
            'normal': transform_supervised,
            'anomaly': transform_supervised_anomaly,
        }
        
        test_transform_dict = {
            'test': transform_unsupervised
        }

        # Initialize datasets with transformation
        train_dataset = LoadDataset(self.data_path, is_train=True, transforms_dict=train_transform_dict, data_info=None)
        # Here I set the unsupervised transformation dict since its for testing
        bound_dataset = LoadDataset(self.data_path, is_train=False, transforms_dict=test_transform_dict, data_info=None)

        # Collect indices for normal, base, and anomaly data
        indices_good = [i for i, (_, y, _) in enumerate(train_dataset) if y == 0]
        indices_unsupervised = [i for i, (_, y, _) in enumerate(train_dataset) if y == -1]
        indices_anomaly = [i for i, (_, y, _) in enumerate(train_dataset) if y == 1]  

        # Prepare Stratified splits for 'normal' and 'base' data only for validation
        sss_good = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
        train_idx_good, val_idx_good = next(sss_good.split(np.zeros(len(indices_good)), np.zeros(len(indices_good))))  

        np.random.shuffle(indices_unsupervised)
        split_point_unsupervised = int(args.val_ratio * len(indices_unsupervised))
        val_idx_unsupervised = indices_unsupervised[:split_point_unsupervised]
        train_idx_unsupervised = indices_unsupervised[split_point_unsupervised:]

        # Anomalies are included only in the training dataset and not to validation one
        train_idx_anomaly = indices_anomaly  

        # Combine indices for training and validation, exclude anomalies from validation
        train_idx = [indices_good[i] for i in train_idx_good] + train_idx_unsupervised + train_idx_anomaly
        val_idx = [indices_good[i] for i in val_idx_good] + val_idx_unsupervised

        # Shuffle the combined training and validation indices to mix the data
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)

        # Create dataloaders
        train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(train_idx))
        val_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(val_idx))

        # Setup for bound_loader and test_loader and stratify them for a balance representation
        labels_bound = [y for _, y, _ in bound_dataset]
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=args.test_ratio, random_state=args.seed)
        bound_idx, test_idx = next(sss_test.split(np.zeros(len(labels_bound)), labels_bound))

        bound_loader = TorchDataLoader(bound_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(bound_idx))
        test_loader = TorchDataLoader(bound_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(test_idx))

        if args.data_info:
            print("\n---------- Dataset Distribution ----------")
            print("Number of training images:", len(train_idx))
            print("Number of validation images:", len(val_idx))
            print("Number of bound images:", len(bound_idx))
            print("Number of testing images:", len(test_idx))

        return train_loader, val_loader, bound_loader, test_loader


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
        img = img.clamp(0, 1)  # Ensure the values are in the range [0, 1]
        img = img.permute(1, 2, 0)  # Convert from CxHxW to HxWxC
        ax.imshow(img.numpy())  # Convert to numpy for imshow
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
            grid_img = grid_img.clamp(0, 1)  # Ensure the grid image is within [0, 1]
            grid_img = grid_img.permute(1, 2, 0)  # Convert to HxWxC for matplotlib
            ax.imshow(grid_img.numpy())  # Display using matplotlib
            ax.axis('off')  # Hide axis
            plt.tight_layout()  
            plt.show()
    