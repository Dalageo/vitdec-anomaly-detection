
import os
import torch
import argparse
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
class CustomDataset(Dataset):
    def __init__(self, data_path, is_train=True, transforms_dict=None, data_info=False):
        super(CustomDataset, self).__init__()
        self.data_path = data_path
        self.is_train = is_train
        self.transforms_dict = transforms_dict
        self.data_info = data_info  

        # Sets the phase based on whether it is training or not (Train or Test)
        self.phase = 'train' if is_train else 'test'
        self.x, self.y, self.category = self.load_dataset_folder()

    def __getitem__(self, idx):
        x_path, y, category = self.x[idx], self.y[idx], self.category[idx]
        x = Image.open(x_path).convert('RGB')
        
        # Debug: Print initial pixel value range
        if self.data_info:
            img_np = np.array(x)
            print(f"Initial pixel range in {x_path}: min={img_np.min()}, max={img_np.max()}")

        # Apply the specific transformation based on the category
        if category in self.transforms_dict:
            x = self.transforms_dict[category](x)
            
        # Debug: Print pixel value range after transformation
        if self.data_info:
            x_tensor = torch.tensor(np.array(x)) / 255.0  # Assuming x is an image after ToTensor (0-1 range)
            print(f"Transformed pixel range for {x_path}: min={x_tensor.min().item()}, max={x_tensor.max().item()}")

        return x, y, category

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        x, y, category = [], [], []
        img_dir = os.path.join(self.data_path, self.phase)

        # Base images (only for unsupervised training)
        if self.is_train:
            normal_images_path = os.path.join(img_dir, 'base')
            normal_paths = [os.path.join(normal_images_path, img) for img in os.listdir(normal_images_path)]
            x.extend(normal_paths)
            # Assign label -1
            y.extend([-1] * len(normal_paths))
            # Add unsupervised for category
            category.extend(['base'] * len(normal_paths))

        # Normal images (class 0 --> supervised training)
        good_images_path = os.path.join(img_dir, 'normal')
        good_paths = [os.path.join(good_images_path, img) for img in os.listdir(good_images_path)]
        x.extend(good_paths)
        # Assign label 0
        y.extend([0] * len(good_paths))
        # Assign normal for category if its for training (because 'normal' folder is also used for testing) while if its for testing asign test to them
        category.extend(['normal'] * len(good_paths) if self.is_train else ['test'] * len(good_paths))

        # Anomaly images (class 1 --> supervised training)
        anomaly_base_path = os.path.join(img_dir, 'anomaly')
        for anomaly_folder in os.listdir(anomaly_base_path):
            anomaly_folder_path = os.path.join(anomaly_base_path, anomaly_folder)
            anomaly_paths = [os.path.join(anomaly_folder_path, img) for img in os.listdir(anomaly_folder_path)]
            x.extend(anomaly_paths)
            # Assign label 1
            y.extend([1] * len(anomaly_paths))
            # Assign anomaly for category if its for training (because 'anomaly' folder is also used for testing) while if its for testing asign test to them
            category.extend(['anomaly'] * len(anomaly_paths) if self.is_train else ['test'] * len(anomaly_paths))
        
        if self.data_info:
            print(f"Loaded {len(x)} images for {self.phase}.")
            print("Class distribution:", Counter(y))
            print("Categories:", Counter(category))
        return x, y, category
    

# ---------------------------
# Load and Transform the Data
# ---------------------------
def get_dataloader(args): 
    # Base transforms (always applied)
    transform = [
        # Resizing using LANCZOS filter for high-quality downsampling
        T.Resize((args.img_res, args.img_res), Image.LANCZOS),
    ]

    # For unsupervised transforms
    if args.us_aug == 1:
        transform_unsupervised = transform + [
            # T.RandomApply([T.ColorJitter(brightness=(1.1, 1.3))], p=0.5),
            T.RandomHorizontalFlip(p=0.5),
            # T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(args.mean, args.std)
        ]
    else:
        # Even if no augmentation is chosen, still add T.ToTensor(), T.Normalize
        transform_unsupervised = transform + [T.ToTensor()] + [T.Normalize(args.mean, args.std)]

    # For supervised normal transforms
    if args.sv_aug == 1:
        transform_supervised_normal = transform + [
            # T.RandomApply([T.ColorJitter(brightness=(1.1, 1.3))], p=0.5),
            T.RandomHorizontalFlip(p=0.5),
            # T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(args.mean, args.std)
        ]
    else:
        transform_supervised_normal = transform + [T.ToTensor()] + [T.Normalize(args.mean, args.std)]

    # For anomaly, just transform + ToTensor + Normalize (no augmentation)
    transform_supervised_anomaly = transform + [T.ToTensor()] + [T.Normalize(args.mean, args.std)]

    # Wrap each list of transforms with T.Compose
    transform_unsupervised = T.Compose(transform_unsupervised)
    transform_supervised_normal = T.Compose(transform_supervised_normal)
    transform_supervised_anomaly = T.Compose(transform_supervised_anomaly)
    
    # Category dicts for transformation
    transforms_dict = {
        'base': transform_unsupervised,
        'normal': transform_supervised_normal,
        'anomaly': transform_supervised_anomaly,
    }

    # Initialize datasets with transformation
    train_dataset = CustomDataset(args.data_path, is_train=True, transforms_dict=transforms_dict, data_info=args.data_info)
    # Here I set the unsupervised transformation dict since its for testing
    bound_dataset = CustomDataset(args.data_path, is_train=False, transforms_dict={'test': transform_unsupervised}, data_info=args.data_info)

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
    