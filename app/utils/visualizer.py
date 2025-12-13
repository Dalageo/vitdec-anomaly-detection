

import torch
import numpy as np
from torchvision import utils
import matplotlib.pyplot as plt
from app.config import MEAN, STD
from app.utils.log_utils import LoggerConfig

logger = LoggerConfig().get_logger(__name__)

# ---------------------
# Visualize Image Class
# ---------------------
class Visualizer:
    def __init__(self):
        self.mean = MEAN
        self.std = STD

    # Denormalize a tensor image
    def denormalize(self, tensor):
        """Denormalize a tensor by scaling it to [0, 255] and converting to bytes."""
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
            logger.info(f"Batch {i+1} labels:", labels)
            logger.info(f"Batch {i+1} categories:", categories)

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


    def plot_show(self, img, ori, epoch):
        """# Plot and show original and reconstructed images side-by-side."""
        # Denormalize and prepare the first image in the batch for display
        img_denorm = self.denormalize(img[0]).clamp(0, 1).detach().cpu().numpy()
        ori_denorm = self.denormalize(ori[0]).clamp(0, 1).detach().cpu().numpy()

        # Transpose tensors from Channel-Height-Width to Height-Width-Channel format for plotting
        img_numpy = np.transpose(img_denorm, (1, 2, 0))
        ori_numpy = np.transpose(ori_denorm, (1, 2, 0))

        # Plotting the original and reconstructed images
        fig, plots = plt.subplots(1, 2)
        fig.set_figwidth(9)
        fig.set_tight_layout(True)

        plots[0].imshow(ori_numpy)
        plots[0].set_title(f"Original - Epoch {epoch}")
        plots[0].axis('off') 

        plots[1].imshow(img_numpy)
        plots[1].set_title(f"Reconstructed - Epoch {epoch}")
        plots[1].axis('off')  

        # Save the plot
        plt.savefig(f'Images/Epoch_{epoch}.png')  # Save the figure
        plt.close(fig)
        
     
    def plot_loss(train_total_losses, val_total_losses, train_cls_losses, val_cls_losses, train_recon_losses, val_recon_losses):
        """Plot training and validation loss curves for total, classification, and reconstruction losses."""
        # Initialize a figure with dimensions (15, 5) for the plots
        plt.figure(figsize=(15, 5))
        
        # Plotting total loss
        plt.subplot(1, 3, 1)
        plt.plot(train_total_losses, label='Training Total Loss')
        plt.plot(val_total_losses, label='Validation Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plotting Classification loss
        plt.subplot(1, 3, 2)
        plt.plot(train_cls_losses, label='Training Classification Loss')
        plt.plot(val_cls_losses, label='Validation Classification Loss')
        plt.title('Classification Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plotting Reconstruction loss
        plt.subplot(1, 3, 3)
        plt.plot(train_recon_losses, label='Training Reconstruction Loss')
        plt.plot(val_recon_losses, label='Validation Reconstruction Loss')
        plt.title('Reconstruction Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')  
        plt.grid(True)

        plt.tight_layout()
        plt.show()