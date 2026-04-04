import os
import torch
import operator
import numpy as np
import seaborn as sns
from torchvision import utils as torchvision_utils
import matplotlib.pyplot as plt
from app.utils.log_utils import LoggerConfig
from app.config import MEAN, STD, PLOT_OUTPUT_PATH
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


logger = LoggerConfig().get_logger(__name__)

# ----------------
# Visualizer Class
# ----------------
class Visualizer:
    def __init__(self):
        self.mean = MEAN
        self.std = STD
        
        self.plot_output_path = PLOT_OUTPUT_PATH

    # Denormalize a tensor image
    def denormalize(self, tensor):
        """Denormalize a tensor by scaling it to [0, 255] and converting to bytes."""
        mean = torch.tensor(self.mean).view(3, 1, 1).to(tensor.device)
        std = torch.tensor(self.std).view(3, 1, 1).to(tensor.device)
        return tensor * std + mean

    
    def imshow(self, img, ax):
        """Display an image on a given axes."""
        img = self.denormalize(img)  # Denormalize the image
        img = img.clamp(0, 1)        # Ensure the values are in the range [0, 1]
        img = img.permute(1, 2, 0)   # Convert from CxHxW to HxWxC
        ax.imshow(img.numpy())       # Convert to numpy for imshow
        ax.axis('off')

    
    def display_data(self, loader, num_batches):
        """Displays images from the first batch of data loader."""
        for i, (images, labels, categories) in enumerate(loader):
            
            if i >= num_batches:
                break
            
            logger.info(f"Batch {i+1} labels: {labels}")
            logger.info(f"Batch {i+1} categories: {categories}")

            # Denormalize the images in the batch before passing to make_grid
            images = torch.stack([self.denormalize(img) for img in images])
            
            # Create a grid with black padding (pad_value=0)
            grid_img = torchvision_utils.make_grid(images, nrow=10, padding=15, pad_value=255)  # Pad with black (0)

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


    def display_reconstruction(self, original_img, recon_img, epoch, save_plot: bool = False):
        """Visualizes the first image of the batch alongside its reconstruction."""
        
        # Denormalize and prepare the first image in the batch for display
        orginal_denorm = self.denormalize(original_img[0]).clamp(0, 1).detach().cpu().numpy()
        recon_denorm = self.denormalize(recon_img[0]).clamp(0, 1).detach().cpu().numpy()

        # Transpose tensors from Channel-Height-Width to Height-Width-Channel format for plotting
        original_numpy = np.transpose(orginal_denorm, (1, 2, 0))
        recon_numpy = np.transpose(recon_denorm, (1, 2, 0))

        # Plotting the original and reconstructed images
        fig, plots = plt.subplots(1, 2)
        fig.set_figwidth(9)
        fig.set_tight_layout(True)

        plots[0].imshow(original_numpy)
        plots[0].set_title(f"Original - Epoch {epoch}")
        plots[0].axis('off') 

        plots[1].imshow(recon_numpy)
        plots[1].set_title(f"Reconstructed - Epoch {epoch}")
        plots[1].axis('off')  

        # Save the plot
        if save_plot:
            os.makedirs('Images', exist_ok=True)
            plt.savefig(f'Images/Epoch_{epoch}.png')  
        
        plt.show()
        plt.close(fig)
        
     
    def plot_learning_curves(self, train_total_losses, val_total_losses, train_cls_losses, val_cls_losses, 
                             train_recon_losses, val_recon_losses, save_plot=False):
        """Plot training and validation loss curves for total, classification, and reconstruction losses."""

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs = range(1, len(train_total_losses) + 1)
        
        def plot_subplot(ax, train_data, val_data, title):
            ax.plot(epochs, train_data, label='Training', color='tab:blue')
            ax.plot(epochs, val_data, label='Validation', color='tab:orange', linestyle='--')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Total Loss
        plot_subplot(axes[0], train_total_losses, val_total_losses, 'Total Loss')

        # Classification Loss
        plot_subplot(axes[1], train_cls_losses, val_cls_losses, 'Classification Loss')

        # Reconstruction Loss
        plot_subplot(axes[2], train_recon_losses, val_recon_losses, 'Reconstruction Loss')

        plt.tight_layout()
        
        if save_plot:
            os.makedirs(os.path.dirname(self.plot_output_path), exist_ok=True)
            plt.savefig(self.plot_output_path, dpi=300)
        
        plt.show()
        plt.close(fig)
        
        
    def plot_anomaly_score_distribution(self, det_scores, gt_list, save_plot=None):
        """Plots the histogram and KDE (Kernel Density Estimate) of anomaly scores 
        for Normal vs. Anomalous classes."""
        
        plt.figure(figsize=(10, 6))

        # Split scores by class
        normal_scores = det_scores[gt_list == 0]
        anomalous_scores = det_scores[gt_list == 1]
        
        logger.info(f"Anomaly Reconstruction Scores")
        logger.info(f"Minimum Anomaly Score: {anomalous_scores.min()}")
        logger.info(f"Maximum Anomaly Score: {anomalous_scores.max()} \n")

        logger.info(f"Normal Reconstruction Scores")
        logger.info(f"Minimum Normal Score: {normal_scores.min()}")
        logger.info(f"Maximum Normal Score: {normal_scores.max()}")

        # Plot Normal Scores
        sns.histplot(normal_scores, bins=30, kde=(len(np.unique(normal_scores)) > 1), 
                     label='Normal', color='blue', alpha=0.6)

        # Plot Anomalous Scores
        sns.histplot(anomalous_scores, bins=30, kde=(len(np.unique(anomalous_scores)) > 1), 
                     label='Anomalous', color='red', alpha=0.6)

        plt.xlabel('Anomaly Score')
        plt.ylabel('Count') 
        plt.title('Distribution of Anomaly Scores')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        if save_plot:
            os.makedirs(os.path.dirname(self.plot_output_path), exist_ok=True)
            plt.savefig(self.plot_output_path, dpi=300)
            
        plt.show()
        plt.close()
        
        
    def plot_confusion_matrices(self, bound_results):
        """Plots side-by-side confusion matrices for the Reconstructor and Classifier.
        This allows for a direct visual and statistical comparison between the 
        baseline reconstructor (using the optimal threshold) and the standalone classifier."""
        
        gt_list = bound_results['gt_list']
        det_scores = bound_results['det_scores']
        class_probabilities = bound_results['probs_list']
        standalone_recon_threshold = bound_results['thresholds']['standalone_recon_threshold']['threshold']
        standalone_recon_threshold_rule = bound_results['thresholds']['standalone_recon_threshold']['rule']
        
        gt_array = np.array(gt_list)
        
        operators = {'>=': operator.ge, '>': operator.gt, '<=': operator.le, '<': operator.lt}
        binary_predictions = operators[standalone_recon_threshold_rule](det_scores, standalone_recon_threshold).astype(int)
        
        # Classifier Predictions
        all_probabilities = np.vstack(class_probabilities) 
        predicted_labels = np.argmax(all_probabilities, axis=1)

        # Calculate Accuracies
        recon_acc = np.mean(binary_predictions == gt_array)
        cls_acc = np.mean(predicted_labels == gt_array)
        
        print(f'Reconstructor - Accuracy: {recon_acc:.4f} --> {recon_acc * 100:.2f}%')
        print(f'Classifier - Accuracy: {cls_acc:.4f} --> {cls_acc * 100:.2f}%\n')

        # Reconstructor Metrics
        recon_p, recon_r, recon_f1, _ = precision_recall_fscore_support(
            gt_array, binary_predictions, average='binary', zero_division=0)
        recon_cm = confusion_matrix(gt_array, binary_predictions)
        recon_text = f'Accuracy: {recon_acc*100:.1f}% | Precision: {recon_p:.2f} | Recall: {recon_r:.2f} | F1: {recon_f1:.2f}'

        # Classifier Metrics
        cls_p, cls_r, cls_f1, _ = precision_recall_fscore_support(
            gt_array, predicted_labels, average='binary', zero_division=0)
        cls_cm = confusion_matrix(gt_array, predicted_labels)
        cls_text = f'Accuracy: {cls_acc*100:.1f}% | Precision: {cls_p:.2f} | Recall: {cls_r:.2f} | F1: {cls_f1:.2f}'

        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        sns.heatmap(recon_cm, annot=True, fmt="d", cmap='Blues', 
                    xticklabels=['Normal', 'Anomalous'], 
                    yticklabels=['Normal', 'Anomalous'], ax=ax[0])
        ax[0].set_title(f'Confusion Matrix - Reconstructor (Using Optimal Threshold)\n{recon_text}')
        ax[0].set_xlabel('Predicted Labels')
        ax[0].set_ylabel('True Labels')

        sns.heatmap(cls_cm, annot=True, fmt="d", cmap='Blues', 
                    xticklabels=['Normal', 'Anomalous'], 
                    yticklabels=['Normal', 'Anomalous'], ax=ax[1])
        ax[1].set_title(f'Confusion Matrix - Classifier\n{cls_text}')
        ax[1].set_xlabel('Predicted Labels')
        ax[1].set_ylabel('True Labels')

        plt.tight_layout()
        plt.show()