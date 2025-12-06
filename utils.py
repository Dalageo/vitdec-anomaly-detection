import time
import torch
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Tuple Conversion Function
# -------------------------
# Convert a single value to a 2-tuple, or return the tuple if it's already one.
def to_2tuple(x):
    return (x, x) if not isinstance(x, tuple) else x


# ------------------------
# Metrics Monitoring Class
# ------------------------
class AverageMeter(object):
    # Computes and stores the average and current value of metrics during training.
    def __init__(self):
        self.reset()

    # Reset all statistics.
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # Update the statistics for the meter
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# -------------------------
# Time Formatting Functions
# -------------------------
# Generate a string that present current time
def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string

# Convert a time in seconds to a more readable format (hours, minutes, seconds).
def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


# ----------------
# Logging Function
# ----------------
# Print a string to both the console and a log file, and flush the log.
def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()
    

# ------------------------------
# Image Denormalization Function
# ------------------------------
# Denormalize a tensor by scaling it to [0, 255] and converting to bytes.
def denormalize(tensor, mean, std):
    device = tensor.device
    mean_tensor = torch.tensor(mean, device=device).view(3, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(3, 1, 1)
    return tensor * std_tensor + mean_tensor

# -----------------------
# Visualization Functions
# -----------------------
# Plot and show original and reconstructed images side-by-side.
def plot_show(img, ori, epoch, mean, std):
    # Denormalize and prepare the first image in the batch for display
    img_denorm = denormalize(img[0], mean, std).clamp(0, 1).detach().cpu().numpy()
    ori_denorm = denormalize(ori[0], mean, std).clamp(0, 1).detach().cpu().numpy()

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

    
# Plot training and validation loss curves for total, classification, and reconstruction losses. 
def plot_loss(train_total_losses, val_total_losses, train_classification_losses, val_classification_losses, train_reconstruction_losses, val_reconstruction_losses):
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
    plt.plot(train_classification_losses, label='Training Classification Loss')
    plt.plot(val_classification_losses, label='Validation Classification Loss')
    plt.title('Classification Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plotting Reconstruction loss
    plt.subplot(1, 3, 3)
    plt.plot(train_reconstruction_losses, label='Training Reconstruction Loss')
    plt.plot(val_reconstruction_losses, label='Validation Reconstruction Loss')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')  
    plt.grid(True)

    plt.tight_layout()
    plt.show()