"""
Evaluating the model using ROC AUC score which measures the model's ability to distinguish between classes (in this case, anomalous vs. normal pixels), with a score of 1.0 representing perfect discrimination and a score of 0.5 representing random guessing.

- **Detection scores** are used for identifying the presence of anomalies in the entire image without pinpointing their exact locations.

- **Segmentation scores** provide a detailed map of anomaly likelihood across an image, useful for precisely locating anomalies
"""
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def get_bound_results(model, test_loader, device, apply_gaussian=False):
    model.eval()
    MSE = nn.MSELoss(reduction='none')
    
    det_scores = []
    test_imgs = []
    recon_imgs = []
    gt_list = []
    probs_list = []  

    with torch.no_grad():
        for (x, label, _) in tqdm(test_loader, desc="Testing"):
            x = x.to(device)
            label = label.to(device)
            
            # Forward pass
            cls_token_logits, reconstructed_output = model(x, return_logits=True, return_reconstruction=True)

            # Convert logits to probabilities
            probs = F.softmax(cls_token_logits, dim=1)
            probs_list.append(probs.cpu().numpy())
            
            # Gaussian filter smoothing
            if apply_gaussian:
                det_sig = 2
                x_hat_np = reconstructed_output.detach().cpu().numpy()
                x_hat_smoothed_np = np.array([
                        gaussian_filter(x_hat_np[i], sigma=(0, det_sig, det_sig)) 
                        for i in range(x_hat_np.shape[0])
                    ])
                
                x_hat_smoothed = torch.tensor(x_hat_smoothed_np, dtype=torch.float32, device=device)
                reconstructed_output = x_hat_smoothed
            
            # Calculate Anomaly Score (MSE)
            mse = MSE(reconstructed_output, x)
            # Average over channels (dim 1)
            mse_pixel_mean_across_channels = mse.mean(dim=1)
            # Average over spatial dimensions (H, W) -> (B,)
            overall_mse_mean = mse_pixel_mean_across_channels.mean(dim=[1, 2])
            
            det_scores.extend(overall_mse_mean.cpu().numpy())      
            gt_list.extend(label.cpu().numpy())                    
            test_imgs.extend(x.cpu().numpy())                      
            recon_imgs.extend(reconstructed_output.cpu().numpy())  
            
    return {
        'det_scores': np.array(det_scores),                 # List of anomaly scores
        'gt_list': np.array(gt_list),                       # List of ground truth labels
        'test_imgs': np.array(test_imgs),                   # Original images
        'recon_imgs': np.array(recon_imgs),                 # Reconstructed images
        'probs_list': np.concatenate(probs_list, axis=0) 
     }
    
    
if __name__ == "__main__":
    
    
    bound_results = get_bound_results()
    