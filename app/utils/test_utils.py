"""
Evaluating the model using ROC AUC score which measures the model's ability to distinguish between classes (in this case, anomalous vs. normal pixels), with a score of 1.0 representing perfect discrimination and a score of 0.5 representing random guessing.

- **Detection scores** are used for identifying the presence of anomalies in the entire image without pinpointing their exact locations.

- **Segmentation scores** provide a detailed map of anomaly likelihood across an image, useful for precisely locating anomalies
"""
import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from app.config import DEVICE, VIT_CHECKPOINT_PATH
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def get_bound_results(model, bound_loader, device, apply_gaussian=False):
    model.eval()
    MSE = nn.MSELoss(reduction='none')
    
    det_scores = []
    test_imgs = []
    recon_imgs = []
    gt_list = []
    probs_list = []  

    with torch.no_grad():
        for (x, label, _) in tqdm(bound_loader, desc="Testing"):
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
    
def get_best_score(det_scores, gt_list):
    """Finds the optimal threshold for anomaly detection by maximizing the F1-score."""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    best_f1 = -np.inf
    best_threshold = 0
    best_preds = None
    
    normal_scores = det_scores[gt_list == 0]
    anomalous_scores = det_scores[gt_list == 1]
    
    best_f1 = -np.inf
    best_threshold = 0

    min_norm_score = normal_scores.min()
    max_norm_score = normal_scores.max()
    
    test_thresholds = np.linspace(min_norm_score, max_norm_score, num=100)  

    for threshold in test_thresholds:
        # Make predictions based on the threshold
        binary_predictions = [1 if score >= threshold else 0 for score in det_scores]

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            gt_list, binary_predictions, average='binary', zero_division=0
        )

        # Store the best threshold and F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Calculate and print all metrics for the best threshold
    binary_predictions = [1 if score >= best_threshold else 0 for score in det_scores]
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_list, binary_predictions, average='binary', zero_division=0
    )
    
    # Calculate accuracy for the best threshold
    accuracy = accuracy_score(gt_list, binary_predictions)
    
    # Print the bounds and metrics for the best F1 score 
    print("---------- Best Threshold ----------")
    print(f"Best Threshold: {best_threshold}")

    print("\n----- Best F1 score metrics -----")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    scores = {
        "best_threshold": best_threshold,
        "accuracy": accuracy.round(4),
        "precision": precision.round(4),
        "recall": recall.round(4),
        "f1_score": f1.round(4)
    }
    
    return scores

def get_classification_range(det_scores, gt_list):
    """
    Calculates the overlapping score range where the classifier will be applied.
    - Scores < lower_bound: 100% Normal
    - Scores > upper_bound: 100% Anomalous
    - lower_bound <= Scores <= upper_bound: Send to classifier
    """
    normal_scores = det_scores[gt_list == 0]
    anomalous_scores = det_scores[gt_list == 1]
    
    # The start of the ambiguity (the lowest score an anomaly ever got)
    lower_bound = anomalous_scores.min()
    
    # The end of the ambiguity (the highest score a normal image ever got)
    upper_bound = normal_scores.max()
    
    print("---------- Stage 1 Routing Logic ----------")
    # Check if there is actually an overlap in your data
    if lower_bound <= upper_bound:
        print(f"1. Guaranteed Normal: score < {lower_bound:.4f}")
        print(f"2. APPLY CLASSIFIER:  {lower_bound:.4f} <= score <= {upper_bound:.4f}")
        print(f"3. Guaranteed Anomaly: score > {upper_bound:.4f}")
    else:
        print("Perfect Separation Detected! No overlap exists.")
        print(f"A secondary classifier is mathematically unnecessary for this dataset.")
        
    return float(lower_bound), float(upper_bound)