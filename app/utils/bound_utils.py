import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from app.utils.log_utils import LoggerConfig
from sklearn.metrics import precision_recall_fscore_support

logger = LoggerConfig().get_logger(__name__)


# -----------------------
# Executes bound workflow
# -----------------------
def get_bound_results(model, bound_loader, device, apply_gaussian=False):
    model.eval()
    MSE = nn.MSELoss(reduction='none')
    
    det_scores = []
    test_imgs = []
    recon_imgs = []
    gt_list = []
    probs_list = []  

    with torch.no_grad():
        for (x, label, _) in tqdm(bound_loader, desc="Testing(Bound)"):
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
            
    det_scores_np = np.array(det_scores)
    gt_list_np = np.array(gt_list)
            
    return {
        'det_scores': det_scores_np,                             # List of anomaly scores
        'gt_list': gt_list_np,                                   # List of ground truth labels
        'test_imgs': np.array(test_imgs),                        # Original images
        'recon_imgs': np.array(recon_imgs),                      # Reconstructed images
        'probs_list': np.concatenate(probs_list, axis=0),        # Classifier probability predictions
        'thresholds': get_thresholds(det_scores_np, gt_list_np)  # Overlap bounds and thresholds
     }
    
    

# ------------------------------
# Decision Boundary Calculations
# ------------------------------
def get_optimal_threshold(det_scores, gt_list):
    """Finds the single threshold that maximizes the F1-score across the entire score range."""
    
    normal_scores = det_scores[gt_list == 0]
    best_f1 = -np.inf
    best_threshold = 0

    min_norm_score = normal_scores.min()
    max_norm_score = normal_scores.max()

    test_thresholds = np.linspace(min_norm_score, max_norm_score, num=100)  
    for threshold in test_thresholds:
        binary_predictions = (det_scores >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            gt_list, binary_predictions, average='binary', zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return {"threshold": float(best_threshold), "rule": ">="}



def get_thresholds(det_scores, gt_list):
    """Calculates the overlapping score range between normal and anomalous samples to define a decision boundary.
    Identifies ranges where scores are uniquely normal or anomalous and isolates the overlapping 'classifier_range' for ambiguous cases."""
    
    if len(np.unique(gt_list)) < 2:
        logger.warning("⚠️ Only one class found in gt_list. Cannot calculate overlap.")
        return None
    
    normal_scores = det_scores[gt_list == 0]
    anomalous_scores = det_scores[gt_list == 1]
    
    min_norm = normal_scores.min()
    max_norm = normal_scores.max()
    
    min_anom = anomalous_scores.min()
    max_anom = anomalous_scores.max()
    
    overlap_start = float(max(min_norm, min_anom))
    overlap_end = float(min(max_norm, max_anom))
    
    thresholds = {
        "classifier_range": (overlap_start, overlap_end),
        "normal_threshold": {"threshold": None, "rule": None},
        "anomaly_threshold": {"threshold": None, "rule": None},
        "standalone_recon_threshold": get_optimal_threshold(det_scores, gt_list)
    }
    
    if overlap_start <= overlap_end:
        if min_anom < min_norm:
            thresholds["anomaly_threshold"] = {"threshold": overlap_start, "rule": "<"}
        else:
            thresholds["normal_threshold"] = {"threshold": overlap_start, "rule": "<"}
            
        if max_norm > max_anom:
            thresholds["normal_threshold"]["threshold"] = overlap_end
            thresholds["normal_threshold"]["rule"] = ">"
        else:
            thresholds["anomaly_threshold"]["threshold"] = overlap_end
            thresholds["anomaly_threshold"]["rule"] = ">"
            
        logger.info(f"Overlapping Range: {overlap_start:.6f} to {overlap_end:.6f}")
        logger.info(f"Normal if Reconstruction Score {thresholds['normal_threshold']['rule']} {thresholds['normal_threshold']['threshold']:.6f}")
        logger.info(f"Anomaly if Reconstruction Score {thresholds['anomaly_threshold']['rule']} {thresholds['anomaly_threshold']['threshold']:.6f}")
        
    else:
        logger.info("Perfect Separation! No classifier range needed.")
    
    logger.info(f"Standalone Reconstructor Optimal Threshold: {thresholds['standalone_recon_threshold']['threshold']:.6f} (Anomaly if Score {thresholds['standalone_recon_threshold']['rule']} Threshold)")
    return thresholds