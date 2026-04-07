import torch
import operator
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def test(model, test_loader, thresholds, device, apply_gaussian=False):
    """Inference function that applies the combined Reconstructor + Classifier logic,
    using pre-calculated thresholds."""
    
    model.eval()
    MSE = nn.MSELoss(reduction='none')
    
    results = []
    overlap_start, overlap_end = thresholds['classifier_range']
    perfect_separation = overlap_start > overlap_end
    
    if not perfect_separation:
        below_is_anomaly = thresholds['anomaly_threshold']['rule'] == '<'
        above_is_anomaly = thresholds['anomaly_threshold']['rule'] == '>'
    else:
        ops = {'>=': operator.ge, '>': operator.gt, '<=': operator.le, '<': operator.lt}
        standalone_rule = thresholds['standalone_recon_threshold']['rule']
        standalone_thresh = thresholds['standalone_recon_threshold']['threshold']

    with torch.no_grad():
        for (x, label, _) in tqdm(test_loader, desc="Testing(Inference)"):
            x = x.to(device)
            label = label.to(device) 
            
            # Forward pass
            cls_token_logits, reconstructed_output = model(x, return_logits=True, return_reconstruction=True)
            
            # Gaussian filter smoothing
            if apply_gaussian:
                det_sig = 2
                x_hat_np = reconstructed_output.detach().cpu().numpy()
                x_hat_smoothed_np = np.array([
                    gaussian_filter(x_hat_np[i], sigma=(0, det_sig, det_sig)) 
                    for i in range(x_hat_np.shape[0])
                ])
                reconstructed_output = torch.tensor(x_hat_smoothed_np, dtype=torch.float32, device=device)
            
            # Calculate Anomaly Score (MSE)
            mse = MSE(reconstructed_output, x)
            mse_pixel_mean = mse.mean(dim=1)
            overall_mse_mean = mse_pixel_mean.mean(dim=[1, 2]).cpu().numpy()
            
            # Get Classifier Probabilities
            probs = F.softmax(cls_token_logits, dim=1).cpu().numpy()
            predicted_classes = np.argmax(probs, axis=1)
            
            # Apply the Combined Logic
            batch_preds = np.zeros_like(overall_mse_mean, dtype=int)
            batch_decision_source = np.empty_like(overall_mse_mean, dtype=object)
            
            if not perfect_separation:
                # Classifier Range (Inside bounds)
                in_range = (overall_mse_mean >= overlap_start) & (overall_mse_mean <= overlap_end)
                batch_preds[in_range] = predicted_classes[in_range]
                batch_decision_source[in_range] = 'classifier'
                
                # Definite Range (Below overlap)
                below_range = overall_mse_mean < overlap_start
                batch_preds[below_range] = 1 if below_is_anomaly else 0
                batch_decision_source[below_range] = 'recon_threshold_rule'
                
                # Definite Range (Above overlap)
                above_range = overall_mse_mean > overlap_end
                batch_preds[above_range] = 1 if above_is_anomaly else 0
                batch_decision_source[above_range] = 'recon_threshold_rule'
                
            else:
                # Perfect Separation Fallback
                batch_preds = ops[standalone_rule](overall_mse_mean, standalone_thresh).astype(int)
                batch_decision_source[:] = 'standalone_reconstructor'

            # Store Results
            for i in range(len(overall_mse_mean)):
                pred_str = 'anomaly' if batch_preds[i] == 1 else 'normal'
                
                results.append({
                    'true_label': label[i].item(),
                    'predicted_label': batch_preds[i].item(),
                    'predicted_class': pred_str,
                    'classifier_probs': probs[i],
                    'reconstruction_error': overall_mse_mean[i].item(),
                    'decision_source': str(batch_decision_source[i])
                })

    return results