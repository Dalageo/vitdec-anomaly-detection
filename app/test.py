import torch
from app.utils.visualizer import Visualizer
from app.utils.test_utils import get_bound_results
from app.utils.load_dataset import MVTecDataModule
from app.model.vit_decoder import get_vitdec
from app.config import VIT_CHECKPOINT_PATH, DEVICE


def model_training():
    data_module = MVTecDataModule(data_info=True)
    bound_loader = data_module.bound_dataloader
    
    vitdec =  get_vitdec(dry_run=True)
    vitdec_path = torch.load(VIT_CHECKPOINT_PATH)
    vitdec.load_state_dict(vitdec_path['model_state_dict'], strict=False)
    
    bound_results = get_bound_results(vitdec, bound_loader, DEVICE)
    image_visualizer = Visualizer()
    image_visualizer.plot_anomaly_score_distribution(det_scores=bound_results["det_scores"], gt_list=bound_results["gt_list"], save_plot=True)  
    