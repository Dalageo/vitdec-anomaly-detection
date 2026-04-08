import torch
from app.utils.test_utils import test
from app.utils.visualizer import Visualizer
from app.model.vit_decoder import get_vitdec
from app.config import VIT_CHECKPOINT_PATH, DEVICE
from app.utils.load_dataset import MVTecDataModule
from app.utils.bound_utils import get_bound_results



def model_evaluation():
    data_module = MVTecDataModule(data_info=True)
    bound_loader = data_module.bound_dataloader
    
    vitdec =  get_vitdec(dry_run=True)
    vitdec_path = torch.load(VIT_CHECKPOINT_PATH)
    vitdec.load_state_dict(vitdec_path['model_state_dict'], strict=False)
    
    bound_results = get_bound_results(vitdec, bound_loader, DEVICE)
    image_visualizer = Visualizer()
    image_visualizer.plot_bound_combined_confusion_matrix(bound_results=bound_results, save_plot=True)  
    
    test_loader = data_module.test_dataloader
    test_results = test(vitdec, test_loader, bound_results['thresholds'], DEVICE)
    
    image_visualizer.plot_test_combined_confusion_matrix(test_results=test_results, save_plot=True)
    

if __name__ == "__main__":
    model_evaluation()