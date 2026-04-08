from app.utils.visualizer import Visualizer
from app.model.vit_decoder import get_vitdec
from app.utils.train_utils import ViTDecTrainer
from app.utils.load_dataset import MVTecDataModule


def model_training():
    data_module = MVTecDataModule(data_info=True)
    train_loader = data_module.train_dataloader
    val_loader = data_module.val_dataloader

    # visualizer = Visualizer()
    # visualizer.display_data(train_loader, num_batches=1)

    vitdec =  get_vitdec(dry_run=True)
    trainer = ViTDecTrainer(vitdec, train_loader, val_loader)
    trainer.train()
    
if __name__ == "__main__":
    model_training()