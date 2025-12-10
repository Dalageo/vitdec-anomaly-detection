# final.py
import torch
import argparse
from vit_decoder import vitdec_init
from train_utils import ModelTrainer  
from load_dataset import get_dataloader, VisualizeImages


def main(args):
    # Load data 
    train_loader, val_loader, bound_loader, test_loader = get_dataloader(args)

    # Optionally visualize images 
    if args.img_batch_info > 0:
        visualizer = VisualizeImages(mean=args.mean, std=args.std)
        print(f"\nDisplaying {args.img_batch_info} batches of images from the training loader:")
        visualizer.check_data(train_loader, img_batch_info=args.img_batch_info)
    
    # Initialize the ViT-Decoder model
    vit_dec = vitdec_init(args)
    
    # Initialize the trainer
    trainer = ModelTrainer(args, vit_dec, train_loader, val_loader)
    # Execute training and validation
    trainer.execute_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified script to load data, build model, and train.")

    # --------------------
    # Dataset Loading Args
    # --------------------
    parser.add_argument('--data_path',      type=str,                   required=True,              help='Path to the dataset')
    parser.add_argument('--sv_aug',         type=bool,                  default=False,              help='Enable supervised augmentation')
    parser.add_argument('--us_aug',         type=bool,                  default=False,              help='Enable unsupervised augmentation')
    parser.add_argument('--val_ratio',      type=float,                 default=0.15)
    parser.add_argument('--test_ratio',     type=float,                 default=0.5)
    parser.add_argument('--batch_size',     type=int,                   default=16,                 help='Batch size for training')
    parser.add_argument('--img_res',        type=int,                   default=384)
    parser.add_argument('--img_batch_info', type=int,                   default=0)
    parser.add_argument('--data_info',      action='store_true',                                    help='Print dataset info')
    parser.add_argument('--mean',           type=float, nargs=3,        default=[0.5, 0.5, 0.5])
    parser.add_argument('--std',            type=float, nargs=3,        default=[0.5, 0.5, 0.5])
    parser.add_argument('--seed',           type=int,                   default=42)

    # ----------------
    # Vit-Decoder Args
    # ----------------
    parser.add_argument('--vit_weights',    type=str,                   default="",                 help='Path to .npz or .pth weights')
    parser.add_argument('--device',         type=str,                   default='cuda')
    parser.add_argument('--num_classes',    type=int,                   default=2)
    parser.add_argument('--output_test',    action='store_true',                                    help='Test with dummy input?')

    # -------------------------
    # Training ViT-Decoder Args
    # -------------------------
    parser.add_argument('--save_dir',       type=str,                   default='./checkpoints',    help='Directory to save checkpoints')
    parser.add_argument('--epochs',         type=int,                   default=50,                 help='Number of training epochs')
    parser.add_argument('--lr_vit',         type=float,                 default=1e-5,               help='Learning rate for Vision Transformer')
    parser.add_argument('--lr_dec',         type=float,                 default=1e-4,               help='Learning rate for Decoder')
    parser.add_argument('--beta1',          type=float,                 default=0.85,               help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2',          type=float,                 default=0.999,              help='Beta2 for Adam optimizer')
    parser.add_argument('--wd_vit',         type=float,                 default=1e-5,               help='Weight decay for Vision Transformer')
    parser.add_argument('--wd_dec',         type=float,                 default=1e-5,               help='Weight decay for Decoder')
    parser.add_argument('--amp',            action='store_true',                                    help='Enable automatic mixed precision')

    args = parser.parse_args()
    
    # Call the main function
    main(args)
    
