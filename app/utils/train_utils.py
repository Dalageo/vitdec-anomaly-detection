import os
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from app.utils.visualizer import Visualizer
from app.utils.log_utils import LoggerConfig
from app.utils.log_utils import AverageMeter, convert_secs2time
from app.config import MEAN, STD, \
                       EPOCHS, BATCH_SIZE, DEVICE, \
                       AMP, BETA_1, BETA_2, \
                       LR_VIT, LR_DEC, WD_VIT, WD_DEC, \
                       LOG_OUTPUT_PATH, CHECKPOINT_PATH
                       
logger = LoggerConfig().get_logger(__name__)

# --------------
# Early Stopping 
# --------------
class EarlyStop:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.checkpoint_path = CHECKPOINT_PATH
        self.verbose = True

    def __call__(self, class_loss, recon_loss, model, epoch=0, optimizer_vit=None, optimizer_dec=None, scheduler=None):
        overall_loss = class_loss + recon_loss
        
        # Check if the new loss is significantly better
        if overall_loss < self.val_loss_min - self.delta:  
            if self.verbose:
                logger.info(f"Overall loss decreased ({self.val_loss_min:.6f} --> {overall_loss:.6f}). Saving model...")
                
            self.save_checkpoint(model, epoch, optimizer_vit, optimizer_dec, scheduler)
            self.val_loss_min = overall_loss 
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")

        if self.counter >= self.patience:
            self.early_stop = True
            logger.warning("Early stopping triggered")
            return True

        return False

    def save_checkpoint(self, model, epoch=0, optimizer_vit=None, optimizer_dec=None, scheduler=None):
        """ Saves model when validation loss decrease. """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': self.val_loss_min,
        }
        
        if optimizer_vit is not None:
            checkpoint['optimizer_vit_state_dict'] = optimizer_vit.state_dict()
        if optimizer_dec is not None:
            checkpoint['optimizer_dec_state_dict'] = optimizer_dec.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_path)
        logger.info("Model saved")
        
        
# -----------------------------------------
# Class for training the custom ViT-Decoder
# -----------------------------------------
class ViTDecTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.visualizer = Visualizer()
        
        # System & Logging Setup
        log_dir = os.path.dirname(LOG_OUTPUT_PATH)
        os.makedirs(log_dir, exist_ok=True)
        self.checkpoint_path = CHECKPOINT_PATH
        self.device = DEVICE
        
        # Data & Training Hyperparameters
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE
        self.mean = MEAN
        self.std = STD
        
        # Optimizer Configuration
        self.lr_vit = LR_VIT
        self.lr_dec = LR_DEC
        self.wd_vit = WD_VIT
        self.wd_dec = WD_DEC
        self.beta1 = BETA_1
        self.beta2 = BETA_2
        self.amp = AMP
        
        # Loss Functions & Tools
        # self.mse = nn.MSELoss()
        self.mse = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler(enabled=AMP) 
        self.early_stop = EarlyStop(patience=5, delta=0.00001)
        
        # Initialize Training Components
        self.get_train_components()
    
        
    # ---------------------------------------
    # Optimizers and scheduler initialization 
    # ---------------------------------------
    def get_train_components(self):
        # Vit's optimizer Initialization
        self.optimizer_vit = torch.optim.Adam(
            self.model.vit_encoder.parameters(),
            lr=self.lr_vit,
            betas=(self.beta1, self.beta2),
            weight_decay=self.wd_vit
        )
        # Decoder's optimizer Initialization
        self.optimizer_dec = torch.optim.Adam(
            self.model.decoder.parameters(),
            lr=self.lr_dec,
            betas=(self.beta1, self.beta2),
            weight_decay=self.wd_dec
        )
        
        # Scheduler's Initialization
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_vit,
            mode='min',
            factor=0.1,
            patience=2
        )


    def load_checkpoint(self):
        """Load model checkpoint and optimizer states to resume training"""
        
        if not os.path.exists(self.checkpoint_path):
            logger.warning(f"Checkpoint not found at {self.checkpoint_path}")
            return 0
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        
        # Load optimizer states
        if 'optimizer_vit_state_dict' in checkpoint:
            self.optimizer_vit.load_state_dict(checkpoint['optimizer_vit_state_dict'])
        if 'optimizer_dec_state_dict' in checkpoint:
            self.optimizer_dec.load_state_dict(checkpoint['optimizer_dec_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
        return start_epoch

    # --------------------------------------
    # Training operations for a single epoch
    # --------------------------------------
    def train_epoch(self, epoch):
        # Set model to training mode
        self.model.train()
        
        sum_recon_loss = 0.0
        sum_cls_loss = 0.0
        total_recon_samples = 0
        total_cls_samples = 0

        for (x, y, _) in tqdm(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            # Zero Gradients
            self.optimizer_vit.zero_grad()
            self.optimizer_dec.zero_grad()
            
            # Masks for processing logic
            base_mask = y == -1
            normal_mask = y == 0
            anomaly_mask = y == 1

            with torch.amp.autocast(device_type=self.device, enabled=self.amp):
                # Initialize batch losses
                recon_loss_batch = torch.tensor(0.0, device=self.device)
                cls_loss_batch = torch.tensor(0.0, device=self.device)
                
                # Reconstruction Logic (Base + Normal)
                recon_mask = base_mask | normal_mask
                if recon_mask.any():
                    _, recon_output = self.model(x[recon_mask], return_logits=False, return_reconstruction=True)
                    recon_loss_batch = self.mse(recon_output, x[recon_mask])

                    # Multiply mean by count to get sum of errors
                    count = recon_mask.sum().item()
                    sum_recon_loss += recon_loss_batch.item() * count
                    total_recon_samples += count
        
                # Classification Logic (Normal + Anomaly)
                cls_mask = normal_mask | anomaly_mask
                if cls_mask.any():
                    cls_logits, _ = self.model(x[cls_mask], return_logits=True, return_reconstruction=False)
                    cls_loss_batch = self.cross_entropy(cls_logits, y[cls_mask])
                    
                    # Multiply mean by count to get sum of errors
                    count = cls_mask.sum().item()
                    sum_cls_loss += cls_loss_batch.item() * count
                    total_cls_samples += count
                    
                # Total Loss
                total_loss = cls_loss_batch + recon_loss_batch
        
            # Backward Pass
            if (recon_mask.any() or cls_mask.any()):
                if self.amp:
                    self.scaler.scale(total_loss).backward()
                    
                    # Only unscale and step optimizers that have gradients
                    # ViT encoder always gets gradients (used in both tasks)
                    self.scaler.unscale_(self.optimizer_vit)
                    self.scaler.step(self.optimizer_vit)
                    
                    # Decoder only gets gradients from reconstruction
                    if recon_mask.any():
                        self.scaler.unscale_(self.optimizer_dec)
                        self.scaler.step(self.optimizer_dec)
                        
                    self.scaler.update()
                else:
                    total_loss.backward()
                    self.optimizer_vit.step()
                    self.optimizer_dec.step()

        # Calculate Averages
        avg_recon_loss = sum_recon_loss / total_recon_samples if total_recon_samples > 0 else 0.0
        avg_cls_loss = sum_cls_loss / total_cls_samples if total_cls_samples > 0 else 0.0
        avg_loss = avg_recon_loss + avg_cls_loss
        
        log_msg = f'Train Epoch: {epoch} | Avg Loss: {avg_loss:.6f} | Avg Reconstruction Loss: {avg_recon_loss:.6f} | Avg Classification Loss: {avg_cls_loss:.6f}'
        logger.info(log_msg)
        
        return avg_loss, avg_recon_loss, avg_cls_loss


    # ----------------------------------------
    # Validation operations for a single epoch
    # ----------------------------------------
    def val_epoch(self, epoch):
        # Switch the model to evaluation mode
        self.model.eval()

        sum_recon_loss = 0.0
        sum_cls_loss = 0.0
        total_recon_samples = 0
        total_cls_samples = 0
        
        last_original = None
        last_recon = None
    
        for (x, y, _) in tqdm(self.val_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            # Masks for processing logic
            base_mask = y == -1
            normal_mask = y == 0
            anomaly_mask = y == 1

            with torch.no_grad():
                # Reconstruction Logic (Base + Normal) 
                recon_mask = base_mask | normal_mask
                if recon_mask.any():
                    _, recon_output = self.model(x[recon_mask], return_logits=False, return_reconstruction=True)
                    loss_batch = self.mse(recon_output, x[recon_mask])
                    
                    # Multiply mean by count to get sum of errors
                    count = recon_mask.sum().item()
                    sum_recon_loss += loss_batch.item() * count
                    total_recon_samples += count
                    
                    # Save images for plotting
                    last_original = x[recon_mask]
                    last_recon = recon_output
                    
                # Classification Logic (Normal + Anomaly)
                cls_mask = normal_mask | anomaly_mask
                if cls_mask.any():
                    cls_logits, _ = self.model(x[cls_mask], return_logits=True, return_reconstruction=False)
                    loss_batch = self.cross_entropy(cls_logits, y[cls_mask])
                    
                    # Multiply mean by count to get sum of errors
                    count = cls_mask.sum().item()
                    sum_cls_loss += loss_batch.item() * count
                    total_cls_samples += count

        # Calculate Correct Averages
        avg_recon_loss = sum_recon_loss / total_recon_samples if total_recon_samples > 0 else 0.0
        avg_cls_loss = sum_cls_loss / total_cls_samples if total_cls_samples > 0 else 0.0
        avg_loss = avg_recon_loss + avg_cls_loss

        if last_recon is not None and epoch % 1 == 0:
            save_plot = (epoch % 1 == 0)
            self.visualizer.plot_show(last_original, last_recon, epoch, save_plot=save_plot)
                
        # Print learning rate for optimizer_vit
        for i, param_group in enumerate(self.optimizer_vit.param_groups):
            logger.info(f'Valid Epoch {epoch}: Learning Rate for Transformer: {param_group["lr"]:.6f}')
        
        # Print learning rates for optimizer_dec
        for i, param_group in enumerate(self.optimizer_dec.param_groups):
            logger.info(f'Valid Epoch {epoch}: Learning Rate for Decoder: {param_group["lr"]:.6f}')

        log_msg = f'Valid Epoch: {epoch} | Avg Loss: {avg_loss:.6f} | Avg Reconstruction Loss: {avg_recon_loss:.6f} | Avg Classification Loss: {avg_cls_loss:.6f}'
        logger.info(log_msg)

        return avg_loss, avg_recon_loss, avg_cls_loss


    # --------------------------
    # Executes training workflow
    # --------------------------
    def train(self, resume=False):
        # Set up time tracking
        start_time = time.time()
        epoch_time = AverageMeter()
        
        # Initialize loss tracking
        train_total_losses, train_recon_losses, train_cls_losses = [], [], []
        val_total_losses, val_recon_losses, val_cls_losses = [], [], []

        # Resume training if needed
        start_epoch = 1
        if resume:
            start_epoch = self.load_checkpoint()
        
        # Main training loop
        for epoch in range(start_epoch, self.epochs + 1):
            # Estimate time remaining
            need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (self.epochs - epoch))
            need_time = f'[Need: {need_hour:02d}:{need_mins:02d}:{need_secs:02d}]'
            log_msg = f' {epoch:3d}/{self.epochs:3d} ----- [{time.strftime("%Y-%m-%d %H:%M:%S")}] {need_time}'
            logger.info(log_msg)
            
            # Receive detailed losses from training
            train_loss, train_recon_loss, train_cls_loss = self.train_epoch(epoch)
            train_total_losses.append(train_loss)
            train_recon_losses.append(train_recon_loss)
            train_cls_losses.append(train_cls_loss)

            # Receive detailed losses from validation
            val_loss, val_recon_loss, val_cls_loss = self.val_epoch(epoch)
            val_total_losses.append(val_loss)
            val_recon_losses.append(val_recon_loss)
            val_cls_losses.append(val_cls_loss)

            # Early stopping checks against total validation loss
            if self.early_stop(val_cls_loss, val_recon_loss, self.model, epoch, 
                          self.optimizer_vit, self.optimizer_dec, self.scheduler):
                log_msg = "Training stopped early due to lack of improvement."
                logger.info(log_msg)
                break

            # Step the scheduler with the validation loss
            self.scheduler.step(val_loss)
            
            # Update epoch time and reset start_time for the next epoch
            epoch_time.update(time.time() - start_time)
            start_time = time.time()

        # Plot training and validation losses.
        self.visualizer.plot_loss(train_total_losses, val_total_losses, train_cls_losses, val_cls_losses, train_recon_losses, val_recon_losses)