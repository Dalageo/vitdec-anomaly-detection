import os
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from app.utils.utils import LoggerConfig
from app.utils import print_log, plot_show, plot_loss, AverageMeter, convert_secs2time
from app.config import AMP, BETA_1, BETA_2, \
                       LR_VIT, LR_DEC, WD_VIT, WD_DEC, \
                       EPOCHS, BATCH_SIZE, DEVICE, \
                       MEAN, STD

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
        self.val_loss_min = np.Inf
        self.save_name = "checkpoint.pt"
        self.verbose = True

    def __call__(self, class_loss, recon_loss, model):
        overall_loss = class_loss + recon_loss
        
        # Check if the new loss is significantly better
        if overall_loss < self.val_loss_min - self.delta:  
            if self.verbose:
                logger.info(f"Overall loss decreased ({self.val_loss_min:.6f} --> {overall_loss:.6f}). Saving model...")
                
            self.save_checkpoint(model)
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

    def save_checkpoint(self, model):
        """ Saves model when validation loss decrease. """
        torch.save(model.state_dict(), self.save_name)
        logger.info("Model saved")
        
        
# -----------------------------------------
# Class for training the custom ViT-Decoder
# -----------------------------------------
class ViTDecTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # System & Logging Setup
        self.save_dir = "app/checkpoints/"
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_path = os.path.join(self.save_dir, 'training_log.txt')
        self.log = open(self.log_path, 'w')
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
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler(enabled=AMP) 
        self.early_stop = EarlyStop(patience=5, delta=0.001)
        
        # Initialize Training Components
        self.get_train_components()
    
    def __del__(self):
        """Ensures the log file is closed when the trainer instance is deleted"""
        if self.log:
            self.log.close()
        
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
                    self.scaler.step(self.optimizer_vit)
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
        
        print_log(f'Train Epoch: {epoch} | Avg Loss: {avg_loss:.6f} | Avg Reconstruction Loss: {avg_recon_loss:.6f} | Avg Classification Loss: {avg_cls_loss:.6f}', self.log)
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
        
        last_recon = None
        last_x = None

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
                    last_recon = recon_output
                    last_x = x[recon_mask]
                    
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

               
        if epoch % 1 == 0 and last_recon is not None:
            plot_show(last_recon, last_x, epoch, self.mean, self.std)

        # Print learning rate for optimizer_vit
        for i, param_group in enumerate(self.optimizer_vit.param_groups):
            logger.info(f'Valid Epoch {epoch}: Learning Rate for Transformer: {param_group["lr"]:.6f}')
        
        # Print learning rates for optimizer_dec
        for i, param_group in enumerate(self.optimizer_dec.param_groups):
            logger.info(f'Valid Epoch {epoch}: Learning Rate for Decoder: {param_group["lr"]:.6f}')

        log_msg = f'Valid Epoch: {epoch} | Avg Loss: {avg_loss:.6f} | Avg Reconstruction Loss: {avg_recon_loss:.6f} | Avg Classification Loss: {avg_cls_loss:.6f}'
        print_log(log_msg, self.log)

        return avg_loss, avg_recon_loss, avg_cls_loss


    # -----------------------------------
    # Executes the full training workflow
    # -----------------------------------
    def train(self):
        # Set up time tracking
        start_time = time.time()
        epoch_time = AverageMeter()
        
        # Initialize loss tracking
        train_total_losses, train_recon_losses, train_cls_losses = [], [], []
        val_total_losses, val_recon_losses, val_cls_losses = [], [], []

        # Main training loop
        for epoch in range(1, self.epochs + 1):
            # Estimate time remaining
            need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (self.epochs - epoch))
            need_time = f'[Need: {need_hour:02d}:{need_mins:02d}:{need_secs:02d}]'
            print_log(f' {epoch:3d}/{self.epochs:3d} ----- [{time.strftime("%Y-%m-%d %H:%M:%S")}] {need_time}', self.log)
            
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
            if self.early_stop(val_cls_loss, val_recon_loss, self.model):
                print_log("Training stopped early due to lack of improvement.", self.log)
                break

            # Step the scheduler with the validation loss
            self.scheduler.step(val_loss)
            
            # Update epoch time and reset start_time for the next epoch
            epoch_time.update(time.time() - start_time)
            start_time = time.time()

        # Plot training and validation losses.
        plot_loss(train_total_losses, val_total_losses, train_cls_losses, val_cls_losses, train_recon_losses, val_recon_losses)