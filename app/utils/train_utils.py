import os
import math
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from app.config import AMP
from app.utils.utils import LoggerConfig
from app.utils import print_log, plot_show, plot_loss, AverageMeter, convert_secs2time

logger = LoggerConfig().get_logger(__name__)

# --------------
# Early Stopping 
# --------------
class EarlyStop:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.verbose = True
        self.save_name = "checkpoint.pt"
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.log = open(self.log_path, 'w')

    def __call__(self, class_loss, recon_loss, model):
        overall_loss = class_loss + recon_loss
        
        # Check if the new loss is significantly better
        if overall_loss < self.val_loss_min - self.delta:  
            if self.verbose:
                logger.info(f"Overall loss decreased ({self.val_loss_min:.6f} --> {overall_loss:.6f}). Saving model...", self.log)
            self.save_checkpoint(overall_loss, model)
            self.val_loss_min = overall_loss 
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}", self.log)

        if self.counter >= self.patience:
            self.early_stop = True
            logger.warning("Early stopping triggered", self.log)
            return True

        return False

    def save_checkpoint(self, model):
        """ Saves model when validation loss decrease. """
        torch.save(model.state_dict(), self.save_name)
        logger.info("Model saved", self.log)
        
        
# -----------------------------------------
# Class for training the custom ViT-Decoder
# -----------------------------------------
class ModelTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = "app/logs/"
        self.init_train_components()
        # Initialize early stopping and AMP scaler
        
        self.early_stop = EarlyStop(patience=5, delta=0.001)
        self.scaler = torch.amp.GradScaler(enabled=AMP) 
        
        # Setup directories and logs
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_path = os.path.join(self.save_dir, 'training_log.txt')
        self.log = open(self.log_path, 'w')
    
    def __del__(self):
        """Ensures the log file is closed when the trainer instance is deleted"""
        if self.log:
            self.log.close()
        
    # ---------------------------------------
    # Optimizers and scheduler initialization 
    # ---------------------------------------
    def init_train_components(self):
        # Vit's optimizer Initialization
        self.optimizer_vit = torch.optim.Adam(
            self.model.vit_encoder.parameters(),
            lr=self.args.lr_vit,
            betas=(self.args.beta1, self.args.beta2),
            weight_decay=self.args.wd_vit
        )
        # Decoder's optimizer Initialization
        self.optimizer_dec = torch.optim.Adam(
            self.model.decoder.parameters(),
            lr=self.args.lr_dec,
            betas=(self.args.beta1, self.args.beta2),
            weight_decay=self.args.wd_dec
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
        MSE = nn.MSELoss()
        CrossEntropy = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_reconstruction_loss = 0.0
        total_classification_loss = 0.0
        num_batches = 0
        num_label_samples = 0  # total number of images used for classification
        num_recon_samples = 0  # total number of images used for reconstruction

        for (x, y, _) in tqdm(self.train_loader):
            x = x.to(self.args.device)
            y = y.to(self.args.device)

            self.optimizer_vit.zero_grad()
            self.optimizer_dec.zero_grad()
            
            # Masks for processing logic
            base_mask = y == -1
            normal_mask = y == 0
            anomaly_mask = y == 1

            # Initialize loss components
            reconstruction_loss = torch.tensor(0.0, device=self.args.device)
            classification_loss = torch.tensor(0.0, device=self.args.device)

            if self.args.amp:
                with torch.amp.autocast(device_type=self.args.device):
                    # Unlabeled data for reconstruction only
                    if base_mask.any():
                        _, reconstructed_output = self.model(x[base_mask], return_logits=False, return_reconstruction=True)
                        reconstruction_loss = MSE(reconstructed_output, x[base_mask])
                        num_recon_samples += base_mask.sum().item()
            
                    # Normal data for both classification and reconstruction
                    if normal_mask.any():
                        cls_token_logits, reconstructed_output = self.model(x[normal_mask], return_logits=True, return_reconstruction=True)
                        classification_loss = CrossEntropy(cls_token_logits, y[normal_mask])
                        reconstruction_loss += MSE(reconstructed_output, x[normal_mask])
                        num_label_samples += normal_mask.sum().item()
                        num_recon_samples += normal_mask.sum().item()
            
                    # Anomalous data for classification only
                    if anomaly_mask.any():
                        cls_token_logits, _ = self.model(x[anomaly_mask], return_logits=True, return_reconstruction=False)
                        classification_loss += CrossEntropy(cls_token_logits, y[anomaly_mask])
                        num_label_samples += anomaly_mask.sum().item()
                
                # Compute the total loss
                loss = classification_loss + reconstruction_loss
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer_vit)
                self.scaler.step(self.optimizer_dec)
                self.scaler.update()

            else:
                # Compute losses without AMP
                if base_mask.any():
                    _, reconstructed_output = self.model(x[base_mask], return_logits=False, return_reconstruction=True)
                    reconstruction_loss = MSE(reconstructed_output, x[base_mask])
                    num_recon_samples += base_mask.sum().item()
        
                if normal_mask.any():
                    cls_token_logits, reconstructed_output = self.model(x[normal_mask], return_logits=True, return_reconstruction=True)
                    classification_loss = CrossEntropy(cls_token_logits, y[normal_mask])
                    reconstruction_loss += MSE(reconstructed_output, x[normal_mask])
                    num_label_samples += normal_mask.sum().item()
                    num_recon_samples += normal_mask.sum().item()
        
                if anomaly_mask.any():
                    cls_token_logits, _ = self.model(x[anomaly_mask], return_logits=True, return_reconstruction=False)
                    classification_loss += CrossEntropy(cls_token_logits, y[anomaly_mask])
                    num_label_samples += anomaly_mask.sum().item()
                    
                loss = classification_loss + reconstruction_loss
                loss.backward()
                self.optimizer_vit.step()
                self.optimizer_dec.step()
            

            total_loss += loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_classification_loss += classification_loss.item()

            num_batches += 1  # Increment for each batch processed

        estimated_label_batches = math.ceil(num_label_samples / self.args.batch_size)
        estimated_recon_batches = math.ceil(num_recon_samples / self.args.batch_size)
        avg_classification_loss = total_classification_loss / estimated_label_batches if estimated_label_batches > 0 else 0
        avg_reconstruction_loss = total_reconstruction_loss / estimated_recon_batches if estimated_recon_batches > 0 else 0
        avg_loss = avg_classification_loss + avg_reconstruction_loss
        
        print_log(f'Train Epoch: {epoch} | Avg Loss: {avg_loss:.6f} | Avg Reconstruction Loss: {avg_reconstruction_loss:.6f} | Avg Classification Loss: {avg_classification_loss:.6f}', self.log)
        return avg_loss, avg_reconstruction_loss, avg_classification_loss


    # ----------------------------------------
    # Validation operations for a single epoch
    # ----------------------------------------
    def val_epoch(self, epoch):
        # Switch the model to evaluation mode
        self.model.eval()
        MSE = nn.MSELoss()
        CrossEntropy = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_classification_loss = 0.0
        total_reconstruction_loss = 0.0
        num_batches = 0
        num_label_samples = 0  # total number of images used for classification
        num_recon_samples = 0  # total number of images used for reconstruction

        for (x, y, _) in tqdm(self.val_loader):
            x = x.to(self.args.device)
            y = y.to(self.args.device)

            # Masks for processing logic
            base_mask = y == -1
            normal_mask = y == 0
            anomaly_mask = y == 1

            with torch.no_grad():
                # Initialize loss components
                reconstruction_loss = torch.tensor(0.0, device=self.args.device)
                classification_loss = torch.tensor(0.0, device=self.args.device)

                # Unlabeled data for reconstruction only
                if (y == -1).any():
                    _, reconstructed_output = self.model(x[base_mask], return_logits=False, return_reconstruction=True)
                    reconstruction_loss = MSE(reconstructed_output, x[base_mask])
                    num_recon_samples += (base_mask).sum().item()

                # Normal data for both classification and reconstruction
                if (y == 0).any():
                    cls_token_logits, reconstructed_output = self.model(x[normal_mask], return_logits=True, return_reconstruction=True)
                    classification_loss = CrossEntropy(cls_token_logits, y[normal_mask])
                    reconstruction_loss += MSE(reconstructed_output, x[normal_mask])
                    num_label_samples += (normal_mask).sum().item()
                    num_recon_samples += (normal_mask).sum().item()

                # Anomalous data for classification only
                if (y == 1).any():
                    cls_token_logits, _ = self.model(x[anomaly_mask], return_logits=True, return_reconstruction=False)
                    classification_loss += CrossEntropy(cls_token_logits, y[anomaly_mask])
                    num_label_samples += (anomaly_mask).sum().item()

                # Sum the losses
                loss = classification_loss + reconstruction_loss

                total_loss += loss.item()
                total_reconstruction_loss += reconstruction_loss.item()
                total_classification_loss += classification_loss.item()

                num_batches += 1  # Increment for each batch processed

        estimated_label_batches = math.ceil(num_label_samples / self.args.batch_size)
        estimated_recon_batches = math.ceil(num_recon_samples / self.args.batch_size)
        avg_classification_loss = total_classification_loss / estimated_label_batches if estimated_label_batches > 0 else 0
        avg_reconstruction_loss = total_reconstruction_loss / estimated_recon_batches if estimated_recon_batches > 0 else 0
        avg_loss = avg_classification_loss + avg_reconstruction_loss

        # Optionally, display reconstructed images every 5 epochs
        if epoch % 1 == 0:
            plot_show(reconstructed_output, x, epoch, self.args.mean, self.args.std)
            
        # Print learning rate
        for i, param_group in enumerate(self.optimizer_vit.param_groups):
            print(f'Valid Epoch {epoch}: Learning Rate for Transformer: {param_group["lr"]:.6f}')
        
        # Print learning rates for optimizer_dec
        for i, param_group in enumerate(self.optimizer_dec.param_groups):
            print(f'Valid Epoch {epoch}: Learning Rate for Decoder: {param_group["lr"]:.6f}')


        log_msg = f'Valid Epoch: {epoch} | Avg Loss: {avg_loss:.6f} | Avg Reconstruction Loss: {avg_reconstruction_loss:.6f}'
        if num_label_samples > 0:
            log_msg += f' | Avg Classification Loss: {avg_classification_loss:.6f}'
        print_log(log_msg, self.log)

        return avg_loss, avg_reconstruction_loss, avg_classification_loss


    # -----------------------------------
    # Executes the full training workflow
    # -----------------------------------
    def execute_training(self):
        # Set up time tracking
        start_time = time.time()
        epoch_time = AverageMeter()
        
        # Initialize loss tracking
        train_total_losses, train_reconstruction_losses, train_classification_losses = [], [], []
        val_total_losses, val_reconstruction_losses, val_classification_losses = [], [], []

        # Main training loop
        for epoch in range(1, self.args.epochs + 1):
            # Estimate time remaining
            need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (self.args.epochs - epoch))
            need_time = f'[Need: {need_hour:02d}:{need_mins:02d}:{need_secs:02d}]'
            print_log(f' {epoch:3d}/{self.args.epochs:3d} ----- [{time.strftime("%Y-%m-%d %H:%M:%S")}] {need_time}', self.log)
            
            # Receive detailed losses from training
            train_loss, train_recon_loss, train_class_loss = self.train_epoch(epoch)
            train_total_losses.append(train_loss)
            train_reconstruction_losses.append(train_recon_loss)
            train_classification_losses.append(train_class_loss)

            # Receive detailed losses from validation
            val_loss, val_recon_loss, val_class_loss = self.val_epoch(epoch)
            val_total_losses.append(val_loss)
            val_reconstruction_losses.append(val_recon_loss)
            val_classification_losses.append(val_class_loss)

            # Early stopping checks against total validation loss
            if self.early_stop(val_class_loss, val_recon_loss, self.model, self.optimizer_vit, self.optimizer_dec, self.log):
                print_log("Training stopped early due to lack of improvement.", self.log)
                break

            # Step the scheduler with the validation loss
            self.scheduler.step(val_loss)
            
            # Update epoch time and reset start_time for the next epoch
            epoch_time.update(time.time() - start_time)
            start_time = time.time()

        # Plot training and validation losses.
        plot_loss(train_total_losses, val_total_losses, train_classification_losses, val_classification_losses, train_reconstruction_losses, val_reconstruction_losses)