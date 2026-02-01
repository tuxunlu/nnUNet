import numpy as np
import torch
import inspect
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# Import the loss we defined in the other file
# Make sure the python path is correct relative to where you run it
from nnunetv2.training.loss.dice_focal_bce_boundary_loss import DiceFocalBCEBoundaryLoss

class nnUNetTrainerDiceFocalBCEBoundary(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        # Call parent constructor safely
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        
        self.num_epochs = 100  # Match other trainers (default is 1000, but we use 100 for consistency)
        
        # Fix 1: Increase foreground oversampling to handle severe class imbalance
        self.oversample_foreground_percent = 0.8  # Increased from 0.33 to 0.6 for better foreground learning
        
        # Fix 2: Increase weight decay for better regularization (further increased to reduce overfitting)
        self.weight_decay = 2e-4  # Increased from 1e-4 to 2e-4 for stronger regularization
        
        # Fix 3: Reduce initial learning rate for more stable training
        self.initial_lr = 1e-3  # Further reduced from 5e-3 to 3e-3 to reduce overfitting
        
        # Fix 4: Early stopping parameters (reduced patience to stop earlier when validation plateaus)
        self.early_stopping_patience = 10  # Reduced from 15 to 10 to stop earlier when validation plateaus
        self.best_val_dice = None
        self.epochs_without_improvement = 0
        self.should_stop_training = False
    
    def _build_loss(self):
        self.print_to_log_file("Using Dice + BCE + Focal + Boundary Loss for training.")

        # Define the loss function with weights optimized to reduce false negatives
        # Increased dice weight (0.6) to prioritize recall, reduced bce/focal (0.2 each)
        # Boundary loss: 0.1 with warmup over 20 epochs to prevent early interference
        loss = DiceFocalBCEBoundaryLoss(
            boundary_weight=0.5,  # Reduced from 1.0 to 0.1 for more conservative weighting
            dice_weight=0.5,      # Increased from 0.5 to 0.6 to prioritize recall (reduce false negatives)
            bce_weight=0.5,       # Reduced from 0.25 to 0.2 (BCE with pos_weight handles imbalance)
            focal_weight=0.5,     # Reduced from 0.25 to 0.2 (focal loss also helps with imbalance)
            alpha=0.5,            # Increased from 0.35 to 0.5 for better hard example focus
            gamma=2,              # Increased from 3 to 4 for stronger focus on hard negatives
            warmup_epochs=20      # Warmup over 20 epochs
        )

        # Apply Deep Supervision if enabled
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            
            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

            # Avoid errors with Distributed Data Parallel (DDP)
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6  # Prevent unused parameter issues
            else:
                weights[-1] = 0  # Last output has no weight

            # Normalize weights
            weights = weights / weights.sum()

            # Wrap with Deep Supervision
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
    
    def on_epoch_start(self):
        """Update loss warmup schedule at the start of each epoch."""
        super().on_epoch_start()
        if hasattr(self.loss, 'set_epoch'):
            self.loss.set_epoch(self.current_epoch)
        
        # Check if we should stop training early
        if self.should_stop_training:
            self.print_to_log_file(f"Early stopping triggered after {self.current_epoch} epochs. "
                                 f"Best validation Dice: {self.best_val_dice:.4f}")
    
    def on_epoch_end(self):
        """Override to add early stopping logic."""
        # Get current validation dice from logger
        if hasattr(self.logger, 'my_fantastic_logging') and 'ema_fg_dice' in self.logger.my_fantastic_logging:
            if len(self.logger.my_fantastic_logging['ema_fg_dice']) > 0:
                current_val_dice = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
                
                # Initialize best_val_dice if needed
                if self.best_val_dice is None:
                    self.best_val_dice = current_val_dice
                    self.epochs_without_improvement = 0
                else:
                    # Check for improvement (use small epsilon to avoid floating point issues)
                    if current_val_dice > self.best_val_dice + 1e-6:
                        improvement = current_val_dice - self.best_val_dice
                        self.best_val_dice = current_val_dice
                        self.epochs_without_improvement = 0
                        self.print_to_log_file(f"Validation Dice improved by {improvement:.4f} to {self.best_val_dice:.4f}")
                    else:
                        self.epochs_without_improvement += 1
                        self.print_to_log_file(f"No improvement for {self.epochs_without_improvement}/{self.early_stopping_patience} epochs. "
                                             f"Best: {self.best_val_dice:.4f}, Current: {current_val_dice:.4f}")
                        
                        # Check if we should stop
                        if self.epochs_without_improvement >= self.early_stopping_patience:
                            self.should_stop_training = True
                            self.print_to_log_file(f"Early stopping: No improvement for {self.early_stopping_patience} epochs. "
                                                 f"Stopping training at epoch {self.current_epoch}")
        
        # Call parent method (this increments current_epoch)
        super().on_epoch_end()
        
        # Stop training if early stopping triggered by setting num_epochs to current_epoch
        if self.should_stop_training:
            self.num_epochs = self.current_epoch
            self.print_to_log_file(f"Training will stop after epoch {self.current_epoch}")

# ==========================================
# SIGNATURE PATCH (Required for your env)
# ==========================================
nnUNetTrainerDiceFocalBCEBoundary.__init__.__signature__ = inspect.signature(nnUNetTrainer.__init__)