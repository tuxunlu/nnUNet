import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.helpers import softmax_helper_dim1
import torch.nn.functional as F
import numpy as np
import inspect
from torch._dynamo import OptimizedModule

from nnunetv2.training.loss.dice_loss import DiceLoss
from nnunetv2.training.loss.focal_loss import FocalLoss
from nnunetv2.training.loss.pixel_contrastive_loss import PixelContrastiveLoss
from nnunetv2.training.loss.dice_focal_bce_contrastive_loss import DiceFocalBCEContrastiveLoss
from nnunetv2.utilities.helpers import dummy_context

# ==========================================
# TRAINER CLASS
# ==========================================

class nnUNetTrainerDiceFocalBCEContrastive(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.device = device
        self.latest_features = None 
        self.num_epochs = 100
        
        # Fix 1: Increase foreground oversampling to handle severe class imbalance
        self.oversample_foreground_percent = 0.6  # Increased from 0.33 to 0.6 for better foreground learning
        
        # Fix 2: Increase weight decay for better regularization (further increased to reduce overfitting)
        self.weight_decay = 2e-4  # Increased from 1e-4 to 2e-4 for stronger regularization
        
        # Fix 3: Reduce initial learning rate for more stable training
        self.initial_lr = 3e-3  # Further reduced from 5e-3 to 3e-3 to reduce overfitting
        
        # Fix 4: Early stopping parameters (reduced patience to stop earlier when validation plateaus)
        self.early_stopping_patience = 10  # Reduced from 15 to 10 to stop earlier when validation plateaus
        self.best_val_dice = None
        self.epochs_without_improvement = 0
        self.should_stop_training = False
        
    def _build_loss(self):
        loss = DiceFocalBCEContrastiveLoss(
            alpha=0.5,   # Increased from 0.35 to 0.5 for better hard example focus
            gamma=4,     # Increased from 3 to 4 for stronger focus on hard negatives
            contrastive_weight=0.05,  # Reduced from 0.1 to 0.05 for more conservative weighting
            warmup_epochs=20  # Warmup over 20 epochs
        )
        return loss

    def initialize(self):
        super().initialize()
        # Get the actual network module (handle DDP and compiled networks)
        mod = self.network
        if hasattr(mod, 'module'):  # DDP wrapped
            mod = mod.module
        if isinstance(mod, OptimizedModule):  # Compiled network
            mod = mod._orig_mod
        
        # Register hook on decoder to capture features for contrastive loss
        if hasattr(mod, 'decoder'):
            def hook_fn(module, input, output):
                self.latest_features = output
            mod.decoder.register_forward_hook(hook_fn)
        else:
            # Fallback: try to hook the last conv layer before output
            # This is a workaround if decoder doesn't exist
            self.print_to_log_file("Warning: network.decoder not found, contrastive loss may not work properly")
            self.latest_features = None

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)
            
            # Extract features for contrastive loss
            features = self.latest_features
            if features is not None:
                if isinstance(features, (list, tuple)):
                    features = features[0]
                # Ensure features have the right shape [B, C, H, W]
                if features is not None and features.dim() == 4:
                    # Features are already in the right format
                    pass
                elif features is not None:
                    # Reshape if needed
                    self.print_to_log_file(f"Warning: Unexpected feature shape: {features.shape}")

            if self.enable_deep_supervision:
                l = self.loss(output[0], target[0], feats=features)
                
                num_scales = len(output)
                weights = np.array([1 / (2 ** i) for i in range(num_scales)])
                
                if num_scales > 1:
                    weights[-1] = 0
                weights = weights / weights.sum()

                for i in range(1, num_scales):
                    if weights[i] != 0:
                        # Only use contrastive loss on the highest resolution output
                        l += weights[i] * self.loss(output[i], target[i], feats=None)
            else:
                l = self.loss(output, target, feats=features)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        """Override to use probability threshold instead of argmax for better alignment with actual validation dice."""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # Use probability threshold instead of argmax for imbalanced data
            # argmax is too conservative - always picks background for imbalanced data
            # This makes pseudo dice more aligned with actual validation dice
            probs = torch.softmax(output, dim=1)
            # Use lower threshold (0.3) to reduce false negatives and align with inference
            fg_prob = probs[:, 1, ...]  # Foreground probability [B, H, W]
            predicted_seg = (fg_prob > 0.3).long()[:, None]  # Threshold-based prediction
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, predicted_seg, 1)
            # Background is where foreground is not predicted
            predicted_segmentation_onehot[:, 0, ...] = 1 - predicted_segmentation_onehot[:, 1, ...]
            del predicted_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
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
# SIGNATURE PATCH
# ==========================================
nnUNetTrainerDiceFocalBCEContrastive.__init__.__signature__ = inspect.signature(nnUNetTrainer.__init__)