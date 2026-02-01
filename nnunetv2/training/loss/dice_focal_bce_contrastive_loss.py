import torch
from torch import nn
from nnunetv2.training.loss.dice_focal_loss import FocalLoss
from nnunetv2.training.loss.dice_focal_loss import DiceLoss
from nnunetv2.training.loss.pixel_contrastive_loss import PixelContrastiveLoss
import torch.nn.functional as F


class DiceFocalBCEContrastiveLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4, temperature=0.07, contrastive_weight=0.1, warmup_epochs=10):
        super().__init__()
        # Increased alpha and gamma for better handling of hard examples and class imbalance
        self.focal = FocalLoss(alpha, gamma)
        self.dice = DiceLoss()
        # Use pos_weight in BCE to handle class imbalance (penalize false negatives more)
        # pos_weight > 1 increases recall (reduces false negatives)
        # For 1:12,450 imbalance, use pos_weight around 10-20
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(15.0))  # Penalize false negatives more
        self.contrastive = PixelContrastiveLoss(temperature=temperature)
        self.contrastive_weight = contrastive_weight
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def forward(self, inputs, targets, feats=None):
        # --- FIX START: Handle Deep Supervision List Input ---
        # If validation step passes a list (Deep Supervision), 
        # we take the first element (highest resolution) to compute the metric.
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]
            if isinstance(targets, (list, tuple)):
                targets = targets[0]
        # --- FIX END ---

        # Normalize target format: ensure targets are [B, H, W] or [B, 1, H, W]
        # Inputs should be [B, num_classes, H, W] (typically [B, 2, H, W] for binary)
        if targets.ndim == inputs.ndim:
            # Targets have same ndim as inputs, check if channel dim is 1
            if targets.shape[1] == 1:
                targets = targets.squeeze(1)
        elif targets.ndim == inputs.ndim - 1:
            # Targets already in [B, H, W] format, keep as is
            pass
        else:
            raise ValueError(f"Unexpected target shape: {targets.shape}, expected same as inputs or one channel less")
        
        # Compute losses - all loss functions expect targets in [B, H, W] format
        loss_dice = self.dice(inputs, targets)
        
        inputs_fg = inputs[:, 1, ...]  # [B, H, W] foreground logits
        targets_float = targets.float()  # [B, H, W] float targets
        
        # Move pos_weight to same device as inputs
        if hasattr(self.bce, 'pos_weight') and self.bce.pos_weight is not None:
            self.bce.pos_weight = self.bce.pos_weight.to(inputs_fg.device)
        
        loss_bce = self.bce(inputs_fg, targets_float)
        loss_focal = self.focal(inputs, targets)
        
        # Increased dice weight to prioritize recall (reduce false negatives)
        # Dice loss is better at handling class imbalance than BCE
        # 0.6 dice + 0.2 bce + 0.2 focal (dice weight increased from 0.5 to 0.6)
        base_loss = 0.6 * loss_dice + 0.2 * loss_bce + 0.2 * loss_focal

        if feats is None or self.contrastive_weight <= 0:
            return base_loss

        # Normalize features before contrastive loss to ensure stable training
        # This helps prevent the contrastive loss from dominating early in training
        if feats.dim() == 4:
            # Normalize features channel-wise: [B, C, H, W] -> normalize across C dimension
            feats_normalized = F.normalize(feats, p=2, dim=1)
        else:
            feats_normalized = feats
        
        # Convert targets to integer labels for contrastive loss
        # Targets are in [B, H, W] format as float (0.0 or 1.0), convert to int (0 or 1)
        targets_int = targets.long()
        
        # Compute contrastive loss
        loss_contrast = self.contrastive(feats_normalized, targets_int)
        
        # Warmup schedule: gradually increase contrastive weight during early training
        # This prevents the contrastive loss from interfering with initial segmentation learning
        if self.warmup_epochs > 0:
            warmup_factor = min(1.0, self.current_epoch / self.warmup_epochs)
            effective_weight = self.contrastive_weight * warmup_factor
        else:
            effective_weight = self.contrastive_weight
        
        # Clip contrastive loss to prevent it from being too large relative to base loss
        # This helps stabilize training, especially early on when features are random
        # Detach base_loss for comparison to avoid gradient issues
        with torch.no_grad():
            base_loss_val = base_loss.item() if isinstance(base_loss, torch.Tensor) else float(base_loss)
            # Cap contrastive loss at 2x the base loss value, or 1.0, whichever is smaller
            # This prevents the contrastive loss from dominating
            max_contrastive = min(base_loss_val * 2.0, 1.0) if base_loss_val > 0 else 1.0
        
        loss_contrast_clipped = torch.clamp(loss_contrast, max=max_contrastive)
        
        return base_loss + (effective_weight * loss_contrast_clipped)
    
    def set_epoch(self, epoch):
        """Call this method to update the current epoch for warmup scheduling."""
        self.current_epoch = epoch
