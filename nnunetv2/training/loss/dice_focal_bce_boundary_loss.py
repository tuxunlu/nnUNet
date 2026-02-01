import torch
from torch import nn
import torch.nn.functional as F
from nnunetv2.training.loss.dice_focal_loss import FocalLoss
from nnunetv2.training.loss.dice_focal_loss import DiceLoss
from nnunetv2.training.loss.boundary_loss import BoundaryLoss


class DiceFocalBCEBoundaryLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4, boundary_weight=0.1, dice_weight=0.6, bce_weight=0.2, focal_weight=0.2, warmup_epochs=20):
        super().__init__()
        # Increased alpha and gamma for better handling of hard examples and class imbalance
        self.focal = FocalLoss(alpha, gamma)
        self.dice = DiceLoss()
        # Use pos_weight in BCE to handle class imbalance (penalize false negatives more)
        # pos_weight > 1 increases recall (reduces false negatives)
        # For 1:12,450 imbalance, use pos_weight around 10-20
        # Store pos_weight as a buffer so it moves with the model to the correct device
        self.register_buffer('bce_pos_weight', torch.tensor(15.0))
        self.boundary = BoundaryLoss()
        
        self.boundary_weight = boundary_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def forward(self, inputs, targets):
        # Handle Deep Supervision List Input
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]
        if isinstance(targets, (list, tuple)):
            targets = targets[0]
        
        # Normalize target format: ensure targets are [B, H, W] or [B, 1, H, W]
        # Inputs should be [B, num_classes, H, W] (typically [B, 2, H, W] for binary)
        if targets.ndim == inputs.ndim:
            # Targets have same ndim as inputs, check if channel dim is 1
            if targets.shape[1] == 1:
                targets = targets.squeeze(1)
            else:
                # If targets have same ndim but channel dim != 1, assume it's already in correct format
                # This handles edge cases where targets might be [B, H, W] but stored as [B, 1, H, W]
                raise ValueError(f"Unexpected target shape: {targets.shape}, expected channel dim to be 1 if same ndim as inputs")
        elif targets.ndim == inputs.ndim - 1:
            # Targets already in [B, H, W] format, keep as is
            pass
        else:
            raise ValueError(f"Unexpected target shape: {targets.shape}, expected same as inputs or one channel less")
        
        # 1. Standard Losses (matching DiceFocalBCELoss weights)
        loss_dice = self.dice(inputs, targets)
        loss_focal = self.focal(inputs, targets)
        
        inputs_fg = inputs[:, 1, ...]  # [B, H, W] foreground logits
        targets_float = targets.float()  # [B, H, W] float targets
        
        # Use F.binary_cross_entropy_with_logits directly with pos_weight buffer
        # This avoids modifying module state during forward pass
        loss_bce = F.binary_cross_entropy_with_logits(inputs_fg, targets_float, pos_weight=self.bce_pos_weight)
        
        # Base loss with standard weights
        base_loss = (self.dice_weight * loss_dice) + \
                    (self.bce_weight * loss_bce) + \
                    (self.focal_weight * loss_focal)
        
        # 2. Boundary Loss with warmup and clipping
        loss_boundary = self.boundary(inputs, targets)
        
        # Warmup schedule: gradually increase boundary weight during early training
        # This prevents the boundary loss from interfering with initial segmentation learning
        if self.warmup_epochs > 0:
            warmup_factor = min(1.0, self.current_epoch / self.warmup_epochs)
            effective_boundary_weight = self.boundary_weight * warmup_factor
        else:
            effective_boundary_weight = self.boundary_weight
        
        # Clip boundary loss to prevent it from being too large relative to base loss
        # This helps stabilize training, especially early on
        with torch.no_grad():
            base_loss_val = base_loss.item() if isinstance(base_loss, torch.Tensor) else float(base_loss)
            # Cap boundary loss at 1.5x the base loss value, or 0.5, whichever is smaller
            max_boundary = min(base_loss_val * 1.5, 0.5) if base_loss_val > 0 else 0.5
        
        loss_boundary_clipped = torch.clamp(loss_boundary, max=max_boundary)
        
        total_loss = base_loss + (effective_boundary_weight * loss_boundary_clipped)
        
        return total_loss
    
    def set_epoch(self, epoch):
        """Call this method to update the current epoch for warmup scheduling."""
        self.current_epoch = epoch