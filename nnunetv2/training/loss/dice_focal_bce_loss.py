import torch
from torch import nn
from nnunetv2.training.loss.dice_focal_loss import FocalLoss
from nnunetv2.training.loss.dice_focal_loss import DiceLoss
import torch.nn.functional as F


class DiceFocalBCELoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=3, temperature=0.07):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

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
        
        loss_bce = self.bce(inputs_fg, targets_float)
        loss_focal = self.focal(inputs, targets)
        
        # Use same weights as DiceFocalBCELoss: 0.5 dice + 0.25 bce + 0.25 focal
        base_loss = 0.5 * loss_dice + 0.25 * loss_bce + 0.25 * loss_focal

        if feats is None or self.contrastive_weight <= 0:
            return base_loss

        # Convert targets to integer labels for contrastive loss
        # Targets are in [B, H, W] format as float (0.0 or 1.0), convert to int (0 or 1)
        targets_int = targets.long()
        return base_loss
