import torch
from torch import nn
import torch.nn.functional as F
from nnunetv2.training.loss.dice_focal_loss import FocalLoss
from nnunetv2.training.loss.dice_focal_loss import DiceLoss
from nnunetv2.training.loss.signed_distance_loss import SDFLoss


class DiceFocalBCESDFLoss(nn.Module):
    """Dice + Focal + BCE + SDF (signed distance function) loss with optional warmup."""

    def __init__(
        self,
        alpha=0.5,
        gamma=4,
        sdf_weight=0.1,
        dice_weight=0.6,
        bce_weight=0.2,
        focal_weight=0.2,
        warmup_epochs=20,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.dice = DiceLoss()
        self.register_buffer("bce_pos_weight", torch.tensor(15.0))
        self.sdf = SDFLoss()

        self.sdf_weight = sdf_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def forward(self, inputs, targets):
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]
        if isinstance(targets, (list, tuple)):
            targets = targets[0]

        if targets.ndim == inputs.ndim:
            if targets.shape[1] == 1:
                targets = targets.squeeze(1)
            else:
                raise ValueError(
                    f"Unexpected target shape: {targets.shape}, expected channel dim 1 if same ndim as inputs"
                )
        elif targets.ndim != inputs.ndim - 1:
            raise ValueError(
                f"Unexpected target shape: {targets.shape}, expected same as inputs or one channel less"
            )

        loss_dice = self.dice(inputs, targets)
        loss_focal = self.focal(inputs, targets)

        inputs_fg = inputs[:, 1, ...]
        targets_float = targets.float()
        loss_bce = F.binary_cross_entropy_with_logits(
            inputs_fg, targets_float, pos_weight=self.bce_pos_weight
        )

        base_loss = (
            self.dice_weight * loss_dice
            + self.bce_weight * loss_bce
            + self.focal_weight * loss_focal
        )

        loss_sdf = self.sdf(inputs, targets)

        if self.warmup_epochs > 0:
            warmup_factor = min(1.0, self.current_epoch / self.warmup_epochs)
            effective_sdf_weight = self.sdf_weight * warmup_factor
        else:
            effective_sdf_weight = self.sdf_weight

        with torch.no_grad():
            base_loss_val = (
                base_loss.item()
                if isinstance(base_loss, torch.Tensor)
                else float(base_loss)
            )
            max_sdf = (
                min(base_loss_val * 1.5, 0.5) if base_loss_val > 0 else 0.5
            )

        loss_sdf_clipped = torch.clamp(loss_sdf, max=max_sdf)
        total_loss = base_loss + (effective_sdf_weight * loss_sdf_clipped)
        return total_loss

    def set_epoch(self, epoch):
        self.current_epoch = epoch
