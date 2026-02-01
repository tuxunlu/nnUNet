"""
Signed Distance Function (SDF) loss for segmentation.
Uses the ground-truth SDF to weight the loss so that errors near the boundary
(small |phi|) are penalized more, encouraging sharper boundaries.
"""
import torch
from torch import nn
from nnunetv2.training.loss.boundary_loss import compute_fast_distance_map


class SDFLoss(nn.Module):
    """
    Boundary-weighted loss using the signed distance function (SDF) of the target.
    Loss = mean( weight(x) * (p(x) - target(x))^2 ) with weight(x) = 1 / (1 + |phi(x)|),
    so that pixels near the boundary (small |phi|) get higher weight.
    """

    def __init__(self, max_iterations=20, eps=1e-6):
        super(SDFLoss, self).__init__()
        self.max_iterations = max_iterations
        self.eps = eps

    def forward(self, inputs, targets):
        """
        inputs: Logits (B, C, Spatial...) - 2D [B, C, H, W] or 3D [B, C, D, H, W]
        targets: Ground truth - [B, 1, Spatial...] or [B, Spatial...]
        """
        probs = torch.softmax(inputs, dim=1)

        # Normalize target format: ensure targets are [B, Spatial...]
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

        targets_unsqueezed = targets.unsqueeze(1).float()  # [B, 1, Spatial...]

        # Samples with no foreground (normal scans): give full weight so model learns to predict 0.
        spatial_dims = tuple(range(2, targets_unsqueezed.dim()))
        no_fg = (targets_unsqueezed.sum(dim=spatial_dims, keepdim=True) == 0)  # [B, 1, 1, ...]

        with torch.no_grad():
            phi = compute_fast_distance_map(targets_unsqueezed, self.max_iterations)
        abs_phi = phi.abs().squeeze(1)  # [B, Spatial...]
        weight = 1.0 / (1.0 + abs_phi + self.eps)
        # For no-foreground samples, weight would be ~0 (phi huge); set weight=1 so loss = mean((prob_fg-0)^2)
        no_fg_spatial = no_fg.squeeze(1).expand_as(weight)
        weight = torch.where(no_fg_spatial, torch.ones_like(weight, device=weight.device, dtype=weight.dtype), weight)

        prob_fg = probs[:, 1, ...]
        target_float = targets.float()
        loss = (weight * (prob_fg - target_float).pow(2)).mean()
        return loss
