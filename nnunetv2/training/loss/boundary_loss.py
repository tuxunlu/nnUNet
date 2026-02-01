import torch
from torch import nn
import torch.nn.functional as F

# ==========================================
# HELPER: GPU-Based Fast Distance Transform
# ==========================================
def compute_fast_distance_map(target: torch.Tensor, max_iterations=20) -> torch.Tensor:
    """
    Approximates the Distance Transform on GPU using iterative erosion (Chessboard distance).
    This is significantly faster than transferring to CPU for Scipy's Euclidean EDT.
    
    Args:
        target: Shape [B, 1, D, H, W] for 3D or [B, 1, H, W] for 2D (Binary 0/1)
        max_iterations: How far to calculate the distance field. 
                        Gradients > 20px away typically don't aid convergence much.
    """
    # Ensure target is float for math ops
    target = target.float()
    
    # Determine if we're dealing with 2D or 3D based on tensor dimensions
    is_3d = target.ndim == 5  # [B, 1, D, H, W]
    
    # 1. Calculate Inside Distance (Distance to nearest Background)
    # We estimate this by iteratively eroding the foreground
    # D_in(p) = Sum(Erosion_i(p))
    
    curr = target.clone()
    dist_in = torch.zeros_like(curr)
    
    # Use appropriate max pooling based on dimensionality
    for _ in range(max_iterations):
        # Erosion = -MaxPool(-X)
        if is_3d:
            curr = -F.max_pool3d(-curr, kernel_size=3, stride=1, padding=1)
        else:
            # 2D case: [B, 1, H, W]
            curr = -F.max_pool2d(-curr, kernel_size=3, stride=1, padding=1)
        
        # If the mask is empty, stop early to save time
        if curr.sum() == 0:
            break
        dist_in += curr

    # 2. Calculate Outside Distance (Distance to nearest Foreground)
    # We estimate this by iteratively eroding the background (inverted target)
    
    curr = 1.0 - target
    dist_out = torch.zeros_like(curr)
    
    for _ in range(max_iterations):
        # Erosion of background
        if is_3d:
            curr = -F.max_pool3d(-curr, kernel_size=3, stride=1, padding=1)
        else:
            # 2D case: [B, 1, H, W]
            curr = -F.max_pool2d(-curr, kernel_size=3, stride=1, padding=1)
        
        if curr.sum() == 0:
            break
        dist_out += curr

    # Combine
    # Inside pixels: - (dist_in - 1)
    # Outside pixels: + dist_out
    # Note: We subtract 1 from dist_in because the boundary itself is usually dist 1
    phi = dist_out * (1 - target) - (dist_in - 1) * target
    
    return phi

# ==========================================
# LOSS CLASSES
# ==========================================

class BoundaryLoss(nn.Module):
    def __init__(self, max_iterations=20):
        super(BoundaryLoss, self).__init__()
        self.max_iterations = max_iterations

    def forward(self, inputs, targets):
        """
        inputs: Logits (B, C, Spatial...) - can be 2D [B, C, H, W] or 3D [B, C, D, H, W]
        targets: Ground Truth - can be [B, 1, Spatial...] or [B, Spatial...]
        """
        probs = torch.softmax(inputs, dim=1)
        
        # Normalize target format: ensure targets are [B, Spatial...]
        # Inputs: [B, C, H, W] (2D) or [B, C, D, H, W] (3D)
        # Targets should be: [B, H, W] (2D) or [B, D, H, W] (3D) after normalization
        if targets.ndim == inputs.ndim:
            # Targets have same ndim as inputs, check if channel dim is 1
            if targets.shape[1] == 1:
                targets = targets.squeeze(1)  # [B, Spatial...]
            else:
                raise ValueError(f"Unexpected target shape: {targets.shape}, expected channel dim to be 1 if same ndim as inputs")
        elif targets.ndim == inputs.ndim - 1:
            # Targets already in [B, Spatial...] format, keep as is
            pass
        else:
            raise ValueError(f"Unexpected target shape: {targets.shape}, expected same as inputs or one channel less")
        
        # Ensure targets are in [B, 1, Spatial...] format for distance map calculation
        # This handles both 2D [B, H, W] -> [B, 1, H, W] and 3D [B, D, H, W] -> [B, 1, D, H, W]
        targets_unsqueezed = targets.unsqueeze(1).float()  # [B, 1, Spatial...]

        # Samples with no foreground (normal scans): teach model to predict all background.
        # For these, use phi=1 so loss = mean(prob_fg) -> penalize any foreground prediction.
        spatial_dims = tuple(range(2, targets_unsqueezed.dim()))
        no_fg = (targets_unsqueezed.sum(dim=spatial_dims, keepdim=True) == 0)  # [B, 1, 1, ...]

        # Generate Distance Map on GPU
        with torch.no_grad():
            dist_map_fg = compute_fast_distance_map(targets_unsqueezed, self.max_iterations)
            # Where no foreground, phi can be huge; replace with 1 so loss = mean(prob_fg)
            dist_map_fg = torch.where(
                no_fg.expand_as(dist_map_fg),
                torch.ones_like(dist_map_fg, device=dist_map_fg.device, dtype=dist_map_fg.dtype),
                dist_map_fg,
            )

        # Calculate Loss: Sum(Prob_FG * Dist_Map_FG)
        probs_fg = probs[:, 1, ...]
        dist_map_fg = dist_map_fg.squeeze(1)
        loss = (probs_fg * dist_map_fg).mean()
        return loss