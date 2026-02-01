import torch
from torch import nn
import torch.nn.functional as F


class ImprovedPixelwiseBoundaryContrastiveLoss(nn.Module):
    """
    Improved pixel-wise boundary contrastive loss with:
    1. Hard negative mining (focus on difficult negatives)
    2. Adaptive temperature scheduling (can be updated per epoch)
    3. Multi-scale support (handles list of features)
    4. Better sampling strategies for small lesions
    
    Key improvements over base version:
    - Hard negative mining: samples negatives that are most similar to positives
    - Temperature scheduling: can decrease temperature over training
    - Multi-scale: applies loss at multiple decoder scales
    - Distance-aware sampling: ensures diverse positive pairs
    """

    def __init__(
        self,
        temperature: float = 0.1,
        temperature_min: float = 0.05,
        temperature_schedule: str = "cosine",  # "cosine", "linear", "constant"
        max_pos_per_image: int = 2048,
        neg_pos_ratio: float = 5.0,
        hard_neg_ratio: float = 0.5,  # Fraction of negatives that are hard negatives
        dilation_kernel_size: int = 11,
        use_hard_negative_mining: bool = True,
        min_pos_distance: int = 2,  # Minimum distance between positive samples (for diversity)
        eps: float = 1e-8,
    ) -> None:
        """
        Args:
            temperature: Initial temperature for the contrastive softmax.
            temperature_min: Minimum temperature (for scheduling).
            temperature_schedule: How to schedule temperature ("cosine", "linear", "constant").
            max_pos_per_image: Maximum number of foreground pixels sampled per image.
            neg_pos_ratio: Maximum ratio of negatives to positives.
            hard_neg_ratio: Fraction of negatives that should be hard negatives (most similar to positives).
            dilation_kernel_size: Kernel size for dilation to create rim band.
            use_hard_negative_mining: Whether to use hard negative mining.
            min_pos_distance: Minimum spatial distance between positive samples (pixels).
            eps: Small constant to avoid division / log of zero.
        """
        super().__init__()

        if dilation_kernel_size < 3 or dilation_kernel_size % 2 == 0:
            raise ValueError(
                "dilation_kernel_size must be an odd integer >= 3, "
                f"got {dilation_kernel_size}"
            )

        self.temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_schedule = temperature_schedule
        self.max_pos_per_image = max_pos_per_image
        self.neg_pos_ratio = neg_pos_ratio
        self.hard_neg_ratio = hard_neg_ratio
        self.dilation_kernel_size = dilation_kernel_size
        self.use_hard_negative_mining = use_hard_negative_mining
        self.min_pos_distance = min_pos_distance
        self.eps = eps
        
        # For temperature scheduling
        self.current_epoch = 0
        self.num_epochs = 20  # Will be updated by trainer

    def set_epoch(self, epoch: int, num_epochs: int = None):
        """Update current epoch for temperature scheduling."""
        self.current_epoch = epoch
        if num_epochs is not None:
            self.num_epochs = num_epochs
        
        # Update temperature based on schedule
        if self.temperature_schedule == "cosine":
            progress = epoch / max(self.num_epochs, 1)
            self.current_temperature = self.temperature_min + (
                self.temperature - self.temperature_min
            ) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        elif self.temperature_schedule == "linear":
            progress = epoch / max(self.num_epochs, 1)
            self.current_temperature = self.temperature - (
                self.temperature - self.temperature_min
            ) * progress
        else:  # constant
            self.current_temperature = self.temperature

    @staticmethod
    def _prepare_labels(labels: torch.Tensor) -> torch.Tensor:
        """Ensure labels have shape (B, 1, *spatial) and create a foreground mask."""
        if labels.dim() < 3:
            raise ValueError(
                "labels must have shape (B, H, W) or (B, D, H, W) "
                "or with an extra channel dimension."
            )

        if labels.dim() == 3:
            labels = labels.unsqueeze(1)
        elif labels.dim() == 4:
            if labels.size(1) != 1:
                labels = labels.unsqueeze(1)
        elif labels.dim() == 5:
            if labels.size(1) != 1:
                raise ValueError(
                    "For 5D labels, expected shape (B, 1, D, H, W). "
                    f"Got: {tuple(labels.shape)}"
                )
        else:
            raise ValueError(
                "Unsupported labels dimensionality. "
                f"Got: {labels.dim()} dims"
            )

        fg_mask = labels > 0
        return fg_mask

    def _dilate_foreground(self, fg_mask: torch.Tensor) -> torch.Tensor:
        """Dilate the foreground mask to obtain a rim band."""
        if fg_mask.dim() not in (4, 5):
            raise ValueError(
                "fg_mask must have shape (B, 1, H, W) or (B, 1, D, H, W). "
                f"Got: {tuple(fg_mask.shape)}"
            )

        b, c, *spatial = fg_mask.shape
        device = fg_mask.device

        if len(spatial) == 2:
            fg = fg_mask.view(b, c, 1, spatial[0], spatial[1]).float()
            dilated = F.max_pool3d(
                fg,
                kernel_size=self.dilation_kernel_size,
                stride=1,
                padding=self.dilation_kernel_size // 2,
            )
            dilated = dilated.view(b, c, *spatial) > 0
        elif len(spatial) == 3:
            fg = fg_mask.float()
            dilated = F.max_pool3d(
                fg,
                kernel_size=self.dilation_kernel_size,
                stride=1,
                padding=self.dilation_kernel_size // 2,
            )
            dilated = dilated > 0
        else:
            raise ValueError(
                "Unsupported spatial dimensionality in fg_mask. "
                f"Got spatial dims: {len(spatial)}"
            )

        rim = dilated & (~fg_mask)
        return rim.to(device=device, dtype=torch.bool)

    def _sample_diverse_positives(self, fg_idx: torch.Tensor, spatial_shape: tuple, device: torch.device) -> torch.Tensor:
        """
        Sample positive pixels ensuring minimum distance between them for diversity.
        This helps avoid sampling pixels that are too close together.
        """
        if fg_idx.numel() <= self.max_pos_per_image:
            return fg_idx
        
        # Convert flat indices to spatial coordinates
        if len(spatial_shape) == 2:
            h, w = spatial_shape
            coords = torch.stack([
                fg_idx // w,
                fg_idx % w
            ], dim=1).float()
        elif len(spatial_shape) == 3:
            d, h, w = spatial_shape
            coords = torch.stack([
                fg_idx // (h * w),
                (fg_idx % (h * w)) // w,
                fg_idx % w
            ], dim=1).float()
        else:
            # Fallback to random sampling
            perm = torch.randperm(fg_idx.numel(), device=device)
            return fg_idx[perm[:self.max_pos_per_image]]
        
        # Greedy sampling: start with random point, then add points that are far enough
        selected = []
        perm = torch.randperm(fg_idx.numel(), device=device)
        remaining = perm.tolist()
        
        # Add first point
        selected.append(remaining.pop(0))
        
        while len(selected) < self.max_pos_per_image and remaining:
            # Find point farthest from all selected points
            selected_coords = coords[selected]
            remaining_coords = coords[remaining]
            
            # Compute distances from each remaining point to all selected points
            distances = torch.cdist(remaining_coords.unsqueeze(0), selected_coords.unsqueeze(0)).squeeze(0)
            min_distances = distances.min(dim=1)[0]
            
            # Filter points that are far enough
            valid_mask = min_distances >= self.min_pos_distance
            if valid_mask.any():
                valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
                # Pick one randomly from valid ones
                idx = valid_indices[torch.randint(valid_indices.numel(), (1,), device=device)].item()
                selected.append(remaining[idx])
                remaining.pop(idx)
            else:
                # If no point is far enough, pick the farthest one
                idx = min_distances.argmax().item()
                selected.append(remaining[idx])
                remaining.pop(idx)
        
        return fg_idx[selected]

    def _hard_negative_mining(self, pos_feats: torch.Tensor, neg_feats: torch.Tensor, 
                             num_hard: int, device: torch.device) -> torch.Tensor:
        """
        Mine hard negatives: negatives that are most similar to positives.
        Returns indices of hard negatives.
        """
        if num_hard == 0 or neg_feats.shape[0] == 0:
            return torch.tensor([], dtype=torch.long, device=device)
        
        # Compute similarity between positives and negatives
        # pos_feats: (P, C), neg_feats: (N, C)
        similarities = torch.matmul(pos_feats, neg_feats.t())  # (P, N)
        # For each negative, get max similarity to any positive
        max_similarities = similarities.max(dim=0)[0]  # (N,)
        
        # Select top-k hardest negatives (highest similarity to positives)
        num_hard = min(num_hard, neg_feats.shape[0])
        _, hard_indices = torch.topk(max_similarities, num_hard, dim=0)
        
        return hard_indices

    def forward(self, features, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute improved pixel-wise boundary contrastive loss.
        
        Args:
            features: Tensor of shape (B, C, H, W) or (B, C, D, H, W), 
                     or list of such tensors for multi-scale.
            labels: Tensor of shape (B, H, W) or (B, 1, H, W) or
                   (B, D, H, W) / (B, 1, D, H, W).
        
        Returns:
            Scalar loss tensor (averaged across scales if multi-scale).
        """
        # Handle multi-scale features
        if isinstance(features, (list, tuple)):
            losses = []
            for i, feat in enumerate(features):
                # For multi-scale, downsample labels to match feature resolution
                labels_scaled = self._downsample_labels_to_match_features(labels, feat)
                loss = self._forward_single_scale(feat, labels_scaled)
                if loss is not None and loss.item() > 0:
                    losses.append(loss)
            if not losses:
                device = features[0].device if features else torch.device('cuda')
                return torch.tensor(0.0, device=device, dtype=features[0].dtype, requires_grad=True)
            # Weighted average: higher weight for higher resolution
            weights = [1.0 / (2 ** i) for i in range(len(losses))]
            weights = torch.tensor(weights, device=losses[0].device)
            weights = weights / weights.sum()
            return sum(w * l for w, l in zip(weights, losses))
        else:
            return self._forward_single_scale(features, labels)
    
    def _downsample_labels_to_match_features(self, labels: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Downsample labels to match feature spatial dimensions."""
        if labels.dim() < 3:
            return labels
        
        # Get target spatial shape from features
        target_spatial = features.shape[2:]
        label_spatial = labels.shape[1:] if labels.dim() == 3 else labels.shape[2:]
        
        # If spatial dimensions match, return as-is
        if target_spatial == label_spatial:
            return labels
        
        # Prepare labels: ensure (B, 1, *spatial)
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)  # (B, 1, H, W)
        elif labels.dim() == 4 and labels.size(1) != 1:
            labels = labels.unsqueeze(1)  # (B, 1, D, H, W)
        
        # Downsample using max pooling (preserves foreground)
        if len(target_spatial) == 2:
            # 2D: (B, 1, H, W) -> (B, 1, H', W')
            scale_h = label_spatial[0] / target_spatial[0]
            scale_w = label_spatial[1] / target_spatial[1]
            kernel_size = (int(scale_h), int(scale_w))
            if kernel_size[0] > 1 or kernel_size[1] > 1:
                labels = F.max_pool2d(labels.float(), kernel_size=kernel_size, stride=kernel_size)
            labels = F.interpolate(labels, size=target_spatial, mode='nearest')
        elif len(target_spatial) == 3:
            # 3D: (B, 1, D, H, W) -> (B, 1, D', H', W')
            scale_d = label_spatial[0] / target_spatial[0]
            scale_h = label_spatial[1] / target_spatial[1]
            scale_w = label_spatial[2] / target_spatial[2]
            kernel_size = (int(scale_d), int(scale_h), int(scale_w))
            if any(k > 1 for k in kernel_size):
                labels = F.max_pool3d(labels.float(), kernel_size=kernel_size, stride=kernel_size)
            labels = F.interpolate(labels, size=target_spatial, mode='nearest')
        
        return labels.long() if labels.dtype == torch.float else labels

    def _forward_single_scale(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for a single scale."""
        if features.dim() not in (4, 5):
            raise ValueError(
                "features must have shape (B, C, H, W) or (B, C, D, H, W). "
                f"Got: {tuple(features.shape)}"
            )

        b, c, *spatial = features.shape
        device = features.device
        current_temp = getattr(self, 'current_temperature', self.temperature)

        # Normalize features for cosine similarity
        feats = features.view(b, c, -1)  # (B, C, P)
        feats = F.normalize(feats, p=2, dim=1)
        feats = feats.transpose(1, 2)  # (B, P, C)

        fg_mask = self._prepare_labels(labels).to(device=device)
        rim_mask = self._dilate_foreground(fg_mask)

        fg_mask_flat = fg_mask.view(b, -1)
        rim_mask_flat = rim_mask.view(b, -1)

        losses = []

        for img_idx in range(b):
            feat_img = feats[img_idx]  # (P, C)
            fg_idx = torch.nonzero(fg_mask_flat[img_idx], as_tuple=False).squeeze(1)
            rim_idx = torch.nonzero(rim_mask_flat[img_idx], as_tuple=False).squeeze(1)

            if fg_idx.numel() < 2 or rim_idx.numel() == 0:
                continue

            # Sample diverse positives
            if fg_idx.numel() > self.max_pos_per_image:
                fg_idx = self._sample_diverse_positives(fg_idx, spatial, device)
            
            num_pos = fg_idx.numel()
            if num_pos < 2:
                continue

            # Sample negatives
            max_neg = int(self.neg_pos_ratio * float(num_pos))
            if max_neg <= 0:
                max_neg = num_pos

            pos_feats = feat_img[fg_idx]  # (P, C)
            
            if rim_idx.numel() > max_neg:
                # Use hard negative mining if enabled
                if self.use_hard_negative_mining and rim_idx.numel() > max_neg:
                    num_hard = int(self.hard_neg_ratio * max_neg)
                    num_random = max_neg - num_hard
                    
                    neg_feats_all = feat_img[rim_idx]  # (N_all, C)
                    hard_indices = self._hard_negative_mining(
                        pos_feats, neg_feats_all, num_hard, device
                    )
                    
                    # Get random negatives from remaining
                    remaining_mask = torch.ones(rim_idx.numel(), dtype=torch.bool, device=device)
                    remaining_mask[hard_indices] = False
                    remaining_idx = torch.nonzero(remaining_mask, as_tuple=False).squeeze(1)
                    
                    if remaining_idx.numel() > num_random:
                        perm = torch.randperm(remaining_idx.numel(), device=device)
                        random_indices = remaining_idx[perm[:num_random]]
                    else:
                        random_indices = remaining_idx
                    
                    selected_neg_indices = torch.cat([hard_indices, random_indices])
                    rim_idx = rim_idx[selected_neg_indices]
                else:
                    perm = torch.randperm(rim_idx.numel(), device=device)
                    rim_idx = rim_idx[perm[:max_neg]]

            neg_feats = feat_img[rim_idx]  # (N, C)

            # Similarities
            pos_pos_logits = torch.matmul(pos_feats, pos_feats.t()) / current_temp  # (P, P)
            pos_neg_logits = torch.matmul(pos_feats, neg_feats.t()) / current_temp  # (P, N)

            # Mask to exclude self from positives
            pos_mask = torch.ones_like(pos_pos_logits, dtype=torch.bool, device=device)
            pos_mask.fill_diagonal_(False)

            # Combine logits
            all_logits = torch.cat([pos_pos_logits, pos_neg_logits], dim=1)  # (P, P+N)
            diag_indices = torch.arange(num_pos, device=device)
            all_logits[diag_indices, diag_indices] = -1e9

            # Log-sum-exp trick
            max_logits, _ = all_logits.max(dim=1, keepdim=True)
            exp_all = torch.exp(all_logits - max_logits)

            exp_pos = exp_all[:, :num_pos] * pos_mask
            numerator = exp_pos.sum(dim=1) + self.eps
            denominator = exp_all.sum(dim=1) + self.eps

            loss_img = -torch.log(numerator / denominator)
            valid_mask = numerator > self.eps
            if valid_mask.any():
                losses.append(loss_img[valid_mask].mean())

        if not losses:
            return torch.tensor(0.0, device=device, dtype=features.dtype, requires_grad=True)

        return torch.stack(losses, dim=0).mean()


__all__ = ["ImprovedPixelwiseBoundaryContrastiveLoss"]
