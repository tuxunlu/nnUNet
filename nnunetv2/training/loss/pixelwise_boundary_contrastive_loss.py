import torch
from torch import nn
import torch.nn.functional as F


class PixelwiseBoundaryContrastiveLoss(nn.Module):
    """
    Pixel-wise boundary contrastive loss.

    This is a variant of supervised pixel contrastive loss inspired by
    Google's supervised pixel contrastive loss implementation
    (`supervised_pixel_contrastive_loss`).

    Key design choices for this implementation:
    - **Within-image only**: positives and negatives are sampled only
      from the same image (no cross-image sampling).
    - **Foreground-only positives**: only pixels inside the foreground
      (lesion) are used as positives.
    - **Boundary-band negatives**: negatives are sampled from a
      *dilated rim* immediately outside the foreground (lesion). This
      encourages features at the lesion boundary to be highly
      discriminative with respect to nearby background.

    Expected inputs
    ----------------
    - features: Tensor of shape (B, C, H, W) or (B, C, D, H, W)
        Feature vectors for each spatial location.
    - labels: Tensor of shape (B, H, W) or (B, 1, H, W) or
        (B, D, H, W) / (B, 1, D, H, W)
        Binary or multi-class segmentation labels.
        Foreground is defined as labels > 0.

    Loss definition (per image)
    ----------------------------
    Let:
      - P be the set of sampled foreground (lesion) pixels.
      - R be the set of sampled rim-background pixels.

    For each anchor pixel i in P:
      - Positives: all *other* pixels j in P (j != i).
      - Negatives: all pixels k in R.

    We use an InfoNCE-style objective with multiple positives:

      L_i = - log( sum_{j in P, j!=i} exp(s_ij / tau)
                   -----------------------------------
                   sum_{j in P, j!=i} exp(s_ij / tau)
                   + sum_{k in R}     exp(s_ik / tau) )

    where s_ij is the cosine similarity between feature vectors of i
    and j, and tau is the temperature.
    The final loss is the mean of L_i over all valid anchors and
    averaged over the batch.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        max_pos_per_image: int = 2048,
        neg_pos_ratio: float = 5.0,
        dilation_kernel_size: int = 11,
        eps: float = 1e-8,
    ) -> None:
        """
        Args:
            temperature: Temperature for the contrastive softmax.
            max_pos_per_image: Maximum number of foreground (lesion)
                pixels sampled per image. If there are fewer available,
                all are used.
            neg_pos_ratio: Maximum ratio of negatives to positives
                sampled from the rim per image. For example, 5.0 means
                at most 5 * num_pos negatives are used.
            dilation_kernel_size: Kernel size used to dilate the
                foreground mask in order to construct the rim band.
                Must be an odd integer >= 3.
            eps: Small constant to avoid division / log of zero.
        """
        super().__init__()

        if dilation_kernel_size < 3 or dilation_kernel_size % 2 == 0:
            raise ValueError(
                "dilation_kernel_size must be an odd integer >= 3, "
                f"got {dilation_kernel_size}"
            )

        self.temperature = temperature
        self.max_pos_per_image = max_pos_per_image
        self.neg_pos_ratio = neg_pos_ratio
        self.dilation_kernel_size = dilation_kernel_size
        self.eps = eps

    @staticmethod
    def _prepare_labels(labels: torch.Tensor) -> torch.Tensor:
        """
        Ensure labels have shape (B, 1, *spatial) and create a foreground mask.
        Foreground is defined as labels > 0.
        """
        if labels.dim() < 3:
            raise ValueError(
                "labels must have shape (B, H, W) or (B, D, H, W) "
                "or with an extra channel dimension."
            )

        if labels.dim() == 3:
            # (B, H, W) -> (B, 1, H, W)
            labels = labels.unsqueeze(1)
        elif labels.dim() == 4:
            # either (B, 1, H, W) or (B, D, H, W)
            if labels.size(1) != 1:
                # assume (B, D, H, W) -> (B, 1, D, H, W)
                labels = labels.unsqueeze(1)
        elif labels.dim() == 5:
            # (B, 1, D, H, W) is fine
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

        # Foreground mask: labels > 0
        fg_mask = labels > 0
        return fg_mask

    def _dilate_foreground(self, fg_mask: torch.Tensor) -> torch.Tensor:
        """
        Dilate the foreground mask to obtain a rim band.

        Args:
            fg_mask: Bool tensor of shape (B, 1, *spatial).

        Returns:
            rim_mask: Bool tensor of shape (B, 1, *spatial) representing
            the dilated band outside the foreground.
        """
        if fg_mask.dim() not in (4, 5):
            raise ValueError(
                "fg_mask must have shape (B, 1, H, W) or (B, 1, D, H, W). "
                f"Got: {tuple(fg_mask.shape)}"
            )

        b, c, *spatial = fg_mask.shape
        device = fg_mask.device

        # Use 3D max pooling for both 2D and 3D by inserting a dummy depth
        if len(spatial) == 2:
            # (B, 1, H, W) -> (B, 1, 1, H, W)
            fg = fg_mask.view(b, c, 1, spatial[0], spatial[1]).float()
            dilated = F.max_pool3d(
                fg,
                kernel_size=self.dilation_kernel_size,
                stride=1,
                padding=self.dilation_kernel_size // 2,
            )
            dilated = dilated.view(b, c, *spatial) > 0
        elif len(spatial) == 3:
            # (B, 1, D, H, W) -> use directly
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

        # Rim: dilated foreground minus the original foreground
        rim = dilated & (~fg_mask)
        # Ensure boolean type on correct device
        return rim.to(device=device, dtype=torch.bool)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the pixel-wise boundary contrastive loss.

        Args:
            features: Tensor of shape (B, C, H, W) or (B, C, D, H, W).
            labels: Tensor of shape (B, H, W) or (B, 1, H, W) or
                (B, D, H, W) / (B, 1, D, H, W).

        Returns:
            Scalar loss tensor.
        """
        if features.dim() not in (4, 5):
            raise ValueError(
                "features must have shape (B, C, H, W) or (B, C, D, H, W). "
                f"Got: {tuple(features.shape)}"
            )

        b, c, *spatial = features.shape
        device = features.device

        # Normalize features for cosine similarity
        feats = features.view(b, c, -1)  # (B, C, P)
        feats = F.normalize(feats, p=2, dim=1)
        feats = feats.transpose(1, 2)  # (B, P, C)

        fg_mask = self._prepare_labels(labels).to(device=device)  # (B, 1, *spatial)
        rim_mask = self._dilate_foreground(fg_mask)  # (B, 1, *spatial)

        fg_mask_flat = fg_mask.view(b, -1)  # (B, P)
        rim_mask_flat = rim_mask.view(b, -1)  # (B, P)

        losses = []

        for img_idx in range(b):
            feat_img = feats[img_idx]  # (P, C)
            fg_idx = torch.nonzero(fg_mask_flat[img_idx], as_tuple=False).squeeze(1)
            rim_idx = torch.nonzero(rim_mask_flat[img_idx], as_tuple=False).squeeze(1)

            # Need at least 2 positives to form positive pairs
            if fg_idx.numel() < 2 or rim_idx.numel() == 0:
                continue

            # Sample positives (foreground)
            if fg_idx.numel() > self.max_pos_per_image:
                perm = torch.randperm(fg_idx.numel(), device=device)
                fg_idx = fg_idx[perm[: self.max_pos_per_image]]

            num_pos = fg_idx.numel()
            if num_pos < 2:
                continue

            # Sample negatives (rim band)
            max_neg = int(self.neg_pos_ratio * float(num_pos))
            if max_neg <= 0:
                max_neg = num_pos  # at least as many negatives as positives

            if rim_idx.numel() > max_neg:
                perm = torch.randperm(rim_idx.numel(), device=device)
                rim_idx = rim_idx[perm[:max_neg]]

            pos_feats = feat_img[fg_idx]  # (P, C)
            neg_feats = feat_img[rim_idx]  # (N, C)

            # Similarities (cosine, since features are normalized)
            # Pos-pos matrix (anchors vs positives)
            pos_pos_logits = torch.matmul(pos_feats, pos_feats.t()) / self.temperature  # (P, P)
            # Negatives
            pos_neg_logits = torch.matmul(pos_feats, neg_feats.t()) / self.temperature  # (P, N)

            # Mask to exclude self from positives
            pos_mask = torch.ones_like(pos_pos_logits, dtype=torch.bool, device=device)
            pos_mask.fill_diagonal_(False)

            # Combine logits for numerical stability
            all_logits = torch.cat([pos_pos_logits, pos_neg_logits], dim=1)  # (P, P+N)
            # Mask self-positive positions (diagonal) by setting them to a very small value
            diag_indices = torch.arange(num_pos, device=device)
            all_logits[diag_indices, diag_indices] = -1e9

            # Log-sum-exp trick
            max_logits, _ = all_logits.max(dim=1, keepdim=True)  # (P, 1)
            exp_all = torch.exp(all_logits - max_logits)  # (P, P+N)

            # Positive terms are only in the first P columns (excluding diagonal)
            exp_pos = exp_all[:, :num_pos] * pos_mask  # (P, P)

            numerator = exp_pos.sum(dim=1) + self.eps  # (P,)
            denominator = exp_all.sum(dim=1) + self.eps  # (P,)

            loss_img = -torch.log(numerator / denominator)

            # Filter out any anchors that accidentally have no positive mass
            valid_mask = numerator > self.eps
            if valid_mask.any():
                losses.append(loss_img[valid_mask].mean())

        if not losses:
            # No valid foreground / rim pairs in the batch; return zero loss
            return torch.tensor(0.0, device=device, dtype=features.dtype, requires_grad=True)

        return torch.stack(losses, dim=0).mean()


__all__ = ["PixelwiseBoundaryContrastiveLoss"]

