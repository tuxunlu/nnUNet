import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelContrastiveLoss(nn.Module):
    """
    Pixel-wise supervised contrastive loss (InfoNCE-style).

    This implementation follows the core idea used in supervised pixel-wise contrastive pretraining for
    semantic segmentation: for each anchor pixel, pixels with the same label are positives and pixels with
    different labels are negatives. Pixels with `ignore_index` are excluded.

    Notes:
    - This is a *within-sample* (per-image) contrastive loss. It does not require an explicit augmented
      view. If you want the exact (I, Î) formulation from some papers, you must provide two views and
      compute the loss across them.
    - Works for 2D and 3D feature maps.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        ignore_index: int = 255,
        max_samples: int = 2048,
        anchor_chunk_size: int = 1024,
        hard_negative_radius: int = 10,
        hard_negative_fraction: float = 0.75,
        foreground_label: int = 1,
    ):
        super().__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.max_samples = max_samples
        self.anchor_chunk_size = anchor_chunk_size
        self.hard_negative_radius = int(hard_negative_radius)
        self.hard_negative_fraction = float(hard_negative_fraction)
        self.foreground_label = int(foreground_label)

    def forward(self, feats, labels):
        # Accept [B, C, ...] feats (2D or 3D), and labels [B, 1, ...] or [B, ...].
        if labels.dim() == feats.dim():
            # labels [B, 1, ...] -> [B, ...]
            labels = labels.squeeze(1)

        # Align label resolution to feature resolution (nnU-Net outputs may be downsampled).
        if labels.shape[1:] != feats.shape[2:]:
            labels = F.interpolate(
                labels.unsqueeze(1).float(),
                size=feats.shape[2:],
                mode="nearest",
            ).squeeze(1).long()

        B, C = feats.shape[:2]
        spatial = feats.shape[2:]
        n_pix = int(torch.tensor(spatial).prod().item()) if len(spatial) > 0 else 1

        # Flatten spatial dims: feats -> [B, N, C], labels -> [B, N]
        feats = feats.reshape(B, C, n_pix).permute(0, 2, 1).contiguous()
        labels = labels.reshape(B, n_pix)

        total_loss = feats.new_tensor(0.0)
        n_imgs = 0

        for b in range(B):
            lb = labels[b]
            valid = lb != self.ignore_index
            if valid.sum() < 2:
                continue

            # -------------------------------
            # Hard negative mining (Option C)
            # -------------------------------
            # Prefer background pixels near the lesion boundary at feature-map resolution.
            # We do this by creating a "ring" = dilate(fg) \ fg, and sampling negatives from that ring first.
            # This avoids the loss being dominated by easy background.
            fg_mask_flat = (lb == self.foreground_label) & valid
            bg_mask_flat = (lb != self.foreground_label) & valid

            # Build ring mask in spatial shape
            fg_mask = fg_mask_flat.reshape((1, 1, *spatial)).float()
            r = max(0, self.hard_negative_radius)
            if r > 0:
                k = 2 * r + 1
                if len(spatial) == 2:
                    dil = F.max_pool2d(fg_mask, kernel_size=k, stride=1, padding=r)
                elif len(spatial) == 3:
                    dil = F.max_pool3d(fg_mask, kernel_size=k, stride=1, padding=r)
                else:
                    # Fallback: no mining for unexpected dims
                    dil = fg_mask
            else:
                dil = fg_mask

            dil_flat = (dil.reshape(-1) > 0.0)
            ring_flat = dil_flat & (~fg_mask_flat) & bg_mask_flat  # bg near fg

            fg_idx = torch.nonzero(fg_mask_flat, as_tuple=False).squeeze(1)
            ring_idx = torch.nonzero(ring_flat, as_tuple=False).squeeze(1)
            bg_idx = torch.nonzero(bg_mask_flat & (~ring_flat), as_tuple=False).squeeze(1)

            # If there are no foreground pixels, fall back to random valid sampling
            if fg_idx.numel() == 0:
                idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
                if idx.numel() > self.max_samples:
                    perm = torch.randperm(idx.numel(), device=idx.device)[: self.max_samples]
                    idx = idx[perm]
            else:
                # Target composition: all available fg (up to half budget), and bg negatives with ring preference
                max_fg = max(2, self.max_samples // 2)
                if fg_idx.numel() > max_fg:
                    perm = torch.randperm(fg_idx.numel(), device=fg_idx.device)[:max_fg]
                    fg_sel = fg_idx[perm]
                else:
                    fg_sel = fg_idx

                neg_budget = max(0, self.max_samples - fg_sel.numel())
                hard_budget = int(round(self.hard_negative_fraction * neg_budget))
                hard_budget = max(0, min(hard_budget, neg_budget))
                rand_budget = neg_budget - hard_budget

                ring_sel = ring_idx
                if ring_sel.numel() > hard_budget:
                    perm = torch.randperm(ring_sel.numel(), device=ring_sel.device)[:hard_budget]
                    ring_sel = ring_sel[perm]
                else:
                    # If ring has fewer than budget, we will fill remainder from bg_idx below
                    pass

                remaining = neg_budget - ring_sel.numel()
                bg_pool = torch.cat([bg_idx, ring_idx.new_empty((0,), dtype=ring_idx.dtype)])  # ensure tensor on same device
                if remaining > 0 and bg_pool.numel() > 0:
                    take = min(remaining, bg_pool.numel())
                    perm = torch.randperm(bg_pool.numel(), device=bg_pool.device)[:take]
                    bg_sel = bg_pool[perm]
                else:
                    bg_sel = bg_pool[:0]

                idx = torch.cat([fg_sel, ring_sel, bg_sel])
                # Ensure we don't exceed max_samples (can happen if fg is large)
                if idx.numel() > self.max_samples:
                    idx = idx[: self.max_samples]

            f = feats[b, idx]  # [K, C]
            y = lb[idx]  # [K]

            # Use only foreground (lesion) pixels as anchors.
            # Require at least 2 foreground pixels, otherwise no positive pairs exist.
            fg_anchor_mask = (y == self.foreground_label)
            if fg_anchor_mask.sum() < 2:
                continue

            f = F.normalize(f, dim=1)
            K = f.shape[0]

            # logits [K, K]
            logits = (f @ f.t()) / self.temperature
            logits = logits - logits.max(dim=1, keepdim=True).values.detach()

            # mask out self-comparisons
            self_mask = torch.eye(K, device=f.device, dtype=torch.bool)

            # positive pairs: same label and not self
            y_col = y.view(-1, 1)
            pos_mask = (y_col == y_col.t()) & (~self_mask)

            # denominator uses all non-self pixels
            exp_logits = torch.exp(logits) * (~self_mask).float()
            denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8)

            # We compute: - log( sum_{pos} exp(sim)/ denom )
            # Anchors are restricted to foreground pixels; if an anchor has no positives,
            # it is excluded from the average.
            pos_exp_sum = (exp_logits * pos_mask.float()).sum(dim=1)
            valid_anchor = pos_mask.any(dim=1) & fg_anchor_mask
            if valid_anchor.sum() == 0:
                continue

            # To reduce peak memory, optionally chunk anchors (doesn't change result).
            if self.anchor_chunk_size and self.anchor_chunk_size < K:
                loss_b = f.new_tensor(0.0)
                n_valid = 0
                for s in range(0, K, self.anchor_chunk_size):
                    e = min(s + self.anchor_chunk_size, K)
                    va = valid_anchor[s:e]
                    if va.sum() == 0:
                        continue
                    num = pos_exp_sum[s:e][va].clamp_min(1e-8)
                    den = denom[s:e].squeeze(1)[va]
                    loss_b = loss_b + (-torch.log(num / den)).sum()
                    n_valid += int(va.sum().item())
                if n_valid > 0:
                    total_loss = total_loss + (loss_b / n_valid)
                    n_imgs += 1
            else:
                num = pos_exp_sum[valid_anchor].clamp_min(1e-8)
                den = denom.squeeze(1)[valid_anchor]
                total_loss = total_loss + (-torch.log(num / den)).mean()
                n_imgs += 1

        if n_imgs == 0:
            return feats.new_tensor(0.0, requires_grad=True)
        return total_loss / n_imgs