# nnunetv2/training/loss/pixel_contrastive.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def _adaptive_pool_nd(x: torch.Tensor, out_hw: int) -> torch.Tensor:
    """
    Downsample embeddings to out_hw x out_hw (2D) or out_hw^3 (3D) via adaptive avg pool.
    Works for [B, C, H, W] or [B, C, D, H, W].
    """
    if x.ndim == 4:
        return F.adaptive_avg_pool2d(x, (out_hw, out_hw))
    elif x.ndim == 5:
        return F.adaptive_avg_pool3d(x, (out_hw, out_hw, out_hw))
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {x.shape}")

def _flatten_feats_labels(feats: torch.Tensor, y: torch.Tensor, ignore_index: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    feats: [B, C, ...] (2D or 3D)
    y    : [B, ...]    (same spatial)
    returns:
      f: [N, C] L2-normalized
      t: [N]    labels
    """
    B, C = feats.shape[:2]
    f = feats.view(B, C, -1).transpose(1, 2).contiguous()  # [B, HW(D), C]
    t = y.view(B, -1)                                      # [B, HW(D)]
    f = F.normalize(f, dim=-1)
    f = f.view(-1, C)   # [N, C]
    t = t.view(-1)      # [N]
    if ignore_index is not None:
        keep = (t != ignore_index)
        f, t = f[keep], t[keep]
    return f, t

class PixelWiseSupConLoss(nn.Module):
    """
    Pixel-wise supervised contrastive loss from Zhao et al., ICCV'21.
    Implements within-image (Eq. 1) and cross-image (Eq. 2) variants.

    Args:
      temperature: τ in the paper (default 0.07)
      down_hw:     resize shortest side of feature map to down_hw for stability (paper downsamples to ~65) 
      ignore_index: label to ignore (e.g., 255); set None if not needed
      variant:     'within' or 'cross' (cross-image averages I→J and J→I)
      max_pixels:  if not None, uniform-samples at most this many pixels per image to cap O(N^2)
    """
    def __init__(self, temperature: float = 0.07, down_hw: int = 64,
                 ignore_index: Optional[int] = None,
                 variant: str = 'cross', max_pixels: Optional[int] = None):
        super().__init__()
        assert variant in ('within', 'cross')
        self.tau = temperature
        self.down_hw = down_hw
        self.ignore_index = ignore_index
        self.variant = variant
        self.max_pixels = max_pixels
        self.eps = 1e-8

    def _maybe_sample(self, f, t):
        if self.max_pixels is None or f.shape[0] <= self.max_pixels:
            return f, t
        idx = torch.randperm(f.shape[0], device=f.device)[:self.max_pixels]
        # IMPORTANT: no torch.no_grad() here (keeps autograd alive)
        return torch.index_select(f, 0, idx), torch.index_select(t, 0, idx)

    def _within_image_loss(self, fA, tA, fB, tB):
        """
        Eq. (1): - (1/N) sum_p ( 1/|PosB(p)| sum_{q in PosB(p)} log ( exp(sim)/sum_k exp(sim) ) )
        Implemented via log_softmax over all pixels in B, masked average over positives.
        """
        logits = (fA @ fB.t()) / self.tau                         # [NA, NB]
        log_prob = logits.log_softmax(dim=1)                      # denom: all NB
        pos = (tA.view(-1, 1) == tB.view(1, -1)).to(log_prob.dtype)
        pos_counts = pos.sum(1).clamp_min(1.0)
        loss_rows = -(pos * log_prob).sum(1) / pos_counts
        return loss_rows.mean()

    def _cross_image_loss_eq2_oneway(self, fI, tI, fIhat, tIhat, fJ, tJ):
        """
        Implements Eq. (2) for anchors in I (one direction).
        Denominator for each anchor p: sum_k exp(sim(Ip, Ĩ_k)) + sum_{k in PosJ(p)} exp(sim(Ip, J_k))
        Numerators: positives from Ĩ and positives from J, each normalized by the SAME denominator.
        """
        # logits to Ĩ and J
        L_I_Ih = (fI @ fIhat.t()) / self.tau            # [NI, NĨ]
        L_I_J  = (fI @ fJ.t())    / self.tau            # [NI, NJ]

        # positive masks
        pos_Ih = (tI.view(-1,1) == tIhat.view(1,-1))    # [NI, NĨ]
        pos_IJ = (tI.view(-1,1) == tJ.view(1,-1))       # [NI, NJ]

        # denom: log( sum_k e^{L_I_Ih[:,k]}  +  sum_{k in PosJ} e^{L_I_J[:,k]} )
        # -> masked logsumexp over concat([L_I_Ih, masked L_I_J (non-positives -> -inf)])
        masked_L_I_J = L_I_J.masked_fill(~pos_IJ, float('-inf'))
        den = torch.logsumexp(torch.cat([L_I_Ih, masked_L_I_J], dim=1), dim=1)   # [NI]

        # log-probs for positives in Ĩ and J w.r.t. the SAME denominator
        logp_Ih = L_I_Ih - den[:, None]                # [NI, NĨ]
        logp_IJ = L_I_J  - den[:, None]                # [NI, NJ]

        # average over all positives from Ĩ and J
        pos_count_Ih = pos_Ih.sum(1)
        pos_count_IJ = pos_IJ.sum(1)
        pos_count_total = (pos_count_Ih + pos_count_IJ).clamp_min(1)

        # Sum logs only at positive locations
        sum_pos_Ih = (pos_Ih.to(logp_Ih.dtype) * logp_Ih).sum(1)
        sum_pos_IJ = (pos_IJ.to(logp_IJ.dtype) * logp_IJ).sum(1)

        loss_rows = -(sum_pos_Ih + sum_pos_IJ) / pos_count_total
        return loss_rows.mean()

    def forward(self, emb_I, y_I, emb_Ihat, y_Ihat, emb_J=None, y_J=None):
        # downsample features to keep memory manageable (paper uses ~65x65)
        emb_I    = _adaptive_pool_nd(emb_I,    self.down_hw)
        emb_Ihat = _adaptive_pool_nd(emb_Ihat, self.down_hw)

        # resize labels to match features (nearest)
        y_I    = F.interpolate(y_I.unsqueeze(1).float(),    size=emb_I.shape[2:],    mode='nearest').squeeze(1).long()
        y_Ihat = F.interpolate(y_Ihat.unsqueeze(1).float(), size=emb_Ihat.shape[2:], mode='nearest').squeeze(1).long()

        # flatten & L2-normalize features
        fI, tI       = _flatten_feats_labels(emb_I,    y_I,    self.ignore_index)
        fIh, tIh     = _flatten_feats_labels(emb_Ihat, y_Ihat, self.ignore_index)
        fI,  tI      = self._maybe_sample(fI,  tI)
        fIh, tIh     = self._maybe_sample(fIh, tIh)

        if self.variant == 'within' or emb_J is None or y_J is None:
            return self._within_image_loss(fI, tI, fIh, tIh)

        # cross-image (Eq. 2) — IMPORTANT: no J-negatives in denominator!
        emb_J  = _adaptive_pool_nd(emb_J, self.down_hw)
        y_J    = F.interpolate(y_J.unsqueeze(1).float(), size=emb_J.shape[2:], mode='nearest').squeeze(1).long()
        fJ, tJ = _flatten_feats_labels(emb_J, y_J, self.ignore_index)
        fJ, tJ = self._maybe_sample(fJ, tJ)

        # I-anchored term
        loss_I = self._cross_image_loss_eq2_oneway(fI, tI, fIh, tIh, fJ, tJ)
        # J-anchored symmetric term (swap I and J)
        loss_J = self._cross_image_loss_eq2_oneway(fJ, tJ, fIh, tIh, fI, tI)

        return 0.5 * (loss_I + loss_J)
