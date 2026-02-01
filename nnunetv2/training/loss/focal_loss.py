import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if targets.ndim == inputs.ndim:
             targets = targets.squeeze(1)
        
        inputs_fg = inputs[:, 1, ...] 
        targets = targets.float()

        BCE_loss = F.binary_cross_entropy_with_logits(inputs_fg, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()