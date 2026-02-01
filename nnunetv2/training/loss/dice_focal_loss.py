import torch
import torch.nn as nn
import torch.nn.functional as F

# Focal Loss for Binary Segmentation
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: Logits from model (2-channel output, we take only foreground).
        targets: Ground truth masks (binary, single-channel).
        """
        inputs = inputs[:, 1, ...]  # Take only foreground logits
        targets = targets.squeeze(1).float()  # Convert target to float

        # Compute standard BCE loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute Focal Loss
        pt = torch.exp(-BCE_loss)  # Probability of the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# Dice Loss for Binary Segmentation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: Logits from model (2-channel output, we take only foreground).
        targets: Ground truth masks (binary, single-channel).
        """
        inputs = torch.sigmoid(inputs[:, 1, ...])  # Convert to probability
        targets = targets.squeeze(1).float()  # Convert target to float

        # Compute Dice coefficient
        intersection = (inputs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice_coeff  # Dice Loss = 1 - Dice Coefficient

# Combined Dice + Focal + BCE Loss
class DiceFocalBCELoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=3):
        super(DiceFocalBCELoss, self).__init__()
        self.focal_loss = FocalLoss(alpha, gamma)
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        inputs: Logits from model (2-channel output, we take only foreground).
        targets: Ground truth masks (binary, single-channel).
        """
        inputs_fg = inputs[:, 1, ...]  # Foreground logits
        targets = targets.squeeze(1).float()  # Convert target to float

        return (
            0.5 * self.dice_loss(inputs, targets) +  
            0.25 * self.bce_loss(inputs_fg, targets) +  # Apply BCE only on foreground
            0.25 * self.focal_loss(inputs, targets)
        )