import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from scipy.ndimage import distance_transform_edt
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1

# -----------------------------------------------------------------------------
# 1. Helper: Compute Signed Distance Map
# -----------------------------------------------------------------------------
def compute_sdf(segmentation, class_indices):
    """
    Computes Signed Distance Map.
    Negative Inside, Positive Outside.
    """
    normalized_sdf = np.zeros((len(class_indices), *segmentation.shape), dtype=np.float32)

    for i, c in enumerate(class_indices):
        posmask = (segmentation == c)
        if posmask.sum() == 0:
            dist_out = distance_transform_edt(1 - posmask)
            dist_in = np.zeros_like(dist_out)
        elif posmask.all():
            dist_out = np.zeros_like(posmask)
            dist_in = distance_transform_edt(posmask)
        else:
            dist_out = distance_transform_edt(1 - posmask)
            dist_in = distance_transform_edt(posmask)
        
        sdf = dist_out - (dist_in - 1)
        normalized_sdf[i] = sdf

    return normalized_sdf

# -----------------------------------------------------------------------------
# 2. Boundary Loss
# -----------------------------------------------------------------------------
class BoundaryLoss(nn.Module):
    def __init__(self, weight=1.0, ignore_label=None):
        super(BoundaryLoss, self).__init__()
        self.weight = weight
        self.ignore_label = ignore_label

    def forward(self, net_output, target):
        if self.weight <= 0:
            return torch.tensor(0.0, device=net_output.device, requires_grad=True)

        probs = softmax_helper_dim1(net_output)
        B, C = probs.shape[0], probs.shape[1]
        
        with torch.no_grad():
            target_cpu = target.cpu().numpy()
            sdf_list = []
            for b in range(B):
                gt_b = target_cpu[b, 0]
                if self.ignore_label is not None:
                    gt_b[gt_b == self.ignore_label] = 0
                
                classes = list(range(C))
                sdf_b = compute_sdf(gt_b, classes)
                sdf_list.append(torch.from_numpy(sdf_b))
            
            gt_sdf = torch.stack(sdf_list).to(net_output.device)

        loss = torch.mean(probs * gt_sdf)
        return self.weight * loss

# -----------------------------------------------------------------------------
# 3. Compound Loss (Handles Deep Supervision Manually)
# -----------------------------------------------------------------------------
class DC_CE_and_Boundary_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, 
                 weight_boundary=0.01, ignore_label=None, 
                 dice_class=MemoryEfficientSoftDiceLoss,
                 deep_supervision_weights=None):
        super().__init__()
        
        self.dc_ce_loss = DC_and_CE_loss(
            soft_dice_kwargs, ce_kwargs, weight_ce, weight_dice, ignore_label, dice_class
        )
        self.boundary_loss = BoundaryLoss(weight=weight_boundary, ignore_label=ignore_label)
        self.deep_supervision_weights = deep_supervision_weights

    def forward(self, net_output, target):
        """
        Handles both list inputs (Deep Supervision) and single tensors.
        """
        # Case 1: Deep Supervision Enabled (Input is a list)
        if isinstance(net_output, list):
            # 1. Dice + CE (Weighted sum over all scales)
            loss_seg = 0.0
            
            # Use weights if provided, otherwise uniform (should typically provide weights)
            weights = self.deep_supervision_weights
            if weights is None:
                weights = [1.0] * len(net_output)
            
            # Normalize weights to sum to 1 just in case, though nnunet usually does this upfront
            # logic: iterate zip(output, target, weight)
            for out_i, target_i, w_i in zip(net_output, target, weights):
                loss_seg += w_i * self.dc_ce_loss(out_i, target_i)
            
            # 2. Boundary Loss (Only on Highest Resolution / Index 0)
            # We skip lower resolutions for performance
            loss_bound = self.boundary_loss(net_output[0], target[0])
            
        # Case 2: Deep Supervision Disabled (Input is a tensor)
        else:
            loss_seg = self.dc_ce_loss(net_output, target)
            loss_bound = self.boundary_loss(net_output, target)

        return loss_seg + loss_bound

# -----------------------------------------------------------------------------
# 4. Trainer
# -----------------------------------------------------------------------------
class nnUNetTrainerFineTuneSDFLoss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        self.num_epochs = 100
        self.initial_lr = 1e-2
        self.boundary_weight = 0.01 

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return optimizer, lr_scheduler

    def _build_loss(self):
        # Calculate Deep Supervision Weights manually
        ds_weights = None
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            # 1/(2^i)
            ds_weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            # Normalize
            ds_weights = ds_weights / ds_weights.sum()
            # Convert to list for easier zip
            ds_weights = list(ds_weights)

        loss = DC_CE_and_Boundary_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, 
            {}, 
            weight_ce=1, 
            weight_dice=1, 
            weight_boundary=self.boundary_weight,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
            deep_supervision_weights=ds_weights # Pass weights here
        )
        return loss