import torch
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice_focal_bce_boundary_loss import DiceFocalBCEBoundaryLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

class nnUNetTrainerFineTune(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # Fine-tuning parameters
        self.num_epochs = 250 
        self.initial_lr = 1e-2 

    def _build_loss(self):
        # Instantiate your compound loss
        loss = DiceFocalBCEBoundaryLoss(
            alpha=0.35, gamma=3,
            boundary_weight=0.0, # Set to 0 if you don't have precomputed distance maps
            dice_weight=1.0,
            bce_weight=1.0,
            focal_weight=1.0
        )
        
        # Wrap for Deep Supervision if enabled
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            
            # Calculate weights: 1/(2^i) -> [1, 0.5, 0.25, ...]
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            
            # Normalize to sum to 1
            weights = weights / weights.sum()
            
            return DeepSupervisionWrapper(loss, weights)
            
        return loss