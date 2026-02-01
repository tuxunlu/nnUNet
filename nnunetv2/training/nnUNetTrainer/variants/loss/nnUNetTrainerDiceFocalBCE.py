import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.helpers import softmax_helper_dim1
import torch.nn.functional as F
import numpy as np
import inspect
from torch._dynamo import OptimizedModule

from nnunetv2.training.loss.dice_loss import DiceLoss
from nnunetv2.training.loss.focal_loss import FocalLoss
from nnunetv2.training.loss.dice_focal_bce_loss import DiceFocalBCELoss

# ==========================================
# TRAINER CLASS
# ==========================================

class nnUNetTrainerDiceFocalBCE(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.device = device
        self.num_epochs = 100
        
    def _build_loss(self):
        loss = DiceFocalBCELoss(
            alpha=0.35, 
            gamma=3, 
        )
        return loss

    def initialize(self):
        super().initialize()
        # Get the actual network module (handle DDP and compiled networks)
        mod = self.network
        if hasattr(mod, 'module'):  # DDP wrapped
            mod = mod.module
        if isinstance(mod, OptimizedModule):  # Compiled network
            mod = mod._orig_mod
        
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)
            
            if self.enable_deep_supervision:
                l = self.loss(output[0], target[0])
                
                num_scales = len(output)
                weights = np.array([1 / (2 ** i) for i in range(num_scales)])
                
                if num_scales > 1:
                    weights[-1] = 0
                weights = weights / weights.sum()

                for i in range(1, num_scales):
                    if weights[i] != 0:
                        # Only use contrastive loss on the highest resolution output
                        l += weights[i] * self.loss(output[i], target[i])
            else:
                l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

# ==========================================
# SIGNATURE PATCH
# ==========================================
nnUNetTrainerDiceFocalBCE.__init__.__signature__ = inspect.signature(nnUNetTrainer.__init__)