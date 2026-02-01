import numpy as np
import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice_focal_loss import DiceFocalBCELoss  # Import the custom loss

class nnUNetTrainerFocalDiceBCELoss(nnUNetTrainer):
    def _build_loss(self):
        self.print_to_log_file("Using Dice + BCE + Focal Loss for training.")

        # Define the loss function
        loss = DiceFocalBCELoss()  # Adjust parameters if needed

        # Apply Deep Supervision if enabled
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

            # Avoid errors with Distributed Data Parallel (DDP)
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6  # Prevent unused parameter issues
            else:
                weights[-1] = 0  # Last output has no weight

            # Normalize weights
            weights = weights / weights.sum()

            # Wrap with Deep Supervision
            loss = DeepSupervisionWrapper(loss, weights)

        return loss