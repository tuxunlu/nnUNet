import numpy as np
import torch
import inspect
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice_focal_bce_sdf_loss import DiceFocalBCESDFLoss


class nnUNetTrainerDiceFocalBCESDF(nnUNetTrainer):
    """Trainer using Dice + Focal + BCE + SDF (signed distance function) loss."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device=device)

        self.num_epochs = 100
        self.oversample_foreground_percent = 0.8
        self.weight_decay = 2e-4
        self.initial_lr = 1e-3
        self.early_stopping_patience = 10
        self.best_val_dice = None
        self.epochs_without_improvement = 0
        self.should_stop_training = False

    def _build_loss(self):
        self.print_to_log_file(
            "Using Dice + BCE + Focal + SDF (signed distance) Loss for training."
        )

        loss = DiceFocalBCESDFLoss(
            sdf_weight=0.5,
            dice_weight=0.5,
            bce_weight=0.5,
            focal_weight=0.5,
            alpha=0.5,
            gamma=2,
            warmup_epochs=10,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])

            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def on_epoch_start(self):
        super().on_epoch_start()
        if hasattr(self.loss, "set_epoch"):
            self.loss.set_epoch(self.current_epoch)

        if self.should_stop_training:
            self.print_to_log_file(
                f"Early stopping triggered after {self.current_epoch} epochs. "
                f"Best validation Dice: {self.best_val_dice:.4f}"
            )

    def on_epoch_end(self):
        if (
            hasattr(self.logger, "my_fantastic_logging")
            and "ema_fg_dice" in self.logger.my_fantastic_logging
        ):
            if len(self.logger.my_fantastic_logging["ema_fg_dice"]) > 0:
                current_val_dice = self.logger.my_fantastic_logging["ema_fg_dice"][-1]

                if self.best_val_dice is None:
                    self.best_val_dice = current_val_dice
                    self.epochs_without_improvement = 0
                else:
                    if current_val_dice > self.best_val_dice + 1e-6:
                        improvement = current_val_dice - self.best_val_dice
                        self.best_val_dice = current_val_dice
                        self.epochs_without_improvement = 0
                        self.print_to_log_file(
                            f"Validation Dice improved by {improvement:.4f} to {self.best_val_dice:.4f}"
                        )
                    else:
                        self.epochs_without_improvement += 1
                        self.print_to_log_file(
                            f"No improvement for {self.epochs_without_improvement}/{self.early_stopping_patience} epochs. "
                            f"Best: {self.best_val_dice:.4f}, Current: {current_val_dice:.4f}"
                        )

                        if self.epochs_without_improvement >= self.early_stopping_patience:
                            self.should_stop_training = True
                            self.print_to_log_file(
                                f"Early stopping: No improvement for {self.early_stopping_patience} epochs. "
                                f"Stopping training at epoch {self.current_epoch}"
                            )

        super().on_epoch_end()

        if self.should_stop_training:
            self.num_epochs = self.current_epoch
            self.print_to_log_file(
                f"Training will stop after epoch {self.current_epoch}"
            )


nnUNetTrainerDiceFocalBCESDF.__init__.__signature__ = inspect.signature(
    nnUNetTrainer.__init__
)
