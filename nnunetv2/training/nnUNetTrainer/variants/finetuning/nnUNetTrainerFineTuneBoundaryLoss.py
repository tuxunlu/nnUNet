import os
from typing import Dict, Tuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from nnunetv2.training.loss.dice_focal_bce_boundary_loss import DiceFocalBCEBoundaryLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _unwrap_model(m):
    """
    Unwrap DDP (`.module`) and torch.compile (`._orig_mod`) to get the actual nn.Module.
    """
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


class nnUNetTrainerFineTuneBoundaryLoss(nnUNetTrainer):
    """
    Fine-tuning trainer that:
    - loads feature extractor weights from a contrastive pretraining checkpoint
      (`checkpoint['network_weights']` saved by nnUNetTrainerContrastive), and
    - fine-tunes with DiceFocalBCEBoundaryLoss using the standard nnU-Net training loop.

    Overfitting mitigations (tunable in __init__):
    - Lower LR for encoder (encoder_lr_scale): pretrained encoder is updated gently.
    - Optional encoder freeze for first N epochs (freeze_encoder_epochs): decoder adapts first.
    - Early stopping on validation Dice (early_stopping_patience).
    - Stronger weight decay and moderate foreground oversampling.

    Provide the checkpoint path via:
    - env var: CONTRASTIVE_PRETRAINED_CHECKPOINT=/path/to/checkpoint_best.pth
    - or init kwarg: contrastive_pretrained_checkpoint=/path/to/checkpoint_best.pth
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50
        self.initial_lr = 1e-4
        self.boundary_weight = 0.5
        # Regularization: moderate oversampling, strong weight decay
        self.oversample_foreground_percent = 0.3
        self.weight_decay = 1e-3

        # Overfitting mitigations
        self.encoder_lr_scale = 0.2  # Encoder LR = initial_lr * encoder_lr_scale (gentler updates)
        self.freeze_encoder_epochs = 5  # Freeze encoder for first N epochs (0 = disabled)
        self.early_stopping_patience = 10
        self.best_val_dice = None
        self.epochs_without_improvement = 0
        self.should_stop_training = False
        self.grad_clip_max_norm = 2.0  # Tighter grad clip than default 12 to limit large updates

        print(f"Contrastive pretrained checkpoint: {os.environ.get('CONTRASTIVE_PRETRAINED_CHECKPOINT')}")
        self.contrastive_pretrained_checkpoint = (
            os.environ.get("CONTRASTIVE_PRETRAINED_CHECKPOINT")
            or self.my_init_kwargs.get("contrastive_pretrained_checkpoint", None)
        )

    def initialize(self):
        super().initialize()
        if self.contrastive_pretrained_checkpoint:
            self.load_contrastive_pretrained_weights(self.contrastive_pretrained_checkpoint)
        # Optional: freeze encoder for first N epochs so only decoder adapts (reduces overfitting)
        if self.freeze_encoder_epochs > 0:
            mod = _unwrap_model(self.network)
            if hasattr(mod, "encoder"):
                for p in mod.encoder.parameters():
                    p.requires_grad = False
                self.print_to_log_file(
                    f"Encoder frozen for first {self.freeze_encoder_epochs} epochs (overfitting mitigation)."
                )
            else:
                self.print_to_log_file("WARNING: No .encoder on network; freeze_encoder_epochs ignored.")

    def configure_optimizers(self):
        mod = _unwrap_model(self.network)
        if hasattr(mod, "encoder") and hasattr(mod, "decoder"):
            # Build disjoint param groups: each parameter in exactly one group
            # (encoder/decoder can share params under torch.compile, causing "in more than one group" error)
            encoder_ids = {id(p) for p in mod.encoder.parameters()}
            decoder_ids = {id(p) for p in mod.decoder.parameters()}
            encoder_only = [p for p in mod.encoder.parameters() if id(p) not in decoder_ids]
            decoder_only = [p for p in mod.decoder.parameters() if id(p) not in encoder_ids]
            all_enc_dec_ids = encoder_ids | decoder_ids
            rest = [p for p in mod.parameters() if id(p) not in all_enc_dec_ids]
            encoder_lr = self.initial_lr * self.encoder_lr_scale
            param_groups = []
            if encoder_only:
                param_groups.append({"params": encoder_only, "lr": encoder_lr})
            decoder_rest = decoder_only + rest
            if decoder_rest:
                param_groups.append({"params": decoder_rest, "lr": self.initial_lr})
            if not param_groups:
                optimizer = torch.optim.SGD(
                    self.network.parameters(),
                    self.initial_lr,
                    weight_decay=self.weight_decay,
                    momentum=0.99,
                    nesterov=True,
                )
                self.print_to_log_file("Layer-wise LR skipped (no disjoint encoder/decoder params); using single LR.")
            else:
                optimizer = torch.optim.SGD(
                    param_groups,
                    weight_decay=self.weight_decay,
                    momentum=0.99,
                    nesterov=True,
                )
                self.print_to_log_file(
                    f"Layer-wise LR: encoder={encoder_lr:.2e} (n={len(encoder_only)}), decoder={self.initial_lr:.2e} (n={len(decoder_rest)})"
                )
        else:
            optimizer = torch.optim.SGD(
                self.network.parameters(),
                self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99,
                nesterov=True,
            )
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return optimizer, lr_scheduler

    def on_epoch_start(self):
        super().on_epoch_start()
        # Needed for DiceFocalBCEBoundaryLoss warmup schedule.
        if hasattr(self, "loss") and hasattr(self.loss, "set_epoch"):
            self.loss.set_epoch(self.current_epoch)
        # Early stopping: skip further epochs if we already decided to stop
        if self.should_stop_training and self.best_val_dice is not None:
            self.print_to_log_file(
                f"Early stopping active. Best validation Dice: {self.best_val_dice:.4f}"
            )

    def on_epoch_end(self):
        # Unfreeze encoder after freeze_encoder_epochs
        if self.freeze_encoder_epochs > 0 and self.current_epoch == self.freeze_encoder_epochs - 1:
            mod = _unwrap_model(self.network)
            if hasattr(mod, "encoder"):
                for p in mod.encoder.parameters():
                    p.requires_grad = True
                self.print_to_log_file(
                    f"Unfroze encoder at end of epoch {self.current_epoch} (overfitting mitigation)."
                )
        # Early stopping on validation Dice
        if hasattr(self.logger, "my_fantastic_logging") and "ema_fg_dice" in self.logger.my_fantastic_logging:
            if len(self.logger.my_fantastic_logging["ema_fg_dice"]) > 0:
                current_val_dice = self.logger.my_fantastic_logging["ema_fg_dice"][-1]
                if self.best_val_dice is None:
                    self.best_val_dice = current_val_dice
                    self.epochs_without_improvement = 0
                else:
                    if current_val_dice > self.best_val_dice + 1e-6:
                        self.best_val_dice = current_val_dice
                        self.epochs_without_improvement = 0
                        self.print_to_log_file(f"Validation Dice improved to {self.best_val_dice:.4f}")
                    else:
                        self.epochs_without_improvement += 1
                        self.print_to_log_file(
                            f"No improvement for {self.epochs_without_improvement}/{self.early_stopping_patience} epochs. "
                            f"Best: {self.best_val_dice:.4f}, Current: {current_val_dice:.4f}"
                        )
                        if self.epochs_without_improvement >= self.early_stopping_patience:
                            self.should_stop_training = True
                            self.print_to_log_file(
                                f"Early stopping: no improvement for {self.early_stopping_patience} epochs."
                            )
        super().on_epoch_end()
        if self.should_stop_training:
            self.num_epochs = self.current_epoch
            self.print_to_log_file(f"Training will stop after epoch {self.current_epoch}")

    def train_step(self, batch: dict) -> dict:
        """Same as base but with tighter gradient clipping to limit large updates (overfitting mitigation)."""
        from torch import autocast
        from nnunetv2.utilities.helpers import dummy_context
        data = batch["data"]
        target = batch["target"]
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)
        max_norm = getattr(self, "grad_clip_max_norm", 12.0)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def _build_loss(self):
        # This loss implementation is binary-only (uses foreground channel index 1).
        num_classes = self.label_manager.num_segmentation_heads
        if num_classes != 2:
            raise RuntimeError(
                f"DiceFocalBCEBoundaryLoss expects 2 classes (binary), but got num_segmentation_heads={num_classes}."
            )
        # Use the same loss hyperparameters as nnUNetTrainerDiceFocalBCEBoundary
        return DiceFocalBCEBoundaryLoss(
            boundary_weight=float(self.boundary_weight),
            dice_weight=0.5,
            bce_weight=0.5,
            focal_weight=0.5,
            alpha=0.5,
            gamma=2,
            warmup_epochs=20,
        )

    @staticmethod
    def _candidate_keys(pretrained_key: str) -> Tuple[str, ...]:
        """
        Generate likely key variants for matching.
        Contrastive checkpoints commonly use 'original_network.' prefix.
        """
        keys = [pretrained_key]
        if pretrained_key.startswith("module."):
            keys.append(pretrained_key[len("module."):])
        if pretrained_key.startswith("original_network."):
            keys.append(pretrained_key[len("original_network."):])
        if pretrained_key.startswith("module.original_network."):
            keys.append(pretrained_key[len("module.original_network."):])
        # Also try adding 'original_network.' in case current model has it (rare).
        keys.append(f"original_network.{pretrained_key}")
        # De-dup while preserving order
        out = []
        seen = set()
        for k in keys:
            if k not in seen:
                out.append(k)
                seen.add(k)
        return tuple(out)

    def load_contrastive_pretrained_weights(self, checkpoint_path: str) -> None:
        if not self.was_initialized:
            self.initialize()

        print(f"Loading contrastive pretrained weights from {checkpoint_path}")
        self.print_to_log_file(f"Loading contrastive pretrained weights from {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "network_weights" not in ckpt:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} does not contain 'network_weights'. "
                "Did you pass a nnUNetTrainerContrastive checkpoint?"
            )
        pretrained_sd: Dict[str, torch.Tensor] = ckpt["network_weights"]

        mod = _unwrap_model(self.network)
        current_sd = mod.state_dict()

        matched: Dict[str, torch.Tensor] = {}
        loaded = 0
        skipped = 0

        for pk, pv in pretrained_sd.items():
            found = False
            for ck in self._candidate_keys(pk):
                if ck in current_sd and current_sd[ck].shape == pv.shape:
                    matched[ck] = pv
                    loaded += 1
                    found = True
                    break
            if not found:
                skipped += 1

        mod.load_state_dict(matched, strict=False)

        self.print_to_log_file(f"Loaded {loaded} weights from contrastive pretraining")
        self.print_to_log_file(f"Skipped {skipped} keys (not found or shape mismatch)")
        if loaded == 0:
            self.print_to_log_file(
                "WARNING: Loaded 0 weights. This usually means the fine-tune network architecture/plans "
                "do not match the contrastive pretraining run (different dataset/config/plans)."
            )

        self.contrastive_pretrained_checkpoint = checkpoint_path