import numpy as np
import torch
from torch import nn
import inspect

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice_focal_bce_boundary_loss import DiceFocalBCEBoundaryLoss
from nnunetv2.training.loss.boundary_loss import BoundaryLoss
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerDualHeadBoundary(nnUNetTrainer):
    """
    Dual-head architecture:
      - First head: standard segmentation logits (C channels).
      - Second head: boundary logits (2 channels: background / boundary foreground).

    Loss:
      - Segmentation head: Dice + BCE + Focal (using DiceFocalBCEBoundaryLoss with boundary_weight=0).
      - Boundary head: BoundaryLoss on the boundary logits, supervised from the same GT mask.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        # keep unpack_dataset in the signature so nnUNet can store it in my_init_kwargs,
        # but do NOT forward it because the base nnUNetTrainer.__init__ does not accept it
        super().__init__(plans, configuration, fold, dataset_json, device=device)

        # Number of extra channels for the boundary branch (bg + boundary fg)
        self.num_boundary_channels = 2
        self.boundary_loss_weight = 0.1

        # Slightly more conservative training defaults (you can tune as needed)
        self.num_epochs = 100
        self.oversample_foreground_percent = 0.6
        self.weight_decay = 2e-4
        self.initial_lr = 1e-3

        # These will be initialized lazily once label_manager is available
        self.seg_loss = None
        self.boundary_loss_fn = None

    # -------------------------------------------------------------------------
    # 1. Network architecture: add extra boundary head channels
    # -------------------------------------------------------------------------
    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """
        Build the base nnU-Net architecture but request TWO additional output channels
        that will serve as the boundary branch (binary classification: bg / boundary).
        """
        # Increase output channels by 2 (boundary head: 2 channels)
        dual_head_num_output_channels = num_output_channels + 2

        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            dual_head_num_output_channels,
            enable_deep_supervision,
        )

    # -------------------------------------------------------------------------
    # 2. Loss construction
    # -------------------------------------------------------------------------
    def _maybe_build_losses(self):
        if self.seg_loss is not None and self.boundary_loss_fn is not None:
            return

        # C = number of segmentation channels defined by the dataset / plans
        num_seg_channels = self.label_manager.num_segmentation_heads

        # Segmentation loss: use DiceFocalBCEBoundaryLoss but disable its internal boundary term
        # because we handle boundary supervision with a separate head + BoundaryLoss.
        self.seg_loss = DiceFocalBCEBoundaryLoss(
            boundary_weight=0.0,  # boundary handled by second head
            dice_weight=0.6,
            bce_weight=0.2,
            focal_weight=0.2,
            alpha=0.5,
            gamma=4,
            warmup_epochs=20,
        )

        # Pure boundary loss that uses logits of the boundary head and the GT mask
        self.boundary_loss_fn = BoundaryLoss()

        # Cache for convenience
        self._num_seg_channels = num_seg_channels

    def _build_loss(self):
        """
        nnUNet base class expects this, but we do all logic directly in train_step / validation_step.
        We return a dummy module so self.loss is defined.
        """
        self._maybe_build_losses()

        class _DummyLoss(nn.Module):
            def forward(self, x, y):
                # Real loss is computed in train_step / validation_step
                return torch.tensor(0.0, device=x.device if isinstance(x, torch.Tensor) else y.device)

        return _DummyLoss()

    # -------------------------------------------------------------------------
    # 3. Training step with dual-branch loss
    # -------------------------------------------------------------------------
    def train_step(self, batch: dict) -> dict:
        self._maybe_build_losses()

        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True):
            net_output = self.network(data)

            # Split segmentation and boundary branches
            if self.enable_deep_supervision and isinstance(net_output, (list, tuple)):
                # Highest resolution output at index 0
                seg_output_main = net_output[0][:, : self._num_seg_channels, ...]
                boundary_output_main = net_output[0][:, self._num_seg_channels :, ...]
                target_main = target[0]
            else:
                seg_output_main = net_output[:, : self._num_seg_channels, ...]
                boundary_output_main = net_output[:, self._num_seg_channels :, ...]
                target_main = target

            # Segmentation loss (mask prediction)
            seg_loss_val = self.seg_loss(seg_output_main, target_main)

            # Boundary loss (boundary branch prediction)
            boundary_loss_val = self.boundary_loss_fn(boundary_output_main, target_main)
            total_loss = seg_loss_val + self.boundary_loss_weight * boundary_loss_val

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
            self.optimizer.step()

        return {
            "loss": total_loss.detach().cpu().numpy(),
            "seg_loss": seg_loss_val.detach().cpu().numpy(),
            "boundary_loss": boundary_loss_val.detach().cpu().numpy(),
        }

    # -------------------------------------------------------------------------
    # 4. Validation step: metrics on mask branch, same dual loss
    # -------------------------------------------------------------------------
    def validation_step(self, batch: dict) -> dict:
        self._maybe_build_losses()

        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast on GPU only
        context = (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )

        with context:
            net_output = self.network(data)

            # Split branches (use highest-res output if deep supervision)
            if self.enable_deep_supervision and isinstance(net_output, (list, tuple)):
                seg_output = net_output[0][:, : self._num_seg_channels, ...]
                boundary_output = net_output[0][:, self._num_seg_channels :, ...]
                target_main = target[0]
            else:
                seg_output = net_output[:, : self._num_seg_channels, ...]
                boundary_output = net_output[:, self._num_seg_channels :, ...]
                target_main = target

            seg_loss_val = self.seg_loss(seg_output, target_main)
            boundary_loss_val = self.boundary_loss_fn(boundary_output, target_main)
            total_loss = seg_loss_val + self.boundary_loss_weight * boundary_loss_val

        # ---- Online evaluation dice, computed from segmentation head only ----
        if self.enable_deep_supervision and isinstance(net_output, (list, tuple)):
            seg_for_eval = seg_output  # already highest-res
            target_for_eval = target_main
        else:
            seg_for_eval = seg_output
            target_for_eval = target_main

        # Shape: [B, C, ...]
        axes = [0] + list(range(2, seg_for_eval.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(seg_for_eval) > 0.5).long()
        else:
            probs = torch.softmax(seg_for_eval, dim=1)
            fg_prob = probs[:, 1, ...]  # foreground prob for binary seg
            predicted_seg = (fg_prob > 0.5).long()[:, None]
            predicted_segmentation_onehot = torch.zeros_like(probs, device=probs.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, predicted_seg, 1)
            predicted_segmentation_onehot[:, 0, ...] = 1 - predicted_segmentation_onehot[:, 1, ...]

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target_for_eval != self.label_manager.ignore_label).float()
                target_for_eval[target_for_eval == self.label_manager.ignore_label] = 0
            else:
                if target_for_eval.dtype == torch.bool:
                    mask = ~target_for_eval[:, -1:]
                else:
                    mask = 1 - target_for_eval[:, -1:]
                target_for_eval = target_for_eval[:, :-1]
        else:
            mask = None

        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot,
            target_for_eval,
            axes=axes,
            mask=mask,
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": total_loss.detach().cpu().numpy(),
            "seg_loss": seg_loss_val.detach().cpu().numpy(),
            "boundary_loss": boundary_loss_val.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    # -------------------------------------------------------------------------
    # 5. Override perform_actual_validation to wrap network for inference
    # -------------------------------------------------------------------------
    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        Override to wrap the network so it only returns segmentation channels during inference.
        The predictor expects num_segmentation_heads channels, but our network outputs
        num_segmentation_heads + 2 (boundary channels).
        """
        self._maybe_build_losses()
        
        # Create a wrapper module that slices only segmentation channels
        class SegmentationOnlyWrapper(nn.Module):
            def __init__(self, network, num_seg_channels):
                super().__init__()
                self.network = network
                self.num_seg_channels = num_seg_channels
            
            def forward(self, x):
                output = self.network(x)
                # During inference, deep supervision is disabled, so we get a single tensor
                # But handle both cases for safety
                if isinstance(output, (list, tuple)):
                    # Return list with only segmentation channels
                    return [out[:, :self.num_seg_channels, ...] for out in output]
                else:
                    # Single output: slice segmentation channels
                    return output[:, :self.num_seg_channels, ...]
        
        # Wrap the current network (which may be DDP-wrapped)
        # The wrapper will handle DDP internally since it calls self.network
        wrapped_network = SegmentationOnlyWrapper(self.network, self._num_seg_channels)
        wrapped_network.eval()
        
        # Temporarily replace self.network with wrapped version
        original_network = self.network
        self.network = wrapped_network
        
        try:
            # Call parent's perform_actual_validation
            super().perform_actual_validation(save_probabilities)
        finally:
            # Restore original network
            self.network = original_network


# ==========================================
# SIGNATURE PATCH (required for CLI)
# ==========================================
nnUNetTrainerDualHeadBoundary.__init__.__signature__ = inspect.signature(nnUNetTrainer.__init__)