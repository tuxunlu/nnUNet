# nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerSupConPretrain.py
import inspect
import os
from typing import Optional, Tuple, Union, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from torch._dynamo import OptimizedModule
from time import time
from os.path import join

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.pixel_contrastive import PixelWiseSupConLoss
from nnunetv2.training.loss.utils.projection_head import ProjectionHead2D, ProjectionHead3D
from nnunetv2.utilities.helpers import dummy_context

Tensor = torch.Tensor


class nnUNetTrainerSupConPretrain(nnUNetTrainer):
    """
    Contrastive-pretraining trainer for nnU-Net v2 following:
      Zhao et al., "Contrastive Learning for Label-Efficient Semantic Segmentation", ICCV 2021.

    Stage A (this trainer): pretrain the encoder with a pixel-wise supervised contrastive loss using two views.
    Stage B (standard trainer): fine-tune with CE/Dice on the usual nnU-Net pipeline, loading the Stage A weights.

    Notes
    -----
    * Deep supervision is DISABLED for this stage.
    * Validation logs mean contrastive loss only (no Dice).
    * The projection head is kept outside the checkpoint; only network weights are saved for fine-tuning.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # Read SupCon knobs from env (or hardcode your defaults here)
        import os
        self.supcon_temperature   = float(os.getenv('SUPCON_TEMPERATURE', '0.07'))
        self.supcon_variant       =       os.getenv('SUPCON_VARIANT', 'cross')   # 'within' or 'cross'
        self.supcon_down_hw       = int(  os.getenv('SUPCON_DOWN_HW', '64'))
        self.supcon_max_pixels    = int(  os.getenv('SUPCON_MAX_PIXELS', '10000'))
        self.supcon_distortion_p  = float(os.getenv('SUPCON_DISTORTION_P', '0.8'))

        # call base WITH the exact base signature
        super().__init__(plans, configuration, fold, dataset_json, device=device)

        # contrastive pretrain stage: no deep supervision
        self.enable_deep_supervision = False

        # record our knobs for reproducibility in the checkpoint meta
        self.my_init_kwargs.update({
            'supcon_temperature':  self.supcon_temperature,
            'supcon_variant':      self.supcon_variant,
            'supcon_down_hw':      self.supcon_down_hw,
            'supcon_max_pixels':   self.supcon_max_pixels,
            'supcon_distortion_p': self.supcon_distortion_p,
        })

        # will be set in initialize()
        self._feat_hook = None
        self._last_feats = None
        self._proj = None
        self._is_3d = None

        # build loss cfg now that label_manager exists
        self._supcon_cfg = dict(
            temperature=self.supcon_temperature,
            variant=self.supcon_variant,
            down_hw=self.supcon_down_hw,
            max_pixels=self.supcon_max_pixels,
            ignore_index=self.label_manager.ignore_label
        )

        self._best_val_loss = float('inf')

    # ----------------------------
    # Initialization & plumbing
    # ----------------------------
    def initialize(self):
        """
        Build network/optimizer/scheduler/loss via base class, then attach a feature hook
        and create the projection head by probing one dummy forward.
        """
        if self.was_initialized:
            raise RuntimeError("initialize called twice")
        super().initialize()

        # Install hooks on the (likely) last encoder-like module BEFORE any further graph captures.
        self._register_feature_hook()

        # Probe shapes with a dummy forward to size the projection head
        self._prepare_projection_head()

        # Replace loss with our SupCon loss object (no DeepSupervisionWrapper since DS is disabled)
        self.loss = self._build_loss()

    def _build_loss(self):
        self.print_to_log_file("Using Pixel-wise Supervised Contrastive Loss for pretraining "
                               f"(variant={self._supcon_cfg['variant']}, tau={self._supcon_cfg['temperature']}, "
                               f"down_hw={self._supcon_cfg['down_hw']}, max_pixels={self._supcon_cfg['max_pixels']})")
        return PixelWiseSupConLoss(**self._supcon_cfg)

    # Robustly pick a module to hook (something 'encoder'-ish or last big block)
    def _find_encoder_like_module(self, root: nn.Module) -> nn.Module:
        named = list(root.named_modules())
        # prefer items with 'encoder' or 'backbone' in their qualified name; fall back to the deepest large module
        candidates = [m for n, m in named if ('encoder' in n.lower() or 'backbone' in n.lower())]
        if len(candidates) == 0:
            # fallback: pick a late module that has many parameters (heuristic)
            sized = sorted([(n, m, sum(p.numel() for p in m.parameters() if p.requires_grad))
                            for n, m in named], key=lambda x: x[2])
            candidates = [m for _, m, _ in sized[-5:]] if sized else [root]
        # pick the last (deepest) candidate
        return candidates[-1]

    def _register_feature_hook(self):
        mod = self._unwrap_model(self.network)
        target = self._find_encoder_like_module(mod)

        def _hook(_mod, _in, out):
            self._last_feats = self._select_tensor_from(out)

        self._feat_hook = target.register_forward_hook(_hook)
        self.print_to_log_file(f"Feature hook attached to: {target.__class__.__name__}")

    def _prepare_projection_head(self):
        self.network.eval()
        with torch.no_grad():
            b = 1
            shp = self.configuration_manager.patch_size
            dummy = torch.randn(b, self.num_input_channels, *shp, device=self.device)
            _ = self.network(dummy)  # fills self._last_feats via hook

        feats = self._last_feats
        if feats is None or not torch.is_tensor(feats):
            # Fallback: use the network output as features (will still pretrain something meaningful)
            self.print_to_log_file("WARNING: Hook didn't capture encoder features; using logits for projection.")
            with torch.no_grad():
                b = 1
                shp = self.configuration_manager.patch_size
                dummy = torch.randn(b, self.num_input_channels, *shp, device=self.device)
                logits = self.network(dummy)
                if isinstance(logits, (list, tuple)):
                    feats = logits[0]
                else:
                    feats = logits

        self._is_3d = (feats.ndim == 5)
        C = feats.shape[1]
        self.network.train()

        if self._is_3d:
            self._proj = ProjectionHead3D(C).to(self.device)
        else:
            self._proj = ProjectionHead2D(C).to(self.device)

        self.print_to_log_file(f"Projection head initialized: {'3D' if self._is_3d else '2D'} with in_ch={C}")

    @staticmethod
    def _unwrap_model(m: nn.Module) -> nn.Module:
        if isinstance(m, nn.DataParallel) or isinstance(m, nn.parallel.DistributedDataParallel):
            m = m.module
        if isinstance(m, OptimizedModule):
            m = m._orig_mod
        return m

    @staticmethod
    def _select_tensor_from(x: Any) -> Optional[Tensor]:
        """
        Given an arbitrary module output (Tensor / list / tuple / dict),
        pick a reasonable feature tensor (largest spatial tensor).
        """
        if torch.is_tensor(x):
            return x
        cand: List[Tensor] = []
        if isinstance(x, (list, tuple)):
            for t in x:
                if torch.is_tensor(t) and t.ndim >= 4:
                    cand.append(t)
        elif isinstance(x, dict):
            for v in x.values():
                if torch.is_tensor(v) and v.ndim >= 4:
                    cand.append(v)
        if len(cand) == 0:
            return None
        # choose the tensor with the largest number of elements
        cand.sort(key=lambda t: t.numel())
        return cand[-1]

    # ----------------------------
    # Distort second view (lightweight, label-preserving)
    # ----------------------------
    @staticmethod
    def _distort(x: Tensor, p: float) -> Tensor:
        if p <= 0:
            return x
        if torch.rand(()) > p:
            return x
        noise = 0.05 * torch.randn_like(x)
        # per-sample scale/bias (broadcast across channels & spatial dims)
        shape = [x.shape[0], 1] + [1] * (x.ndim - 2)
        scale = 0.1 * (2 * torch.rand(shape, device=x.device) - 1.0) + 1.0
        bias = 0.1 * (2 * torch.rand(shape, device=x.device) - 1.0)
        return x * scale + bias + noise

    
    def _standardize_labels(self, target: torch.Tensor, ref_feats: torch.Tensor) -> torch.Tensor:
        """
        Convert nnU-Net targets to an integer labelmap with shape [B, H, W] (2D) or [B, D, H, W] (3D).
        Handles both classical labelmaps ([B,1,...]) and regions training where target is one-hot
        and the last channel encodes the ignore region.

        Returns y (long) with ignore pixels set to self._supcon_cfg['ignore_index'] (if not None).
        """
        y = target
        ignore_index = self._supcon_cfg.get('ignore_index', None)

        if self.plans_manager.get_label_manager(self.dataset_json).has_regions:
            # Regions: target is one-hot per region + an 'ignore' channel at the end (see nnU-Net docs/issues)
            # Ref: nnU-Net discussions on ignore label & regions usage
            # - Ignore label feature & masking concept
            # - Regions training packs ignore mask in last channel
            # Convert to integer labels by argmax over region channels (exclude last ignore channel).
            if y.dtype == torch.bool:
                # boolean one-hot (rare): last channel True/False indicates NOT-ignored
                not_ignored_mask = ~y[:, -1:]         # shape [B,1,...]
            else:
                # numeric: last channel is 1 for labeled, 0 for ignore; keep in [0,1]
                not_ignored_mask = y[:, -1:].clone()  # [B,1,...] in {0,1}

            region_logits = y[:, :-1]                 # [B, R, ...]
            y_idx = region_logits.argmax(1)           # [B, ...] integers in [0, R-1]

            if ignore_index is not None:
                # set ignored positions to ignore_index
                ignore_mask = (not_ignored_mask == 0)
                if ignore_mask.dtype != torch.bool:
                    ignore_mask = ignore_mask.bool()
                y_idx = y_idx.long()
                y_idx[ignore_mask.squeeze(1)] = ignore_index

            y = y_idx.long()

        else:
            # Classical labelmaps: squeeze singleton channel if present -> [B, ..., ...]
            if y.ndim >= 4 and y.shape[1] == 1:
                y = y[:, 0]
            y = y.long()

        # Sanity: match spatial rank to ref_feats (loss will resample spatial size later)
        # ref_feats: [B, C, H, W] or [B, C, D, H, W]
        # y must be [B, H, W] or [B, D, H, W] (no channel dim!)
        # No further action needed; we only ensure no channel axis remains.
        return y

    # ----------------------------
    # Train / Val steps (contrastive only)
    # ----------------------------
    def _embed_and_labels(self, data: torch.Tensor, target: Union[torch.Tensor, List[torch.Tensor]]):
        """
        Returns three (features, labels) tuples:
        (I, y), (Î, y), (J, yJ)
        where Î is a light distortion of I, and J is a batch-rolled partner (for cross-image positives).
        Labels are standardized to integer maps with shape [B, H, W] or [B, D, H, W].
        """
        # Deep supervision could still hand us a list, be defensive
        if isinstance(target, list):
            target = target[0]
        target = target.to(self.device, non_blocking=True)

        # ---------------- View I ----------------
        _ = self.network(data)
        feats_I = self._last_feats
        if feats_I is None:
            out = self.network(data)
            feats_I = out[0] if isinstance(out, (list, tuple)) else out
        z_I = self._proj(feats_I)
        y = self._standardize_labels(target, feats_I)        # <--- fix: strip channel / handle regions+ignore

        # ---------------- View Î (distorted) ----------------
        data2 = self._distort(data, self.supcon_distortion_p)
        _ = self.network(data2)
        feats_Ihat = self._last_feats
        if feats_Ihat is None:
            out2 = self.network(data2)
            feats_Ihat = out2[0] if isinstance(out2, (list, tuple)) else out2
        z_Ihat = self._proj(feats_Ihat)
        # same y for Î (distortion is label-preserving)

        # ---------------- Cross-image partner J ----------------
        dataJ = torch.roll(data, shifts=1, dims=0)
        _ = self.network(dataJ)
        feats_J = self._last_feats
        if feats_J is None:
            outJ = self.network(dataJ)
            feats_J = outJ[0] if isinstance(outJ, (list, tuple)) else outJ
        z_J = self._proj(feats_J)
        yJ = torch.roll(y, shifts=1, dims=0)

        return (z_I, y), (z_Ihat, y), (z_J, yJ)

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # autocast like base class
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            (fI, yI), (fIhat, yIhat), (fJ, yJ) = self._embed_and_labels(data, target)
            loss = self.loss(fI, yI, fIhat, yIhat, fJ, yJ)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': loss.detach().cpu().numpy()}

    def on_validation_epoch_start(self):
        self.network.eval()

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            (fI, yI), (fIhat, yIhat), (fJ, yJ) = self._embed_and_labels(data, target)
            loss = self.loss(fI, yI, fIhat, yIhat, fJ, yJ)

        # We only log loss in this stage
        return {'loss': loss.detach().cpu().numpy()}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """
        Override base method: only aggregate and log validation losses (no Dice in pretraining).
        """
        losses = [o['loss'] for o in val_outputs]
        if self.is_ddp:
            world_size = torch.distributed.get_world_size()
            gathered = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered, losses)
            losses = np.hstack(gathered)
        loss_here = float(np.mean(losses))
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def on_train_end(self):
        # clean up hook
        try:
            if self._feat_hook is not None:
                self._feat_hook.remove()
                self._feat_hook = None
        finally:
            super().on_train_end()

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_val_loss is None or self.logger.my_fantastic_logging['val_losses'][-1] < self._best_val_loss:
            self._best_val_loss = self.logger.my_fantastic_logging['val_losses'][-1]
            self.print_to_log_file(f"Yayy! New best val loss: {np.round(self._best_val_loss, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
