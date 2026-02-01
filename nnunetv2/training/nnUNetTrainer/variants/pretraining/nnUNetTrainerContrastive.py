import torch
from torch import nn
from typing import Union, Tuple, List
import numpy as np
from time import time
from batchgenerators.utilities.file_and_folder_operations import join
from tqdm import tqdm

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.pixel_contrastive_loss import PixelContrastiveLoss
from nnunetv2.utilities.collate_outputs import collate_outputs


class ProjectionHead(nn.Module):
    """
    Simple pixel-wise projection head for contrastive pretraining.

    Follows the "MLP on top of encoder" idea from pixel-wise supervised
    contrastive segmentation:
      encoder features  ->  1x1 conv (C -> C) + ReLU  ->  1x1 conv (C -> D)
      then L2-normalize along channel dimension.

    This is applied at the highest-resolution decoder feature map so that
    embeddings stay aligned with pixel/voxel locations.
    """

    def __init__(self, in_channels: int, out_channels: int = 256, is_3d: bool = False):
        super().__init__()
        conv = nn.Conv3d if is_3d else nn.Conv2d
        self.proj = nn.Sequential(
            conv(in_channels, in_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            conv(in_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # unit-norm embeddings as in InfoNCE-style contrastive learning
        return torch.nn.functional.normalize(x, p=2, dim=1)


class ContrastivePretrainingNetwork(nn.Module):
    """
    Wrapper around a standard nnU-Net architecture that exposes one or more
    dense pixel/voxel embedding maps instead of class logits.

    - Reuses the encoder/decoder as feature extractor.
    - Optionally attaches projection heads at several decoder scales
      (multi-scale contrastive pretraining, similar in spirit to
      `nnUNetTrainerBoundaryContrastive`).
    - Projection heads are discarded after pretraining; only the
    - encoder/decoder weights are kept for fine-tuning.
    """

    def __init__(
        self,
        original_network: nn.Module,
        num_input_channels: int,
        patch_size,
        proj_dim: int = 256,
        use_multi_scale: bool = True,
        num_scales: int = 3,
    ):
        super().__init__()
        self.original_network = original_network
        self.use_multi_scale = use_multi_scale
        self.num_scales = num_scales
        is_3d = len(patch_size) == 3

        # Probe the decoder once with a dummy patch to infer channel count and
        # how many decoder scales are available.
        try:
            any_param = next(self.original_network.parameters())
            device = any_param.device
        except StopIteration:
            device = torch.device("cpu")

        dummy = torch.zeros(
            (1, num_input_channels, *patch_size),
            dtype=torch.float32,
            device=device,
        )
        training_state = self.original_network.training
        self.original_network.eval()
        with torch.no_grad():
            try:
                skips = self.original_network.encoder(dummy)
                dec_out = self.original_network.decoder(skips)
                if isinstance(dec_out, (list, tuple)):
                    n_scales_avail = len(dec_out)
                    high_res_feats = dec_out[0]
                else:
                    n_scales_avail = 1
                    high_res_feats = dec_out
                in_channels = int(high_res_feats.shape[1])
                self.num_scales = min(self.num_scales, n_scales_avail) if self.use_multi_scale else 1
            except Exception as e:
                raise RuntimeError(f"ContrastivePretrainingNetwork initialization failed: {e}")
            finally:
                self.original_network.train(training_state)

        # One projection head per used scale (or just one if use_multi_scale=False).
        self.projection_heads = nn.ModuleList(
            [ProjectionHead(in_channels, out_channels=proj_dim, is_3d=is_3d) for _ in range(self.num_scales)]
        )

    def forward(self, x: torch.Tensor):
        """
        Returns:
        - If `use_multi_scale=True` and decoder provides multiple outputs:
            list of embedding maps `[emb_s0, emb_s1, ...]` (highest resolution first)
        - Otherwise:
            single embedding map tensor `[B, D, ...]` at highest resolution.
        """
        skips = self.original_network.encoder(x)
        dec_out = self.original_network.decoder(skips)

        if isinstance(dec_out, (list, tuple)) and self.use_multi_scale:
            embeddings: List[torch.Tensor] = []
            for i in range(self.num_scales):
                feats_i = dec_out[i]
                emb_i = self.projection_heads[i](feats_i)
                embeddings.append(emb_i)
            return embeddings
        else:
            if isinstance(dec_out, (list, tuple)):
                feats = dec_out[0]
            else:
                feats = dec_out
            return self.projection_heads[0](feats)


class nnUNetTrainerContrastive(nnUNetTrainer):
    """
    Pixel-wise supervised contrastive pretraining for nnU-Net.

    Design:
    - Uses the standard nnU-Net backbone (encoder + decoder).
    - Adds a temporary projection head to produce unit-normalized
      embeddings for each pixel/voxel.
    - Optimizes a label-based InfoNCE loss (PixelContrastiveLoss):
        * anchors are foreground pixels
        * positives are pixels with the same label
        * negatives are primarily hard background pixels around the lesions
          (ring-based mining) + some random background.
    - Strong foreground oversampling on the data loader so that small,
      sparse lesions appear frequently in training patches.

    After pretraining:
    - Only the encoder/decoder weights are saved in the checkpoint
      (projection head is dropped).
    - Fine-tuning trainers can load `checkpoint['network_weights']`
      (see nnUNetTrainerFineTuneBoundaryLoss).
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Contrastive pretraining is a short warm-up stage
        self.enable_deep_supervision = False
        self.num_epochs = 20
        self.initial_lr = 1e-2

        # For highly imbalanced, small & sparse lesions:
        # - aggressively oversample patches that contain foreground so that
        #   anchors/positives exist in most batches.
        self.oversample_foreground_percent = 0.8

        # Multi-scale contrastive pretraining (decoder feature hierarchy).
        # We keep this relatively lightweight: top 3 decoder scales at most.
        self.use_multi_scale = True
        self.num_scales = 3

        # Track best validation contrastive loss (lower is better)
        self.best_val_loss = float("inf")

    def _do_i_compile(self) -> bool:
        """
        Keep torch.compile disabled here.

        The pretraining stage is relatively short and the custom wrapper
        network with projection head tends to interact poorly with some
        compile backends, so we prefer robustness over marginal speed.
        """
        return False

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """
        Build the standard nnU-Net architecture and wrap it into a
        `ContrastivePretrainingNetwork` that exposes dense embeddings
        instead of logits.
        """
        backbone = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        patch_size = self.configuration_manager.patch_size
        return ContrastivePretrainingNetwork(
            backbone,
            num_input_channels,
            patch_size,
            proj_dim=256,
            use_multi_scale=self.use_multi_scale,
            num_scales=self.num_scales,
        )

    def _build_loss(self) -> nn.Module:
        """
        Pixel-wise supervised contrastive loss (InfoNCE).

        Smart sampling for small/sparse lesions is handled internally by
        `PixelContrastiveLoss`:
        - All available foreground pixels are included up to roughly half
          of the sampling budget (`max_samples`), so tiny lesions still
          contribute strongly.
        - Negatives are drawn with a strong preference for background
          pixels in a ring around lesions (hard negatives) plus a smaller
          set of random background.
        - Labels at `ignore_index` are completely masked out.

        Hyperparameters are tuned for small, sparse 3D lesions on typical
        nnU-Net full-resolution patches:
        - `max_samples`: capped to keep memory reasonable but large
          enough to represent boundary statistics even when lesions are
          small.
        - `hard_negative_radius`: a few feature voxels around lesions,
          corresponding to a thick anatomical boundary band in input
          space.
        - `hard_negative_fraction`: heavily biases the negative pool to
          those near-lesion voxels to sharpen boundary discrimination.
        """
        ignore_index = self.label_manager.ignore_label if self.label_manager.has_ignore_label else 255
        return PixelContrastiveLoss(
            temperature=0.07,
            ignore_index=ignore_index,
            # Focus computation on a compact, boundary-heavy subset of pixels.
            max_samples=3072,
            anchor_chunk_size=768,
            # Lesions are small: a moderate radius in feature space covers
            # several voxels around each lesion as hard negatives.
            hard_negative_radius=6,
            # Strongly prefer hard negatives (near lesions) over random
            # background; keeps gradients focused on the difficult
            # boundary region.
            hard_negative_fraction=0.9,
            foreground_label=1,
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        # Explicitly do nothing: deep supervision is off during contrastive pretraining.
        return None

    # ------------------------------------------------------------------
    # Training & validation loop (contrastive-only)
    # ------------------------------------------------------------------
    def on_epoch_start(self):
        """
        Standard nnU-Net bookkeeping plus a per-epoch tqdm progress bar.
        """
        super().on_epoch_start()
        if self.local_rank == 0:
            self.epoch_pbar = tqdm(
                total=self.num_iterations_per_epoch,
                desc=f"Epoch {self.current_epoch}",
                leave=True,
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            )

    def train_step(self, batch: dict) -> dict:
        """
        Single training iteration:
        - forward encoder+decoder+projection head
        - compute pixel-wise contrastive loss using ground-truth labels
        - backprop & optimizer step

        Note: `PixelContrastiveLoss` performs its own within-patch pixel
        sampling, focusing on (1) all/most lesion pixels and (2) hard
        background near lesion boundaries. Combined with the high
        `oversample_foreground_percent`, this is tailored to very small
        and sparse lesions.
        """
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
            labels = target[0]
        else:
            target = target.to(self.device, non_blocking=True)
            labels = target

        self.optimizer.zero_grad()
        embeddings = self.network(data)

        # Multi-scale: average contrastive loss across decoder scales.
        if isinstance(embeddings, (list, tuple)):
            n_scales = len(embeddings)
            # Heavier weight on highest resolution; exponential decay for deeper scales.
            weights = torch.tensor([1.0 / (2 ** i) for i in range(n_scales)], device=self.device, dtype=torch.float32)
            weights = weights / weights.sum()
            loss = embeddings[0].new_tensor(0.0)
            for i, emb_i in enumerate(embeddings):
                loss = loss + weights[i] * self.loss(emb_i, labels)
        else:
            loss = self.loss(embeddings, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
        self.optimizer.step()

        if self.local_rank == 0 and hasattr(self, "epoch_pbar"):
            self.epoch_pbar.set_postfix(loss=f"{loss.item():.4f}")
            self.epoch_pbar.update(1)

        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        """
        Validation uses the exact same contrastive objective; we only
        log the mean loss per epoch (no Dice here).
        """
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = target[0].to(self.device, non_blocking=True)
        else:
            target = target.to(self.device, non_blocking=True)

        with torch.no_grad():
            embeddings = self.network(data)
            if isinstance(embeddings, (list, tuple)):
                n_scales = len(embeddings)
                weights = torch.tensor(
                    [1.0 / (2 ** i) for i in range(n_scales)],
                    device=self.device,
                    dtype=torch.float32,
                )
                weights = weights / weights.sum()
                loss = embeddings[0].new_tensor(0.0)
                for i, emb_i in enumerate(embeddings):
                    loss = loss + weights[i] * self.loss(emb_i, target)
            else:
                loss = self.loss(embeddings, target)

        return {"loss": loss.detach().cpu().numpy()}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """
        Aggregate validation losses across workers and log the mean.
        """
        outputs_collated = collate_outputs(val_outputs)
        loss_here = float(np.mean(outputs_collated["loss"]))
        self.logger.log("val_losses", loss_here, self.current_epoch)

    def on_epoch_end(self):
        """
        Standard nnU-Net end-of-epoch bookkeeping plus:
        - best-loss tracking
        - periodic checkpointing that only stores encoder/decoder weights.
        """
        if self.local_rank == 0 and hasattr(self, "epoch_pbar"):
            self.epoch_pbar.close()
            del self.epoch_pbar

        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)

        # Log epoch-wise train/val loss
        self.print_to_log_file(
            "train_loss",
            np.round(self.logger.my_fantastic_logging["train_losses"][-1], decimals=4),
        )
        self.print_to_log_file(
            "val_loss",
            np.round(self.logger.my_fantastic_logging["val_losses"][-1], decimals=4),
        )

        epoch_time = (
            self.logger.my_fantastic_logging["epoch_end_timestamps"][-1]
            - self.logger.my_fantastic_logging["epoch_start_timestamps"][-1]
        )
        self.print_to_log_file(f"Epoch time: {np.round(epoch_time, decimals=2)} s")

        # Periodic checkpoint ("latest")
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))

        # Track and save best model according to validation contrastive loss
        current_val_loss = self.logger.my_fantastic_logging["val_losses"][-1]
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.print_to_log_file(
                f"New best contrastive val loss: {np.round(self.best_val_loss, decimals=4)}"
            )
            self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))

        if self.local_rank == 0:
            try:
                self.logger.plot_progress_png(self.output_folder)
            except Exception:
                # plotting is best-effort only
                pass

        self.current_epoch += 1

    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        Skip full segmentation validation for contrastive pretraining.
        We only care about contrastive loss here; Dice is evaluated in
        the subsequent fine-tuning stage.
        """
        self.print_to_log_file(
            "Skipping perform_actual_validation during contrastive pretraining."
        )

    def save_checkpoint(self, filename: str) -> None:
        """
        Save *only* the feature extractor (encoder + decoder) weights.

        The projection head is intentionally excluded so that fine-tuning
        trainers can load a clean segmentation backbone via:
          ckpt = torch.load(path)
          net.load_state_dict(ckpt['network_weights'], strict=False)
        """
        if self.local_rank != 0:
            return
        if self.disable_checkpointing:
            self.print_to_log_file("No checkpoint written, checkpointing is disabled")
            return

        from torch._dynamo import OptimizedModule

        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        full_state_dict = mod.state_dict()

        # Strip projection-head parameters (single- or multi-scale) so only
        # encoder/decoder stay in the checkpoint.
        feature_extractor_state_dict = {
            k: v
            for k, v in full_state_dict.items()
            if not (k.startswith("projection_head.") or k.startswith("projection_heads."))
        }

        checkpoint = {
            "network_weights": feature_extractor_state_dict,
            "optimizer_state": self.optimizer.state_dict(),
            "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            "logging": self.logger.get_checkpoint(),
            "_best_ema": self._best_ema,
            "current_epoch": self.current_epoch + 1,
            "init_args": self.my_init_kwargs,
            "trainer_name": self.__class__.__name__,
            "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
            "is_contrastive_pretrained": True,
        }
        torch.save(checkpoint, filename)
        self.print_to_log_file(
            f"Saved contrastive-pretrained checkpoint (feature extractor only) to {filename}"
        )