import torch
from torch import nn
from typing import Union, Tuple, List
import numpy as np
from time import time
from batchgenerators.utilities.file_and_folder_operations import join
from tqdm import tqdm  # Import tqdm

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.pixelwise_boundary_contrastive_loss import PixelwiseBoundaryContrastiveLoss
from nnunetv2.training.loss.pixelwise_boundary_contrastive_loss_improved import ImprovedPixelwiseBoundaryContrastiveLoss
from nnunetv2.utilities.collate_outputs import collate_outputs

class ProjectionHead(nn.Module):
    """
    Projection head as per the paper: three 1x1 convolution layers with 256 channels
    followed by a unit-normalization layer. The first two layers use ReLU activation.
    """
    def __init__(self, in_channels, out_channels=256, is_3d=False):
        super().__init__()
        conv_op = nn.Conv3d if is_3d else nn.Conv2d
        # First conv: in_channels -> in_channels (ReLU)
        self.conv1 = conv_op(in_channels, in_channels, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)
        # Second conv: in_channels -> in_channels (ReLU)
        self.conv2 = conv_op(in_channels, in_channels, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)
        # Third conv: in_channels -> out_channels (no activation, will normalize)
        self.conv3 = conv_op(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        # Unit normalization as per the paper
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

class ContrastivePretrainingNetwork(nn.Module):
    """
    Improved contrastive pretraining network that:
    1. Uses multi-scale features from decoder (like deep supervision)
    2. Extracts features from encoder skip connections for cross-scale consistency
    3. Applies projection heads at multiple scales
    """
    def __init__(self, original_network, num_input_channels, patch_size, use_multi_scale=True, num_scales=3):
        super().__init__()
        self.original_network = original_network
        self.use_multi_scale = use_multi_scale
        self.num_scales = num_scales
        is_3d = len(patch_size) == 3
        
        # Robust Initialization
        try:
            param = next(self.original_network.parameters())
            device = param.device
        except StopIteration:
            device = torch.device('cpu') 

        dummy_shape = (1, num_input_channels, *patch_size)
        dummy_input = torch.zeros(dummy_shape, dtype=torch.float32, device=device)
        training_state = self.original_network.training
        self.original_network.eval()
        
        with torch.no_grad():
            try:
                skips = self.original_network.encoder(dummy_input)
                decoder_outputs = self.original_network.decoder(skips)
                
                # Determine number of scales available
                if isinstance(decoder_outputs, (list, tuple)):
                    num_available_scales = len(decoder_outputs)
                    high_res_features = decoder_outputs[0]
                else:
                    num_available_scales = 1
                    high_res_features = decoder_outputs
                
                in_channels = high_res_features.shape[1]
                self.num_scales = min(self.num_scales, num_available_scales) if self.use_multi_scale else 1
                
                # Initialize projection heads for each scale
                self.projection_heads = nn.ModuleList()
                for i in range(self.num_scales):
                    self.projection_heads.append(ProjectionHead(in_channels, out_channels=256, is_3d=is_3d))
                
            except Exception as e:
                raise RuntimeError(f"Contrastive Init Failed: {e}")
            finally:
                self.original_network.train(training_state)

    def forward(self, x):
        skips = self.original_network.encoder(x)
        decoder_outputs = self.original_network.decoder(skips)
        
        if isinstance(decoder_outputs, (list, tuple)):
            # Multi-scale: return projections from multiple decoder levels
            if self.use_multi_scale:
                projections = []
                for i in range(min(self.num_scales, len(decoder_outputs))):
                    proj = self.projection_heads[i](decoder_outputs[i])
                    projections.append(proj)
                return projections
            else:
                return self.projection_heads[0](decoder_outputs[0])
        else:
            return self.projection_heads[0](decoder_outputs)

class nnUNetTrainerBoundaryContrastive(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.best_val_loss = float('inf')
        self.initial_lr = 1e-2
        self.num_epochs = 20
        # Improved contrastive learning settings
        self.use_improved_loss = True  # Use improved loss with hard negative mining
        self.use_multi_scale = True  # Use multi-scale features

    def _do_i_compile(self):
        return False

    def build_network_architecture(self, architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        network = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision
        )
        patch_size = self.configuration_manager.patch_size
        return ContrastivePretrainingNetwork(
            network, 
            num_input_channels, 
            patch_size,
            use_multi_scale=self.use_multi_scale,
            num_scales=3  # Use top 3 decoder scales
        )
        
    def _build_loss(self):
        if self.use_improved_loss:
            # Improved loss with hard negative mining, adaptive temperature, multi-scale support
            return ImprovedPixelwiseBoundaryContrastiveLoss(
                temperature=0.1,
                temperature_min=0.05,
                temperature_schedule="cosine",  # Decrease temperature over training
                max_pos_per_image=2048,
                neg_pos_ratio=5.0,
                hard_neg_ratio=0.5,  # 50% hard negatives
                dilation_kernel_size=11,
                use_hard_negative_mining=True,
                min_pos_distance=2,  # Ensure diverse positive sampling
            )
        else:
            # Original loss
            return PixelwiseBoundaryContrastiveLoss(
                temperature=0.07,
                max_pos_per_image=2048,
                neg_pos_ratio=5.0,
                dilation_kernel_size=11,
            )

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def on_epoch_start(self):
        """
        Initialize tqdm progress bar at the start of the epoch.
        Update loss temperature schedule if using improved loss.
        """
        super().on_epoch_start()
        if self.use_improved_loss and hasattr(self.loss, 'set_epoch'):
            self.loss.set_epoch(self.current_epoch, self.num_epochs)
        if self.local_rank == 0:
            self.epoch_pbar = tqdm(total=self.num_iterations_per_epoch, 
                                   desc=f"Epoch {self.current_epoch}",
                                   leave=True,
                                   bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    def train_step(self, batch: dict):
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        output = self.network(data)
        
        labels = target[0] if isinstance(target, list) else target
        
        # Handle multi-scale outputs: if output is a list, loss will handle it
        # For multi-scale, we need to downsample labels to match each scale
        if isinstance(output, (list, tuple)) and self.use_multi_scale:
            # For now, use full-resolution labels for all scales
            # The loss will handle downsampling internally if needed
            l = self.loss(output, labels)
        else:
            l = self.loss(output, labels)
        
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()

        # Update tqdm
        if self.local_rank == 0 and hasattr(self, 'epoch_pbar'):
            self.epoch_pbar.set_postfix(loss=f"{l.item():.4f}")
            self.epoch_pbar.update(1)

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict):
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)
        target = target[0].to(self.device, non_blocking=True) if isinstance(target, list) else target.to(self.device, non_blocking=True)

        with torch.no_grad():
            output = self.network(data)
            # Handle multi-scale outputs
            if isinstance(output, (list, tuple)) and self.use_multi_scale:
                l = self.loss(output, target)
            else:
                l = self.loss(output, target)
        return {'loss': l.detach().cpu().numpy()}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        loss_here = np.mean(outputs_collated['loss'])
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def on_epoch_end(self):
        # Close tqdm
        if self.local_rank == 0 and hasattr(self, 'epoch_pbar'):
            self.epoch_pbar.close()
            del self.epoch_pbar

        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)
        
        # Log losses
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        
        epoch_time = self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1]
        self.print_to_log_file(f"Epoch time: {np.round(epoch_time, decimals=2)} s")

        # Checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # Best Model Tracking
        current_val_loss = self.logger.my_fantastic_logging['val_losses'][-1]
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.print_to_log_file(f"Yayy! New best Val Loss: {np.round(self.best_val_loss, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            try:
                self.logger.plot_progress_png(self.output_folder)
            except Exception:
                pass 

        self.current_epoch += 1

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.print_to_log_file("Skipping perform_actual_validation during contrastive pretraining.")
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Override to save only the feature extractor weights (encoder + decoder),
        excluding the projection head, as per the paper's methodology.
        After contrastive pretraining, we discard the projection head.
        """
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                from torch._dynamo import OptimizedModule
                
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod
                
                # Get full state dict
                full_state_dict = mod.state_dict()
                
                # Filter out projection head weights - only keep feature extractor (encoder + decoder)
                feature_extractor_state_dict = {}
                for key, value in full_state_dict.items():
                    # Skip projection head weights
                    if not key.startswith('projection_head.'):
                        feature_extractor_state_dict[key] = value
                
                checkpoint = {
                    'network_weights': feature_extractor_state_dict,
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                    'is_contrastive_pretrained': True,  # Flag to indicate this is a contrastive pretrained checkpoint
                }
                torch.save(checkpoint, filename)
                self.print_to_log_file(f"Saved contrastive pretrained checkpoint (feature extractor only) to {filename}")
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')