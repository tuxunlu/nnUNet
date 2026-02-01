import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Union, Tuple, List
import numpy as np
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context
from torch import autocast
from torch._dynamo import OptimizedModule
from tqdm import tqdm  # Import tqdm

# ------------------------------------------------------------------
# 1. Pixel Contrastive Loss (Provided Reference)
# ------------------------------------------------------------------
class PixelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, ignore_index=255, max_samples=2048, batch_chunk_size=16):
        super().__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.max_samples = max_samples
        self.batch_chunk_size = batch_chunk_size
        self.background_label = 0  # We assume class 0 is background

    def forward(self, feats, labels):
        # 1. Align labels
        if labels.dim() == feats.dim():
            labels = labels.squeeze(1)
        if labels.shape[-2:] != feats.shape[-2:]:
            labels = F.interpolate(labels.unsqueeze(1).float(), size=feats.shape[2:], mode='nearest').squeeze(1).long()
        
        B, C, H, W = feats.shape
        N_pixels = H * W
        
        # Flatten: [B, C, H, W] -> [B, N, C]
        feats = feats.permute(0, 2, 3, 1).reshape(B, N_pixels, C)
        labels = labels.reshape(B, N_pixels)

        # ------------------------------------------------------------------
        # 2. Explicit Sampling (Positives First)
        # ------------------------------------------------------------------
        sampled_feats_list = []
        sampled_labels_list = []
        sampled_valid_mask_list = []

        device = feats.device

        for b in range(B):
            # Identify masks for this image
            valid_mask = (labels[b] != self.ignore_index)
            pos_mask = (labels[b] != self.background_label) & valid_mask
            neg_mask = (labels[b] == self.background_label) & valid_mask
            
            # Get indices
            pos_indices = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
            neg_indices = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)
            
            n_pos = pos_indices.numel()
            n_neg = neg_indices.numel()
            
            # Selection Logic
            if n_pos >= self.max_samples:
                # Case A: Too many positives -> Randomly subsample positives only
                perm = torch.randperm(n_pos, device=device)[:self.max_samples]
                selected_indices = pos_indices[perm]
            else:
                # Case B: Not enough positives -> Take ALL positives, fill with negatives
                needed = self.max_samples - n_pos
                if n_neg > needed:
                    perm = torch.randperm(n_neg, device=device)[:needed]
                    neg_subset = neg_indices[perm]
                    selected_indices = torch.cat([pos_indices, neg_subset])
                else:
                    # Case C: Not enough valid pixels total -> Take everything
                    selected_indices = torch.cat([pos_indices, neg_indices])
            
            # Gather Data
            # feats[b]: [N, C] -> [K_actual, C]
            s_feats = feats[b, selected_indices]
            s_labels = labels[b, selected_indices]
            
            # Pad if we found fewer than max_samples (Case C)
            K_actual = selected_indices.numel()
            if K_actual < self.max_samples:
                pad_amt = self.max_samples - K_actual
                # Pad feats with 0
                s_feats = F.pad(s_feats, (0, 0, 0, pad_amt)) 
                # Pad labels with ignore_index
                s_labels = F.pad(s_labels, (0, pad_amt), value=self.ignore_index)
                
                # Create mask: 1 for valid, 0 for padded
                s_mask = torch.cat([
                    torch.ones(K_actual, device=device, dtype=torch.bool),
                    torch.zeros(pad_amt, device=device, dtype=torch.bool)
                ])
            else:
                s_mask = torch.ones(self.max_samples, device=device, dtype=torch.bool)

            sampled_feats_list.append(s_feats)
            sampled_labels_list.append(s_labels)
            sampled_valid_mask_list.append(s_mask)

        # Stack to create batch: [B, max_samples, ...]
        sampled_feats = torch.stack(sampled_feats_list)     # [B, K, C]
        sampled_labels = torch.stack(sampled_labels_list)   # [B, K]
        sampled_valid_mask = torch.stack(sampled_valid_mask_list) # [B, K]
        
        K = self.max_samples

        # ------------------------------------------------------------------
        # 3. Create Pairs (Vectorized)
        # ------------------------------------------------------------------
        feats_rolled = torch.roll(sampled_feats, shifts=-1, dims=0)
        labels_rolled = torch.roll(sampled_labels, shifts=-1, dims=0)
        mask_rolled = torch.roll(sampled_valid_mask, shifts=-1, dims=0)
        
        # [B, 2K, C]
        feats_pair = torch.cat([sampled_feats, feats_rolled], dim=1)
        labels_pair = torch.cat([sampled_labels, labels_rolled], dim=1)
        valid_pair_mask = torch.cat([sampled_valid_mask, mask_rolled], dim=1)

        # Normalize once
        feats_pair = F.normalize(feats_pair, dim=2)
        
        # ------------------------------------------------------------------
        # 4. Chunked Loss Computation
        # ------------------------------------------------------------------
        total_loss = 0.0
        total_valid_chunks = 0
        
        for i in range(0, B, self.batch_chunk_size):
            end = min(i + self.batch_chunk_size, B)
            
            f_chunk = feats_pair[i:end]      # [Chunk, 2K, C]
            l_chunk = labels_pair[i:end]     # [Chunk, 2K]
            v_chunk = valid_pair_mask[i:end] # [Chunk, 2K]
            
            if v_chunk.sum() == 0:
                continue
            
            # --- Compute Logits ---
            logits = torch.bmm(f_chunk, f_chunk.transpose(1, 2)) / self.temperature
            logits_max, _ = torch.max(logits, dim=2, keepdim=True)
            logits = logits - logits_max.detach()

            # --- Construct Masks ---
            v_matrix = torch.bmm(v_chunk.unsqueeze(2).float(), v_chunk.unsqueeze(1).float())
            l_view = l_chunk.unsqueeze(2)
            pos_mask = torch.eq(l_view, l_view.transpose(1, 2)).float()
            pos_mask = pos_mask * v_matrix
            
            same_img = torch.eye(2, device=feats.device).repeat_interleave(K, dim=0).repeat_interleave(K, dim=1)
            same_img = same_img.unsqueeze(0) 
            
            neg_mask = (1.0 - pos_mask) * same_img
            neg_mask = neg_mask * v_matrix
            
            eye = torch.eye(2 * K, device=feats.device).unsqueeze(0)
            pos_mask = pos_mask * (1 - eye)
            neg_mask = neg_mask * (1 - eye)
            
            # --- Loss Calculation ---
            exp_logits = torch.exp(logits)
            sum_neg = (exp_logits * neg_mask).sum(dim=2, keepdim=True)
            denominator = exp_logits + sum_neg
            log_probs = logits - torch.log(denominator + 1e-8)
            
            num_pos = torch.clamp(pos_mask.sum(dim=2, keepdim=True), min=1.0)
            log_prob_sum = (pos_mask * log_probs).sum(dim=2)
            mean_log_prob = log_prob_sum / num_pos.squeeze(2)
            
            valid_anchors = (pos_mask.sum(dim=2) > 0).float()
            loss_chunk = -mean_log_prob * valid_anchors
            loss_chunk = loss_chunk.sum(dim=1) / torch.clamp(valid_anchors.sum(dim=1), min=1.0)
            
            total_loss += loss_chunk.sum()
            total_valid_chunks += f_chunk.shape[0]

        if total_valid_chunks == 0:
            return torch.tensor(0.0, device=feats.device, requires_grad=True)
            
        return total_loss / total_valid_chunks

# ------------------------------------------------------------------
# 2. Projection Head
# ------------------------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_channels, out_channels=128, is_3d=False):
        super().__init__()
        conv_op = nn.Conv3d if is_3d else nn.Conv2d
        self.conv1 = conv_op(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_op(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

# ------------------------------------------------------------------
# 3. Network Wrapper for Fine-Tuning
# ------------------------------------------------------------------
class FineTuneNetwork(nn.Module):
    def __init__(self, original_network, num_input_channels, patch_size, contrastive_dim=128):
        super().__init__()
        self.original_network = original_network
        # Flag to control output format
        self.return_logits_only = False
        
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
                outputs = self.original_network(dummy_input)
                if isinstance(outputs, (list, tuple)):
                    high_res_features = outputs[0]
                else:
                    high_res_features = outputs
                in_channels = high_res_features.shape[1]
            except Exception as e:
                raise RuntimeError(f"Contrastive Init Failed: {e}")
            finally:
                self.original_network.train(training_state)
        
        self.projection_head = ProjectionHead(in_channels, contrastive_dim, is_3d)

    def forward(self, x):
        outputs = self.original_network(x)
        
        # If we only need logits (e.g. for standard inference/validation), return them directly
        if self.return_logits_only:
            return outputs

        if isinstance(outputs, (list, tuple)):
            features = outputs[0]
        else:
            features = outputs
            
        proj = self.projection_head(features)
        
        return outputs, proj

# ------------------------------------------------------------------
# 4. The Trainer (Modified with TQDM)
# ------------------------------------------------------------------
class nnUNetTrainerContrastiveFineTune(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.contrastive_loss_weight = 1.0
        self.contrastive_temperature = 0.07

        self.logger.my_fantastic_logging['val_loss_seg'] = []
        self.logger.my_fantastic_logging['val_loss_cont'] = []

        self.num_epochs = 100
        self.initial_lr = 1e-2

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return optimizer, lr_scheduler

    def build_network_architecture(self, architecture_class_name, arch_init_kwargs, 
                                   arch_init_kwargs_req_import, num_input_channels, 
                                   num_output_channels, enable_deep_supervision=True):
        original_network = super().build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels, enable_deep_supervision
        )
        patch_size = self.configuration_manager.patch_size
        # Use the updated FineTuneNetwork definition above
        network = FineTuneNetwork(original_network, num_input_channels, patch_size)
        return network

    def _build_loss(self):
        self.seg_loss = super()._build_loss()
        self.contrastive_loss = PixelContrastiveLoss(
            temperature=self.contrastive_temperature, 
            ignore_index=self.label_manager.ignore_label if self.label_manager.ignore_label is not None else 255
        )
        return self.seg_loss

    # --- FIX 1: Override to unwrap the network ---
    def set_deep_supervision_enabled(self, enabled: bool):
        # 1. Handle DDP wrapper
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
            
        # 2. Handle torch.compile wrapper
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
            
        # 3. Handle FineTuneNetwork wrapper
        if hasattr(mod, 'original_network'):
            mod = mod.original_network
        
        # Now mod is the actual UNet (e.g., ResidualEncoderUNet)
        if hasattr(mod, 'decoder'):
            mod.decoder.deep_supervision = enabled

    # --- FIX 2: Handle inference during final validation ---
    def perform_actual_validation(self, save_probabilities: bool = False):
        # Temporarily switch network to return only logits so standard predictor works
        if self.is_ddp:
            net = self.network.module
        else:
            net = self.network
        
        if isinstance(net, OptimizedModule):
            net = net._orig_mod
            
        # Set flag
        if hasattr(net, 'return_logits_only'):
            net.return_logits_only = True
            
        try:
            super().perform_actual_validation(save_probabilities)
        finally:
            # Restore flag
            if hasattr(net, 'return_logits_only'):
                net.return_logits_only = False

    def on_epoch_start(self):
        """
        Initialize tqdm progress bar at the start of the epoch.
        """
        super().on_epoch_start()
        if self.local_rank == 0:
            self.epoch_pbar = tqdm(total=self.num_iterations_per_epoch, 
                                   desc=f"Epoch {self.current_epoch}",
                                   leave=True,
                                   bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_outputs, proj_features = self.network(data)
            
            l_seg = self.loss(seg_outputs, target)
            
            if isinstance(target, list):
                target_cont = target[0]
            else:
                target_cont = target
            
            l_cont = self.contrastive_loss(proj_features, target_cont)
            l_total = l_seg + self.contrastive_loss_weight * l_cont

            # l_cont = torch.tensor(0.0, device=self.device, requires_grad=True)
            # l_total = l_seg

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l_total).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l_total.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
            
        # Update tqdm
        if self.local_rank == 0 and hasattr(self, 'epoch_pbar'):
            self.epoch_pbar.set_postfix(
                loss=f"{l_total.item():.4f}", 
                seg=f"{l_seg.item():.4f}", 
                cont=f"{l_cont.item():.4f}"
            )
            self.epoch_pbar.update(1)
            
        return {'loss': l_total.detach().cpu().numpy(), 'loss_seg': l_seg.detach().cpu().numpy(), 'loss_cont': l_cont.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_outputs, proj_features = self.network(data)
            l_seg = self.loss(seg_outputs, target)
            
            if isinstance(target, list):
                target_cont = target[0]
            else:
                target_cont = target
            l_cont = self.contrastive_loss(proj_features, target_cont)
            l_total = l_seg + self.contrastive_loss_weight * l_cont

        if self.enable_deep_supervision:
            output_tensor = seg_outputs[0]
            target_tensor = target[0]
        else:
            output_tensor = seg_outputs
            target_tensor = target

        axes = [0] + list(range(2, output_tensor.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output_tensor) > 0.5).long()
        else:
            output_seg = output_tensor.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output_tensor.shape, device=output_tensor.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target_tensor != self.label_manager.ignore_label).float()
                target_tensor[target_tensor == self.label_manager.ignore_label] = 0
            else:
                if target_tensor.dtype == torch.bool:
                    mask = ~target_tensor[:, -1:]
                else:
                    mask = 1 - target_tensor[:, -1:]
                target_tensor = target_tensor[:, :-1]
        else:
            mask = None

        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target_tensor, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            'loss': l_total.detach().cpu().numpy(),
            'loss_seg': l_seg.detach().cpu().numpy(),
            'loss_cont': l_cont.detach().cpu().numpy(),
            'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard
        }

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        loss_total = np.mean(outputs_collated['loss'])
        loss_seg = np.mean(outputs_collated['loss_seg'])
        loss_cont = np.mean(outputs_collated['loss_cont'])
        self.logger.log('val_losses', loss_total, self.current_epoch)
        self.logger.log('val_loss_seg', loss_seg, self.current_epoch)
        self.logger.log('val_loss_cont', loss_cont, self.current_epoch)
        super().on_validation_epoch_end(val_outputs)

    def on_epoch_end(self):
        # Close tqdm before logging so the progress bar doesn't interfere with print statements
        if self.local_rank == 0 and hasattr(self, 'epoch_pbar'):
            self.epoch_pbar.close()
            del self.epoch_pbar

        super().on_epoch_end()
        # Safe access to logs
        seg_loss = self.logger.my_fantastic_logging.get('val_loss_seg', [0])[-1]
        cont_loss = self.logger.my_fantastic_logging.get('val_loss_cont', [0])[-1]
        self.print_to_log_file(f"Seg Loss: {np.round(seg_loss, 4)}, Cont Loss: {np.round(cont_loss, 4)}")