import torch
import torch.nn as nn

from mmcv.cnn import constant_init, kaiming_init

from ..registry import NECKS
from ...core.ops import conv_1x1x1_bn, HSwish, Normalize, soft_dtw


@NECKS.register_module()
class VideoAligner(nn.Module):
    """
    Implementation of the paper: https://arxiv.org/abs/2103.17260
    """

    merge_modes = ['concat', 'sum']

    def __init__(self, in_channels, spatial_size=7, temporal_size=1, hidden_size=512, embedding_size=256,
                 smoothness=0.1, margin=2, window_size=1, reg_weight=1.0, merge_mode='concat'):
        super().__init__()

        self.smoothness = float(smoothness)
        assert self.smoothness > 0.0
        self.margin = float(margin)
        assert self.margin >= 0.0
        self.window_size = int(window_size)
        assert self.window_size > 0
        self.reg_weight = float(reg_weight)
        assert self.reg_weight > 0.0
        self.hidden_size = int(hidden_size)
        assert self.hidden_size > 0
        self.embd_size = int(embedding_size)
        assert self.embd_size > 0
        self.merge_mode = merge_mode
        assert self.merge_mode in self.merge_modes

        self.in_channels = in_channels if isinstance(in_channels, (tuple, list)) else [in_channels]
        self.temporal_size = temporal_size if isinstance(temporal_size, (tuple, list)) else [temporal_size]
        assert len(self.in_channels) == len(self.temporal_size)

        spatial_size = spatial_size if isinstance(spatial_size, (tuple, list)) else [spatial_size]
        self.spatial_size = [ss if isinstance(ss, (tuple, list)) else (ss, ss) for ss in spatial_size]
        assert len(self.spatial_size) == len(self.temporal_size)

        self.trg_temporal_size = max(self.temporal_size)
        self.mapper = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool3d((1,) + self.spatial_size[input_id], stride=1, padding=0),
                conv_1x1x1_bn(self.in_channels[input_id], self.hidden_size, as_list=False),
                nn.Upsample(size=(self.trg_temporal_size, 1, 1), mode='trilinear', align_corners=False)
                if self.temporal_size[input_id] < self.trg_temporal_size else nn.Sequential(),
                HSwish()
            )
            for input_id in range(len(self.in_channels))
        ])

        merged_channels = self.hidden_size
        if self.merge_mode == 'concat':
            merged_channels *= len(self.in_channels)
        self.embedding = nn.Sequential(
            conv_1x1x1_bn(merged_channels, self.embd_size, as_list=False),
            Normalize(dim=1, p=2)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1.0, 0.0)
            elif isinstance(m, nn.Parameter):
                m.data.normal_()

    def forward(self, inputs, return_extra_data=False):
        temporal_embd = None
        if self.training:
            assert len(inputs) >= len(self.mapper)
            internal_outs = [
                self.mapper[input_id](inputs[input_id])
                for input_id in range(len(self.mapper))
            ]

            if self.merge_mode == 'concat':
                y = torch.cat(internal_outs, dim=1)
            else:
                y = sum(internal_outs)

            temporal_embd = self.embedding(y)

        # returns the input unchanged
        if return_extra_data:
            return inputs, dict(temporal_embd=temporal_embd)
        else:
            return inputs

    def loss(self, temporal_embd=None, labels=None, dataset_id=None, num_clips=1):
        losses = dict()

        if temporal_embd is None or labels is None:
            return losses

        with torch.no_grad():
            batch_size = temporal_embd.size(0)
            batch_range = torch.arange(batch_size, device=labels.device)
            top_diagonal_pairs = batch_range.view(-1, 1) < batch_range.view(1, -1)
            same_class_pairs = labels.view(-1, 1) == labels.view(1, -1)
            valid_pairs = same_class_pairs * top_diagonal_pairs

            if dataset_id is not None:
                same_dataset_pairs = dataset_id.view(-1, 1) == dataset_id.view(1, -1)
                valid_pairs = same_dataset_pairs * valid_pairs

            if num_clips > 1:
                instances_range = torch.arange(0, batch_size, num_clips, device=labels.device)
                instances_batch_range = instances_range.view(-1, 1).repeat(1, 2)
                different_instance_pairs = batch_range.view(-1, 1) < instances_batch_range.view(1, -1)
                valid_pairs = different_instance_pairs * valid_pairs

            valid_samples_mask = torch.any(valid_pairs, dim=-1)
            num_valid_pairs = torch.sum(valid_samples_mask, dim=0).item()
            if num_valid_pairs == 0:
                losses['loss/align'] = torch.zeros([], dtype=temporal_embd.dtype, device=temporal_embd.device)
                losses['loss/align_reg'] = torch.zeros([], dtype=temporal_embd.dtype, device=temporal_embd.device)
                return losses

            valid_pairs_subset = valid_pairs[valid_samples_mask]
            valid_pairs_ids = torch.argmax(valid_pairs_subset.int(), dim=-1)

        temporal_embd = temporal_embd.view(-1, self.embd_size, self.trg_temporal_size)
        left_embd = temporal_embd[valid_samples_mask]
        right_embd = temporal_embd[valid_pairs_ids]

        pair_distances = (1.0 - torch.matmul(left_embd.transpose(1, 2), right_embd)).clamp_min(0.0)
        left_distances = (1.0 - torch.matmul(left_embd.transpose(1, 2), left_embd)).clamp_min(0.0)
        right_distances = (1.0 - torch.matmul(right_embd.transpose(1, 2), right_embd)).clamp_min(0.0)

        main_losses = soft_dtw(1e-2 + pair_distances, self.smoothness, 0)
        losses['loss/align'] = main_losses.mean()

        left_reg_loss = self._contrastive_idm_loss(left_distances, self.margin, self.window_size)
        right_reg_loss = self._contrastive_idm_loss(right_distances, self.margin, self.window_size)
        reg_loss = (self.reg_weight / float(num_valid_pairs)) * (left_reg_loss + right_reg_loss)
        losses['loss/align_reg'] = reg_loss

        return losses

    @staticmethod
    def _contrastive_idm_loss(dist_matrix, margin, window_size):
        with torch.no_grad():
            temporal_size = dist_matrix.size(1)
            temporal_range = torch.arange(temporal_size, device=dist_matrix.device)
            range_diff = temporal_range.view(-1, 1) - temporal_range.view(1, -1)
            mode_mask = torch.abs(range_diff) > window_size

            outer_weights = range_diff ** 2 + 1.0
            inner_weights = torch.reciprocal(outer_weights)

        inner_losses = inner_weights.unsqueeze(0) * dist_matrix
        outer_losses = outer_weights.unsqueeze(0) * (margin - dist_matrix).clamp_min(0.0)
        losses = torch.where(mode_mask.unsqueeze(0), outer_losses, inner_losses)

        weight = 1.0 / float(temporal_size * temporal_size)
        loss = weight * torch.sum(losses)

        return loss




