import cv2
import torch
import torch.nn as nn

from mmcv.cnn import constant_init, kaiming_init

from ..registry import NECKS
from ...core.ops import conv_1x1x1_bn, normalize


@NECKS.register_module()
class EMDRegularizer(nn.Module):
    """ Based on the paper: https://arxiv.org/abs/2103.07350
    """

    modes = ['pairs', 'classmates', 'random']

    def __init__(self, in_channels, mode='pairs', hidden_size=256, loss_weight=1.0):
        super().__init__()

        self.mode = mode
        assert self.mode in self.modes
        self.loss_weight = float(loss_weight)
        assert self.loss_weight > 0.0
        self.hidden_size = int(hidden_size)
        assert self.hidden_size > 0
        self.scale = self.hidden_size ** (-0.5)

        self.in_channels = in_channels if isinstance(in_channels, (tuple, list)) else [in_channels]
        num_inputs = len(self.in_channels)
        assert num_inputs > 0

        self.mappers = nn.ModuleList([
            nn.Sequential(
                conv_1x1x1_bn(self.in_channels[input_id], self.hidden_size, as_list=False),
                nn.AvgPool3d(kernel_size=(3, 3, 3), stride=1, padding=1, count_include_pad=False)
            )
            for input_id in range(num_inputs)
        ])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1.0, 0.0)
            elif isinstance(m, nn.Parameter):
                m.data.normal_()

    def forward(self, inputs, return_extra_data=False):
        # assert len(inputs) == len(self.mappers) + 1
        # filtered_inputs = inputs[:-1]

        filtered_inputs = [inputs[-1]]
        assert len(filtered_inputs) == len(self.mappers)

        features = None
        if self.training:
            features = []
            for input_feature, mapper in zip(filtered_inputs, self.mappers):
                output_feature = mapper(input_feature)
                features.append(output_feature.view(output_feature.size(0), self.hidden_size, -1))
            features = torch.cat(features, dim=2)

        # returns the input unchanged
        if return_extra_data:
            return inputs, dict(features=features)
        else:
            return inputs

    def loss(self, features=None, **kwargs):
        losses = dict()

        if features is None:
            return losses

        features_a, features_b = self._split_features(features, self.mode, **kwargs)
        if features_a is None or features_b is None:
            losses['loss/emd_sfr'] = torch.zeros([], dtype=features.dtype, device=features.device)
            return losses

        assert features_a.size(0) == features_b.size(0)
        num_pairs = features_a.size(0)

        assert features_a.size(2) == features_b.size(2)
        num_nodes = features_a.size(2)
        cost_scale = 1.0 / float(num_nodes)

        cost_matrix = self._get_cost_matrix(features_a, features_b)
        weights_a = self._get_weights(features_a, features_b)
        weights_b = self._get_weights(features_b, features_a)

        pair_losses = []
        for pair_id in range(num_pairs):
            local_weights_a = weights_a[pair_id]
            local_weights_b = weights_b[pair_id]
            if torch.sum(local_weights_a > 0.0) == 0 or torch.sum(local_weights_b > 0.0) == 0:
                continue

            flow = self._solve_emd(cost_matrix[pair_id], local_weights_a, local_weights_b)

            cost = torch.sum(flow * cost_matrix[pair_id])
            pair_losses.append(cost_scale * cost)

        if len(pair_losses) > 0:
            loss_weight = self.loss_weight / float(len(pair_losses))
            losses['loss/emd_sfr'] = loss_weight * sum(pair_losses)
        else:
            losses['loss/emd_sfr'] = torch.zeros([], dtype=features.dtype, device=features.device)

        return losses

    @staticmethod
    def _split_features(features, mode, labels=None, dataset_id=None, num_clips=1, **kwargs):
        if mode == 'pairs':
            assert num_clips == 2

            features_a = features[::num_clips]
            features_b = features[1::num_clips]
        elif mode == 'classmates':
            with torch.no_grad():
                batch_size = features.size(0)
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
            if num_valid_pairs > 0:
                valid_pairs_subset = valid_pairs[valid_samples_mask].float()
                rand_weights = 1.0 + torch.rand_like(valid_pairs_subset)
                valid_pairs_subset_weights = valid_pairs_subset * rand_weights
                valid_pairs_ids = torch.argmax(valid_pairs_subset_weights, dim=-1)

                features_a = features[valid_samples_mask]
                features_b = features[valid_pairs_ids]
            else:
                features_a, features_b = None, None
        else:
            batch_size = features.size(0)
            assert batch_size % 2 == 0

            idx = torch.randperm(batch_size, device=features.device)
            features_a = features[idx[:(batch_size // 2)]]
            features_b = features[idx[(batch_size // 2):]]

        return features_a, features_b

    @staticmethod
    def _get_cost_matrix(features_a, features_b):
        norm_a = normalize(features_a, dim=1, p=2)
        norm_b = normalize(features_b, dim=1, p=2)
        dist_matrix = 1.0 - torch.matmul(norm_a.transpose(1, 2), norm_b)

        return dist_matrix.clamp_min(0.0)

    @staticmethod
    def _get_weights(ref, trg):
        with torch.no_grad():
            mean_trg = normalize(trg.mean(dim=2, keepdim=True), dim=1, p=2)
            weights = torch.sum(ref * mean_trg, dim=1).clamp_min(0.0)

            sum_weights = torch.sum(weights, dim=1, keepdim=True)
            scales = torch.where(sum_weights > 0.0,
                                 torch.reciprocal(sum_weights),
                                 torch.ones_like(sum_weights))
            num_nodes = weights.size(1)
            norm_weights = (float(num_nodes) * scales) * weights

        return norm_weights

    @staticmethod
    def _solve_emd(cost_matrix, weights_a, weights_b):
        data_type = cost_matrix.dtype
        device = cost_matrix.device

        cost_matrix = cost_matrix.detach().cpu().numpy()
        weights_a = weights_a.detach().cpu().numpy()
        weights_b = weights_b.detach().cpu().numpy()

        _, _, flow = cv2.EMD(weights_a, weights_b, cv2.DIST_USER, cost_matrix)

        return torch.from_numpy(flow).to(device).type(data_type)





