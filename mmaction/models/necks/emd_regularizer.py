import cv2
import torch
import torch.nn as nn

from mmcv.cnn import constant_init, kaiming_init

from ..registry import NECKS
from ...core.ops import conv_1x1x1_bn, Normalize, normalize


@NECKS.register_module()
class EMDRegularizer(nn.Module):
    def __init__(self, in_channels, hidden_size=256, loss_weight=1.0):
        super().__init__()

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
                Normalize(dim=1, p=2)
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

    def loss(self, features=None, num_clips=1, **kwargs):
        losses = dict()

        if features is None:
            return losses

        if num_clips > 1:
            assert num_clips == 2

            features_a = features[::num_clips]
            features_b = features[1::num_clips]
        else:
            batch_size = features.size(0)
            idx = torch.randperm(batch_size, device=features.device)

            features_a = features[idx[:(batch_size // 2)]]
            features_b = features[idx[(batch_size // 2):]]

        assert features_a.size(0) == features_b.size(0)
        num_pairs = features_a.size(0)

        assert features_a.size(2) == features_b.size(2)
        num_nodes = features_a.size(2)
        cost_scale = 1.0 / float(num_nodes)

        cost_matrix = 1.0 - torch.matmul(features_a.transpose(1, 2), features_b)
        weights_a = self._get_weights(features_a, features_b)
        weights_b = self._get_weights(features_b, features_a)

        pair_losses = []
        for pair_id in range(num_pairs):
            flow = self._solve_emd(cost_matrix[pair_id], weights_a[pair_id], weights_b[pair_id])

            cost = torch.sum(flow * cost_matrix[pair_id])
            pair_losses.append(cost_scale * cost)

        loss_weight = self.loss_weight / float(num_pairs)
        losses['loss/emd_sfr'] = loss_weight * sum(pair_losses)

        return losses

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





