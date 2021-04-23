import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import constant_init, kaiming_init

from ..registry import NECKS
from ...core.ops import conv_1x1x1_bn, HSwish


class ChannelReducer(nn.Module):
    def __init__(self, dim=1, keepdim=False):
        super().__init__()

        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


@NECKS.register_module()
class SelfFeatureRegularizer(nn.Module):
    def __init__(self, in_channels, spatial_size=7, temporal_size=1, hidden_size=256, reg_weight=1.0):
        super().__init__()

        self.reg_weight = float(reg_weight)
        assert self.reg_weight > 0.0
        self.hidden_size = int(hidden_size)
        assert self.hidden_size > 0
        self.scale = self.hidden_size ** (-0.5)

        self.in_channels = in_channels if isinstance(in_channels, (tuple, list)) else [in_channels]
        num_inputs = len(self.in_channels)
        assert num_inputs > 1

        self.temporal_size = temporal_size if isinstance(temporal_size, (tuple, list)) else [temporal_size]
        assert len(self.temporal_size) == num_inputs

        spatial_size = spatial_size if isinstance(spatial_size, (tuple, list)) else [spatial_size]
        self.spatial_size = [ss if isinstance(ss, (tuple, list)) else (ss, ss) for ss in spatial_size]
        assert len(self.spatial_size) == num_inputs

        self.keys = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool3d((self.temporal_size[input_id],) + self.spatial_size[input_id], stride=1, padding=0),
                conv_1x1x1_bn(self.in_channels[input_id], self.hidden_size, as_list=False),
                HSwish(),
                conv_1x1x1_bn(self.hidden_size, self.hidden_size, as_list=False),
            )
            for input_id in range(num_inputs)
        ])

        self.student_tokens = nn.Parameter(torch.Tensor(1, num_inputs - 1, self.hidden_size))
        self.student_tokens.data.normal_(std=0.02)

        self.teacher_token = nn.Parameter(torch.Tensor(1, 1, self.hidden_size))
        self.teacher_token.data.normal_(std=0.02)

        self.student_mappers = nn.ModuleList([
            ChannelReducer(dim=1, keepdim=True)
            for _ in range(num_inputs - 1)
        ])
        self.teacher_reducer = ChannelReducer(dim=1, keepdim=True)
        self.teacher_mappers = nn.ModuleList([
            nn.Upsample(size=(self.temporal_size[input_id],) + self.spatial_size[input_id],
                        mode='trilinear', align_corners=False)
            for input_id in range(num_inputs - 1)
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
        assert len(inputs) == len(self.keys) + 1
        inputs = inputs[:-2] + [inputs[-1]]

        attention, student_features, teacher_features = None, None, None
        if self.training:
            keys = [
                key_module(input_feature).view(-1, 1, self.hidden_size)
                for input_feature, key_module in zip(inputs, self.keys)
            ]
            student_keys = torch.cat(keys[:-1], dim=1)
            teacher_key = keys[-1].view(-1, self.hidden_size, 1)

            key_prod = torch.matmul(student_keys, teacher_key).squeeze(2)
            token_prod = torch.sum(self.teacher_token * self.student_tokens, dim=-1)
            attention = F.softmax(self.scale * (key_prod + token_prod), dim=-1)

            student_features = [mapper(x_) for x_, mapper in zip(inputs[:-1], self.student_mappers)]

            teacher_feature = self.teacher_reducer(inputs[-1])
            teacher_features = [mapper(teacher_feature) for mapper in self.teacher_mappers]

        # returns the input unchanged
        if return_extra_data:
            return inputs, dict(attention=attention,
                                student_features=student_features,
                                teacher_features=teacher_features)
        else:
            return inputs

    def loss(self, attention=None, student_features=None, teacher_features=None, **kwargs):
        losses = dict()

        if attention is None or student_features is None or teacher_features is None:
            return losses

        all_losses = [
            self._reg_loss(student_feature, teacher_feature).view(-1, 1)
            for student_feature, teacher_feature in zip(student_features, teacher_features)
        ]

        weighted_losses = attention * torch.cat(all_losses, dim=1)
        losses['loss/sfr'] = torch.mean(torch.sum(weighted_losses, dim=1))

        return losses

    @staticmethod
    def _reg_loss(x, y):
        sqr_diff = (x - y) ** 2
        return torch.mean(sqr_diff, dim=(1, 2, 3, 4))




