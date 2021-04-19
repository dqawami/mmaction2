import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import NECKS
from ...core.ops import conv_1x1x1_bn, HSwish


@NECKS.register_module()
class VideoAligner(nn.Module):
    def __init__(self, in_channels, spatial_size=7, temporal_size=1, hidden_size=512, embedding_size=256):
        super().__init__()

        self.in_channels = in_channels
        self.spatial_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)
        self.temporal_size = temporal_size
        self.hidden_size = hidden_size
        self.embd_size = embedding_size

        self.mapper = nn.Sequential(
            *conv_1x1x1_bn(self.in_channels, self.hidden_size),
            HSwish(),
            *conv_1x1x1_bn(self.hidden_size, self.embd_size),
        )
        self.spatial_pool = nn.AvgPool3d((1,) + self.spatial_size, stride=1, padding=0)

    def init_weights(self):
        pass

    def forward(self, x, return_extra_data=False):
        y = None
        if not self.training:
            y = self.spatial_pool(x)
            y = self.mapper(y)

        # returns the input unchanged
        if return_extra_data:
            return x, dict(features=y)
        else:
            return x

    def loss(self, features, labels, dataset_id):
        losses = dict()

        return losses
