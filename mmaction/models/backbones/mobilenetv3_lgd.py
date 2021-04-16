import torch
import torch.nn as nn

from ...core.ops import HSwish, conv_1x1x1_bn
from ..registry import BACKBONES
from .mobilenetv3_s3d import MobileNetV3_S3D


class ContextPool3d(nn.Module):
    def __init__(self, in_channels):
        super(ContextPool3d, self).__init__()

        self.encoder = conv_1x1x1_bn(in_channels, 1, as_list=False)

    def forward(self, x):
        _, c, t, h, w = x.size()

        keys = self.encoder(x)
        attention_map = torch.softmax(keys.view(-1, t * h * w, 1), dim=1)

        context = torch.matmul(x.view(-1, c, t * h * w), attention_map)
        out = context.view(-1, c, 1, 1, 1)

        return out


class PoolingBlock(nn.Module):
    modes = ['average', 'attention']

    def __init__(self, in_planes, out_planes, factor=3, norm='none', mode='average'):
        super(PoolingBlock, self).__init__()

        assert mode in self.modes

        hidden_dim = int(factor * in_planes)
        layers = [
            nn.AdaptiveAvgPool3d((1, 1, 1)) if mode == 'average' else ContextPool3d(in_planes),
            *conv_1x1x1_bn(in_planes, hidden_dim, norm=norm),
            HSwish(),
            *conv_1x1x1_bn(hidden_dim, out_planes, norm=norm),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_planes, out_planes, factor=3, norm='none'):
        super(UpsampleBlock, self).__init__()

        hidden_dim = int(factor * in_planes)
        layers = [
            *conv_1x1x1_bn(in_planes, hidden_dim, norm=norm),
            HSwish(),
            *conv_1x1x1_bn(hidden_dim, out_planes, norm=norm),
        ]
        self.conv = nn.Sequential(*layers)

        self._reset_weights()

    def forward(self, x):
        return self.conv(x)

    def _reset_weights(self):
        last_stage = self.conv[len(self.conv) - 1]

        if hasattr(last_stage, 'weight') and last_stage.weight is not None:
            last_stage.weight.data.zero_()
        if hasattr(last_stage, 'bias') and last_stage.bias is not None:
            last_stage.bias.data.zero_()


class GlobBlock(nn.Module):
    def __init__(self, in_planes, out_planes, factor=3, norm='none'):
        super(GlobBlock, self).__init__()

        self.identity = in_planes == out_planes

        hidden_dim = int(factor * in_planes)
        layers = [
            *conv_1x1x1_bn(in_planes, hidden_dim, norm=norm),
            HSwish(),
            *conv_1x1x1_bn(hidden_dim, out_planes, norm=norm),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


@BACKBONES.register_module()
class MobileNetV3_LGD(MobileNetV3_S3D):
    def __init__(self, mix_paths, pool_method='average', channel_factor=3, **kwargs):
        super(MobileNetV3_LGD, self).__init__(**kwargs)

        assert len(mix_paths) == len(self.cfg)
        assert mix_paths[0] == 0

        mix_idx = [idx for idx in range(len(self.cfg)) if mix_paths[idx] > 0]
        assert len(mix_idx) > 0

        self.glob_to_local_idx = mix_idx
        self.local_to_glob_idx = [mix_idx[0] - 1] + mix_idx[:-1]
        self.glob_idx = mix_idx[:-1]

        self.glob_channels_num = [self.channels_num[idx] for idx in self.local_to_glob_idx]
        self.channel_factor = channel_factor

        self.lgd_upsample = nn.ModuleDict({
            f'upsample_{idx}': UpsampleBlock(
                glob_channels,
                self.channels_num[idx],
                factor=self.channel_factor,
                norm=self.weight_norm
            )
            for idx, glob_channels in zip(self.glob_to_local_idx, self.glob_channels_num)
        })
        self.lgd_pool = nn.ModuleDict({
            f'pooling_{idx}': PoolingBlock(
                self.channels_num[idx],
                self.channels_num[idx],
                factor=self.channel_factor,
                norm=self.weight_norm,
                mode=pool_method
            )
            for idx in self.local_to_glob_idx
        })
        self.lgd_glob = nn.ModuleDict({
            f'glob_{idx}': GlobBlock(
                glob_channels,
                self.channels_num[idx],
                factor=self.channel_factor,
                norm=self.weight_norm
            )
            for idx, glob_channels in zip(self.glob_idx, self.glob_channels_num[:-1])
        })

    def forward(self, x, return_extra_data=False, enable_extra_modules=True):
        y = self._norm_input(x)

        local_y = y
        glob_y = None

        local_outs = []
        feature_data, att_data, sgs_data = dict(), dict(), dict()
        for module_idx in range(len(self.features)):
            local_y = self._infer_module(
                local_y, module_idx, return_extra_data, enable_extra_modules, feature_data, att_data
            )

            if self.sgs_modules is not None and module_idx in self.sgs_idx:
                sgs_module_name = 'sgs_{}'.format(module_idx)
                sgs_module = self.sgs_modules[sgs_module_name]

                if self.enable_sgs_loss:
                    local_y, sgs_extra_data = sgs_module(local_y, return_extra_data=True)
                    sgs_data[sgs_module_name] = sgs_extra_data
                else:
                    local_y = sgs_module(local_y)

            if module_idx in self.glob_to_local_idx:
                assert glob_y is not None

                upsample_module = self.lgd_upsample[f'upsample_{module_idx}']
                local_y = upsample_module(glob_y) + local_y

            if module_idx in self.local_to_glob_idx:
                pooling_module = self.lgd_pool[f'pooling_{module_idx}']
                pooled_local_y = pooling_module(local_y)

                if glob_y is not None:
                    glob_module = self.lgd_glob[f'glob_{module_idx}']
                    glob_y = glob_module(glob_y) + pooled_local_y
                else:
                    glob_y = pooled_local_y

            if module_idx in self.out_ids:
                local_outs.append(local_y)

        local_outs = self._out_conv(local_outs, return_extra_data, enable_extra_modules, att_data)

        if return_extra_data:
            return local_outs, dict(feature_data=feature_data, att_data=att_data, sgs_data=sgs_data)
        else:
            return local_outs
