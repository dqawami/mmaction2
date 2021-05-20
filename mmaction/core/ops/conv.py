import torch.nn as nn

from .conv3d import Conv3d


def conv_kxkxk_bn(in_planes, out_planes, spatial_k=3, temporal_k=3,
                  spatial_stride=1, temporal_stride=1, groups=1,
                  as_list=True, norm='none', center_weight=None):
    layers = [Conv3d(in_planes, out_planes, bias=False, groups=groups,
                     kernel_size=(temporal_k, spatial_k, spatial_k),
                     padding=((temporal_k - 1) // 2, (spatial_k - 1) // 2, (spatial_k - 1) // 2),
                     stride=(temporal_stride, spatial_stride, spatial_stride),
                     normalization=norm, center_weight=center_weight),
              nn.BatchNorm3d(out_planes)]
    return layers if as_list else nn.Sequential(*layers)


def conv_1xkxk_bn(in_planes, out_planes, k=3, spatial_stride=1, groups=1,
                  as_list=True, norm='none', center_weight=None):
    layers = [Conv3d(in_planes, out_planes, bias=False, groups=groups,
                     kernel_size=(1, k, k), padding=(0, (k - 1) // 2, (k - 1) // 2),
                     stride=(1, spatial_stride, spatial_stride),
                     normalization=norm, center_weight=center_weight),
              nn.BatchNorm3d(out_planes)]
    return layers if as_list else nn.Sequential(*layers)


def conv_kx1x1_bn(in_planes, out_planes, k, temporal_stride=1, groups=1,
                  as_list=True, norm='none', center_weight=None):
    layers = [Conv3d(in_planes, out_planes, bias=False, groups=groups,
                     kernel_size=(k, 1, 1), padding=((k - 1) // 2, 0, 0),
                     stride=(temporal_stride, 1, 1),
                     normalization=norm, center_weight=center_weight),
              nn.BatchNorm3d(out_planes)]
    return layers if as_list else nn.Sequential(*layers)


def conv_1x1x1_bn(in_planes, out_planes, as_list=True, norm='none', center_weight=None, bias=False, bn=True):
    conv_layer = Conv3d(
        in_planes, out_planes, bias=bias, kernel_size=1,
        padding=0, stride=1,
        normalization=norm, center_weight=center_weight
    )

    if bn:
        layers = [conv_layer,
                  nn.BatchNorm3d(out_planes)]
        return layers if as_list else nn.Sequential(*layers)
    else:
        return conv_layer
