import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init

from ...core.utils import load_checkpoint, inflate_weights
from ...core.ops import (Dropout, Conv3d, HSigmoid, HSwish, conv_kxkxk_bn, conv_1xkxk_bn,
                         conv_kx1x1_bn, conv_1x1x1_bn, gumbel_sigmoid, SimilarityGuidedSampling)
from ...utils import get_root_logger
from ..registry import BACKBONES
from ..losses import TotalVarianceLoss
from .mobilenetv3 import make_divisible, MobileNetV3


class SELayer_3D(nn.Module):
    def __init__(self, in_channels, reduction=4, reduce_temporal=False, norm='none'):
        super(SELayer_3D, self).__init__()

        self.reduce_temporal = reduce_temporal

        self.avg_pool = nn.AdaptiveAvgPool3d((1 if reduce_temporal else None, 1, 1))
        self.fc = nn.Sequential(
            Conv3d(in_channels, in_channels // reduction, kernel_size=1, padding=0, normalization=norm),
            nn.ReLU(inplace=True),
            Conv3d(in_channels // reduction, in_channels, kernel_size=1, padding=0, normalization=norm),
            HSigmoid()
        )

    def forward(self, x):
        if torch.onnx.is_in_onnx_export() and not self.reduce_temporal:
            glob_context = F.avg_pool3d(x, (1, int(x.shape[3]), int(x.shape[4])), stride=1, padding=0)
        else:
            glob_context = self.avg_pool(x)

        mask = self.fc(glob_context)
        return mask * x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1.0, 0.0)


class ResidualAttention(nn.Module):
    def __init__(self, in_channels, kernels=3, scale=1.0, add_temporal=True, gumbel=True,
                 norm='none', residual=True, reg_weight=1.0, adaptive_threshold=True,
                 threshold=0.5, neg_fraction=0.7, tv_loss=True, gt_regression=False, enable_loss=True):
        super(ResidualAttention, self).__init__()

        self.gumbel = gumbel
        self.residual = residual
        self.scale = scale
        assert self.scale > 0.0
        self.reg_weight = reg_weight
        assert self.reg_weight > 0.0
        self.adaptive_threshold = adaptive_threshold
        self.threshold = threshold
        assert 0.0 < self.threshold < 1.0
        self.neg_fraction = neg_fraction
        assert 0.0 < self.neg_fraction < 1.0
        self.enable_loss = enable_loss
        self.gt_regression = gt_regression

        if enable_loss and tv_loss:
            temp_kernel = 3 if add_temporal else 1
            self.tv_loss = TotalVarianceLoss(
                spatial_kernels=3, temporal_kernels=temp_kernel, num_channels=1,
                hard_values=True, limits=(0.0, 1.0), threshold=0.5
            )
        else:
            self.tv_loss = None

        self.spatial_logits = nn.Sequential(
            *conv_1xkxk_bn(in_channels, in_channels, kernels, 1, groups=in_channels, norm=norm),
            HSwish(),
            *conv_1x1x1_bn(in_channels, 1, norm=norm)
        )

        if add_temporal:
            self.temporal_logits = nn.Sequential(
                nn.Sequential(),
                *conv_kx1x1_bn(in_channels, in_channels, kernels, 1, groups=in_channels, norm=norm),
                HSwish(),
                *conv_1x1x1_bn(in_channels, 1, norm=norm)
            )
        else:
            self.temporal_logits = None

    def forward(self, x, return_extra_data=False):
        spatial_logits = self.spatial_logits(x)

        if self.temporal_logits is not None:
            pooled_x = F.avg_pool3d(x, (1, int(x.shape[3]), int(x.shape[4])), stride=1, padding=0)
            temporal_logits = self.temporal_logits(pooled_x)
            logits = spatial_logits + temporal_logits
        else:
            logits = spatial_logits

        if self.gumbel and self.training:
            soft_mask = gumbel_sigmoid(self.scale * logits)
        else:
            soft_mask = torch.sigmoid(self.scale * logits)

        if self.residual:
            out = soft_mask * x + x
        else:
            out = soft_mask * x

        if return_extra_data:
            return out, dict(attention_logits=logits, attention_scores=soft_mask)
        else:
            return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1.0, 0.0)

    def loss(self, attention_logits, attention_scores, attention_mask=None, **kwargs):
        if not self.enable_loss:
            return None

        spatial_size = attention_scores.size(3) * attention_scores.size(4)

        if attention_mask is not None and self.gt_regression:
            with torch.no_grad():
                trg_values = F.interpolate(attention_mask, size=attention_logits.size()[2:], mode='nearest')

            losses = F.binary_cross_entropy_with_logits(attention_logits, trg_values, reduction='none')

            pos_mask = trg_values > 0
            pos_losses = losses[pos_mask]
            neg_losses = losses[~pos_mask]

            num_trg_losses = 0
            out_loss = 0.0
            if pos_losses.numel() > 0:
                out_loss += 1.0 / float(pos_losses.numel()) * pos_losses.sum()
                num_trg_losses += 1
            if neg_losses.numel() > 0:
                out_loss += 1.0 / float(neg_losses.numel()) * neg_losses.sum()
                num_trg_losses += 1
            out_loss = out_loss / float(num_trg_losses)
        elif self.tv_loss is not None:
            out_loss = self.tv_loss(attention_scores)
        else:
            conf = attention_scores.view(-1, spatial_size)

            if self.adaptive_threshold:
                num_values = int(self.neg_fraction * spatial_size)
                threshold, _ = conf.kthvalue(num_values, dim=-1, keepdim=True)
            else:
                threshold = self.threshold

            losses = torch.where(conf < threshold, conf, 1.0 - conf)
            out_loss = losses.mean()

        return self.reg_weight * out_loss


class DynamicAttention(nn.Module):
    def __init__(self, in_channels, kernels=3, scale=1.0, norm='none', residual=True, **kwargs):
        super(DynamicAttention, self).__init__()

        self.scale = scale
        self.residual = residual

        self.key = nn.Sequential(
            *conv_1xkxk_bn(in_channels, in_channels, kernels, 1, groups=in_channels, norm=norm),
            HSwish(),
            *conv_1x1x1_bn(in_channels, in_channels, norm=norm),
            nn.AdaptiveMaxPool3d((1, 1, 1))
        )
        self.value = nn.Sequential(
            *conv_1xkxk_bn(in_channels, in_channels, kernels, 1, groups=in_channels, norm=norm),
            HSwish(),
            *conv_1x1x1_bn(in_channels, in_channels, norm=norm)
        )

    def forward(self, x):
        key = self.key(x)
        values = self.value(x)

        responses = (key * values).sum(dim=1, keepdim=True)
        soft_mask = torch.sigmoid(self.scale * responses)

        if self.residual:
            out = soft_mask * x + x
        else:
            out = soft_mask * x

        return out


class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels, internal_factor=3.0, norm='none'):
        super(GlobalContextBlock, self).__init__()

        self.encoder = conv_1x1x1_bn(in_channels, 1, norm=norm, as_list=False)

        hidden_dim = int(internal_factor * in_channels)
        layers = [
            *conv_1x1x1_bn(in_channels, hidden_dim, norm=norm),
            HSwish(),
            *conv_1x1x1_bn(hidden_dim, in_channels, norm=norm)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        _, c, t, h, w = x.size()

        keys = self.encoder(x)
        attention_map = torch.softmax(keys.view(-1, t * h * w, 1), dim=1)

        context = torch.matmul(x.view(-1, c, t * h * w), attention_map)

        y = self.conv(context.view(-1, c, 1, 1, 1))
        out = x + y

        return out


class InvertedResidual_S3D(nn.Module):
    def __init__(self, in_planes, hidden_dim, out_planes, spatial_kernels, temporal_kernels,
                 spatial_stride, temporal_stride, use_se, use_hs,
                 temporal_avg_pool=False, dropout_cfg=None, internal_dropout=True,
                 dw_temporal=True, norm='none', scale=None, center_weight=None):
        super(InvertedResidual_S3D, self).__init__()
        assert spatial_stride in [1, 2]

        self.identity = spatial_stride == 1 and temporal_stride == 1 and in_planes == out_planes
        self.scale = scale
        self.internal_dropout = internal_dropout

        if in_planes == hidden_dim:
            conv_layers = [
                # dw
                *conv_1xkxk_bn(hidden_dim, hidden_dim, spatial_kernels, spatial_stride,
                               groups=hidden_dim, norm=norm),
                HSwish() if use_hs else nn.ReLU(inplace=True),

                # Squeeze-and-Excite
                SELayer_3D(hidden_dim, norm=norm) if use_se else nn.Sequential()
            ]
        else:
            # pw
            conv_layers = [
                *conv_1x1x1_bn(in_planes, hidden_dim, norm=norm),
                HSwish() if use_hs else nn.ReLU(inplace=True)
            ]

            # dw
            if dw_temporal:
                conv_layers.extend(conv_kxkxk_bn(
                    hidden_dim, hidden_dim, spatial_kernels, temporal_kernels,
                    spatial_stride, temporal_stride, groups=hidden_dim, norm=norm
                ))
            else:
                conv_layers.extend(conv_1xkxk_bn(
                    hidden_dim, hidden_dim, spatial_kernels, spatial_stride,
                    groups=hidden_dim, norm=norm
                ))

            # Squeeze-and-Excite
            conv_layers.extend([
                SELayer_3D(hidden_dim, norm=norm) if use_se else nn.Sequential(),
                HSwish() if use_hs else nn.ReLU(inplace=True)
            ])

        # pw-linear
        if dw_temporal:
            conv_layers.extend(conv_1x1x1_bn(hidden_dim, out_planes, norm=norm))
        else:
            if temporal_avg_pool and temporal_stride > 1:
                conv_layers.extend([*conv_kx1x1_bn(hidden_dim, out_planes, temporal_kernels, 1,
                                                   norm=norm, center_weight=center_weight),
                                    nn.AvgPool3d(kernel_size=(temporal_stride, 1, 1),
                                                 stride=(temporal_stride, 1, 1),
                                                 padding=((temporal_stride - 1) // 2, 0, 0))
                                    ])
            else:
                conv_layers.extend(conv_kx1x1_bn(hidden_dim, out_planes, temporal_kernels, temporal_stride,
                                                 norm=norm, center_weight=center_weight))

        self.conv = nn.Sequential(*conv_layers)

        if dropout_cfg is not None and self.identity:
            self.dropout = Dropout(**dropout_cfg)
        else:
            self.dropout = None

    def forward(self, x):
        y = self.conv(x)

        if self.dropout is not None and self.internal_dropout:
            y = self.dropout(y, x)

        if self.identity and self.scale is not None and self.scale != 1.0:
            y *= self.scale

        out = x + y if self.identity else y

        if self.dropout is not None and not self.internal_dropout:
            out = self.dropout(out)

        return out


@BACKBONES.register_module()
class MobileNetV3_S3D(nn.Module):
    def __init__(self,
                 mode,
                 num_input_layers=3,
                 pretrained=None,
                 pretrained2d=True,
                 width_mult=1.0,
                 pool1_stride_t=2,
                 temporal_kernels=3,
                 temporal_strides=1,
                 bn_eval=False,
                 bn_frozen=False,
                 use_temporal_avg_pool=False,
                 use_dw_temporal=0,
                 use_gcb=0,
                 use_st_att=0,
                 attention_cfg=None,
                 input_bn=False,
                 out_conv=True,
                 out_attention=False,
                 out_ids=None,
                 internal_dropout=True,
                 dropout_cfg=None,
                 use_dropout=0,
                 weight_norm='none',
                 center_conv_weight=None,
                 sgs_cfg=None):
        super(MobileNetV3_S3D, self).__init__()

        # setting of inverted residual blocks
        self.mode = mode
        assert self.mode in ['large', 'small']
        self.cfg = MobileNetV3.arch_settings[self.mode]
        self.width_mult = width_mult
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.temporal_kernels = temporal_kernels \
            if not isinstance(temporal_kernels, int) else (temporal_kernels,) * len(self.cfg)
        self.temporal_strides = temporal_strides \
            if not isinstance(temporal_strides, int) else (temporal_strides,) * len(self.cfg)
        self.use_dw_temporal = use_dw_temporal \
            if not isinstance(use_dw_temporal, int) else (use_dw_temporal,) * len(self.cfg)
        self.use_st_att = use_st_att \
            if not isinstance(use_st_att, int) else (use_st_att,) * len(self.cfg)
        self.use_gcb = use_gcb \
            if not isinstance(use_gcb, int) else (use_gcb,) * len(self.cfg)
        self.use_dropout = use_dropout \
            if not isinstance(use_dropout, int) else (use_dropout,) * len(self.cfg)

        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.weight_norm = weight_norm
        self.dropout_cfg = dropout_cfg
        self.internal_dropout = internal_dropout

        if input_bn:
            self.input_bn = nn.BatchNorm3d(num_input_layers)
        else:
            self.input_bn = None

        # building first layer
        input_channel = make_divisible(16 * width_mult, 8)
        self.channels_num = [input_channel]

        first_stage_layers = [*conv_1xkxk_bn(num_input_layers, input_channel,
                                             k=3, spatial_stride=2, norm=weight_norm)]
        if pool1_stride_t > 1:
            first_stage_layers.append(nn.AvgPool3d(kernel_size=(pool1_stride_t, 1, 1),
                                                   stride=(pool1_stride_t, 1, 1),
                                                   padding=((pool1_stride_t - 1) // 2, 0, 0)))
        first_stage_layers.append(HSwish())
        layers = [nn.Sequential(*first_stage_layers)]
        num_layers_before = len(layers)

        attention_block = ResidualAttention
        attention_cfg = dict() if attention_cfg is None else attention_cfg
        residual_block = InvertedResidual_S3D

        # building inverted residual blocks
        self.attentions, self.gcb_modules = nn.ModuleDict(), nn.ModuleDict()
        for layer_id, layer_params in enumerate(self.cfg):
            k, exp_size, c, use_se, use_hs, s = layer_params
            output_channel = make_divisible(c * width_mult, 8)
            spatial_stride = s
            temporal_stride = self.temporal_strides[layer_id]
            use_dw_temporal = self.use_dw_temporal[layer_id] > 0
            module_dropout_cfg = dropout_cfg if self.use_dropout[layer_id] > 0 else None

            layers.append(residual_block(
                input_channel, exp_size, output_channel, k, self.temporal_kernels[layer_id],
                spatial_stride, temporal_stride, use_se, use_hs,
                use_temporal_avg_pool, module_dropout_cfg, internal_dropout,
                use_dw_temporal, weight_norm, center_weight=center_conv_weight)
            )

            if self.use_st_att[layer_id] > 0:
                att_name = 'st_att_{}'.format(layer_id + num_layers_before)
                self.attentions[att_name] = attention_block(output_channel, **attention_cfg)

            if self.use_gcb[layer_id] > 0:
                gcb_name = 'gcb_{}'.format(layer_id + num_layers_before)
                self.gcb_modules[gcb_name] = GlobalContextBlock(output_channel, norm=weight_norm)

            input_channel = output_channel
            self.channels_num.append(output_channel)

        self.features = nn.ModuleList(layers)

        if sgs_cfg is not None:
            assert len(sgs_cfg.idx) >= -1
            assert len(sgs_cfg.idx) == len(sgs_cfg.bins)

            self.sgs_modules = nn.ModuleDict()
            self.sgs_idx = []
            for layer_id, num_bins in zip(sgs_cfg.idx, sgs_cfg.bins):
                sgs_layer_id = layer_id + num_layers_before
                self.sgs_idx.append(sgs_layer_id)

                sgs_name = 'sgs_{}'.format(sgs_layer_id)
                self.sgs_modules[sgs_name] = SimilarityGuidedSampling(
                    self.channels_num[sgs_layer_id],
                    num_bins,
                    internal_factor=sgs_cfg.internal_factor,
                    norm=weight_norm
                )

            self.enable_sgs_loss = sgs_cfg.get('enable_loss', True)
        else:
            self.sgs_modules = None

        # building last several layers
        if out_conv:
            exp_size = self.cfg[-1][1]
            out_channels = make_divisible(exp_size * width_mult, 8)

            self.conv = nn.Sequential(
                nn.Sequential(*conv_1x1x1_bn(input_channel, out_channels, norm=weight_norm),
                              HSwish()),
                SELayer_3D(out_channels, norm=weight_norm) if mode == 'small' else nn.Sequential()
            )

            if out_attention:
                self.attentions['out_st_att'] = attention_block(out_channels, **attention_cfg)
        else:
            self.conv = None

        if out_ids is None:
            self.out_ids = [len(self.features) - 1]
        else:
            self.out_ids = [out_id + num_layers_before for out_id in out_ids]

    def forward(self, x, return_extra_data=False, enable_extra_modules=True):
        y = self._norm_input(x)

        outs = []
        feature_data, att_data, sgs_data = dict(), dict(), dict()
        for module_idx in range(len(self.features)):
            y = self._infer_module(
                y, module_idx, return_extra_data, enable_extra_modules, feature_data, att_data
            )

            if self.sgs_modules is not None and module_idx in self.sgs_idx:
                sgs_module_name = 'sgs_{}'.format(module_idx)
                sgs_module = self.sgs_modules[sgs_module_name]

                if self.enable_sgs_loss:
                    y, sgs_extra_data = sgs_module(y, return_extra_data=True)
                    sgs_data[sgs_module_name] = sgs_extra_data
                else:
                    y = sgs_module(y)

            if module_idx in self.out_ids:
                outs.append(y)

        assert len(outs) > 0
        outs = self._out_conv(outs, return_extra_data, enable_extra_modules, att_data)

        if return_extra_data:
            return outs, dict(feature_data=feature_data, att_data=att_data, sgs_data=sgs_data)
        else:
            return outs

    def _norm_input(self, x):
        if self.input_bn is not None:
            x = self.input_bn(x)

        return x

    def _infer_module(self, x, module_idx, return_extra_data, enable_extra_modules, feature_data, att_data):
        module = self.features[module_idx]

        if return_extra_data and hasattr(module, 'loss'):
            y, feature_extra_data = module(x, return_extra_data=True)
            feature_data[module_idx] = feature_extra_data
        else:
            y = module(x)

        attention_module_name = 'st_att_{}'.format(module_idx)
        if attention_module_name in self.attentions:
            attention_module = self.attentions[attention_module_name]

            if return_extra_data and hasattr(attention_module, 'loss'):
                y, att_extra_data = attention_module(y, return_extra_data=True)
                att_data[attention_module_name] = att_extra_data
            elif enable_extra_modules:
                y = attention_module(y)
            else:
                y += y  # simulate residual block

        gcb_module_name = 'gcb_{}'.format(module_idx)
        if gcb_module_name in self.gcb_modules:
            gcb_module = self.gcb_modules[gcb_module_name]
            y = gcb_module(y)

        return y

    def _out_conv(self, outs, return_extra_data, enable_extra_modules, att_data):
        if self.conv is not None:
            y = self.conv(outs[-1])

            if 'out_st_att' in self.attentions:
                out_attention = self.attentions['out_st_att']
                if return_extra_data and hasattr(out_attention, 'loss'):
                    y, out_att_extra_data = out_attention(y, return_extra_data=True)
                    att_data['out_st_att'] = out_att_extra_data
                elif enable_extra_modules:
                    y = out_attention(y)
                else:
                    y += y  # simulate residual block

            outs.append(y)

        return outs

    def loss(self, feature_data=None, att_data=None, sgs_data=None, **kwargs):
        losses = dict()
        losses.update(self._process_loss(feature_data, self.features, 'loss/freg', **kwargs))
        losses.update(self._process_loss(att_data, self.attentions, 'loss/att', **kwargs))
        losses.update(self._process_loss(sgs_data, self.sgs_modules, 'loss/sgs', **kwargs))
        return losses

    @staticmethod
    def _process_loss(data, modules, name, **kwargs):
        if data is None or len(data) == 0:
            return dict()

        out_losses = []
        for module_name, module_data in data.items():
            module = modules[module_name]

            loss_value = module.loss(**module_data, **kwargs)
            if loss_value is not None:
                out_losses.append(loss_value)

        if len(out_losses) > 0:
            return {name: torch.mean(torch.stack(out_losses))}
        else:
            return dict()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.data.fill_(1)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.data.normal_(0, 0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.zero_()

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger(log_level='INFO')
            if self.pretrained2d:
                inflate_weights(self, self.pretrained, logger=logger)
            else:
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            self._init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def train(self, train_mode=True):
        super(MobileNetV3_S3D, self).train(train_mode)

        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()

                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False

        return self
