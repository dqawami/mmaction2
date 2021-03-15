import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import constant_init, kaiming_init

from .base import BaseHead
from ..registry import HEADS
from ...core.ops import conv_1x1x1_bn, normalize, AngleMultipleLinear, KernelizedClassifier


@HEADS.register_module()
class ClsHead(BaseHead):
    def __init__(self,
                 spatial_type=None,
                 temporal_size=1,
                 spatial_size=7,
                 init_std=0.01,
                 embedding=False,
                 enable_rebalance=False,
                 rebalance_num_groups=3,
                 rebalance_alpha=0.9,
                 classification_layer='linear',
                 embd_size=128,
                 num_centers=1,
                 st_scale=5.0,
                 reg_threshold=0.1,
                 enable_sampling=False,
                 adaptive_sampling=False,
                 sampling_angle_std=None,
                 reg_weight=1.0,
                 enable_class_mixing=False,
                 class_mixing_alpha=0.1,
                 enable_bias=False,
                 enable_bn=True,
                 **kwargs):
        super(ClsHead, self).__init__(**kwargs)

        self.embd_size = embd_size
        self.temporal_feature_size = temporal_size
        self.spatial_feature_size = \
            spatial_size \
            if not isinstance(spatial_size, int) \
            else (spatial_size, spatial_size)
        self.init_std = init_std

        self.avg_pool = None
        if spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.enable_rebalance = enable_rebalance and rebalance_num_groups > 1
        if self.enable_rebalance:
            assert classification_layer == 'linear', 'Re-balancing supports linear head only'
            assert self.class_sizes is not None, 'Re-balancing requires class_sizes'

            self.rebalance_alpha = rebalance_alpha
            assert 0.0 <= self.rebalance_alpha <= 1.0
            rebalance_zero_mask, imbalance_ratios = self._build_rebalance_masks(
                self.class_sizes, rebalance_num_groups
            )
            np.set_printoptions(precision=3)
            print(f'[INFO] Balance ratios for dataset with {self.num_classes} '
                  f'classes : {imbalance_ratios}')
            self.register_buffer('rebalance_zero_mask', torch.from_numpy(rebalance_zero_mask))
            rebalance_weights = np.where(rebalance_zero_mask > 0.0,
                                         1.0 / np.sum(rebalance_zero_mask, axis=1, keepdims=True),
                                         np.zeros_like(rebalance_zero_mask))
            self.register_buffer('rebalance_weights', torch.from_numpy(rebalance_weights))

        self.with_embedding = embedding and self.embd_size > 0
        if self.with_embedding:
            if self.enable_rebalance:
                assert not enable_sampling, 'Re-balancing does not support embd sampling'
                assert not enable_class_mixing, 'Re-balancing does not support embd mixing'

                self.fc_pre_angular = nn.ModuleList([
                    conv_1x1x1_bn(self.in_channels, self.embd_size,
                                  as_list=False, bias=enable_bias, bn=enable_bn)
                    for _ in range(rebalance_num_groups)
                ])
            else:
                self.fc_pre_angular = conv_1x1x1_bn(self.in_channels, self.embd_size,
                                                    as_list=False, bias=enable_bias, bn=enable_bn)

            if classification_layer == 'linear':
                self.fc_angular = AngleMultipleLinear(self.embd_size, self.num_classes, num_centers,
                                                      st_scale, reg_weight, reg_threshold)
            elif classification_layer == 'kernel':
                assert not enable_class_mixing, 'Kernelized classifier does not support class mixing'

                self.fc_angular = KernelizedClassifier(self.embd_size, self.num_classes, num_centers)
            else:
                raise ValueError(f'Unknown classification layer: {classification_layer}')
        else:
            if self.enable_rebalance:
                self.internal_num_channels = int(1.3 * self.in_channels)
                self.fc_pre_cls = nn.ModuleList([
                    conv_1x1x1_bn(self.in_channels, self.internal_num_channels,
                                  as_list=False, bias=enable_bias, bn=enable_bn)
                    for _ in range(rebalance_num_groups)
                ])
                self.fc_cls_out = nn.Linear(self.internal_num_channels, self.num_classes)
            else:
                self.fc_pre_cls = None
                self.fc_cls_out = nn.Linear(self.in_channels, self.num_classes)

        self.enable_sampling = (self.with_embedding and
                                enable_sampling and
                                sampling_angle_std is not None and
                                sampling_angle_std > 0.0)
        self.adaptive_sampling = (self.enable_sampling and
                                  adaptive_sampling and
                                  self.class_sizes is not None)
        if self.enable_sampling:
            assert sampling_angle_std < 0.5 * np.pi

            if self.adaptive_sampling:
                counts = np.ones([self.num_classes], dtype=np.float32)
                for class_id, class_size in self.class_sizes.items():
                    counts[class_id] = class_size

                class_angle_std = sampling_angle_std * np.power(counts, -1. / 4.)
                self.register_buffer('sampling_angle_std', torch.from_numpy(class_angle_std))
            else:
                self.sampling_angle_std = sampling_angle_std

        self.enable_class_mixing = enable_class_mixing
        self.alpha_class_mixing = class_mixing_alpha

    def init_weights(self):
        if self.with_embedding:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                    constant_init(m, 1.0, 0.0)
        else:
            nn.init.normal_(self.fc_cls_out.weight, 0, self.init_std)
            nn.init.constant_(self.fc_cls_out.bias, 0)

    @staticmethod
    def _build_rebalance_masks(classes_meta, num_groups):
        assert 1 < num_groups <= 3
        assert len(classes_meta) >= num_groups
        num_borders = num_groups - 1

        ordered_class_sizes = list(sorted(classes_meta.items(), key=lambda tup: -tup[1]))
        class_ids = np.array([class_id for class_id, _ in ordered_class_sizes], dtype=np.int32)
        class_sizes = np.array([class_size for _, class_size in ordered_class_sizes], dtype=np.float32)

        all_border_combinations = itertools.combinations(range(1, len(ordered_class_sizes)), num_borders)
        all_border_combinations = np.array(list(all_border_combinations))

        ratios = [class_sizes[0] / class_sizes[all_border_combinations[:, 0] - 1]]
        for ii in range(num_borders - 1):
            starts = all_border_combinations[:, ii]
            ends = all_border_combinations[:, ii + 1] - 1
            ratios.append(class_sizes[starts] / class_sizes[ends])
        ratios.append(class_sizes[all_border_combinations[:, -1]] / class_sizes[-1])

        ratios = np.stack(ratios, axis=1)
        costs = np.max(ratios, axis=1) - np.min(ratios, axis=1)
        best_match_idx = np.argmin(costs)
        best_border_combination = all_border_combinations[best_match_idx]
        best_ratios = ratios[best_match_idx]

        groups = [class_ids[:best_border_combination[0]]]
        for ii in range(num_borders - 1):
            groups.append(class_ids[best_border_combination[ii]:best_border_combination[ii + 1]])
        groups.append(class_ids[best_border_combination[-1]:])

        num_classes = max(classes_meta.keys()) + 1
        mask = np.zeros([num_groups, num_classes], dtype=np.float32)
        for group_id in range(num_groups):
            for ii in range(group_id, num_groups):
                mask[group_id, groups[ii]] = 1.0

        return mask.reshape([1, num_groups, num_classes]), best_ratios

    def _squash_features(self, x):
        if x.ndimension() == 4:
            x = x.unsqueeze(2)

        if self.avg_pool is not None:
            x = self.avg_pool(x)

        return x

    @staticmethod
    def _mix_embd(norm_embd, labels, norm_centers, num_classes, alpha_class_mixing):
        with torch.no_grad():
            sampled_ids = torch.randint_like(labels, 0, num_classes - 1)
            sampled_neg_ids = torch.where(sampled_ids < labels, sampled_ids, sampled_ids + 1)
            random_centers = norm_centers[sampled_neg_ids]

        alpha = alpha_class_mixing * torch.rand_like(labels, dtype=norm_embd.dtype)
        mixed_embd = (1.0 - alpha.view(-1, 1)) * norm_embd + alpha.view(-1, 1) * random_centers
        norm_embd = normalize(mixed_embd, dim=1)

        return norm_embd

    @staticmethod
    def _sample_embd(norm_embd, labels, batch_size, adaptive_sampling, sampling_angle_std):
        with torch.no_grad():
            unit_directions = F.normalize(torch.randn_like(norm_embd), dim=1)
            dot_prod = torch.sum(norm_embd * unit_directions, dim=1, keepdim=True)
            orthogonal_directions = unit_directions - dot_prod * norm_embd

            if adaptive_sampling and labels is not None:
                all_angle_std = sampling_angle_std.expand(batch_size, -1)
                class_indices = torch.arange(batch_size, device=labels.device)
                angle_std = all_angle_std[class_indices, labels].view(-1, 1)
            else:
                angle_std = sampling_angle_std

            angles = angle_std * torch.randn_like(dot_prod)
            alpha = torch.clamp_max(torch.where(angles > 0.0, angles, torch.neg(angles)), 0.5 * np.pi)
            cos_alpha = torch.cos(alpha)
            sin_alpha = torch.sin(alpha)

        out_norm_embd = cos_alpha * norm_embd + sin_alpha * orthogonal_directions

        return out_norm_embd

    def forward(self, x, labels=None, return_extra_data=False, **kwargs):
        x = self._squash_features(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.with_embedding:
            if self.enable_rebalance:
                unnorm_embd = [module(x) for module in self.fc_pre_angular]
                norm_embd = [normalize(embd.view(-1, self.embd_size), dim=1) for embd in unnorm_embd]
                split_scores = [self.fc_angular(embd) for embd in norm_embd]

                all_scores = torch.cat([score.unsqueeze(1) for score in split_scores], dim=1)
                main_cls_score = torch.sum(all_scores * self.rebalance_weights, dim=1)

                extra_cls_score = split_scores
            else:
                unnorm_embd = self.fc_pre_angular(x)
                norm_embd = normalize(unnorm_embd.view(-1, self.embd_size), dim=1)

                if self.training:
                    if self.enable_class_mixing:
                        norm_class_centers = normalize(self.fc_angular.weight.permute(1, 0), dim=1)
                        norm_embd = self._mix_embd(
                            norm_embd, labels, norm_class_centers, self.num_classes, self.alpha_class_mixing
                        )

                    if self.enable_sampling:
                        norm_embd = self._sample_embd(
                            norm_embd, labels, x.shape[0], self.adaptive_sampling, self.sampling_angle_std
                        )

                main_cls_score = self.fc_angular(norm_embd)
                extra_cls_score = None
        else:
            norm_embd = None
            if self.enable_rebalance:
                unnorm_embd = [module(x).view(-1, self.internal_num_channels) for module in self.fc_pre_cls]
                split_scores = [self.fc_cls_out(embd) for embd in unnorm_embd]

                all_scores = torch.cat([score.unsqueeze(1) for score in split_scores], dim=1)
                main_cls_score = torch.sum(all_scores * self.rebalance_weights, dim=1)

                extra_cls_score = split_scores
            else:
                extra_cls_score = None
                main_cls_score = self.fc_cls_out(x.view(-1, self.in_channels))

        if return_extra_data:
            return main_cls_score, norm_embd, extra_cls_score
        else:
            return main_cls_score

    def loss(self, main_cls_score, labels, norm_embd, name, extra_cls_score, **kwargs):
        losses = dict()

        main_cls_loss = self.head_loss(main_cls_score, labels)
        if hasattr(self.head_loss, 'last_scale'):
            losses['scale/cls' + name] = self.head_loss.last_scale

        if self.enable_rebalance:
            with torch.no_grad():
                all_indexed_labels_mask = torch.zeros_like(main_cls_score, dtype=torch.float32)\
                    .scatter_(1, labels.view(-1, 1), 1)
                indexed_labels_mask = all_indexed_labels_mask.unsqueeze(1) * self.rebalance_zero_mask

                valid_samples_mask = indexed_labels_mask.sum(dim=2) > 0.0

            group_losses = []
            for group_id, group_cls_score in enumerate(extra_cls_score):
                with torch.no_grad():
                    group_samples_mask = valid_samples_mask[:, group_id]
                    if torch.sum(group_samples_mask) == 0:
                        continue

                    group_labels_mask = indexed_labels_mask[:, group_id]
                    group_logits_mask = self.rebalance_zero_mask[:, group_id].view(-1) > 0.0

                    group_labels = group_labels_mask[group_samples_mask][:, group_logits_mask]
                    group_targets = torch.argmax(group_labels, dim=1)

                group_cls_score = group_cls_score[group_samples_mask][:, group_logits_mask]

                group_loss_args = dict(increment_step=False) if self.with_embedding else dict()
                group_loss = self.head_loss(group_cls_score, group_targets, **group_loss_args)

                group_losses.append(group_loss)

            fused_group_loss = sum(group_losses) / float(len(group_losses))
            losses['loss/cls' + name] = \
                (1.0 - self.rebalance_alpha) * main_cls_loss + self.rebalance_alpha * fused_group_loss
        else:
            losses['loss/cls' + name] = main_cls_loss

        if self.losses_extra is not None and not self.enable_rebalance:
            for extra_loss_name, extra_loss in self.losses_extra.items():
                losses[extra_loss_name.replace('_', '/') + name] = extra_loss(
                    norm_embd, main_cls_score, labels)

        if self.with_embedding and hasattr(self.fc_angular, 'loss'):
            losses.update(self.fc_angular.loss(name))

        return losses

    @property
    def last_scale(self):
        if hasattr(self.head_loss, 'last_scale'):
            return self.head_loss.last_scale
        else:
            return None
