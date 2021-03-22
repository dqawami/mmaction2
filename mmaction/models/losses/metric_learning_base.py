from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseWeightedLoss
from .. import builder
from ...core.ops import entropy


class BaseMetricLearningLoss(BaseWeightedLoss):
    _loss_filter_types = ['positives', 'top_k']

    def __init__(self, scale_cfg, pr_product=False, conf_penalty_weight=None,
                 filter_type=None, top_k=None, class_sizes=None,
                 enable_class_weighting=False, **kwargs):
        super(BaseMetricLearningLoss, self).__init__(**kwargs)

        self._enable_pr_product = pr_product
        self._conf_penalty_weight = conf_penalty_weight
        self._filter_type = filter_type
        self._top_k = top_k
        if self._filter_type == 'top_k':
            assert self._top_k is not None and self._top_k >= 1

        self.scale_scheduler = builder.build_scheduler(scale_cfg)
        self._last_scale = 0.0

        self.class_sizes = class_sizes
        if enable_class_weighting and self.class_sizes is not None:
            self.num_classes = max(list(self.class_sizes.keys())) + 1
            weights = self._estimate_class_weights(self.class_sizes, self.num_classes)
            self.register_buffer('class_weights', torch.from_numpy(weights))
        else:
            self.num_classes = None
            self.class_weights = None

    @property
    def with_regularization(self):
        return self._conf_penalty_weight is not None and self._conf_penalty_weight > 0.0

    @property
    def with_class_weighting(self):
        return self.class_weights is not None

    @property
    def with_filtering(self):
        return self._filter_type is not None and self._filter_type in self._loss_filter_types

    @property
    def with_pr_product(self):
        return self._enable_pr_product

    @property
    def last_scale(self):
        return self._last_scale

    def update_state(self, num_iters_per_epoch):
        assert num_iters_per_epoch > 0
        self.scale_scheduler.iters_per_epoch = num_iters_per_epoch

    @staticmethod
    def _estimate_class_weights(class_sizes, num_classes, num_steps=1000, num_samples=14, scale=1.0, eps=1e-4):
        class_ids = np.array(list(class_sizes.keys()), dtype=np.int32)
        counts = np.array(list(class_sizes.values()), dtype=np.float32)

        frequencies = counts / np.sum(counts)
        init_weights = np.reciprocal(frequencies + eps)

        average_weights = list()
        for _ in range(num_steps):
            ids = np.random.choice(class_ids, num_samples, p=frequencies)
            values = init_weights[ids]
            average_weights.append(np.mean(values))

        weights = scale / np.median(average_weights) * init_weights

        out_weights = np.zeros([num_classes], dtype=np.float32)
        for class_id, class_weight in zip(class_ids, weights):
            out_weights[class_id] = class_weight

        return out_weights

    @staticmethod
    def _pr_product(prod):
        alpha = torch.sqrt(1.0 - prod.pow(2.0))
        out_prod = alpha.detach() * prod + prod.detach() * (1.0 - alpha)

        return out_prod

    def _regularization(self, cos_theta, scale):
        probs = F.softmax(scale * cos_theta, dim=-1)
        entropy_values = entropy(probs, dim=-1)
        out_values = np.negative(self._conf_penalty_weight) * entropy_values

        return out_values

    def _reweight(self, losses, labels):
        with torch.no_grad():
            loss_weights = torch.gather(self.class_weights, 0, labels.view(-1))

        weighted_losses = loss_weights * losses

        return weighted_losses

    def _filter_losses(self, losses):
        if self._filter_type == 'positives':
            losses = losses[losses > 0.0]
        elif self._filter_type == 'top_k':
            valid_losses = losses[losses > 0.0]

            if valid_losses.numel() > 0:
                num_top_k = int(min(valid_losses.numel(), self._top_k))
                losses, _ = torch.topk(valid_losses, k=num_top_k)
            else:
                losses = valid_losses.new_zeros((0,))

        return losses

    def _forward(self, output, labels, increment_step=True):
        if increment_step:
            self._last_scale = self.scale_scheduler.get_scale_and_increment_step()
        else:
            self._last_scale = self.scale_scheduler.get_scale()

        if self.with_pr_product:
            output = self._pr_product(output)

        losses = self._calculate(output, labels, self._last_scale)

        if self.with_regularization:
            losses += self._regularization(output, self._last_scale)

        if self.with_class_weighting:
            losses = self._reweight(losses, labels)

        if self.with_filtering:
            losses = self._filter_losses(losses)

        return losses.mean() if losses.numel() > 0 else losses.sum()

    @abstractmethod
    def _calculate(self, output, labels, scale):
        pass
