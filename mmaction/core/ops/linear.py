import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from .math import normalize


class AngleMultipleLinear(nn.Module):
    """Based on SoftTriplet loss: https://arxiv.org/pdf/1909.05235.pdf
    """

    def __init__(self, in_features, num_classes, num_centers=1, scale=10.0, reg_weight=0.2, reg_threshold=0.2):
        super(AngleMultipleLinear, self).__init__()

        self.in_features = in_features
        assert in_features > 0
        self.num_classes = num_classes
        assert num_classes >= 2
        self.num_centers = num_centers
        assert num_centers >= 1
        self.scale = scale
        assert scale > 0.0

        weight_shape = [in_features, num_classes, num_centers] if num_centers > 1 else [in_features, num_classes]
        self.weight = Parameter(torch.Tensor(*weight_shape))
        self.weight.data.normal_().renorm_(2, 1, 1e-5).mul_(1e5)

        self.enable_regularization = reg_weight is not None and reg_weight > 0.0
        if self.enable_regularization:
            self.reg_weight = reg_weight
            if num_centers == 1:
                self.reg_threshold = reg_threshold
                assert self.reg_threshold >= 0.0

                reg_valid_mask = np.triu(np.ones((num_classes, num_classes), dtype=np.float32), k=1)
            else:
                self.reg_weight /= num_classes
                if num_centers > 2:
                    self.reg_weight /= (num_centers - 1) * (num_centers - 2)

                reg_valid_mask = np.tile(np.triu(np.ones((1, num_centers, num_centers), dtype=np.float32), k=1),
                                         (num_classes, 1, 1))

            self.register_buffer('reg_mask', torch.from_numpy(reg_valid_mask))
        else:
            self.reg_weight = None
            self.reg_mask = None

    def forward(self, normalized_x):
        normalized_x = normalized_x.view(-1, self.in_features)
        normalized_weights = normalize(self.weight.view(self.in_features, -1), dim=0)

        prod = normalized_x.mm(normalized_weights)
        if not torch.onnx.is_in_onnx_export():
            prod = prod.clamp(-1.0, 1.0)

        if self.num_centers > 1:
            prod = prod.view(-1, self.num_classes, self.num_centers)

            prod_weights = F.softmax(self.scale * prod, dim=-1)
            scores = torch.sum(prod_weights * prod, dim=-1)
        else:
            scores = prod

        return scores

    def loss(self, name):
        out_losses = dict()

        if self.enable_regularization:
            normalized_weights = F.normalize(self.weight, dim=0)
            if self.num_centers == 1:
                all_pairwise_scores = normalized_weights.permute(1, 0).matmul(normalized_weights)
                valid_pairwise_scores = all_pairwise_scores[self.reg_mask > 0.0]
                losses = valid_pairwise_scores[valid_pairwise_scores > self.reg_threshold] - self.reg_threshold
                out_losses['loss/cpush' + name] =\
                    self.reg_weight * losses.mean() if losses.numel() > 0 else losses.sum()
            else:
                all_pairwise_scores = normalized_weights.permute(1, 2, 0).matmul(normalized_weights.permute(1, 0, 2))
                valid_pairwise_scores = all_pairwise_scores[self.reg_mask > 0.0]
                losses = 1.0 - valid_pairwise_scores
                out_losses['loss/st_reg' + name] = self.reg_weight * losses.sum()

        return out_losses


class SymmetricalLayer(nn.Module):
    """
    Init version: https://github.com/IoannisKansizoglou/Symmetrical-Feature-Space
    """

    def __init__(self, in_features, num_classes):
        super().__init__()

        self.in_features = in_features
        assert in_features > 0
        self.num_classes = num_classes
        assert num_classes > 1

        self.weight = nn.Parameter(torch.FloatTensor(2, self.in_features))
        self.weight.data.normal_()

        steps = torch.arange(self.num_classes, dtype=torch.float32)
        thetas = 2.0 * math.pi / float(self.num_classes) * steps
        self.register_buffer('thetas', thetas)

        eye_matrix = torch.eye(self.in_features)
        self.register_buffer('eye_matrix', eye_matrix)

    def _generate_centers(self, v1, v2):
        n1 = normalize(v1, dim=0, p=2)

        n2 = normalize(v2, dim=0, p=2)
        n2 = normalize(n2 - torch.dot(n1, n2) * n1, dim=0, p=2)

        ger_sub = torch.outer(n2, n1) - torch.outer(n1, n2)
        ger_add = torch.outer(n1, n1) + torch.outer(n2, n2)
        sin_thetas = torch.unsqueeze(torch.unsqueeze(torch.sin(self.thetas), dim=-1), dim=-1)
        cos_thetas = torch.unsqueeze(torch.unsqueeze(torch.cos(self.thetas) - 1, dim=-1), dim=-1)
        R = self.eye_matrix + ger_sub * sin_thetas + ger_add * cos_thetas

        return torch.einsum('bij,j->bi', R, n1)

    def forward(self, normalized_x):
        normalized_x = normalized_x.view(-1, self.in_features)
        centers = self._generate_centers(self.weight[0], self.weight[1])

        scores = F.linear(normalized_x, centers)

        return scores
