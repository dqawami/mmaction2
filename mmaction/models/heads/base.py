from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from ...core import top_k_accuracy
from ..builder import build_loss


class AvgConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Default: 1.
    """

    def __init__(self, dim=1):
        super().__init__()

        self.dim = dim

    def forward(self, input):
        """Defines the computation performed at every call."""
        return input.mean(dim=self.dim, keepdim=True)


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base class for head.

    All Head should subclass it.
    All subclass should overwrite:
    - Methods:``init_weights``, initializing weights in some modules.
    - Methods:``forward``, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: arxiv.org/abs/1906.02629. Default: 0.
    """

    def __init__(self,
                 num_classes=None,
                 class_sizes=None,
                 in_channels=2048,
                 consensus=None,
                 loss_cls=None,
                 losses_extra=None,
                 multi_class=False,
                 label_smooth_eps=0.0,
                 dropout_ratio=None):
        super().__init__()

        self.in_channels = in_channels
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps

        self.num_classes = num_classes
        self.class_sizes = class_sizes
        if self.class_sizes is None:
            assert self.num_classes is not None
        else:
            self.num_classes = max(class_sizes.keys()) + 1

        loss_cls = loss_cls if loss_cls is not None else dict(type='CrossEntropyLoss')
        self.head_loss = build_loss(loss_cls, class_sizes=class_sizes)

        self.losses_extra = None
        if losses_extra is not None:
            self.losses_extra = nn.ModuleDict()
            for loss_name, extra_loss_cfg in losses_extra.items():
                self.losses_extra[loss_name] = build_loss(extra_loss_cfg)

        self.dropout = None
        if dropout_ratio is not None and dropout_ratio > 0.0:
            self.dropout = nn.Dropout(p=dropout_ratio)

        self.consensus = None
        if consensus is not None:
            consensus_ = consensus.copy()
            consensus_type = consensus_.pop('type')
            if consensus_type == 'AvgConsensus':
                self.consensus = AvgConsensus(**consensus_)

    @abstractmethod
    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        pass

    def update_state(self, *args):
        if hasattr(self.head_loss, 'update_state'):
            self.head_loss.update_state(*args)

    @abstractmethod
    def forward(self, x):
        """Defines the computation performed at every call."""
        pass

    def loss(self, cls_score, labels, **kwargs):
        """Calculate the loss given output ``cls_score`` and target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)

        if not self.multi_class:
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(top_k_acc[1], device=cls_score.device)
        elif self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        losses['loss/cls'] = self.head_loss(cls_score, labels)

        return losses
