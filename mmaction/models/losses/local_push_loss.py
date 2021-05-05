import torch

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class LocalPushLoss(BaseWeightedLoss):
    def __init__(self, margin=0.1, smart_margin=True, **kwargs):
        super(LocalPushLoss, self).__init__(**kwargs)

        self.margin = margin
        assert self.margin >= 0.0

        self.smart_margin = smart_margin

    def _forward(self, all_norm_embd, cos_theta, labels):
        pos_samples_mask = labels.view(-1) >= 0
        if torch.sum(pos_samples_mask) == 0:
            return torch.zeros([], dtype=all_norm_embd.dtype, device=all_norm_embd.device)

        pos_labels = labels.view(-1)[pos_samples_mask]
        pos_norm_embd = all_norm_embd[pos_samples_mask]
        pos_cos_theta = cos_theta[pos_samples_mask]

        similarity = pos_norm_embd.matmul(all_norm_embd.permute(1, 0))

        with torch.no_grad():
            pairs_mask = pos_labels.view(-1, 1) != labels.view(1, -1)

            if self.smart_margin:
                batch_inds = torch.arange(pos_cos_theta.size(0), device=pos_labels.device)
                center_similarity = pos_cos_theta[batch_inds, pos_labels]
                threshold = center_similarity.clamp(min=self.margin).view(-1, 1) - self.margin
            else:
                threshold = self.margin
            similarity_mask = similarity > threshold

            mask = pairs_mask & similarity_mask

        filtered_similarity = torch.where(mask, similarity - threshold, torch.zeros_like(similarity))
        losses = filtered_similarity.max(dim=-1)[0]

        return losses.mean()
