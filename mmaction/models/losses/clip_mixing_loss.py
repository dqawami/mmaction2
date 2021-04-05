import torch

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class ClipMixingLoss(BaseWeightedLoss):
    MODES = 'embd', 'logits'

    def __init__(self, mode='', default_scale=10.0, num_clips=2, **kwargs):
        super(ClipMixingLoss, self).__init__(**kwargs)

        assert mode in self.MODES
        self.mode = mode
        self.default_scale = default_scale

        self.num_clips = num_clips
        assert self.num_clips > 1

    def _forward(self, all_logits, labels, all_norm_embd, scale=None):
        with torch.no_grad():
            labels = labels.view(labels.view(-1).size(0) // self.num_clips, self.num_clips)
            valid_tuples_mask = torch.all(labels == labels[:, 0].view(-1, 1), dim=1)

        if valid_tuples_mask.sum() == 0:
            return torch.zeros([], dtype=all_logits.dtype, device=all_logits.device)

        if self.mode == 'embd':
            all_norm_embd = all_norm_embd.view(all_norm_embd.size(0) // self.num_clips, self.num_clips, -1)
            norm_embd = all_norm_embd[valid_tuples_mask]

            similarity = torch.matmul(norm_embd, norm_embd.permute(0, 2, 1))
            losses = 1.0 - similarity

            ind_range = torch.arange(self.num_clips, dtype=torch.int64, device=norm_embd.device)
            mask = ind_range.view(-1, 1) < ind_range.view(1, -1)
            mask = mask.view(-1, self.num_clips, self.num_clips).repeat(norm_embd.size(0), 1, 1)

            valid_losses = losses[mask]
        else:
            scale = scale if scale is not None else self.default_scale
            all_logits = all_logits.view(all_logits.size(0) // self.num_clips, self.num_clips, -1)
            logits = scale * all_logits[valid_tuples_mask]

            with torch.no_grad():
                probs = torch.softmax(logits, dim=2)
                trg_probs = probs.mean(dim=1, keepdim=True)

            log_probs = torch.log_softmax(logits, dim=2)
            valid_losses = (trg_probs * log_probs).sum(dim=2).neg()

        return valid_losses.mean()
