import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad


class RSC(nn.Module):
    def __init__(self, retain_p):
        super(RSC, self).__init__()

        self.retain_p = float(retain_p)
        assert 0. < self.retain_p < 1.

    def forward(self, features, scores, labels):
        return rsc(features, scores, labels, self.retain_p)


def rsc(features, scores, labels, retain_p=0.77):
    """Representation Self-Challenging module (RSC).
       Based on the paper: https://arxiv.org/abs/2007.02454
    """

    pos_samples_mask = labels.view(-1) >= 0
    pos_labels = labels.view(-1)[pos_samples_mask]
    pos_scores = scores[pos_samples_mask]

    batch_range = torch.arange(pos_scores.size(0), device=pos_scores.device)
    gt_scores = pos_scores[batch_range, pos_labels.view(-1)]
    z_grads = grad(outputs=gt_scores,
                   inputs=features,
                   grad_outputs=torch.ones_like(gt_scores),
                   create_graph=True)[0]

    with torch.no_grad():
        z_grads_cpu = z_grads.cpu().numpy()
        z_grad_thresholds_cpu = np.quantile(z_grads_cpu, retain_p, axis=(1, 2, 3, 4), keepdims=True)
        zero_mask = z_grads > torch.from_numpy(z_grad_thresholds_cpu).to(z_grads.device)

        unchanged_mask = torch.randint(2, [z_grads.size(0)], dtype=torch.bool, device=z_grads.device)
        unchanged_mask = unchanged_mask.view(-1, 1, 1, 1, 1)

    scale = 1.0 / float(retain_p)
    filtered_features = scale * torch.where(zero_mask, torch.zeros_like(features), features)
    out_features = torch.where(unchanged_mask, features, filtered_features)

    return out_features
