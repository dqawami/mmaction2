import torch
import torch.nn as nn
import torch.nn.functional as F


class Dropout(nn.Module):
    DISTRIBUTIONS = ['bernoulli', 'gaussian', 'info_drop', 'focused_drop']

    def __init__(self, p=0.5, mu=0.5, sigma=0.2, dist='bernoulli', kernel=3, temperature=0.05,
                 random_limits=(0.3, 0.6), focused_prob=0.1):
        super(Dropout, self).__init__()

        self.dist = dist
        assert self.dist in Dropout.DISTRIBUTIONS

        self.p = float(p)
        assert 0. <= self.p <= 1.

        self.mu = float(mu)
        self.sigma = float(sigma)
        assert self.sigma > 0.

        self.kernel = kernel
        assert self.kernel >= 3
        self.temperature = temperature
        assert self.temperature > 0.0

        self.random_limits = random_limits
        self.focused_prob = focused_prob

    def forward(self, x, x_original=None):
        if not self.training:
            return x

        if self.dist == 'bernoulli':
            out = F.dropout(x, self.p, self.training)
        elif self.dist == 'gaussian':
            with torch.no_grad():
                soft_mask = x.new_empty(x.size()).normal_(self.mu, self.sigma).clamp_(0., 1.)

            scale = 1. / self.mu
            out = scale * soft_mask * x
        elif self.dist == 'info_drop':
            assert x_original is not None

            out = info_dropout(x_original, self.kernel, x, self.p, self.temperature)
        elif self.dist == 'focused_drop':
            out = focused_dropout(x, self.random_limits, self.focused_prob)
        else:
            out = x

        return out


def info_dropout(in_features, kernel, out_features, drop_rate, temperature=0.05, eps=1e-12):
    """
    Implementation of the paper: https://arxiv.org/abs/2008.04254
    """

    assert isinstance(kernel, int)
    assert kernel % 2 == 1

    in_shape = in_features.size()
    assert len(in_shape) in (4, 5)

    with torch.no_grad():
        if len(in_shape) == 5:
            b, c, t, h, w = in_shape
            out_mask_shape = b, 1, t, h, w

            in_features = in_features.permute(0, 2, 1, 3, 4)
            b *= t
        else:
            b, c, h, w = in_shape
            out_mask_shape = b, 1, h, w
        in_features = in_features.reshape(-1, c, h, w)

        padding = (kernel - 1) // 2
        unfolded_features = F.unfold(in_features, kernel, padding=padding)
        unfolded_features = unfolded_features.view(b, c, kernel * kernel, -1)

        distances = ((unfolded_features - in_features.view(-1, c, 1, h * w)) ** 2).sum(dim=1)
        weights = (0.5 * distances / distances.mean(dim=(1, 2), keepdim=True).clamp_min(eps)).neg().exp()

        middle = kernel * kernel // 2
        log_info = (weights[:, :middle].sum(dim=1) + weights[:, (middle + 1):].sum(dim=1) + eps).log()

        prob_weights = (1. / float(temperature) * log_info).exp() + eps
        probs = prob_weights / prob_weights.sum(dim=-1, keepdim=True)

        drop_num_samples = max(1, int(drop_rate * float(probs.size(-1))))
        drop_indices = torch.multinomial(probs, num_samples=drop_num_samples, replacement=True)

        out_mask = torch.ones_like(probs)
        out_mask[torch.arange(out_mask.size(0), device=out_mask.device).view(-1, 1), drop_indices] = 0.0

    out_scale = 1.0 / (1.0 - drop_rate)
    out = out_scale * out_features * out_mask.view(out_mask_shape)

    return out


def focused_dropout(x, random_limits=(0.3, 0.6), prob=0.1):
    """
    Implementation of thr paper: https://arxiv.org/abs/2103.15425
    """

    assert isinstance(random_limits, (tuple, list))
    assert len(random_limits) == 2
    min_prob, max_prob = random_limits
    assert 0 < min_prob < max_prob < 1

    in_shape = x.size()
    assert len(in_shape) in (4, 5)

    if len(in_shape) == 5:
        b, c, t, h, w = in_shape
        out_mask_shape = b, 1, t, h, w
        x = x.permute(0, 2, 1, 3, 4)
        b *= t
    else:
        b, c, h, w = in_shape
        out_mask_shape = b, 1, h, w
    participation_mask_shape = tuple([out_mask_shape[0]] + [1] * (len(out_mask_shape) - 1))

    x = x.reshape(-1, c, h, w)

    with torch.no_grad():
        channel_weights = x.mean(dim=(2, 3))
        ref_channel_ids = torch.argmax(channel_weights, dim=1)

        batch_inds = torch.arange(ref_channel_ids.size(0), device=ref_channel_ids.device)
        ref_channels = x[batch_inds, ref_channel_ids]

        ref_value = torch.max(ref_channels.view(-1, h * w), dim=1)[0].view(-1, 1, 1)
        rand_values = torch.rand_like(ref_channels, device=ref_channels.device)
        thresholds = ref_value * ((max_prob - min_prob) * rand_values + min_prob)
        drop_mask = torch.where(ref_channels > thresholds,
                                torch.ones_like(ref_channels),
                                torch.zeros_like(ref_channels))

        drop_mask = drop_mask.view(out_mask_shape)
        participation_mask = torch.rand(participation_mask_shape, device=drop_mask.device) > prob
        out_mask = torch.where(participation_mask,
                               torch.ones_like(drop_mask),
                               drop_mask)

    out = out_mask * x.view(in_shape)

    return out
