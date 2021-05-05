import torch
import torch.nn as nn
import torch.nn.functional as F

from .math import normalize


class PRISM(nn.Module):
    """Filters labels according to the distance to set of centers.

    The original paper: https://arxiv.org/abs/2103.16047
    """

    def __init__(self, num_classes, feature_length, buffer_size=1, min_num_updates=10,
                 clear_prob_margin=0.5, default_scale=10.0):
        super().__init__()

        self.num_classes = int(num_classes)
        assert self.num_classes > 0
        self.feature_length = int(feature_length)
        assert self.feature_length > 0
        self.buffer_size = int(buffer_size)
        assert self.buffer_size > 0
        self.min_num_updates = max(int(min_num_updates), self.buffer_size)
        assert self.min_num_updates > 0
        self.clear_prob_margin = float(clear_prob_margin)
        assert self.clear_prob_margin >= 0.0
        self.scale = float(default_scale)
        assert self.scale >= 0.0

        buffer_size = [self.num_classes, self.buffer_size, self.feature_length]
        self.register_buffer('feature_buffer', torch.zeros(buffer_size))

        self.start_pos = [0] * self.num_classes
        self.num_updates = [0] * self.num_classes

    def forward(self, features, labels, scale=None):
        features = features.view(-1, self.feature_length)
        labels = labels.view(-1)

        fixed_labels = torch.full_like(labels, -1)

        enable_cleaning = all(n_ >= self.min_num_updates for n_ in self.num_updates)
        unique_labels = torch.unique(labels).detach().cpu().numpy()
        for class_id in unique_labels:
            if class_id < 0:
                continue

            class_mask = labels == class_id
            class_features = features[class_mask]

            if enable_cleaning:
                clear_mask = self._estimate_clear_features(class_features, class_id, scale)
                clear_features = class_features[clear_mask]

                local_labels = torch.full_like(clear_mask, -1, dtype=labels.dtype)
                local_labels[clear_mask] = class_id
                fixed_labels[class_mask] = local_labels
            else:
                clear_features = class_features

                fixed_labels[class_mask] = class_id

            self._store_features(clear_features, class_id)

        return fixed_labels

    def _store_features(self, new_features, class_id):
        num_features = new_features.size(0)
        if num_features == 0:
            return
        elif num_features > self.buffer_size:
            raise ValueError(f'Num features ({num_features}) is bigger than '
                             f'tbe buffer size ({self.buffer_size})')

        start_pos = self.start_pos[class_id]

        num_free_lines = self.buffer_size - start_pos
        if num_free_lines >= num_features:
            with torch.no_grad():
                self.feature_buffer[class_id, start_pos:(start_pos + num_features)] = new_features

            start_pos += num_features
        else:
            num_extra_lines = num_features - num_free_lines
            with torch.no_grad():
                self.feature_buffer[class_id, start_pos:] = new_features[:num_free_lines]
                self.feature_buffer[class_id, :num_extra_lines] = new_features[num_free_lines:]

            start_pos = num_extra_lines

        self.start_pos[class_id] = start_pos % self.buffer_size
        self.num_updates[class_id] += num_features

    def _estimate_clear_features(self, features, class_id, scale=None):
        scale = scale if scale is not None else self.scale

        with torch.no_grad():
            class_centers = torch.mean(self.feature_buffer, dim=1)
            norm_class_centers = normalize(class_centers, dim=1, p=2)

            set_similarities = torch.matmul(norm_class_centers, torch.transpose(features, 0, 1))
            set_probs = torch.softmax(scale * set_similarities, dim=0)

            # all_similarities = torch.matmul(self.feature_buffer.view(-1, self.feature_length),
            #                                 torch.transpose(features, 0, 1))
            # all_similarities = all_similarities.view(self.num_classes, self.buffer_size, -1)
            #
            # set_similarities = torch.max(all_similarities, dim=1)[0]
            # set_probs = torch.softmax(scale * set_similarities, dim=0)

            clear_prob = set_probs[class_id].view(-1)
            clear_mask = clear_prob > self.clear_prob_margin

        return clear_mask
