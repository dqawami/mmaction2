import torch
import torch.nn as nn
import torch.nn.functional as F


class PRISM(nn.Module):
    """Filters labels according to the distance to set of centers.

    The original paper: https://arxiv.org/abs/2103.16047
    """

    def __init__(self, num_classes, feature_length, buffer_size=1, margin=0.0):
        super().__init__()

        self.num_classes = int(num_classes)
        assert self.num_classes > 0
        self.feature_length = int(feature_length)
        assert self.feature_length > 0
        self.buffer_size = int(buffer_size)
        assert self.buffer_size > 0
        self.margin = float(margin)
        assert self.margin >= 0.0

        buffer_size = [num_classes, self.buffer_size, self.feature_length]
        self.register_buffer('feature_buffer', torch.zeros(buffer_size))

        self.start_pos = [0] * num_classes
        self.size = [0] * num_classes

    def forward(self, features, labels):
        features = features.view(-1, self.feature_length)
        labels = labels.view(-1)

        fixed_labels = torch.full_like(labels, -1)

        unique_labels = torch.unique(labels).detach().cpu().numpy()
        for class_id in unique_labels:
            if class_id < 0:
                continue

            class_mask = labels == class_id
            class_features = features[class_mask]

            if self.size[class_id] < self.buffer_size:
                clear_features = class_features

                fixed_labels[class_mask] = class_id
            else:
                valid_mask = self._estimate_clear_features(class_features, class_id)
                clear_features = class_features[valid_mask]

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
        self.size[class_id] = min(self.size[class_id] + num_features, self.buffer_size)

    def _estimate_clear_features(self, features, class_id):
        pass
