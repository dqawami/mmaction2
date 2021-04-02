import copy
import math

import torch
from torch.utils.data import DistributedSampler as _DistributedSampler


class BalancedDistributedSampler(_DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, num_instances=1):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.num_instances = num_instances
        assert self.num_instances > 0

        num_groups = 0
        for dataset_items in dataset.clustered_ids.values():
            for record_ids in dataset_items.values():
                num_same_label_records = len(record_ids)
                num_extra_records = len(record_ids) % self.num_instances
                num_groups += (num_same_label_records + num_extra_records) // self.num_instances

        self.num_samples = math.ceil(num_groups * num_instances / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        clustered_ids = self.dataset.clustered_ids

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)

        grouped_records = []
        for dataset_items in clustered_ids.values():
            for record_ids in dataset_items.values():
                same_label_records = copy.deepcopy(record_ids)

                num_extra_records = len(record_ids) % self.num_instances
                if num_extra_records > 0:
                    if self.shuffle:
                        rand_ids = torch.randperm(len(record_ids), generator=g).tolist()
                        extra_record_ids = [record_ids[ii] for ii in rand_ids[:num_extra_records]]
                    else:
                        extra_record_ids = record_ids[:num_extra_records]

                    same_label_records.extend(extra_record_ids)

                num_groups = len(same_label_records) // self.num_instances
                for group_ind in range(num_groups):
                    start_pos = group_ind * self.num_instances
                    end_pos = (group_ind + 1) * self.num_instances
                    grouped_records.append(same_label_records[start_pos:end_pos])

        if self.shuffle:
            group_inds = torch.randperm(len(grouped_records), generator=g).tolist()
            grouped_records = [grouped_records[group_ind] for group_ind in group_inds]

        indices = []
        for group in grouped_records:
            indices.extend(group)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        samples_start_pos = self.rank * self.num_samples
        samples_end_pos = (self.rank + 1) * self.num_samples
        indices = indices[samples_start_pos:samples_end_pos]
        assert len(indices) == self.num_samples

        return iter(indices)
