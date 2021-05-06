import os.path as osp
from collections import defaultdict

import torch

from .recognition_dataset import RecognitionDataset
from .registry import DATASETS


@DATASETS.register_module()
class VideoDataset(RecognitionDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
    """

    def __init__(self,
                 source,
                 root_dir,
                 ann_file,
                 data_subdir,
                 pipeline,
                 kpts_subdir=None,
                 load_kpts=False,
                 test_mode=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=0,
                 modality='RGB',
                 logger=None,
                 filter_min_fraction=0.7):
        super().__init__(source, root_dir, ann_file, data_subdir, pipeline,
                         kpts_subdir, load_kpts, test_mode,
                         multi_class, num_classes, start_index, modality, logger)

        self.filter_min_fraction = filter_min_fraction

    def _load_annotations(self, ann_file, data_prefix=None):
        """Load annotation file to get video information."""

        if ann_file.endswith('.json'):
            return self.load_json_annotations()

        video_infos = []
        with open(ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                    onehot = torch.zeros(self.num_classes)
                    onehot[label] = 1.0
                else:
                    filename, label = line_split
                    label = int(label)

                if data_prefix is not None:
                    filename = osp.join(data_prefix, filename)

                video_infos.append(dict(
                    filename=filename,
                    label=onehot if self.multi_class else label,
                    matched_weights=defaultdict(float),
                    filter_ready=False,
                ))

        return video_infos

    def update_meta_info(self, pred_labels, pred_conf, sample_idx, clip_starts, clip_ends, total_frames):
        for idx, pred_label, pred_weight, start, end, num_frames in \
                zip(sample_idx, pred_labels, pred_conf, clip_starts, clip_ends, total_frames):
            video_info = self.records[idx]
            video_label = video_info['label']
            video_matched_weights = video_info['matched_weights']

            weight = pred_weight if video_label == pred_label else -pred_weight
            for ii in range(start, end):
                video_matched_weights[ii] += weight

            filter_ready = float(len(video_matched_weights)) / float(num_frames) > self.filter_min_fraction
            video_info['filter_ready'] = filter_ready

    def get_filter_active_samples_ratio(self):
        num_active_samples = len([True for record in self.records if record['filter_ready']])
        return float(num_active_samples) / float(len(self.records))
