import copy
import os.path as osp

import torch

from .recognition_dataset import RecognitionDataset
from .registry import DATASETS


@DATASETS.register_module()
class RawframeDataset(RecognitionDataset):
    """Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        some/directory-1 163 1
        some/directory-2 122 1
        some/directory-3 258 2
        some/directory-4 234 2
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a multi-class annotation file:


    .. code-block:: txt

        some/directory-1 163 1 3 5
        some/directory-2 122 1 2
        some/directory-3 258 2
        some/directory-4 234 2 4 6 8
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a with_offset annotation file (clips from long videos), each
    line indicates the directory to frames of a video, the index of the start
    frame, total frames of the video clip and the label of a video clip, which
    are split with a whitespace.


    .. code-block:: txt

        some/directory-1 12 163 3
        some/directory-2 213 122 4
        some/directory-3 100 258 5
        some/directory-4 98 234 2
        some/directory-5 0 295 3
        some/directory-6 50 121 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        with_offset (bool): Determines whether the offset information is in
            ann_file. Default: False.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Default: False.
        num_classes (int): Number of classes in the dataset. Default: None.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
                            Default: 'RGB'.
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
                 filename_tmpl='img_{:05}.jpg',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 logger=None):
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset

        super().__init__(source, root_dir, ann_file, data_subdir, pipeline,
                         kpts_subdir, load_kpts, test_mode,
                         multi_class, num_classes, start_index, modality, logger)

    def _load_annotations(self, ann_file, data_prefix=None):
        """Load annotation file to get video information."""

        if ann_file.endswith('.json'):
            return self.load_json_annotations()

        video_infos = []
        with open(ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_info = {}
                idx = 0
                # idx for frame_dir
                frame_dir = line_split[idx]
                if data_prefix is not None:
                    frame_dir = osp.join(data_prefix, frame_dir)
                video_info['frame_dir'] = frame_dir
                idx += 1
                if self.with_offset:
                    # idx for offset and total_frames
                    video_info['offset'] = int(line_split[idx])
                    video_info['total_frames'] = int(line_split[idx + 1])
                    idx += 2
                else:
                    # idx for total_frames
                    video_info['total_frames'] = int(line_split[idx])
                    idx += 1
                # idx for label[s]
                label = [int(x) for x in line_split[idx:]]
                assert len(label), f'missing label in line: {line}'
                if self.multi_class:
                    assert self.num_classes is not None
                    onehot = torch.zeros(self.num_classes)
                    onehot[label] = 1.0
                    video_info['label'] = onehot
                else:
                    assert len(label) == 1
                    video_info['label'] = label[0]

                video_info['filename_tmpl'] = self.filename_tmpl

                video_infos.append(video_info)

        return video_infos
