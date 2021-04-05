import copy
import os.path as osp

from .recognition_dataset import RecognitionDataset
from .registry import DATASETS


@DATASETS.register_module()
class StreamDataset(RecognitionDataset):
    """StreamDataset dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, the label of a video,
    start/end frames of the clip, start/end frames of the video and
    video frame rate, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        some/directory-1 1 0 120 0 120 30.0
        some/directory-2 1 0 120 0 120 30.0
        some/directory-3 2 0 120 0 120 30.0
        some/directory-4 2 0 120 0 120 30.0
        some/directory-5 3 0 120 0 120 30.0
        some/directory-6 3 0 120 0 120 30.0


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
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
                 action_type_file=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg',
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 logger=None):
        assert not multi_class, 'StreamDataset does not support multi-class labels'

        self.filename_tmpl = filename_tmpl

        super().__init__(source, root_dir, ann_file, data_subdir, pipeline,
                         kpts_subdir, load_kpts, test_mode,
                         multi_class, num_classes, start_index, modality, logger)

        if action_type_file is not None:
            assert isinstance(action_type_file, dict)
            assert len(self.dataset_ids_map) == 1

            source_name = self.dataset_ids_map[0]
            if source_name in action_type_file:
                action_type_file = osp.join(self.root_dir, action_type_file[source_name])
                self.records = self._update_action_type_info(self.records, action_type_file)

    @staticmethod
    def _update_action_type_info(records, action_type_file):
        if not osp.exists(action_type_file):
            return records

        action_type_map = dict()
        with open(action_type_file) as input_stream:
            for line in input_stream:
                line_parts = line.strip().split(':')
                if len(line_parts) != 2:
                    continue

                action_type_map[int(line_parts[0])] = line_parts[1]

        for record in records:
            label = record['label']
            record['action_type'] = action_type_map[label]

        return records

    def _load_annotations(self, ann_file, data_prefix=None):
        """Load annotation file to get video information."""

        if ann_file.endswith('.json'):
            return self.load_json_annotations()

        video_infos = []
        with open(ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                if len(line_split) != 7:
                    continue

                video_info = {
                    'label': int(line_split[1]),
                    'clip_start': int(line_split[2]),
                    'clip_end': int(line_split[3]),
                    'video_start': int(line_split[4]),
                    'video_end': int(line_split[5]),
                    'fps': float(line_split[6]),
                    'filename_tmpl': self.filename_tmpl,
                    'action_type': 'dynamic',
                }

                video_info['clip_len'] = video_info['clip_end'] - video_info['clip_start']
                assert video_info['clip_len'] > 0

                video_info['video_len'] = video_info['video_end'] - video_info['video_start']
                assert video_info['video_len'] > 0

                frame_dir = line_split[0]
                video_info['rel_frame_dir'] = frame_dir

                if data_prefix is not None:
                    frame_dir = osp.join(data_prefix, frame_dir)
                video_info['frame_dir'] = frame_dir

                video_infos.append(video_info)

        return video_infos
