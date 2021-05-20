import os.path as osp

from .recognition_dataset import RecognitionDataset
from .registry import DATASETS


@DATASETS.register_module()
class RawframeDataset(RecognitionDataset):
    """RawframeDataset dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, the label of a video,
    start/end frames of the clip, start/end frames of the video and
    video frame rate, which are split with a whitespace.

    Example of a full annotation file:

    .. code-block:: txt

        some/directory-1 1 0 120 0 120 30.0
        some/directory-2 1 0 120 0 120 30.0
        some/directory-3 2 0 120 0 120 30.0
        some/directory-4 2 0 120 0 120 30.0
        some/directory-5 3 0 120 0 120 30.0
        some/directory-6 3 0 120 0 120 30.0

    Example of a simple annotation file:

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
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Default: False.
        num_classes (int): Number of classes in the dataset. Default: None.
        modality (str): Modality of data. Support 'RGB', 'Flow'. Default: 'RGB'.
    """

    def __init__(self,
                 start_index=1,
                 filename_tmpl='img_{:05}.jpg',
                 **kwargs):
        self.filename_tmpl = filename_tmpl

        super().__init__(start_index=start_index, **kwargs)

    def _parse_data_source(self, data_source, data_prefix):
        record = dict(
            filename_tmpl=self.filename_tmpl,
        )

        frame_dir = data_source
        record['rel_frame_dir'] = frame_dir

        if data_prefix is not None:
            frame_dir = osp.join(data_prefix, frame_dir)
        record['frame_dir'] = frame_dir

        return record
