import os.path as osp

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
    """

    def __init__(self,
                 start_index=0,
                 **kwargs):
        super().__init__(start_index=start_index, **kwargs)

    def _parse_data_source(self, data_source, data_prefix):
        filename = data_source
        if data_prefix is not None:
            filename = osp.join(data_prefix, filename)

        return dict(
            filename=filename,
        )
