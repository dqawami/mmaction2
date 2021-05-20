import io
import os
import os.path as osp
import shutil
import warnings

import mmcv
import numpy as np
from mmcv.fileio import FileClient
from torch.nn.modules.utils import _pair

from ...utils import get_random_string, get_shm_dir, get_thread_id
from ..registry import PIPELINES


@PIPELINES.register_module()
class SampleFrames(object):
    """Sample frames from the video.

    Required keys are "filename", "total_frames", "start_index" , added or
    modified keys are "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False,
                 start_index=None,
                 enable_negatives=False):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.enable_negatives = enable_negatives
        self.test_mode = test_mode
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _get_train_clips(self, num_frames, frames_meta_info=None):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """

        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        frame_weights = None
        if frames_meta_info is not None:
            est_frame_weights = frames_meta_info['matched_weights']
            start_shift = frames_meta_info['start_shift']

            frame_weights = np.zeros([num_frames], dtype=np.float32)
            for frame_id, frame_weight in est_frame_weights.items():
                local_frame_id = frame_id - start_shift
                if 0 <= local_frame_id < num_frames:
                    frame_weights[local_frame_id] += frame_weight

            frame_weights = np.where(frame_weights < 0.0,
                                     np.zeros_like(frame_weights),
                                     1.0 + frame_weights)

        is_positives = [True] * self.num_clips

        if avg_interval > 0:
            if frame_weights is not None:
                frame_cumsum = np.cumsum(frame_weights)
                frame_pad_cumsum = np.pad(frame_cumsum, (1, 0), 'constant')
                window_weights = frame_cumsum[(ori_clip_len - 1):] - frame_pad_cumsum[:-ori_clip_len]

                sum_weights = np.sum(window_weights)
                if sum_weights > 0:
                    probs = window_weights / sum_weights
                else:
                    probs = np.full_like(window_weights, 1.0 / float(len(window_weights)))
                    is_positives = [False] * self.num_clips

                clip_offsets = np.sort(
                    np.random.choice(np.arange(len(probs)), self.num_clips, p=probs)
                )
            else:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(np.random.randint(num_frames - ori_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)

        if frame_weights is not None and avg_interval <= 0:
            sum_weights = np.sum(frame_weights)
            if sum_weights <= 0:
                is_positives = [False] * self.num_clips

        return clip_offsets, is_positives

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """

        ori_clip_len = self.clip_len * self.frame_interval

        if num_frames > ori_clip_len - 1:
            avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)

        is_positives = [True] * self.num_clips

        return clip_offsets, is_positives

    def _sample_clips(self, num_frames, frames_meta_info=None):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets, is_positives = self._get_test_clips(num_frames)
        else:
            clip_offsets, is_positives = self._get_train_clips(num_frames, frames_meta_info)

        return clip_offsets, is_positives

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['clip_len']
        start_index = results['start_index'] + results['clip_start']

        frames_meta_info = None
        if results.get('sample_filtering', False) and results.get('filter_ready', False):
            frames_meta_info = dict(
                matched_weights=results['matched_weights'],
                start_shift=results['clip_start']
            )

        clip_offsets, is_positives = self._sample_clips(total_frames, frames_meta_info)
        frame_inds = np.arange(self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(clip_offsets[:, None] + frame_inds)

        if self.temporal_jitter:
            per_frame_offsets = np.random.randint(self.frame_interval, size=len(frame_inds))
            frame_inds += per_frame_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        frame_inds = frame_inds.astype(np.int)
        clip_starts = np.min(frame_inds, axis=1)
        clip_ends = np.max(frame_inds, axis=1) + 1

        results['frame_inds'] = frame_inds + start_index
        results['clip_starts'] = clip_starts
        results['clip_ends'] = clip_ends
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        results['dataset_id'] = [results['dataset_id']] * self.num_clips\
            if self.num_clips > 1 else results['dataset_id']
        results['total_frames'] = [results['total_frames']] * self.num_clips \
            if self.num_clips > 1 else results['total_frames']
        if 'sample_idx' in results:
            results['sample_idx'] = [results['sample_idx']] * self.num_clips \
                if self.num_clips > 1 else results['sample_idx']

        original_label = results['label']
        if self.enable_negatives:
            clip_labels = [original_label if pl else -1 for pl in is_positives]
        else:
            clip_labels = [original_label] * self.num_clips
        results['label'] = clip_labels if self.num_clips > 1 else clip_labels[0]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(' \
                   f'clip_len={self.clip_len}, ' \
                   f'frame_interval={self.frame_interval}, ' \
                   f'num_clips={self.num_clips})'
        return repr_str


@PIPELINES.register_module()
class UntrimmedSampleFrames(object):
    """Sample frames from the untrimmed video.

    Required keys are "filename", "total_frames", added or modified keys are
    "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): The length of sampled clips. Default: 1.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 16.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 1.
    """

    def __init__(self, clip_len=1, frame_interval=16, start_index=1):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.start_index = start_index

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']

        clip_centers = np.arange(self.frame_interval // 2, total_frames,
                                 self.frame_interval)
        num_clips = clip_centers.shape[0]
        frame_inds = clip_centers[:, None] + np.arange(
            -(self.clip_len // 2), self.clip_len -
            (self.clip_len // 2))[None, :]
        # clip frame_inds to legal range
        frame_inds = np.clip(frame_inds, 0, total_frames - 1)

        frame_inds = np.concatenate(frame_inds) + self.start_index
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = num_clips
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(clip_len={self.clip_len}, ' \
                   f'frame_interval={self.frame_interval})'
        return repr_str


@PIPELINES.register_module()
class DenseSampleFrames(SampleFrames):
    """Select frames from the video by dense sample strategy.

    Required keys are "filename", added or modified keys are "total_frames",
    "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        sample_range (int): Total sample range for dense sample.
            Default: 64.
        num_sample_positions (int): Number of sample start positions, Which is
            only used in test mode. Default: 10.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 sample_range=64,
                 num_sample_positions=10,
                 temporal_jitter=False,
                 out_of_bound_opt='loop',
                 test_mode=False):
        super().__init__(
            clip_len,
            frame_interval,
            num_clips,
            temporal_jitter,
            out_of_bound_opt=out_of_bound_opt,
            test_mode=test_mode)
        self.sample_range = sample_range
        self.num_sample_positions = num_sample_positions

    def _get_train_clips(self, num_frames):
        """Get clip offsets by dense sample strategy in train mode.

        It will calculate a sample position and sample interval and set
        start index 0 when sample_pos == 1 or randomly choose from
        [0, sample_pos - 1]. Then it will shift the start index by each
        base offset.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        sample_position = max(1, 1 + num_frames - self.sample_range)
        interval = self.sample_range // self.num_clips
        start_idx = 0 if sample_position == 1 else np.random.randint(
            0, sample_position - 1)
        base_offsets = np.arange(self.num_clips) * interval
        clip_offsets = (base_offsets + start_idx) % num_frames
        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets by dense sample strategy in test mode.

        It will calculate a sample position and sample interval and evenly
        sample several start indexes as start positions between
        [0, sample_position-1]. Then it will shift each start index by the
        base offsets.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        sample_position = max(1, 1 + num_frames - self.sample_range)
        interval = self.sample_range // self.num_clips
        start_list = np.linspace(
            0, sample_position - 1, num=self.num_sample_positions, dtype=int)
        base_offsets = np.arange(self.num_clips) * interval
        clip_offsets = list()
        for start_idx in start_list:
            clip_offsets.extend((base_offsets + start_idx) % num_frames)
        clip_offsets = np.array(clip_offsets)
        return clip_offsets


@PIPELINES.register_module()
class SparseSampleFrames(object):
    def __init__(self,
                 clip_len,
                 num_clips=1,
                 test_mode=False,
                 start_index=None):

        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _get_train_inds(self, num_frames):
        if num_frames >= self.clip_len:
            avg_interval = float(num_frames) / float(self.clip_len)
            base_offsets = np.arange(self.clip_len) * avg_interval
            frame_offsets = (base_offsets + np.random.rand(self.clip_len) * avg_interval).astype(np.int)
        else:
            frame_offsets = np.sort(np.random.randint(num_frames, size=self.clip_len))

        return frame_offsets

    def _get_test_inds(self, num_frames):
        avg_interval = float(num_frames) / float(self.clip_len)

        if num_frames >= self.clip_len:
            base_offsets = np.arange(self.clip_len) * avg_interval
            frame_offsets = (base_offsets + 0.5 * avg_interval).astype(np.int)
        else:
            frame_offsets = np.arange(num_frames)

        return frame_offsets

    def _get_inds(self, num_frames):
        if self.test_mode:
            frame_inds = self._get_test_inds(num_frames)
        else:
            frame_inds = self._get_train_inds(num_frames)

        return frame_inds

    def __call__(self, results):
        total_frames = results['total_frames']
        start_index = results['start_index']

        all_frame_inds = []
        for clip_id in range(self.num_clips):
            frame_inds = self._get_inds(total_frames)
            frame_inds = np.array(frame_inds).astype(np.int)
            frame_inds = np.where(frame_inds < 0, frame_inds, frame_inds + start_index)
            all_frame_inds.append(frame_inds)

        results['frame_inds'] = np.concatenate(all_frame_inds)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = 1
        results['num_clips'] = self.num_clips

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(' \
                   f'clip_len={self.clip_len})'
        return repr_str


@PIPELINES.register_module()
class StreamSampleFrames(object):
    """Sample frames from the video stream.

    Required keys are "filename", "clip_start", "clip_end", "video_start",
    "video_end", "fps", "start_index" , added or
    modified keys are "frame_inds", "trg_fps" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        trg_fps (int): Target frame rate. Default: 15.0
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self,
                 clip_len,
                 trg_fps=15.0,
                 num_clips=1,
                 temporal_jitter=False,
                 min_intersection=1.0,
                 neg_prob=-1,
                 ignore_outside=False,
                 test_mode=False):

        self.clip_len = clip_len
        self.trg_fps = trg_fps
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.ignore_outside = ignore_outside
        self.test_mode = test_mode

        self.enable_negatives = neg_prob is not None and neg_prob > 0
        self.neg_prob = float(neg_prob) if self.enable_negatives else None

        self.min_intersection = min_intersection
        if isinstance(min_intersection, float):
            self.min_static_intersect = min_intersection
            self.min_dynamic_intersect = min_intersection
        elif isinstance(min_intersection, dict):
            self.min_static_intersect = float(min_intersection['static'])
            self.min_dynamic_intersect = float(min_intersection['dynamic'])
        else:
            raise ValueError(f'Unknown format of min_intersection: {type(min_intersection)}')

    def _estimate_time_step(self, video_fps):
        return max(1, int(np.round(float(video_fps) / float(self.trg_fps))))

    def _estimate_clip_lengths(self, time_step):
        return time_step * self.clip_len, self.clip_len

    def _generate_indices(self, record, time_step, input_length, output_length):
        if self.test_mode:
            return self._get_test_indices(record, time_step, input_length, output_length)
        else:
            return self._get_train_indices(record, time_step, input_length, output_length)

    def _get_train_indices(self, record, time_step, input_length, output_length):
        if record['video_len'] < input_length:
            num_valid_frames = record['video_len'] // time_step
            if num_valid_frames == 0:
                num_valid_frames = record['video_len']

            if self.temporal_jitter:
                offsets = np.random.randint(low=0, high=time_step, size=num_valid_frames, dtype=np.int32)
            else:
                offsets = np.zeros(num_valid_frames, dtype=np.int32)

            shift_start = record['video_start']
            indices = np.array([shift_start + i * time_step + offsets[i] for i in range(num_valid_frames)])

            num_rest = output_length - num_valid_frames
            if num_rest > 0:
                num_before = np.random.randint(num_rest + 1)
                before_fill_value = -1 if self.ignore_outside else indices[0]
                before_values = np.full(num_before, before_fill_value, dtype=np.int32)

                num_after = num_rest - num_before
                after_fill_value = -1 if self.ignore_outside else indices[-1]
                after_values = np.full(num_after, after_fill_value, dtype=np.int32)

                indices = np.concatenate((before_values, indices, after_values))

            neg_label = False
        else:
            min_intersection_ratio = self.min_static_intersect \
                if record['action_type'] == 'static' \
                else self.min_dynamic_intersect
            bumpy_length = int(np.round(float(1.0 - min_intersection_ratio) *
                                        float(min(input_length, record['clip_len']))))

            valid_neg_segments = []

            neg_before_start = record['video_start']
            neg_before_end = record['clip_start'] + bumpy_length + 1
            neg_before_length = neg_before_end - neg_before_start
            if neg_before_length >= input_length:
                valid_neg_segments.append((neg_before_start, neg_before_end))

            neg_after_start = record['clip_end'] - bumpy_length
            neg_after_end = record['video_end']
            neg_after_length = neg_after_end - neg_after_start
            if neg_after_length >= input_length:
                valid_neg_segments.append((neg_after_start, neg_after_end))

            neg_label = self.enable_negatives and len(valid_neg_segments) > 0 and np.random.rand() < self.neg_prob
            if neg_label:
                neg_segment_start, neg_segment_end = valid_neg_segments[np.random.randint(len(valid_neg_segments))]
                shift_start = neg_segment_start
                shift_end = neg_segment_end - input_length + 1
            else:
                if record['clip_len'] < input_length:
                    shift_start = max(record['video_start'],
                                      record['clip_end'] - bumpy_length - input_length)
                    shift_end = min(record['video_end'] - input_length + 1,
                                    record['clip_start'] + bumpy_length + 1)
                else:
                    shift_start = max(record['video_start'],
                                      record['clip_start'] - bumpy_length)
                    shift_end = min(record['video_end'] - input_length + 1,
                                    record['clip_end'] + bumpy_length - input_length + 1)

            if self.temporal_jitter:
                offsets = np.random.randint(low=0, high=time_step, size=output_length, dtype=np.int32)
            else:
                offsets = np.zeros(output_length, dtype=np.int32)

            start_pos = np.random.randint(low=shift_start, high=shift_end)
            indices = np.array([start_pos + i * time_step + offsets[i] for i in range(output_length)])

        return indices, not neg_label

    def _get_test_indices(self, record, time_step, input_length, output_length):
        if record['video_len'] < input_length:
            shift_start = record['video_start']
            indices = np.array([shift_start + i * time_step for i in range(record['video_len'] // time_step)])

            num_rest = output_length - len(indices)
            if num_rest > 0:
                num_before = num_rest // 2
                num_after = num_rest - num_before
                indices = np.concatenate((np.full(num_before, indices[0], dtype=np.int32),
                                          indices,
                                          np.full(num_after, indices[-1], dtype=np.int32)))
        else:
            if record['clip_len'] < input_length:
                shift_start = max(record['video_start'], record['clip_end'] - input_length)
                shift_end = min(record['video_end'] - input_length + 1, record['clip_start'] + 1)
            else:
                shift_start = record['clip_start']
                shift_end = record['clip_end'] - input_length + 1
            start_pos = (shift_start + shift_end) // 2

            indices = np.array([start_pos + i * time_step for i in range(output_length)])

        return indices, True

    def __call__(self, results):
        """Perform the StreamSampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        frame_interval = self._estimate_time_step(results['fps'])
        input_length, output_length = self._estimate_clip_lengths(frame_interval)

        start_index = results['start_index']
        all_frame_inds, all_labels, all_dataset_ids, clip_starts, clip_ends = [], [], [], [], []
        for clip_id in range(self.num_clips):
            clip_frame_inds, is_positive = self._generate_indices(
                results, frame_interval, input_length, output_length
            )

            valid_clip_frame_inds = [cfi for cfi in clip_frame_inds if cfi >= 0]
            assert len(valid_clip_frame_inds) > 0
            clip_starts.append(np.min(valid_clip_frame_inds))
            clip_ends.append(np.max(valid_clip_frame_inds) + 1)

            clip_frame_inds = np.array(clip_frame_inds).astype(np.int)
            clip_frame_inds = np.where(clip_frame_inds < 0, clip_frame_inds, clip_frame_inds + start_index)
            all_frame_inds.append(clip_frame_inds)

            all_labels.append(results['label'] if is_positive else -1)
            all_dataset_ids.append(results['dataset_id'])

        results['frame_inds'] = np.array(all_frame_inds)
        results['clip_starts'] = clip_starts
        results['clip_ends'] = clip_ends
        results['num_clips'] = self.num_clips
        results['clip_len'] = self.clip_len
        results['label'] = all_labels if self.num_clips > 1 else all_labels[0]
        results['dataset_id'] = all_dataset_ids if self.num_clips > 1 else all_dataset_ids[0]
        results['total_frames'] = [results['total_frames']] * self.num_clips \
            if self.num_clips > 1 else results['total_frames']
        if 'sample_idx' in results:
            results['sample_idx'] = [results['sample_idx']] * self.num_clips \
                if self.num_clips > 1 else results['sample_idx']

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(' \
                   f'clip_len={self.clip_len}, ' \
                   f'trg_fps={self.trg_fps}, ' \
                   f'num_clips={self.num_clips}, ' \
                   f'temporal_jitter={self.temporal_jitter}, ' \
                   f'min_intersection={self.min_intersection}, ' \
                   f'neg_prob={self.neg_prob}, ' \
                   f'ignore_outside={self.ignore_outside}, ' \
                   f'test_mode={self.test_mode})'
        return repr_str


@PIPELINES.register_module()
class SampleProposalFrames(SampleFrames):
    """Sample frames from proposals in the video.

    Required keys are "total_frames" and "out_proposals", added or
    modified keys are "frame_inds", "frame_interval", "num_clips",
    'clip_len' and 'num_proposals'.

    Args:
        clip_len (int): Frames of each sampled output clip.
        body_segments (int): Number of segments in course period.
        aug_segments (list[int]): Number of segments in starting and
            ending period.
        aug_ratio (int | float | tuple[int | float]): The ratio
            of the length of augmentation to that of the proposal.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        test_interval (int): Temporal interval of adjacent sampled frames
            in test mode. Default: 6.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        mode (str): Choose 'train', 'val' or 'test' mode.
            Default: 'train'.
    """

    def __init__(self,
                 clip_len,
                 body_segments,
                 aug_segments,
                 aug_ratio,
                 frame_interval=1,
                 test_interval=6,
                 temporal_jitter=False,
                 mode='train'):
        super().__init__(
            clip_len,
            frame_interval=frame_interval,
            temporal_jitter=temporal_jitter)
        self.body_segments = body_segments
        self.aug_segments = aug_segments
        self.aug_ratio = _pair(aug_ratio)
        if not mmcv.is_tuple_of(self.aug_ratio, (int, float)):
            raise TypeError(f'aug_ratio should be int, float'
                            f'or tuple of int and float, '
                            f'but got {type(aug_ratio)}')
        assert len(self.aug_ratio) == 2
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.test_interval = test_interval

    def _get_train_indices(self, valid_length, num_segments):
        """Get indices of different stages of proposals in train mode.

        It will calculate the average interval for each segment,
        and randomly shift them within offsets between [0, average_duration].
        If the total number of frames is smaller than num segments, it will
        return all zero indices.

        Args:
            valid_length (int): The length of the starting point's
                valid interval.
            num_segments (int): Total number of segments.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        avg_interval = (valid_length + 1) // num_segments
        if avg_interval > 0:
            base_offsets = np.arange(num_segments) * avg_interval
            offsets = base_offsets + np.random.randint(
                avg_interval, size=num_segments)
        else:
            offsets = np.zeros((num_segments, ), dtype=np.int)

        return offsets

    def _get_val_indices(self, valid_length, num_segments):
        """Get indices of different stages of proposals in validation mode.

        It will calculate the average interval for each segment.
        If the total number of valid length is smaller than num segments,
        it will return all zero indices.

        Args:
            valid_length (int): The length of the starting point's
                valid interval.
            num_segments (int): Total number of segments.

        Returns:
            np.ndarray: Sampled frame indices in validation mode.
        """
        if valid_length >= num_segments:
            avg_interval = valid_length / float(num_segments)
            base_offsets = np.arange(num_segments) * avg_interval
            offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
        else:
            offsets = np.zeros((num_segments, ), dtype=np.int)

        return offsets

    def _get_proposal_clips(self, proposal, num_frames):
        """Get clip offsets in train mode.

        It will calculate sampled frame indices in the proposal's three
        stages: starting, course and ending stage.

        Args:
            proposal (object): The proposal object.
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        # proposal interval: [start_frame, end_frame)
        start_frame = proposal.start_frame
        end_frame = proposal.end_frame
        ori_clip_len = self.clip_len * self.frame_interval

        duration = end_frame - start_frame
        assert duration != 0
        valid_length = duration - ori_clip_len

        valid_starting = max(0,
                             start_frame - int(duration * self.aug_ratio[0]))
        valid_ending = min(num_frames - ori_clip_len + 1,
                           end_frame - 1 + int(duration * self.aug_ratio[1]))

        valid_starting_length = start_frame - valid_starting - ori_clip_len
        valid_ending_length = (valid_ending - end_frame + 1) - ori_clip_len

        if self.mode == 'train':
            starting_offsets = self._get_train_indices(valid_starting_length,
                                                       self.aug_segments[0])
            course_offsets = self._get_train_indices(valid_length,
                                                     self.body_segments)
            ending_offsets = self._get_train_indices(valid_ending_length,
                                                     self.aug_segments[1])
        elif self.mode == 'val':
            starting_offsets = self._get_val_indices(valid_starting_length,
                                                     self.aug_segments[0])
            course_offsets = self._get_val_indices(valid_length,
                                                   self.body_segments)
            ending_offsets = self._get_val_indices(valid_ending_length,
                                                   self.aug_segments[1])
        starting_offsets += valid_starting
        course_offsets += start_frame
        ending_offsets += end_frame

        offsets = np.concatenate(
            (starting_offsets, course_offsets, ending_offsets))
        return offsets

    def _get_train_clips(self, num_frames, proposals):
        """Get clip offsets in train mode.

        It will calculate sampled frame indices of each proposal, and then
        assemble them.

        Args:
            num_frames (int): Total number of frame in the video.
            proposals (list): Proposals fetched.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        clip_offsets = []
        for proposal in proposals:
            proposal_clip_offsets = self._get_proposal_clips(
                proposal[0][1], num_frames)
            clip_offsets = np.concatenate(
                [clip_offsets, proposal_clip_offsets])

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        It will calculate sampled frame indices based on test interval.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        return np.arange(
            0, num_frames - ori_clip_len, self.test_interval, dtype=np.int)

    def _sample_clips(self, num_frames, proposals):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.
            proposals (list | None): Proposals fetched.
                It is set to None in test mode.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.mode == 'test':
            clip_offsets = self._get_test_clips(num_frames)
        else:
            assert proposals is not None
            clip_offsets = self._get_train_clips(num_frames, proposals)

        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']

        out_proposals = results.get('out_proposals', None)
        clip_offsets = self._sample_clips(total_frames, out_proposals)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        start_index = results['start_index']
        frame_inds = np.mod(frame_inds, total_frames) + start_index

        results['frame_inds'] = np.array(frame_inds).astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = (
            self.body_segments + self.aug_segments[0] + self.aug_segments[1])
        if self.mode in ['train', 'val']:
            results['num_proposals'] = len(results['out_proposals'])

        return results


@PIPELINES.register_module()
class PyAVInit(object):
    """Using pyav to initialize the video.

    PyAV: https://github.com/mikeboers/PyAV

    Required keys are "filename",
    added or modified keys are "video_reader", and "total_frames".

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        kwargs (dict): Args for file client.
    """

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the PyAV initiation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import av
        except ImportError:
            raise ImportError('Please run "conda install av -c conda-forge" '
                              'or "pip install av" to install PyAV first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        container = av.open(file_obj)

        results['video_reader'] = container

        num_frames = container.streams.video[0].frames
        results['total_frames'] = num_frames

        if 'clip_end' in results:
            assert 0 <= results['clip_start'] < results['clip_end'] <= num_frames
            assert 0 <= results['video_start'] < results['video_end'] <= num_frames
        else:
            results['clip_start'] = 0
            results['clip_end'] = num_frames
            results['video_start'] = 0
            results['video_end'] = num_frames
            results['clip_len'] = results['clip_end'] - results['clip_start']
            results['video_len'] = results['video_end'] - results['video_start']
            results['fps'] = container.streams.video[0].average_rate

        return results


@PIPELINES.register_module()
class PyAVDecode(object):
    """Using pyav to decode the video.

    PyAV: https://github.com/mikeboers/PyAV

    Required keys are "video_reader" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".

    Args:
        multi_thread (bool): If set to True, it will apply multi
            thread processing. Default: False.
    """

    def __init__(self, multi_thread=False):
        self.multi_thread = multi_thread

    def __call__(self, results):
        """Perform the PyAV loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
        imgs = list()

        if self.multi_thread:
            container.streams.video[0].thread_type = 'AUTO'
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        # set max indice to make early stop
        max_inds = max(results['frame_inds'])
        i = 0
        for frame in container.decode(video=0):
            if i > max_inds + 1:
                break
            imgs.append(frame.to_rgb().to_ndarray())
            i += 1

        results['video_reader'] = None
        del container

        # the available frame in pyav may be less than its length,
        # which may raise error
        results['imgs'] = [imgs[i % len(imgs)] for i in results['frame_inds']]

        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(multi_thread={self.multi_thread})'
        return repr_str


@PIPELINES.register_module()
class DecordInit(object):
    """Using decord to initialize the video_reader.

    Decord: https://github.com/dmlc/decord

    Required keys are "filename",
    added or modified keys are "video_reader" and "total_frames".
    """

    def __init__(self, io_backend='disk', num_threads=1, **kwargs):
        self.io_backend = io_backend
        self.num_threads = num_threads
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the PyAV loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import decord
        except ImportError:
            raise ImportError('Please run "pip install decord" to install Decord first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        container = decord.VideoReader(file_obj, num_threads=self.num_threads)

        results['video_reader'] = container

        num_frames = len(container)
        results['total_frames'] = num_frames

        if 'clip_end' in results:
            assert 0 <= results['clip_start'] < results['clip_end'] <= num_frames
            assert 0 <= results['video_start'] < results['video_end'] <= num_frames
        else:
            results['clip_start'] = 0
            results['clip_end'] = num_frames
            results['video_start'] = 0
            results['video_end'] = num_frames
            results['clip_len'] = results['clip_end'] - results['clip_start']
            results['video_len'] = results['video_end'] - results['video_start']
            results['fps'] = container.get_avg_fps()

        return results


@PIPELINES.register_module()
class DecordDecode(object):
    """Using decord to decode the video.

    Decord: https://github.com/dmlc/decord

    Required keys are "video_reader", "filename" and "frame_inds",
    added or modified keys are "imgs" and "original_shape".
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, results):
        """Perform the Decord loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        # Generate frame index mapping in order
        frame_dict = {
            idx: container[idx].asnumpy()
            for idx in np.unique(frame_inds)
        }

        imgs = [frame_dict[idx] for idx in frame_inds]

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@PIPELINES.register_module()
class OpenCVInit(object):
    """Using OpenCV to initalize the video_reader.

    Required keys are "filename", added or modified keys are "new_path",
    "video_reader" and "total_frames".
    """

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None
        random_string = get_random_string()
        thread_id = get_thread_id()
        self.tmp_folder = osp.join(get_shm_dir(),
                                   f'{random_string}_{thread_id}')
        os.mkdir(self.tmp_folder)

    def __call__(self, results):
        """Perform the OpenCV initiation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.io_backend == 'disk':
            new_path = results['filename']
        else:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend, **self.kwargs)

            thread_id = get_thread_id()
            # save the file of same thread at the same place
            new_path = osp.join(self.tmp_folder, f'tmp_{thread_id}.mp4')
            with open(new_path, 'wb') as f:
                f.write(self.file_client.get(results['filename']))

        container = mmcv.VideoReader(new_path)
        results['new_path'] = new_path
        results['video_reader'] = container

        num_frames = len(container)
        results['total_frames'] = num_frames

        if 'clip_end' in results:
            assert 0 <= results['clip_start'] < results['clip_end'] <= num_frames
            assert 0 <= results['video_start'] < results['video_end'] <= num_frames
        else:
            results['clip_start'] = 0
            results['clip_end'] = num_frames
            results['video_start'] = 0
            results['video_end'] = num_frames
            results['clip_len'] = results['clip_end'] - results['clip_start']
            results['video_len'] = results['video_end'] - results['video_start']
            results['fps'] = container.fps()

        return results

    def __del__(self):
        shutil.rmtree(self.tmp_folder)


@PIPELINES.register_module()
class OpenCVDecode(object):
    """Using OpenCV to decode the video.

    Required keys are "video_reader", "filename" and "frame_inds", added or
    modified keys are "imgs", "img_shape" and "original_shape".
    """

    def __init__(self):
        pass

    def __call__(self, results):
        """Perform the OpenCV loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        for frame_ind in results['frame_inds']:
            cur_frame = container[frame_ind]
            # last frame may be None in OpenCV
            while isinstance(cur_frame, type(None)):
                frame_ind -= 1
                cur_frame = container[frame_ind]
            imgs.append(cur_frame)

        results['video_reader'] = None
        del container

        imgs = np.array(imgs)
        # The default channel order of OpenCV is BGR, thus we change it to RGB
        imgs = imgs[:, :, :, ::-1]
        results['imgs'] = list(imgs)
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@PIPELINES.register_module()
class RawFrameDecode(object):
    """Load and decode frames with given indices.

    Required keys are "frame_dir", "filename_tmpl" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = results['frame_inds'].flatten()

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']
        offset = results.get('offset', 0)

        imgs = list()
        valid_image_ids, empty_image_ids = [], []
        for i, frame_idx in enumerate(results['frame_inds']):
            if frame_idx < 0:
                imgs.append(None)
                empty_image_ids.append(i)
                continue

            frame_idx += offset

            if modality == 'RGB':
                filepath = osp.join(directory, filename_tmpl.format(frame_idx))
                img_bytes = self.file_client.get(filepath)
                image = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            elif modality == 'Flow':
                x_filepath = osp.join(directory,
                                      filename_tmpl.format('x', frame_idx))
                y_filepath = osp.join(directory,
                                      filename_tmpl.format('y', frame_idx))
                x_img_bytes = self.file_client.get(x_filepath)
                x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
                y_img_bytes = self.file_client.get(y_filepath)
                y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
                image = [x_frame, y_frame]
            else:
                raise NotImplementedError

            imgs.append(image)
            valid_image_ids.append(i)

        assert len(valid_image_ids) > 0

        if len(empty_image_ids) > 0:
            valid_image = imgs[valid_image_ids[0]]
            for empty_idx in empty_image_ids:
                image = np.zeros_like(valid_image)
                if modality == 'Flow':
                    image = [image, np.zeros_like(valid_image)]

                imgs[empty_idx] = image

        if modality == 'Flow':
            imgs = [im for tup in imgs for im in tup]

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(io_backend={self.io_backend}, ' \
                   f'decoding_backend={self.decoding_backend})'
        return repr_str


@PIPELINES.register_module()
class FrameSelector(RawFrameDecode):
    """Deprecated class for ``RawFrameDecode``."""

    def __init__(self, *args, **kwargs):
        warnings.warn('"FrameSelector" is deprecated, please switch to'
                      '"RawFrameDecode"')
        super().__init__(*args, **kwargs)


@PIPELINES.register_module()
class LoadLocalizationFeature(object):
    """Load Video features for localizer with given video_name list.

    Required keys are "video_name" and "data_prefix",
    added or modified keys are "raw_feature".

    Args:
        raw_feature_ext (str): Raw feature file extension.  Default: '.csv'.
    """

    def __init__(self, raw_feature_ext='.csv'):
        valid_raw_feature_ext = ('.csv', )
        if raw_feature_ext not in valid_raw_feature_ext:
            raise NotImplementedError
        self.raw_feature_ext = raw_feature_ext

    def __call__(self, results):
        """Perform the LoadLocalizationFeature loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        video_name = results['video_name']
        data_prefix = results['data_prefix']

        data_path = osp.join(data_prefix, video_name + self.raw_feature_ext)
        raw_feature = np.loadtxt(
            data_path, dtype=np.float32, delimiter=',', skiprows=1)

        results['raw_feature'] = np.transpose(raw_feature, (1, 0))

        return results


@PIPELINES.register_module()
class GenerateLocalizationLabels(object):
    """Load video label for localizer with given video_name list.

    Required keys are "duration_frame", "duration_second", "feature_frame",
    "annotations", added or modified keys are "gt_bbox".
    """

    def __call__(self, results):
        """Perform the GenerateLocalizationLabels loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        video_frame = results['duration_frame']
        video_second = results['duration_second']
        feature_frame = results['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second
        annotations = results['annotations']

        gt_bbox = []

        for annotation in annotations:
            current_start = max(
                min(1, annotation['segment'][0] / corrected_second), 0)
            current_end = max(
                min(1, annotation['segment'][1] / corrected_second), 0)
            gt_bbox.append([current_start, current_end])

        gt_bbox = np.array(gt_bbox)
        results['gt_bbox'] = gt_bbox
        return results


@PIPELINES.register_module()
class LoadProposals(object):
    """Loading proposals with given proposal results.

    Required keys are "video_name"
    added or modified keys are 'bsp_feature', 'tmin', 'tmax',
    'tmin_score', 'tmax_score' and 'reference_temporal_iou'.

    Args:
        top_k (int): The top k proposals to be loaded.
        pgm_proposals_dir (str): Directory to load proposals.
        pgm_features_dir (str): Directory to load proposal features.
        proposal_ext (str): Proposal file extension. Default: '.csv'.
        feature_ext (str): Feature file extension. Default: '.npy'.
    """

    def __init__(self,
                 top_k,
                 pgm_proposals_dir,
                 pgm_features_dir,
                 proposal_ext='.csv',
                 feature_ext='.npy'):
        self.top_k = top_k
        self.pgm_proposals_dir = pgm_proposals_dir
        self.pgm_features_dir = pgm_features_dir
        valid_proposal_ext = ('.csv', )
        if proposal_ext not in valid_proposal_ext:
            raise NotImplementedError
        self.proposal_ext = proposal_ext
        valid_feature_ext = ('.npy', )
        if feature_ext not in valid_feature_ext:
            raise NotImplementedError
        self.feature_ext = feature_ext

    def __call__(self, results):
        """Perform the LoadProposals loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        video_name = results['video_name']
        proposal_path = osp.join(self.pgm_proposals_dir,
                                 video_name + self.proposal_ext)
        if self.proposal_ext == '.csv':
            pgm_proposals = np.loadtxt(
                proposal_path, dtype=np.float32, delimiter=',', skiprows=1)

        pgm_proposals = np.array(pgm_proposals[:self.top_k])
        tmin = pgm_proposals[:, 0]
        tmax = pgm_proposals[:, 1]
        tmin_score = pgm_proposals[:, 2]
        tmax_score = pgm_proposals[:, 3]
        reference_temporal_iou = pgm_proposals[:, 5]

        feature_path = osp.join(self.pgm_features_dir,
                                video_name + self.feature_ext)
        if self.feature_ext == '.npy':
            bsp_feature = np.load(feature_path).astype(np.float32)

        bsp_feature = bsp_feature[:self.top_k, :]

        results['bsp_feature'] = bsp_feature
        results['tmin'] = tmin
        results['tmax'] = tmax
        results['tmin_score'] = tmin_score
        results['tmax_score'] = tmax_score
        results['reference_temporal_iou'] = reference_temporal_iou

        return results


@PIPELINES.register_module()
class LoadKpts(object):
    """Loads key-points.
    """

    def __init__(self, out_name='kpts'):
        self.out_name = out_name

    @staticmethod
    def _load_kpts(filepath):
        with open(filepath) as kpts_stream:
            raw_kpts = mmcv.load(kpts_stream, file_format='json')

        kpts = dict()
        for kpt_id, frame_data in raw_kpts.items():
            kpts[int(kpt_id)] = {int(frame_id): kpt for frame_id, kpt in frame_data.items()}

        return kpts

    def __call__(self, results):
        assert 'kpts_file' in results
        assert osp.exists(results['kpts_file'])

        results[self.out_name] = self._load_kpts(results['kpts_file'])

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__} (' \
                   f'out_name={self.out_name})'
        return repr_str


@PIPELINES.register_module()
class GenerateKptsMask(object):
    """Generates key-point masks.
    """

    def __init__(self, sigma_scale=0.1, out_name='attention_mask'):
        self.sigma_scale = sigma_scale
        self.out_name = out_name

    @staticmethod
    def _generate_mask_by_kpts(frame_id, kpts, size, sigma_scale):
        assert len(size) == 2
        out_height, out_width = size
        sigma_sqr = (sigma_scale * np.sqrt(out_height ** 2 + out_width ** 2)) ** 2

        mask_accumulator = np.zeros(size, dtype=np.float32)
        for kpt_data in kpts.values():
            if frame_id in kpt_data:
                center_x, center_y = [float(value) for value in kpt_data[frame_id][:2]]

                x, y = np.meshgrid(np.arange(out_width), np.arange(out_height))
                dist_sqr = (x - center_x) ** 2 + (y - center_y) ** 2

                local_mask = np.exp(-0.5 * dist_sqr / sigma_sqr)
                mask_accumulator += local_mask

        max_value = np.max(mask_accumulator)
        if max_value > 0.0:
            mask_accumulator /= max_value
            out_mask = np.where(
                mask_accumulator > 0.5,
                np.ones(size, dtype=np.uint8),
                np.zeros(size, dtype=np.uint8)
            )
        else:
            out_mask = np.zeros(size, dtype=np.uint8)

        return out_mask.reshape(size)

    @staticmethod
    def _load_kpts(filepath):
        with open(filepath) as kpts_stream:
            raw_kpts = mmcv.load(kpts_stream, file_format='json')

        kpts = dict()
        for kpt_id, frame_data in raw_kpts.items():
            kpts[int(kpt_id)] = {int(frame_id): kpt for frame_id, kpt in frame_data.items()}

        return kpts

    def __call__(self, results):
        assert 'kpts_file' in results
        assert osp.exists(results['kpts_file'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = results['frame_inds'].flatten()

        kpts = self._load_kpts(results['kpts_file'])
        imgs = results['imgs']
        offset = results.get('offset', 0)

        assert len(imgs) > 0, 'At least one image should be loaded before mask sampling'

        masks = []
        valid_mask_ids, empty_mask_ids = [], []
        for i, frame_idx in enumerate(results['frame_inds']):
            mask = None
            if frame_idx >= 0:
                mask = self._generate_mask_by_kpts(
                    frame_idx + offset, kpts, imgs[0].shape[:2], self.sigma_scale
                )

            ids_list = empty_mask_ids if mask is None else valid_mask_ids
            ids_list.append(i)

            masks.append(mask)

        if len(empty_mask_ids) > 0:
            valid_mask = masks[valid_mask_ids[0]]
            for empty_idx in empty_mask_ids:
                masks[empty_idx] = np.zeros_like(valid_mask)

        results[self.out_name] = masks

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__} (' \
                   f'sigma_scale={self.sigma_scale}, ' \
                   f'out_name={self.out_name})'
        return repr_str
