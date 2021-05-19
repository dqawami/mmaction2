from abc import ABCMeta, abstractmethod
from collections import defaultdict
from os.path import exists, join

import torch
from mmcv.utils import print_log

from ..core import (mean_class_accuracy, top_k_accuracy, mean_top_k_accuracy,
                    mean_average_precision, ranking_mean_average_precision,
                    invalid_pred_info, confusion_matrix)
from .base import BaseDataset


SIMPLE_RECORD_SIZE = 2
DETAILED_RECORD_SIZE = 7


class RecognitionDataset(BaseDataset, metaclass=ABCMeta):
    """Base class for action recognition datasets.
    """

    allowed_metrics = [
        'top_k_accuracy', 'mean_top_k_accuracy', 'mean_class_accuracy',
        'mean_average_precision', 'ranking_mean_average_precision',
        'confusion_matrix', 'invalid_info'
    ]

    def __init__(self,
                 action_type_file=None,
                 filter_min_fraction=0.8,
                 **kwargs):
        self.filter_min_fraction = filter_min_fraction

        super().__init__(**kwargs)

        if action_type_file is not None:
            assert isinstance(action_type_file, dict)
            assert len(self.dataset_ids_map) == 1

            source_name = self.dataset_ids_map[0]
            if source_name in action_type_file:
                action_type_file = join(self.root_dir, action_type_file[source_name])
                action_type_map = self._load_action_type_map(action_type_file)
                if action_type_map is not None:
                    self.records = self._update_action_type_info(self.records, action_type_map)

    @staticmethod
    def _load_action_type_map(file_path):
        if not exists(file_path):
            return None

        action_type_map = dict()
        with open(file_path) as input_stream:
            for line in input_stream:
                line_parts = line.strip().split(':')
                if len(line_parts) != 2:
                    continue

                action_type_map[int(line_parts[0])] = line_parts[1]

        return action_type_map

    @staticmethod
    def _update_action_type_info(records, action_type_map):
        for record in records:
            label = record['label']
            record['action_type'] = action_type_map[label]

        return records

    def _load_annotations(self, ann_file, data_prefix=None):
        """Load annotation file to get video information."""

        if ann_file.endswith('.json'):
            return self.load_json_annotations()

        video_infos = []
        with open(ann_file, 'r') as input_stream:
            for line in input_stream:
                line_split = line.strip().split()

                if self.multi_class or self.with_offset:
                    record = self._parse_original_record(line_split[1:], self.multi_class,
                                                         self.with_offset, self.num_classes)
                elif len(line_split) == SIMPLE_RECORD_SIZE:
                    record = self._parse_simple_record(line_split[1:])
                elif len(line_split) == DETAILED_RECORD_SIZE:
                    record = self._parse_detailed_record(line_split[1:])
                else:
                    continue

                record.update(self._parse_data_source(line_split[0], data_prefix))
                record.update(self._get_extra_info())

                video_infos.append(record)

        return video_infos

    @staticmethod
    def _parse_original_record(line_splits, multi_class, with_offset, num_classes):
        record = dict()

        idx = 0
        if with_offset:
            # idx for offset and total_frames
            record['offset'] = int(line_splits[idx])
            record['total_frames'] = int(line_splits[idx + 1])
            idx += 2
        else:
            # idx for total_frames
            record['total_frames'] = int(line_splits[idx])
            idx += 1

        record['clip_start'] = 0
        record['clip_end'] = record['total_frames']
        record['video_start'] = 0
        record['video_end'] = record['total_frames']
        record['fps'] = 30.0

        # idx for label[s]
        label = [int(x) for x in line_splits[idx:]]
        assert len(label), 'missing label'
        if multi_class:
            assert num_classes is not None
            label_vector = torch.zeros(num_classes)
            label_vector[label] = 1.0
            record['label'] = label_vector
        else:
            assert len(label) == 1
            record['label'] = int(label[0])

        return record

    @staticmethod
    def _parse_simple_record(line_splits):
        record = dict(
            label=int(line_splits[0]),
        )

        return record

    @staticmethod
    def _parse_detailed_record(line_splits):
        record = dict(
            label=int(line_splits[0]),
            clip_start=int(line_splits[1]),
            clip_end=int(line_splits[2]),
            video_start=int(line_splits[3]),
            video_end=int(line_splits[4]),
            fps=float(line_splits[5]),
        )

        record['clip_len'] = record['clip_end'] - record['clip_start']
        assert record['clip_len'] > 0

        record['video_len'] = record['video_end'] - record['video_start']
        assert record['video_len'] > 0

        return record

    @staticmethod
    def _get_extra_info():
        return dict(
            matched_weights=defaultdict(float),
            filter_ready=False,
            action_type='dynamic',
        )

    @abstractmethod
    def _parse_data_source(self, data_source, data_prefix):
        pass

    def _evaluate(self, results, metrics='top_k_accuracy', topk=(1, 5), logger=None):
        """Evaluation in action recognition dataset.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            logger (obj): Training logger. Defaults: None.
            topk (tuple[int]): K value for top_k_accuracy metric.
                Defaults: (1, 5).
            logger (logging.Logger | None): Logger for recording.
                Default: None.

        Return:
            dict: Evaluation results dict.
        """

        if isinstance(topk, int):
            topk = (topk,)
        elif not isinstance(topk, tuple):
            raise TypeError(f'topk must be int or tuple of int, but got {type(topk)}')

        all_gt_labels = [ann['label'] for ann in self.records]
        all_dataset_ids = [ann['dataset_id'] for ann in self.records]

        split_results, split_gt_labels = defaultdict(list), defaultdict(list)
        for ind, result in enumerate(results):
            dataset_id = all_dataset_ids[ind]
            dataset_name = self.dataset_ids_map[dataset_id]

            split_results[dataset_name].append(result.reshape([-1]))
            split_gt_labels[dataset_name].append(all_gt_labels[ind])

        eval_results = dict()
        for dataset_name in split_results.keys():
            dataset_results = split_results[dataset_name]
            dataset_gt_labels = split_gt_labels[dataset_name]

            dataset_results = self._evaluate_dataset(
                dataset_results, dataset_gt_labels, dataset_name, metrics, topk, logger
            )
            eval_results.update(dataset_results)

        return eval_results

    @staticmethod
    def _evaluate_dataset(results, gt_labels, name, metrics, topk, logger=None):
        eval_results = dict()
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'val/{name}/top{k}_acc'] = acc
                    log_msg.append(f'\n{name}/top{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric == 'mean_top_k_accuracy':
                log_msg = []
                for k in topk:
                    acc = mean_top_k_accuracy(results, gt_labels, k)
                    eval_results[f'val/{name}/mean_top{k}_acc'] = acc
                    log_msg.append(f'\n{name}/mean_top{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results[f'val/{name}/mean_class_accuracy'] = mean_acc
                log_msg = f'\n{name}/mean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric == 'mean_average_precision':
                mAP = mean_average_precision(results, gt_labels)
                eval_results[f'val/{name}/mAP'] = mAP
                log_msg = f'\n{name}/mAP\t{mAP:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric == 'ranking_mean_average_precision':
                mAP = ranking_mean_average_precision(results, gt_labels)
                eval_results[f'val/{name}/rank_mAP'] = mAP
                log_msg = f'\n{name}/rank_mAP\t{mAP:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric == 'confusion_matrix':
                cm = confusion_matrix(results, gt_labels)
                eval_results[f'val/{name}/conf_matrix'] = cm
                log_msg = f'\n{name}/conf_matrix evaluated'
                print_log(log_msg, logger=logger)
                continue

            if metric == 'invalid_info':
                invalid_ids, invalid_conf, invalid_pred = invalid_pred_info(results, gt_labels, k=1)
                eval_results[f'val/{name}/invalid_info'] = \
                    dict(ids=invalid_ids, conf=invalid_conf, pred=invalid_pred)
                log_msg = f'\n{name}/invalid is collected'
                print_log(log_msg, logger=logger)
                continue

        return eval_results

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
