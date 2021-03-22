from abc import ABCMeta
from collections import defaultdict

from mmcv.utils import print_log

from ..core import (mean_class_accuracy, top_k_accuracy, mean_top_k_accuracy,
                    mean_average_precision, ranking_mean_average_precision,
                    invalid_pred_info, confusion_matrix)
from .base import BaseDataset


class RecognitionDataset(BaseDataset, metaclass=ABCMeta):
    """Base class for action recognition datasets.
    """

    allowed_metrics = [
        'top_k_accuracy', 'mean_top_k_accuracy', 'mean_class_accuracy',
        'mean_average_precision', 'ranking_mean_average_precision',
        'confusion_matrix', 'invalid_info'
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _evaluate(self,
                  results,
                  metrics='top_k_accuracy',
                  topk=(1, 5),
                  logger=None):
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
