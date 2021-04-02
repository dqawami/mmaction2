import platform
import random
from functools import partial
from copy import deepcopy

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import build_from_cfg
from torch.utils.data import DataLoader

from .dataset_wrappers import RepeatDataset
from .registry import DATASETS
from .samplers import DistributedSampler, BalancedDistributedSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def _to_list(value, num=None):
    if isinstance(value, (tuple, list)):
        if num is not None:
            assert len(value) == num, f'Invalid len of argument: {len(value)} but expected {num}'

        return value
    else:
        return [value] * (num if num is not None else 1)


def build_dataset(cfg, target, default_args=None):
    """Build a dataset from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        target (str): Target name. One of : "train", "val", "test".
        default_args (dict, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """

    source_cfg = cfg['dataset'] if hasattr(cfg, 'type') and cfg['type'] == 'RepeatDataset' else cfg
    target_cfg = source_cfg[target]

    assert 'root_dir' in source_cfg, 'Data config does not contain \'root_dir\' field'
    assert 'source' in target_cfg, 'Data config does not contain \'sources\' field'
    assert 'ann_file' in target_cfg, 'Data config does not contain \'ann_file\' field'

    sources = _to_list(target_cfg['source'])
    num_sources = len(sources)

    ann_files = _to_list(target_cfg['ann_file'], num_sources)
    shared_info = {k: _to_list(v, num_sources) for k, v in source_cfg.get('shared', dict()).items()}

    datasets = []
    for dataset_id in range(num_sources):
        dataset_cfg = deepcopy(target_cfg)

        dataset_cfg['root_dir'] = source_cfg['root_dir']
        dataset_cfg['source'] = sources[dataset_id]
        dataset_cfg['ann_file'] = ann_files[dataset_id]
        for shared_key, shared_value in shared_info.items():
            dataset_cfg[shared_key] = shared_value[dataset_id]

        datasets.append(build_from_cfg(dataset_cfg, DATASETS, default_args))

    dataset = sum(datasets)

    if hasattr(cfg, 'type') and cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(dataset, cfg['times'])

    return dataset


def build_dataloader(dataset,
                     videos_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (:obj:`Dataset`): A PyTorch dataset.
        videos_per_gpu (int): Number of videos on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data
            loading for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed
            training. Default: 1.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int | None): Seed to be used. Default: None.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default: False
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        kwargs (dict, optional): Any keyword argument to be used to initialize
            DataLoader.

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        # sampler = DistributedSampler(dataset, world_size, rank, shuffle=shuffle)
        sampler = BalancedDistributedSampler(dataset, world_size, rank, shuffle=shuffle, num_instances=2)
        shuffle = False
        batch_size = videos_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * videos_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)\
              if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=videos_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Init the random seed for various workers."""
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
