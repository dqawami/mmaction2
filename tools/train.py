import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config
from mmcv.runner import init_dist, set_random_seed

from mmaction import __version__
from mmaction.apis import train_model
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.utils import collect_env, get_root_logger, ExtendedDictAction
from mmaction.core.utils import propagate_root_dir
from mmaction.integration.nncf import check_nncf_is_enabled, get_nncf_metadata

MODEL_SOURCES = 'modelzoo://', 'torchvision://', 'open-mmlab://', 'http://', 'https://'


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config',
                        help='train config file path')
    parser.add_argument('--data_dir',
                        help='the dir with dataset')
    parser.add_argument('--work_dir',
                        help='the dir to save logs and models')
    parser.add_argument('--tensorboard_dir',
                        help='the dir to save tensorboard logs')
    parser.add_argument('--resume_from',
                        help='the checkpoint file to resume from')
    parser.add_argument('--load_from',
                        help='the checkpoint file to init weights from')
    parser.add_argument('--load2d_from',
                        help='the checkpoint file to init 2D weights from')
    parser.add_argument('--no_validate', action='store_true',
                        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int,
                            help='number of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu_ids', type=int, nargs='+',
                            help='ids of gpus to use (only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    parser.add_argument('--num_videos', type=int,
                        help='number of videos per GPU')
    parser.add_argument('--num_workers', type=int,
                        help='number of CPU workers per GPU')
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--update_config', nargs='+', action=ExtendedDictAction,
                        help='arguments in dict')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def is_valid(model_path):
    if model_path is None:
        return False

    return osp.exists(model_path) or model_path.startswith(MODEL_SOURCES)


def update_config(cfg, args):
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.tensorboard_dir is not None:
        hooks = [hook for hook in cfg.log_config.hooks if hook.type == 'TensorboardLoggerHook']
        for hook in hooks:
            hook.log_dir = args.tensorboard_dir

    # work_dir is determined in this priority:
    # CLI > config file > default (base filename)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if is_valid(args.resume_from):
        cfg.resume_from = args.resume_from

    if is_valid(args.load_from):
        cfg.load_from = args.load_from

    if is_valid(args.load2d_from):
        cfg.model.backbone.pretrained = args.load2d_from
        cfg.model.backbone.pretrained2d = True

    if args.num_videos is not None and args.num_videos > 0:
        cfg.data.videos_per_gpu = args.num_videos

    if args.num_workers is not None and args.num_workers > 0:
        cfg.data.workers_per_gpu = args.num_workers

    if cfg.checkpoint_config is not None:
        # save mmaction version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmaction_version=__version__,
            config=cfg.text
        )
        if cfg.get('nncf_config'):
            nncf_metadata = get_nncf_metadata()
            cfg.checkpoint_config.meta.update(nncf_metadata)
    else:
        assert not cfg.get('nncf_config'), (
            "NNCF is enabled, but checkpoint_config is not set -- "
            "cannot store NNCF metainfo into checkpoints")

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    return cfg


def main():
    # parse arguments
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.update_config is not None:
        cfg.merge_from_dict(args.update_config)
    cfg = update_config(cfg, args)
    cfg = propagate_root_dir(cfg, args.data_dir)

    # init distributed env first, since logger depends on the dist info.
    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()

    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config: {cfg.text}')

    if cfg.get('nncf_config'):
        check_nncf_is_enabled()
        logger.info('NNCF config: {}'.format(cfg.nncf_config))
        meta.update(get_nncf_metadata())

    # set random seeds
    cfg.seed = args.seed
    meta['seed'] = args.seed
    if cfg.get('seed'):
        logger.info(f'Set random seed to {cfg.seed}, deterministic: {args.deterministic}')
        set_random_seed(cfg.seed, deterministic=args.deterministic)

    # build datasets
    datasets = [build_dataset(cfg.data, 'train', dict(logger=logger))]
    logger.info(f'Train datasets:\n{str(datasets[0])}')

    if len(cfg.workflow) == 2:
        if not args.no_validate:
            warnings.warn('val workflow is duplicated with `--validate`, '
                          'it is recommended to use `--validate`. see '
                          'https://github.com/open-mmlab/mmaction2/pull/123')
        datasets.append(build_dataset(copy.deepcopy(cfg.data), 'val', dict(logger=logger)))
        logger.info(f'Val datasets:\n{str(datasets[-1])}')

    # filter dataset labels
    if cfg.get('classes'):
        datasets = [dataset.filter(cfg.classes) for dataset in datasets]

    # build model
    model = build_model(
        cfg.model,
        train_cfg=cfg.train_cfg,
        test_cfg=cfg.test_cfg,
        class_sizes=datasets[0].class_sizes,
        class_maps=datasets[0].class_maps
    )

    # define ignore layers
    ignore_prefixes = []
    if hasattr(cfg, 'reset_layer_prefixes') and isinstance(cfg.reset_layer_prefixes, (list, tuple)):
        ignore_prefixes += cfg.reset_layer_prefixes
    ignore_suffixes = ['num_batches_tracked']
    if hasattr(cfg, 'reset_layer_suffixes') and isinstance(cfg.reset_layer_suffixes, (list, tuple)):
        ignore_suffixes += cfg.reset_layer_suffixes

    # train model
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
        ignore_prefixes=tuple(ignore_prefixes),
        ignore_suffixes=tuple(ignore_suffixes)
    )


if __name__ == '__main__':
    main()
