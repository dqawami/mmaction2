import argparse
import os

import mmcv
import torch
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmaction.apis import multi_gpu_test, single_gpu_test
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import ExtendedDictAction
from mmaction.core.utils import propagate_root_dir, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 test (and eval) a model')
    parser.add_argument('config',
                        help='test config file path')
    parser.add_argument('checkpoint',
                        help='checkpoint file')
    parser.add_argument('--data_dir', type=str,
                        help='the dir with dataset')
    parser.add_argument('--out', default=None,
                        help='output result file in pickle format')
    parser.add_argument('--fuse_conv_bn', action='store_true',
                        help='Whether to fuse conv and bn, this will slightly increase the inference speed')
    parser.add_argument('--eval', type=str, nargs='+',
                        help='evaluation metrics, which depends on the dataset, e.g.,'
                             ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument('--gpu_collect', action='store_true',
                        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir',
                        help='tmp directory used for collecting results from multiple '
                             'workers, available when gpu-collect is not specified')
    parser.add_argument('--options', nargs='+', help='custom options')
    parser.add_argument('--average_clips', choices=['score', 'prob'], default='score',
                        help='average type when averaging test clips')
    parser.add_argument('--num_workers', type=int,
                        help='number of CPU workers per GPU')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--update_config', nargs='+', action=ExtendedDictAction,
                        help='Update configuration file by parameters specified here.')
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def update_config(cfg, args):
    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.num_workers is not None and args.num_workers > 0:
        cfg.data.workers_per_gpu = args.num_workers

    if cfg.test_cfg is None:
        cfg.test_cfg = dict(average_clips=args.average_clips)
    else:
        cfg.test_cfg.average_clips = args.average_clips

    cfg.data.train.test_mode = True
    cfg.data.val.test_mode = True
    cfg.data.test.test_mode = True

    cfg.data.train.transforms = None

    return cfg


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v

    return cfg1


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.update_config is not None:
        cfg.merge_from_dict(args.update_config)
    cfg = update_config(cfg, args)
    cfg = propagate_root_dir(cfg, args.data_dir)

    # Load output_config from cfg
    output_config = cfg.get('output_config', {})
    # Overwrite output_config from args.out
    output_config = merge_configs(output_config, dict(out=args.out))

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    # Overwrite eval_config from args.eval
    eval_config = merge_configs(eval_config, dict(metrics=args.eval))
    # Add options from args.option
    eval_config = merge_configs(eval_config, args.options)

    assert output_config or eval_config, \
        ('Please specify at least one operation (save or eval the '
         'results) with the argument "--out" or "--eval"')

    # init distributed env first, since logger depends on the dist info.
    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()

    # build the dataset
    dataset = build_dataset(cfg.data, 'test', dict(test_mode=True))
    if cfg.get('classes'):
        target_class_ids = list(map(int, cfg.classes.split(',')))
        dataset = dataset.filter(target_class_ids)
    if rank == 0:
        print(f'Test datasets:\n{str(dataset)}')

    # build the dataloader
    data_loader = build_dataloader(
        dataset,
        videos_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False
    )

    # build the model and load checkpoint
    model = build_model(
        cfg.model,
        train_cfg=None,
        test_cfg=cfg.test_cfg,
        class_sizes=dataset.class_sizes,
        class_maps=dataset.class_maps
    )
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    # load model weights
    load_checkpoint(model, args.checkpoint, map_location='cpu', force_matching=True)

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    if rank == 0:
        if output_config:
            out = output_config['out']
            print(f'\nwriting results to {out}')
            dataset.dump_results(outputs, **output_config)

        if eval_config:
            eval_res = dataset.evaluate(outputs, **eval_config)

            print('\nFinal metrics:')
            for name, val in eval_res.items():
                print(f'{name}: {val:.04f}')


if __name__ == '__main__':
    main()
