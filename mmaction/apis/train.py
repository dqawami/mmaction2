import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, EpochBasedRunner, build_optimizer
from mmcv.runner.hooks import Fp16OptimizerHook, EMAHook

from ..core import DistEvalHook, EvalHook, load_checkpoint, DistOptimizerHook, SampleInfoAggregatorHook
from ..datasets import build_dataloader, build_dataset
from ..utils import get_root_logger
from ..models import build_params_manager
from ..integration.nncf import CompressionHook, wrap_nncf_model
from .fake_input import get_fake_input


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None,
                ignore_prefixes=None,
                ignore_suffixes=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
        ignore_suffixes: Ignored suffixes
        ignore_prefixes: Ignored prefixes
    """
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', {}),
        workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting, **cfg.data.get('train_dataloader', {}))
    data_loaders = [
        build_dataloader(ds, **dataloader_setting) for ds in dataset
    ]

    if hasattr(model, 'update_state'):
        num_steps_per_epoch = len(data_loaders[0])
        model.update_state(num_steps_per_epoch)

    if cfg.load_from:
        load_checkpoint(model,
                        cfg.load_from,
                        strict=False,
                        logger=logger,
                        show_converted=True,
                        ignore_prefixes=ignore_prefixes, 
                        ignore_suffixes=ignore_suffixes)
                    
    if torch.cuda.is_available():
        model = model.cuda()

    nncf_enable_compression = bool(cfg.get('nncf_config'))
    if nncf_enable_compression:
        compression_ctrl, model = wrap_nncf_model(model, cfg, data_loaders[0], get_fake_input)
    else:
        compression_ctrl = None

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters
        )
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids
        )

    if nncf_enable_compression and distributed:
        compression_ctrl.distributed()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta
    )
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed
        )
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    ema_cfg = cfg.get('ema_config', None)
    if ema_cfg:
        runner.register_hook(EMAHook(**ema_cfg))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    params_manager_cfg = cfg.get('params_config', None)
    if params_manager_cfg is not None:
        runner.register_hook(build_params_manager(params_manager_cfg))

    if model.module.with_sample_filtering:
        runner.register_hook(SampleInfoAggregatorHook(
            cfg.train_cfg.sample_filtering.get('warmup_epochs', 0)
        ))

    if validate:
        val_dataset = build_dataset(cfg.data, 'val', dict(test_mode=True))
        runner.logger.info(f'Val datasets:\n{str(val_dataset)}')

        num_test_videos_per_gpu = cfg.data['test_videos_per_gpu']\
            if 'test_videos_per_gpu' in cfg.data else cfg.data.get('videos_per_gpu', {})

        dataloader_setting = dict(
            videos_per_gpu=num_test_videos_per_gpu,
            workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)

        eval_hook = DistEvalHook if distributed else EvalHook
        eval_cfg = cfg.get('evaluation', {})
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if nncf_enable_compression:
        runner.register_hook(CompressionHook(compression_ctrl=compression_ctrl))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
