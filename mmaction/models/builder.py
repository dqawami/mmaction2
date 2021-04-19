import torch.nn as nn
from mmcv.utils import build_from_cfg

from .registry import (BACKBONES, SPATIAL_TEMPORAL_MODULES, HEADS, LOCALIZERS, LOSSES, RECOGNIZERS,
                       SCALAR_SCHEDULERS, PARAMS_MANAGERS, NECKS)


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, it is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """

    if cfg is None:
        return None
    elif isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_reducer(cfg):
    """Build spatio-temporal reducer."""
    return build(cfg, SPATIAL_TEMPORAL_MODULES)


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)


def build_head(cfg, class_sizes=None):
    """Build head."""
    if class_sizes is None:
        return build(cfg, HEADS)
    else:
        assert isinstance(class_sizes, (list, tuple))
        heads = [build(cfg, HEADS, dict(class_sizes=cs)) for cs in class_sizes]

        if len(heads) > 1:
            return nn.ModuleList(heads)
        else:
            return heads[0]


def build_recognizer(cfg, train_cfg=None, test_cfg=None, class_sizes=None, class_maps=None):
    """Build recognizer."""
    return build(cfg, RECOGNIZERS,
                 dict(train_cfg=train_cfg, test_cfg=test_cfg, class_sizes=class_sizes, class_maps=class_maps))


def build_loss(cfg, class_sizes=None):
    """Build loss."""
    kwargs = dict()
    if class_sizes is not None:
        kwargs['class_sizes'] = class_sizes

    return build(cfg, LOSSES, kwargs)


def build_localizer(cfg):
    """Build localizer."""
    return build(cfg, LOCALIZERS)


def build_model(cfg, train_cfg=None, test_cfg=None, class_sizes=None, class_maps=None):
    """Build model."""
    args = cfg.copy()

    obj_type = args.pop('type')
    if obj_type in LOCALIZERS:
        return build_localizer(cfg)
    elif obj_type in RECOGNIZERS:
        return build_recognizer(cfg, train_cfg, test_cfg, class_sizes, class_maps)


def build_scheduler(cfg):
    return build(cfg, SCALAR_SCHEDULERS)


def build_params_manager(cfg):
    return build(cfg, PARAMS_MANAGERS)
