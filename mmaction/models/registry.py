from mmcv.utils import Registry

BACKBONES = Registry('backbone')
HEADS = Registry('head')
RECOGNIZERS = Registry('recognizer')
LOSSES = Registry('loss')
LOCALIZERS = Registry('localizer')
SCALAR_SCHEDULERS = Registry('scalar_scheduler')
PARAMS_MANAGERS = Registry('params_manager')
SPATIAL_TEMPORAL_MODULES = Registry('spatial_temporal_module')
NECKS = Registry('necks')
