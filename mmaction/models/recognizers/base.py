import copy
import re
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16

from .. import builder
from ...core.ops import rsc, NormRegularizer, balance_losses
from ...integration.nncf import is_in_nncf_tracing, no_nncf_trace


class EvalModeSetter:
    def __init__(self, module, m_type):
        self.module = module
        self.modes_storage = dict()

        self.m_types = m_type
        if not isinstance(self.m_types, (tuple, list)):
            self.m_types = [self.m_types]

    def __enter__(self):
        for name, module in self.module.named_modules():
            matched = any(isinstance(module, m_type) for m_type in self.m_types)
            if matched:
                self.modes_storage[name] = module.training
                module.train(mode=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.module.named_modules():
            if name in self.modes_storage:
                module.train(mode=self.modes_storage[name])


class BaseRecognizer(nn.Module, metaclass=ABCMeta):
    """Base class for recognizers.

    All recognizers should subclass it.
    All subclass should overwrite:

    - Methods:``reshape_images``, supporting the input reshape.

    Args:
        backbone (dict): Backbone modules to extract feature.
        reducer (dict): Spatial-temporal modules to reduce feature. Default: None.
        cls_head (dict): Classification head to process feature.
        class_sizes (list): Number of samples for each class in each task. Default: None.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        bn_eval (bool): Whether to switch all BN in eval mode. Default: False.
        bn_frozen (bool): Whether to disable backprop for all BN. Default: False.
    """

    meta_fields = ['pred_labels', 'pred_conf', 'sample_idx', 'clip_starts', 'clip_ends', 'total_frames']

    def __init__(self,
                 backbone,
                 cls_head,
                 reducer=None,
                 neck=None,
                 class_sizes=None,
                 class_maps=None,
                 train_cfg=None,
                 test_cfg=None,
                 bn_eval=False,
                 bn_frozen=False,
                 reg_cfg=None):
        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.fp16_enabled = False
        self.multi_head = class_sizes is not None and len(class_sizes) > 1
        self.with_self_challenging = hasattr(train_cfg, 'self_challenging') and train_cfg.self_challenging.enable
        self.with_clip_mixing = hasattr(train_cfg, 'clip_mixing') and train_cfg.clip_mixing.enable
        self.with_loss_norm = hasattr(train_cfg, 'loss_norm') and train_cfg.loss_norm.enable
        self.with_sample_filtering = hasattr(train_cfg, 'sample_filtering') and train_cfg.sample_filtering.enable

        if class_maps is None:
            num_classes = cls_head.num_classes
            self.CLASSES = {0: {ii: ii for ii in range(num_classes)}}
        else:
            self.CLASSES = copy.deepcopy(class_maps)

        self.train_meta = {}

        self.backbone = builder.build_backbone(backbone)
        self.spatial_temporal_module = builder.build_reducer(reducer)
        self.cls_head = builder.build_head(cls_head, class_sizes)
        self.neck = builder.build_neck(neck)

        self.num_clips = 1
        if self.with_clip_mixing:
            self.clip_mixing_loss = builder.build_loss(dict(
                type='ClipMixingLoss',
                num_clips=train_cfg.clip_mixing.num_clips,
                mode=train_cfg.clip_mixing.get('mode', 'logits'),
                reweight_targets=train_cfg.clip_mixing.get('reweight_targets', False),
                loss_weight=train_cfg.clip_mixing.get('weight', 1.0)
            ))
            self.num_clips = train_cfg.clip_mixing.num_clips

        self.losses_meta = None
        if self.with_loss_norm:
            assert 0.0 < train_cfg.loss_norm.gamma <= 1.0

            self.losses_meta = dict()

        self.regularizer = NormRegularizer(**reg_cfg) if reg_cfg is not None else None

        self.init_weights()

    def init_weights(self):
        for module in self.children():
            if hasattr(module, 'init_weights'):
                module.init_weights()

        heads = self.cls_head if self.multi_head else [self.cls_head]
        for head in heads:
            if hasattr(head, 'init_weights'):
                head.init_weights()

    def update_state(self, *args, **kwargs):
        for module in self.children():
            if hasattr(module, 'update_state'):
                module.update_state(*args, **kwargs)

        heads = self.cls_head if self.multi_head else [self.cls_head]
        for head in heads:
            if hasattr(head, 'update_state'):
                head.update_state(*args, **kwargs)

    @auto_fp16()
    def _forward_module_train(self, module, x, losses, squeeze=False, squeeze_dim=-1, **kwargs):
        if module is None:
            out = x
        elif hasattr(module, 'loss'):
            out, extra_data = module(x, return_extra_data=True)
            losses.update(module.loss(**extra_data, **kwargs))
        else:
            out = module(x)

        if squeeze and isinstance(out, (list, tuple)):
            assert len(out) > 0
            out = out[squeeze_dim]

        return out

    @auto_fp16()
    def _extract_features_test(self, imgs):
        """Extract features through a backbone.

        Args:
            imgs (torch.Tensor): The input images.

        Returns:
            torch.tensor: The extracted features.
        """

        y = self.backbone(imgs)

        if isinstance(y, (list, tuple)):
            assert len(y) > 0
            y = y[-1]

        if self.spatial_temporal_module is not None:
            y = self.spatial_temporal_module(y)

        return y

    def _average_clip(self, cls_score):
        """Averaging class score over multiple clips.

        Using different averaging types ('score' or 'prob' or None,
        which defined in test_cfg) to computed the final averaged
        class score.

        Args:
            cls_score (torch.Tensor): Class score to be averaged.

        return:
            torch.Tensor: Averaged class score.
        """
        if 'average_clips' not in self.test_cfg.keys():
            raise KeyError('"average_clips" must defined in test_cfg\'s keys')

        average_clips = self.test_cfg['average_clips']
        if average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        if average_clips == 'prob':
            cls_score = F.softmax(cls_score, dim=1).mean(dim=0, keepdim=True)
        elif average_clips == 'score':
            cls_score = cls_score.mean(dim=0, keepdim=True)

        return cls_score

    @abstractmethod
    def reshape_input(self, imgs, masks=None):
        pass

    @abstractmethod
    def reshape_input_inference(self, imgs, masks=None):
        pass

    @staticmethod
    def _infer_head(head_module, *args, **kwargs):
        out = head_module(*args, **kwargs)

        if isinstance(out, (tuple, list)):
            assert len(out) == 3
            return out
        else:
            return out, None, None

    @staticmethod
    def _filter(x, mask):
        if x is None:
            return None
        elif mask is None:
            return x
        elif isinstance(x, (tuple, list)):
            return [_x[mask] for _x in x]
        else:
            return x[mask]

    def forward_train(self, imgs, labels, dataset_id=None, attention_mask=None, **kwargs):
        imgs, attention_mask, head_args = self.reshape_input(imgs, attention_mask)
        losses = dict()

        features = self._forward_module_train(
            self.backbone, imgs, losses, attention_mask=attention_mask
        )
        features = self._forward_module_train(
            self.neck, features, losses, squeeze=True, squeeze_dim=-1,
            labels=labels, dataset_id=dataset_id, num_clips=self.num_clips
        )
        features = self._forward_module_train(
            self.spatial_temporal_module, features, losses
        )

        enable_rsc = not is_in_nncf_tracing()

        with no_nncf_trace():

            if self.with_self_challenging and not features.requires_grad:
                features.requires_grad = True

            if self.with_sample_filtering:
                pred_labels = torch.zeros_like(labels.view(-1))
                pred_conf = torch.zeros_like(labels.view(-1), dtype=features.dtype)

            heads = self.cls_head if self.multi_head else [self.cls_head]
            for head_id, cl_head in enumerate(heads):
                trg_mask = (dataset_id == head_id).view(-1) if dataset_id is not None else None

                trg_labels = self._filter(labels.view(-1), trg_mask)
                trg_num_samples = trg_labels.numel()
                if trg_num_samples == 0:
                    continue

                if self.with_self_challenging:
                    trg_features = self._filter(features, trg_mask)
                    trg_main_scores, _, _ = self._infer_head(
                        cl_head,
                        *([trg_features] + head_args),
                        labels=trg_labels.view(-1)
                    )

                    # TODO: Find permanent solution that will replace RSC for NNCF
                    if enable_rsc:
                        trg_features = rsc(
                            trg_features,
                            trg_main_scores,
                            trg_labels, 1.0 - self.train_cfg.self_challenging.drop_p
                        )

                    with EvalModeSetter(cl_head, m_type=(nn.BatchNorm2d, nn.BatchNorm3d)):
                        trg_main_scores, trg_norm_embd, trg_extra_scores = self._infer_head(
                            cl_head,
                            *([trg_features] + head_args),
                            labels=trg_labels.view(-1),
                            return_extra_data=True
                        )
                else:
                    all_main_scores, all_norm_embd, all_extra_scores = self._infer_head(
                        cl_head,
                        *([features] + head_args),
                        labels=labels.view(-1),
                        return_extra_data=True
                    )

                    trg_main_scores = self._filter(all_main_scores, trg_mask)
                    trg_extra_scores = self._filter(all_extra_scores, trg_mask)
                    trg_norm_embd = self._filter(all_norm_embd, trg_mask)

                # main head loss
                losses.update(cl_head.loss(
                    main_cls_score=trg_main_scores,
                    extra_cls_score=trg_extra_scores,
                    labels=trg_labels.view(-1),
                    norm_embd=trg_norm_embd,
                    name=str(head_id)
                ))

                # clip mixing loss
                if self.with_clip_mixing:
                    clip_mixing_scale = self.train_cfg.clip_mixing.get('scale', cl_head.last_scale)
                    losses['loss/clip_mix' + str(head_id)] = self.clip_mixing_loss(
                        trg_main_scores, trg_labels.view(-1), trg_norm_embd, clip_mixing_scale
                    )

                if self.with_sample_filtering:
                    with torch.no_grad():
                        pred_conf[trg_mask], pred_labels[trg_mask] = torch.max(trg_main_scores, dim=1)

            if self.regularizer is not None:
                losses['loss/reg'] = self.regularizer(self.backbone)

            if self.with_sample_filtering:
                self._add_train_meta_info(pred_labels=pred_labels, pred_conf=pred_conf, **kwargs)

        return losses

    def _add_train_meta_info(self, **kwargs):
        for meta_name in self.meta_fields:
            assert meta_name in kwargs.keys(), f'There is no {meta_name} in meta info'
            assert kwargs[meta_name] is not None, f'The value of {meta_name} is None'

            self.train_meta[meta_name] = kwargs[meta_name].detach().clone().view(-1)

    def forward_test(self, imgs, dataset_id=None):
        """Defines the computation performed at every call when evaluation and
        testing."""

        imgs, _, head_args = self.reshape_input(imgs)

        y = self._extract_features_test(imgs)

        with no_nncf_trace():

            if self.multi_head:
                assert dataset_id is not None

                head_outs = []
                for cls_head in self.cls_head:
                    head_y = cls_head(y, *head_args)
                    head_out = self._average_clip(head_y)
                    head_outs.append(head_out.cpu().numpy())

                out = []
                dataset_id = dataset_id.view(-1).cpu().numpy()
                for idx, head_id in enumerate(dataset_id):
                    out.extend(head_outs[head_id][idx].reshape([1, -1]))
            else:
                y = self.cls_head(y, *head_args)
                out = self._average_clip(y).cpu().numpy()

        return out

    def forward_inference(self, imgs):
        """Used for computing network FLOPs and ONNX export.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """

        if self.multi_head:
            raise NotImplementedError('Inference does not support multi-head architectures')

        imgs, _, head_args = self.reshape_input_inference(imgs)
        y = self._extract_features_test(imgs)
        with no_nncf_trace():
            out = self.cls_head(y, *head_args)

        return out

    def forward(self, imgs, label=None, return_loss=True, dataset_id=None, **kwargs):
        """Define the computation performed at every call."""

        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')

            return self.forward_train(imgs, label, dataset_id, **kwargs)
        else:
            return self.forward_test(imgs, dataset_id)

    @staticmethod
    def _parse_losses(losses, multi_head, enable_loss_norm=False, losses_meta=None, gamma=0.9):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, float):
                log_vars[loss_name] = loss_value
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        if enable_loss_norm and losses_meta is not None:
            loss_groups = defaultdict(list)
            single_losses = []
            for _key, _value in log_vars.items():
                if 'loss' not in _key:
                    continue

                end_digits_match = re.search(r'\d+$', _key)
                if end_digits_match is None:
                    single_losses.append(_value)
                else:
                    end_digits = end_digits_match.group()
                    loss_group_name = _key[:-len(end_digits)]
                    loss_groups[loss_group_name].append((_key, _value))

            group_losses = []
            for loss_group in loss_groups.values():
                group_losses.extend(balance_losses(loss_group, losses_meta, gamma))

            loss = sum(single_losses + group_losses)
        else:
            loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            if not isinstance(loss_value, torch.Tensor):
                continue

            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                if not multi_head or loss_name == 'loss':
                    loss_value = loss_value.clone().detach()
                    dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(
            losses,
            self.multi_head,
            self.with_loss_norm,
            self.losses_meta,
            self.train_cfg.loss_norm.gamma if self.with_loss_norm else 0.9
        )

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """

        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(
            losses,
            self.multi_head,
            self.with_loss_norm,
            self.losses_meta,
            self.train_cfg.loss_norm.gamma if self.with_loss_norm else 0.9
        )

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def train(self, train_mode=True):
        super(BaseRecognizer, self).train(train_mode)

        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()

                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False

        return self
