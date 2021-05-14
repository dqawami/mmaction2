from mmcv.runner.hooks import Hook
import torch

from ..registry import PARAMS_MANAGERS


@PARAMS_MANAGERS.register_module()
class FreezeLayers(Hook):
    def __init__(self, epochs=0, open_layers=None, **kwargs):
        super(FreezeLayers, self).__init__(**kwargs)

        self.epochs = epochs
        self.open_layers = open_layers
        if isinstance(self.open_layers, str):
            self.open_layers = [self.open_layers]

        self.enable = self.epochs > 0 and self.open_layers is not None and len(self.open_layers) > 0

    def before_train_epoch(self, runner):
        cur_epoch = runner.epoch

        model = runner.model.module
        if self.enable and cur_epoch < self.epochs:
            runner.logger.info('* Only train {} (epoch: {}/{})'.format(self.open_layers, cur_epoch + 1, self.epochs))
            self.open_specified_layers(model, self.open_layers)
        else:
            self.open_all_layers(model)

    @staticmethod
    def open_all_layers(model):
        model.train()
        for p in model.parameters():
            if p.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
                # only Tensors of floating point dtype can require gradients
                continue
            p.requires_grad = True

    @staticmethod
    def open_specified_layers(model, open_layers):
        for name, module in model.named_modules():
            if any([open_substring in name for open_substring in open_layers]):
                module.train()
                for p in module.parameters():

                    p.requires_grad = True
            else:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False
