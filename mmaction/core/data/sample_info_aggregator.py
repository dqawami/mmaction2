import torch
import torch.distributed as dist

from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class SampleInfoAggregatorHook(Hook):
    def __init__(self):
        pass

    def after_train_iter(self, runner):
        local_meta = runner.model.module.train_meta
        sync_meta = {
            meta_name: self._sync(meta_data, runner.rank, runner.world_size)
            for meta_name, meta_data in local_meta.items()
        }

        dataset = runner.data_loader.dataset
        dataset.enable_sample_filtering = True
        dataset.update_meta_info(**sync_meta)

    def after_epoch(self, runner):
        dataset = runner.data_loader.dataset
        if dataset.enable_sample_filtering:
            runner.log_buffer.update({'filter_active_samples': dataset.get_filter_active_samples_ratio()})

    @staticmethod
    def _sync(data, rank, world_size):
        if dist.is_available() and dist.is_initialized():
            batch_size = data.size(0)

            shared_shape = [world_size * batch_size] + list(data.shape[1:])
            shared_data = torch.zeros(shared_shape, dtype=data.dtype, device=data.device)

            shared_data[rank*batch_size:(rank + 1)*batch_size] = data
            dist.all_reduce(shared_data, dist.ReduceOp.SUM)

            out_data = shared_data
        else:
            out_data = data

        return out_data.cpu().numpy()
