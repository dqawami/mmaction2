# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import sys
import argparse

import numpy as np
import mmcv

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.utils import ExtendedDictAction
from mmaction.core.utils import propagate_root_dir
from mmcv.runner import set_random_seed


def update_config(cfg, args):
    if args.num_workers is not None and args.num_workers > 0:
        cfg.data.workers_per_gpu = args.num_workers

    cfg.data.test.test_mode = True

    normalize_idx = [i for i, v in enumerate(cfg.data.test.pipeline) if v['type'] == 'Normalize'][0]
    cfg.data.test.pipeline[normalize_idx]['mean'] = [0.0, 0.0, 0.0]
    cfg.data.test.pipeline[normalize_idx]['std'] = [1.0, 1.0, 1.0]
    cfg.data.test.pipeline[normalize_idx]['to_bgr'] = False

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


def collect_stat(data_loader):
    mean_data, std_data = [], []
    progress_bar = mmcv.ProgressBar(len(data_loader.dataset))

    for data in data_loader:
        input_data = data['imgs'].detach().squeeze().cpu().numpy()

        mean_data.append(np.mean(input_data, axis=(2, 3, 4)))
        std_data.append(np.std(input_data, axis=(2, 3, 4)))

        batch_size = len(input_data)
        for _ in range(batch_size):
            progress_bar.update()

    mean_data = np.concatenate(mean_data, axis=0)
    std_data = np.concatenate(std_data, axis=0)

    return mean_data, std_data


def filter_stat(mean_data, std_data, min_value=1.0):
    mask = np.all(mean_data > min_value, axis=1) & np.all(std_data > min_value, axis=1)
    return mean_data[mask], std_data[mask]


def dump_stat(mean_data, std_data, out_filepath):
    assert mean_data.shape == std_data.shape

    with open(out_filepath, 'w') as output_stream:
        for mean_value, std_value in zip(mean_data, std_data):
            mean_value_str = ','.join(str(v) for v in mean_value)
            std_value_str = ','.join(str(v) for v in std_value)

            output_stream.write(f'{mean_value_str} {std_value_str}\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Test model deployed to ONNX or OpenVINO')
    parser.add_argument('config', help='path to configuration file')
    parser.add_argument('out', help='path to save stat')
    parser.add_argument('--data_dir', type=str,
                        help='the dir with dataset')
    parser.add_argument('--num_workers', type=int,
                        help='number of CPU workers per GPU')
    parser.add_argument('--update_config', nargs='+', action=ExtendedDictAction,
                        help='Update configuration file by parameters specified here.')
    args = parser.parse_args()

    return args


def main(args):
    # load config
    cfg = mmcv.Config.fromfile(args.config)
    if args.update_config is not None:
        cfg.merge_from_dict(args.update_config)
    cfg = update_config(cfg, args)
    cfg = propagate_root_dir(cfg, args.data_dir)

    if cfg.get('seed'):
        print(f'Set random seed to {cfg.seed}')
        set_random_seed(cfg.seed)

    # build the dataset
    dataset = build_dataset(cfg.data, 'test', dict(test_mode=True))
    print(f'Test datasets:\n{str(dataset)}')

    # build the dataloader
    data_loader = build_dataloader(
        dataset,
        videos_per_gpu=20,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    # collect results
    mean_data, std_data = collect_stat(data_loader)

    # filter data
    mean_data, std_data = filter_stat(mean_data, std_data, min_value=1.0)

    # dump stat
    dump_stat(mean_data, std_data, args.out)


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)