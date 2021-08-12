# Copyright (C) 2020 Intel Corporation
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

import mmcv
from openvino.inference_engine import IECore  # pylint: disable=no-name-in-module

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.utils import ExtendedDictAction
from mmaction.core.utils import propagate_root_dir


def update_config(cfg, args):
    if args.num_workers is not None and args.num_workers > 0:
        cfg.data.workers_per_gpu = args.num_workers

    if cfg.test_cfg is None:
        cfg.test_cfg = dict(average_clips=args.average_clips)
    else:
        cfg.test_cfg.average_clips = args.average_clips

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


def build_class_map(dataset_classes, model_classes):
    model_inv_class_map = {v: k for k, v in model_classes.items()}
    out_class_map = {k: model_inv_class_map[v] for k, v in dataset_classes.items()}

    return out_class_map


def collect_results(model, data_loader):
    results = []
    progress_bar = mmcv.ProgressBar(len(data_loader.dataset))

    for data in data_loader:
        input_data = data['imgs'].cpu().numpy()
        results.extend(model(input_data))

        batch_size = len(input_data)
        for _ in range(batch_size):
            progress_bar.update()

    return results


def load_ie_core(device='CPU', cpu_extension=None):
    ie = IECore()
    if device == 'CPU' and cpu_extension:
        ie.add_extension(cpu_extension, 'CPU')

    return ie


class IEModel:
    def __init__(self, model_path, ie_core, device='CPU', num_requests=1):
        if model_path.endswith((".xml", ".bin")):
            model_path = model_path[:-4]
        self.net = ie_core.read_network(model_path + ".xml", model_path + ".bin")
        assert len(self.net.input_info) == 1, "One input is expected"

        self.exec_net = ie_core.load_network(
            network=self.net, device_name=device, num_requests=num_requests
        )

        self.input_name = next(iter(self.net.input_info))
        if len(self.net.outputs) > 1:
            raise Exception("One output is expected")
        else:
            self.output_name = next(iter(self.net.outputs))

        self.input_size = self.net.input_info[self.input_name].input_data.shape
        self.output_size = self.exec_net.requests[0].output_blobs[self.output_name].buffer.shape
        self.num_requests = num_requests

    def infer(self, data):
        input_data = {self.input_name: data}
        infer_result = self.exec_net.infer(input_data)

        return infer_result[self.output_name]


class ActionRecognizer(IEModel):
    def __init__(self, model_path, ie_core, class_map, device='CPU', num_requests=1):
        super().__init__(model_path, ie_core, device, num_requests)

        self.class_ids = [class_map[k] for k in sorted(class_map.keys())]

    def __call__(self, input_data):
        raw_output = self.infer(input_data)
        result = raw_output[:, self.class_ids]

        return result


def main(args):
    assert args.model.endswith('.xml')

    # load config
    cfg = mmcv.Config.fromfile(args.config)
    if args.update_config is not None:
        cfg.merge_from_dict(args.update_config)
    cfg = update_config(cfg, args)
    cfg = propagate_root_dir(cfg, args.data_dir)

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    # Overwrite eval_config from args.eval
    eval_config = merge_configs(eval_config, dict(metrics=args.eval))
    # Add options from args.option
    eval_config = merge_configs(eval_config, args.options)

    assert eval_config, 'Please specify at eval operation with the argument "--eval"'

    # build the dataset
    dataset = build_dataset(cfg.data, 'test', dict(test_mode=True))
    assert dataset.num_datasets == 1
    if cfg.get('classes'):
        dataset = dataset.filter(cfg.classes)
    print(f'Test datasets:\n{str(dataset)}')

    # build the dataloader
    data_loader = build_dataloader(
        dataset,
        videos_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    # build class mapping between model.classes and dataset.classes
    assert cfg.get('model_classes') is not None
    model_classes = {k: v for k, v in enumerate(cfg.model_classes)}
    class_map = build_class_map(dataset.class_maps[0], model_classes)

    # load model
    ie_core = load_ie_core()
    model = ActionRecognizer(args.model, ie_core, class_map)

    # collect results
    outputs = collect_results(model, data_loader)

    # get metrics
    if eval_config:
        eval_res = dataset.evaluate(outputs, **eval_config)

        print('\nFinal metrics:')
        for name, val in eval_res.items():
            print(f'{name}: {val:.04f}')


def parse_args():
    parser = argparse.ArgumentParser(description='Test model deployed to OpenVINO')
    parser.add_argument('config', help='path to configuration file')
    parser.add_argument('model', help='path to xml file in case of OpenVINO.')
    parser.add_argument('--data_dir', type=str,
                        help='the dir with dataset')
    parser.add_argument('--eval', type=str, nargs='+',
                        help='evaluation metrics, which depends on the dataset, e.g.,'
                             ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument('--options', nargs='+', help='custom options')
    parser.add_argument('--average_clips', choices=['score', 'prob'], default='score',
                        help='average type when averaging test clips')
    parser.add_argument('--num_workers', type=int,
                        help='number of CPU workers per GPU')
    parser.add_argument('--update_config', nargs='+', action=ExtendedDictAction,
                        help='Update configuration file by parameters specified here.')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)
