import sys
import argparse
import json
from os import makedirs
from os.path import exists, dirname, basename, splitext, join
from subprocess import run, CalledProcessError, DEVNULL

import torch
import onnx
import mmcv
from mmcv.runner import set_random_seed

from mmaction.apis import get_fake_input, init_recognizer
from mmaction.integration.nncf import (check_nncf_is_enabled,
                                       get_nncf_config_from_meta,
                                       is_checkpoint_nncf,
                                       wrap_nncf_model)

from mmaction.models import build_recognizer
from mmaction.core import load_checkpoint
from mmaction.utils import ExtendedDictAction


def convert_to_onnx(net, input_size, output_file_path, opset, check=True):
    dummy_input = torch.randn((1, *input_size))
    input_names = ['input']
    output_names = ['output']

    dynamic_axes = {'input': {0: 'batch_size', 1: 'channels', 2: 'length', 3: 'height', 4: 'width'},
                    'output': {0: 'batch_size', 1: 'scores'}}

    net = net.cpu()

    with torch.no_grad():
        torch.onnx.export(
            net,
            dummy_input,
            output_file_path,
            verbose=False,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )

    net_from_onnx = onnx.load(output_file_path)
    if check:
        try:
            onnx.checker.check_model(net_from_onnx)
            print('ONNX check passed.')
        except onnx.onnx_cpp2py_export.checker.ValidationError as ex:
            print('ONNX check failed: {}.'.format(ex))


def export_to_openvino(cfg, onnx_model_path, output_dir_path, input_shape=None, input_format='rgb'):
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    onnx_model = onnx.load(onnx_model_path)
    output_names = set(out.name for out in onnx_model.graph.output)
    # Clear names of the nodes that produce network's output blobs.
    for node in onnx_model.graph.node:
        if output_names.intersection(node.output):
            node.ClearField('name')
    onnx.save(onnx_model, onnx_model_path)
    output_names = ','.join(output_names)

    normalize = [v for v in cfg.data.test.pipeline if v['type'] == 'Normalize'][0]

    mean_values = normalize['mean']
    scale_values = normalize['std']
    command_line = f'mo.py --input_model="{onnx_model_path}" ' \
                   f'--mean_values="{mean_values}" ' \
                   f'--scale_values="{scale_values}" ' \
                   f'--output_dir="{output_dir_path}" ' \
                   f'--output="{output_names}"'

    assert input_format.lower() in ['bgr', 'rgb']

    if input_shape is not None:
        command_line += f' --input_shape="{input_shape}"'
    if not normalize['to_bgr'] and input_format.lower() == 'bgr' or \
       normalize['to_bgr'] and input_format.lower() == 'rgb':
        command_line += ' --reverse_input_channels'

    try:
        run('mo.py -h', stdout=DEVNULL, stderr=DEVNULL, shell=True, check=True)
    except CalledProcessError:
        print('OpenVINO Model Optimizer not found, please source '
              'openvino/bin/setupvars.sh before running this script.')
        return

    print(command_line)
    run(command_line, shell=True, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Export model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help="path to file with model's weights")
    parser.add_argument('output_dir', help='path to directory to save exported models in')
    parser.add_argument('meta_info', help='path to file to save meta info in')
    parser.add_argument('--opset', type=int, default=10, help='ONNX opset')
    parser.add_argument('--update_config', nargs='+', action=ExtendedDictAction,
                        help='Update configuration file by parameters specified here.')

    subparsers = parser.add_subparsers(title='target', dest='target', help='target model format')
    subparsers.required = True
    parser_onnx = subparsers.add_parser('onnx', help='export to ONNX')
    parser_openvino = subparsers.add_parser('openvino', help='export to OpenVINO')
    parser_openvino.add_argument('--input_format', choices=['BGR', 'RGB'], default='RGB',
                                 help='Input image format for exported model.')

    return parser.parse_args()


def main(args):
    cfg = mmcv.Config.fromfile(args.config)
    if args.update_config is not None:
        cfg.merge_from_dict(args.update_config)
    cfg.data.videos_per_gpu = 1

    if cfg.get('seed'):
        print(f'Set random seed to {cfg.seed}')
        set_random_seed(cfg.seed)

    class_maps = None
    if cfg.get('classes'):
        class_maps = {0: {k: v for k, v in enumerate(sorted(cfg.classes))}}

    model = build_recognizer(
        cfg.model,
        train_cfg=None,
        test_cfg=cfg.test_cfg,
        class_maps=class_maps
    )

    model.eval()

    load_checkpoint(model, args.checkpoint, force_matching=True)
    if hasattr(model, 'forward_inference'):
        model.forward = model.forward_inference

    input_time_size = cfg.input_clip_length
    input_image_size = (tuple(cfg.input_img_size)
                        if isinstance(cfg.input_img_size, (list, tuple))
                        else (cfg.input_img_size, cfg.input_img_size))
    input_size = (3, input_time_size) + input_image_size

    # BEGIN nncf part
    was_model_compressed = is_checkpoint_nncf(args.checkpoint)
    cfg_contains_nncf = cfg.get('nncf_config')

    if cfg_contains_nncf and not was_model_compressed:
        raise RuntimeError('Trying to make export with NNCF compression '
                           'a model snapshot that was NOT trained with NNCF')

    if was_model_compressed and not cfg_contains_nncf:
        # reading NNCF config from checkpoint
        nncf_part = get_nncf_config_from_meta(args.checkpoint)
        for k, v, in nncf_part.items():
            cfg[k] = v

    if cfg.get('nncf_config'):
        if torch.cuda.is_available():
            model.cuda()
        alt_ssd_export = getattr(args, 'alt_ssd_export', False)
        assert not alt_ssd_export, \
                'Export of NNCF-compressed model is incompatible with --alt_ssd_export'
        check_nncf_is_enabled()
        cfg.load_from = args.checkpoint
        cfg.resume_from = None
        compression_ctrl, model = wrap_nncf_model(model, cfg, None, get_fake_input, export=True)
        compression_ctrl.prepare_for_export()
    # END nncf part

    onnx_model_path = join(args.output_dir, splitext(basename(args.config))[0] + '.onnx')
    base_output_dir = dirname(onnx_model_path)
    if not exists(base_output_dir):
        makedirs(base_output_dir)

    convert_to_onnx(model, input_size, onnx_model_path, opset=args.opset, check=True)

    if args.target == 'openvino':
        input_shape = (1,) + input_size
        export_to_openvino(cfg, onnx_model_path, args.output_dir, input_shape, args.input_format)

    meta = {'model_classes': model.CLASSES[0]}
    with open(args.meta_info, 'w') as output_meta_stream:
        json.dump(meta, output_meta_stream)


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)
