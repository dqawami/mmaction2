import json
import math
import argparse
from os import makedirs, listdir
from os.path import exists, join, isfile, isdir
from collections import OrderedDict, defaultdict
from operator import itemgetter

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


TRG_KPT_IDS = [4, 7]
NET_STRIDE = 8
NET_UPSAMPLE_RATIO = 4


def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                  dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_dw_no_bn(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                  dilation=dilation, groups=in_channels, bias=False),
        nn.ELU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.ELU(inplace=True),
    )


class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        self.model = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5
        )
        self.cpm = Cpm(512, num_channels)

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

    def forward(self, x):
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

        return stages_output


def create_network(weights, use_cuda):
    net = PoseEstimationWithMobileNet()

    checkpoint = torch.load(weights, map_location='cpu')
    load_state(net, checkpoint)

    if use_cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = False

    net.eval()

    return net


def load_state(net, checkpoint):
    source_state = checkpoint['state_dict']
    target_state = net.state_dict()
    new_target_state = OrderedDict()
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

    net.load_state_dict(new_target_state)


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride

    pad = list()
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)

    return padded_img, pad


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, use_cuda,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if use_cuda:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def extract_keypoints(heatmap, min_conf, threshold):
    heatmap[heatmap < min_conf] = 0.0
    heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode='constant')

    heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 1:heatmap_with_borders.shape[1]-1]
    heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 2:heatmap_with_borders.shape[1]]
    heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 0:heatmap_with_borders.shape[1]-2]
    heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1]-1]
    heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0]-2, 1:heatmap_with_borders.shape[1]-1]

    heatmap_peaks = (heatmap_center > heatmap_left) &\
                    (heatmap_center > heatmap_right) &\
                    (heatmap_center > heatmap_up) &\
                    (heatmap_center > heatmap_down)
    heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0]-1, 1:heatmap_center.shape[1]-1]

    keypoints = list(zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]))
    keypoints = sorted(keypoints, key=itemgetter(0))

    out_keypoints = []
    suppressed = np.zeros(len(keypoints), np.bool)
    for i in range(len(keypoints)):
        if suppressed[i]:
            continue

        for j in range(i+1, len(keypoints)):
            suppressed[j] = math.sqrt((keypoints[i][0] - keypoints[j][0]) ** 2 +
                                      (keypoints[i][1] - keypoints[j][1]) ** 2) < threshold

        out_keypoints.append((keypoints[i][0], keypoints[i][1], heatmap[keypoints[i][1], keypoints[i][0]]))

    return out_keypoints


def scale_keypoints(kpts, stride, upsample_ratio, pad, scale):
    converted_records = []
    for kpt in kpts:
        record = (kpt[0] * stride / upsample_ratio - pad[1]) / scale, \
                 (kpt[1] * stride / upsample_ratio - pad[0]) / scale, \
                 kpt[2]
        converted_records.append(record)

    return converted_records


class Track:
    def __init__(self, frame_id=-1, kpt=None):
        self.frame_ids = []
        self.kpts = []

        if frame_id >= 0 and kpt is not None:
            self.add(frame_id, kpt)

    def add(self, frame_id, kpt):
        self.frame_ids.append(frame_id)
        self.kpts.append(kpt)

    @property
    def last_kpt(self):
        assert len(self.kpts) > 0
        return self.kpts[-1]

    @property
    def length(self):
        return len(self.frame_ids)

    @property
    def data(self):
        return {frame_id: kpt for frame_id, kpt in zip(self.frame_ids, self.kpts)}


def select_single_person_track(kpts):
    active_track = None
    for frame_id, candidates in kpts.items():
        if len(candidates) == 0:
            continue

        candidate_kpts = [(kpt[0], kpt[1]) for kpt in candidates]
        candidate_scores = [kpt[2] for kpt in candidates]

        if active_track is None:
            active_track = Track(frame_id, candidate_kpts[np.argmax(candidate_scores)])
            continue

        previous = np.array([active_track.last_kpt], dtype=np.float32).reshape([-1, 2])
        current = np.array(candidate_kpts, dtype=np.float32).reshape([-1, 2])

        distance_matrix = (previous[:, 0].reshape([-1, 1]) - current[:, 0].reshape([1, -1])) ** 2 + \
                          (previous[:, 1].reshape([-1, 1]) - current[:, 1].reshape([1, -1])) ** 2
        distance_matrix = np.sqrt(distance_matrix)

        match_ind = np.argmin(distance_matrix, axis=1)[0]
        active_track.add(frame_id, candidate_kpts[match_ind])

    return active_track.data if active_track is not None else None


def ensure_dir_exists(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)


def parse_images(root_dir, extension):
    base_names = [d for d in listdir(root_dir) if isdir(join(root_dir, d))]

    out_paths = dict()
    total_num_images = 0
    for base_name in tqdm(base_names, desc='Parsing images'):
        frames_dir = join(root_dir, base_name)
        frames = {int(f.split('.')[0]): join(frames_dir, f) for f in listdir(frames_dir)
                  if isfile(join(frames_dir, f)) and f.endswith(extension)}

        frame_ids = list(frames.keys())
        start_frame_id, end_frame_id = np.min(frame_ids), np.max(frame_ids) + 1
        assert len(frame_ids) == end_frame_id - start_frame_id

        out_paths[base_name] = frames
        total_num_images += len(frames)

    return out_paths, total_num_images


def main():
    def _str2bool(v):
        return v.lower() in ("yes", "y", "true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--height_size', type=int, default=256, required=False)
    parser.add_argument('--cuda', default=True, type=_str2bool, required=False)
    parser.add_argument('--override', action='store_true')
    args = parser.parse_args()

    assert exists(args.input_dir)
    assert exists(args.model)
    ensure_dir_exists(args.output_dir)

    net = create_network(args.model, args.cuda)

    all_paths, total_num_images = parse_images(args.input_dir, 'jpg')
    print('Found {} images.'.format(total_num_images))

    num_valid_videos = 0
    pbar = tqdm(total=total_num_images, desc='Processing')
    for video_name, frames in all_paths.items():
        frame_ids = list(frames)
        frame_ids.sort()

        out_data_path = join(args.output_dir, '{}.json'.format(video_name))
        if not args.override and exists(out_data_path):
            pbar.update(len(frame_ids))
            continue

        video_kpts = defaultdict(dict)
        for frame_id in frame_ids:
            frame_path = frames[frame_id]
            frame = cv2.imread(frame_path)

            heatmaps, _, scale, pad = infer_fast(
                net, frame, args.height_size, NET_STRIDE, NET_UPSAMPLE_RATIO, args.cuda
            )

            for kpt_idx in TRG_KPT_IDS:
                extracted_kpts = extract_keypoints(heatmaps[:, :, kpt_idx], min_conf=0.1, threshold=6)
                scaled_kpts = scale_keypoints(extracted_kpts, NET_STRIDE, NET_UPSAMPLE_RATIO, pad, scale)
                video_kpts[kpt_idx][frame_id] = scaled_kpts

            pbar.update(1)

        out_tracks = dict()
        for kpt_idx in video_kpts.keys():
            track = select_single_person_track(video_kpts[kpt_idx])
            if track is not None:
                out_tracks[kpt_idx] = track

        if len(out_tracks) > 0:
            num_valid_videos += 1

        with open(out_data_path, 'w') as output_stream:
            json.dump(out_tracks, output_stream)

    pbar.close()

    print('Finished: {} / {} valid videos.'.format(num_valid_videos, len(all_paths)))


if __name__ == '__main__':
    main()
