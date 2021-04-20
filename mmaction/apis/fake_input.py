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
#

import numpy as np
import torch
from mmcv.parallel import collate, scatter

from ..datasets.pipelines import Compose


def get_fake_data(orig_img_shape, stream_sample_frames):
    data = {}
    data['clip_len'] = stream_sample_frames.clip_len
    data['num_clips'] = stream_sample_frames.num_clips
    data['imgs'] = [np.zeros(orig_img_shape, dtype=np.uint8), ] * data['clip_len']
    data['modality'] = 'RGB'

    return data


def get_fake_input(cfg, orig_img_shape=(128, 128, 3), device='cuda'):
    test_pipeline = cfg.data.test.pipeline[2:]
    test_pipeline = Compose(test_pipeline)
    data = get_fake_data(orig_img_shape, stream_sample_frames=cfg.data.test.pipeline[0])
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    return data
