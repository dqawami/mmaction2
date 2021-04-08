import random
import os.path as osp
from collections.abc import Sequence

import mmcv
import numpy as np
from torch.nn.modules.utils import _pair

from ..registry import PIPELINES


def _init_lazy_if_proper(results, lazy):
    """Initialize lazy operation properly.

    Make sure that a lazy operation is properly initialized,
    and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    if 'img_shape' not in results:
        results['img_shape'] = results['imgs'][0].shape[:2]
    if lazy:
        if 'lazy' not in results:
            img_h, img_w = results['img_shape']
            lazyop = dict()
            lazyop['original_shape'] = results['img_shape']
            lazyop['crop_bbox'] = np.array([0, 0, img_w, img_h],
                                           dtype=np.float32)
            lazyop['flip'] = False
            lazyop['flip_direction'] = None
            lazyop['interpolation'] = None
            results['lazy'] = lazyop
    else:
        assert 'lazy' not in results, 'Use Fuse after lazy operations'


@PIPELINES.register_module()
class Fuse(object):
    """Fuse lazy operations.

    Fusion order:
        crop -> resize -> flip

    Required keys are "imgs", "img_shape" and "lazy", added or modified keys
    are "imgs", "lazy".
    Required keys in "lazy" are "crop_bbox", "interpolation", "flip_direction".
    """

    def __call__(self, results):
        if 'lazy' not in results:
            raise ValueError('No lazy operation detected')
        lazyop = results['lazy']
        imgs = results['imgs']

        # crop
        left, top, right, bottom = lazyop['crop_bbox'].round().astype(int)
        imgs = [img[top:bottom, left:right] for img in imgs]

        # resize
        img_h, img_w = results['img_shape']
        if lazyop['interpolation'] is None:
            interpolation = 'bilinear'
        else:
            interpolation = lazyop['interpolation']
        imgs = [
            mmcv.imresize(img, (img_w, img_h), interpolation=interpolation)
            for img in imgs
        ]

        # flip
        if lazyop['flip']:
            for img in imgs:
                mmcv.imflip_(img, lazyop['flip_direction'])

        results['imgs'] = imgs
        del results['lazy']

        return results


@PIPELINES.register_module()
class RandomCrop(object):
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "imgs" and "img_shape", added or
    modified keys are "imgs", "lazy"; Required keys in "lazy" are "flip",
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        size (int): The output size of the images.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, size, lazy=False):
        if not isinstance(size, int):
            raise TypeError(f'Size must be an int, but got {type(size)}')
        self.size = size
        self.lazy = lazy

    def __call__(self, results):
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        img_h, img_w = results['img_shape']
        assert self.size <= img_h and self.size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > self.size:
            y_offset = int(np.random.randint(0, img_h - self.size))
        if img_w > self.size:
            x_offset = int(np.random.randint(0, img_w - self.size))

        new_h, new_w = self.size, self.size

        results['crop_bbox'] = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            results['imgs'] = [
                img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
                for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(size={self.size}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class RandomResizedCrop(object):
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "imgs", "img_shape", "crop_bbox" and "lazy",
    added or modified keys are "imgs", "crop_bbox" and "lazy"; Required keys
    in "lazy" are "flip", "crop_bbox", added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 lazy=False):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.lazy = lazy
        if not mmcv.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmcv.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(np.random.uniform(np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area

        candidate_crop_w = np.round(np.sqrt(target_areas * aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas / aspect_ratios)).astype(np.int32)

        valid_crop_bbox = None
        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)

                valid_crop_bbox = x_offset, y_offset, x_offset + crop_w, y_offset + crop_h
                break

        # Fallback
        if valid_crop_bbox is None:
            crop_size = min(img_h, img_w)
            x_offset = (img_w - crop_size) // 2
            y_offset = (img_h - crop_size) // 2
            valid_crop_bbox = x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

        return valid_crop_bbox

    def __call__(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        img_h, img_w = results['img_shape']

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        results['crop_bbox'] = np.array([left, top, right, bottom])
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            results['imgs'] = [
                img[top:bottom, left:right] for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class MultiScaleCrop(object):
    """Crop images with a list of randomly selected scales.

    Randomly select the w and h scales from a list of scales. Scale of 1 means
    the base size, which is the minimal of image weight and height. The scale
    level of w and h is controlled to be smaller than a certain value to
    prevent too large or small aspect ratio.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox", "img_shape", "lazy" and "scales". Required keys in "lazy" are
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        input_size (int | tuple[int]): (w, h) of network input.
        scales (tuple[float]): Weight and height scales to be selected.
        max_wh_scale_gap (int): Maximum gap of w and h scale levels.
            Default: 1.
        random_crop (bool): If set to True, the cropping bbox will be randomly
            sampled, otherwise it will be sampler from fixed regions.
            Default: False.
        num_fixed_crops (int): If set to 5, the cropping bbox will keep 5
            basic fixed regions: "upper left", "upper right", "lower left",
            "lower right", "center".If set to 13, the cropping bbox will append
            another 8 fix regions: "center left", "center right",
            "lower center", "upper center", "upper left quarter",
            "upper right quarter", "lower left quarter", "lower right quarter".
            Default: 5.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 input_size,
                 scales=(1, ),
                 max_wh_scale_gap=1,
                 random_crop=False,
                 num_fixed_crops=5,
                 lazy=False):
        self.input_size = _pair(input_size)
        if not mmcv.is_tuple_of(self.input_size, int):
            raise TypeError(f'Input_size must be int or tuple of int, '
                            f'but got {type(input_size)}')

        if not isinstance(scales, tuple):
            raise TypeError(f'Scales must be tuple, but got {type(scales)}')

        if num_fixed_crops not in [5, 13]:
            raise ValueError(f'Num_fix_crops must be in {[5, 13]}, '
                             f'but got {num_fixed_crops}')

        self.scales = scales
        self.max_wh_scale_gap = max_wh_scale_gap
        self.random_crop = random_crop
        self.num_fixed_crops = num_fixed_crops
        self.lazy = lazy

    def __call__(self, results):
        """Performs the MultiScaleCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        img_h, img_w = results['img_shape']
        base_size = min(img_h, img_w)
        crop_sizes = [int(base_size * s) for s in self.scales]

        candidate_sizes = []
        for i, h in enumerate(crop_sizes):
            for j, w in enumerate(crop_sizes):
                if abs(i - j) <= self.max_wh_scale_gap:
                    candidate_sizes.append([w, h])

        crop_size = random.choice(candidate_sizes)
        for i in range(2):
            if abs(crop_size[i] - self.input_size[i]) < 3:
                crop_size[i] = self.input_size[i]

        crop_w, crop_h = crop_size

        if self.random_crop:
            x_offset = random.randint(0, img_w - crop_w)
            y_offset = random.randint(0, img_h - crop_h)
        else:
            w_step = (img_w - crop_w) // 4
            h_step = (img_h - crop_h) // 4
            candidate_offsets = [
                (0, 0),  # upper left
                (4 * w_step, 0),  # upper right
                (0, 4 * h_step),  # lower left
                (4 * w_step, 4 * h_step),  # lower right
                (2 * w_step, 2 * h_step),  # center
            ]
            if self.num_fixed_crops == 13:
                extra_candidate_offsets = [
                    (0, 2 * h_step),  # center left
                    (4 * w_step, 2 * h_step),  # center right
                    (2 * w_step, 4 * h_step),  # lower center
                    (2 * w_step, 0 * h_step),  # upper center
                    (1 * w_step, 1 * h_step),  # upper left quarter
                    (3 * w_step, 1 * h_step),  # upper right quarter
                    (1 * w_step, 3 * h_step),  # lower left quarter
                    (3 * w_step, 3 * h_step)  # lower right quarter
                ]
                candidate_offsets.extend(extra_candidate_offsets)
            x_offset, y_offset = random.choice(candidate_offsets)

        new_h, new_w = crop_h, crop_w

        results['crop_bbox'] = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['img_shape'] = (new_h, new_w)
        results['scales'] = self.scales

        if not self.lazy:
            results['imgs'] = [
                img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
                for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'input_size={self.input_size}, scales={self.scales}, '
                    f'max_wh_scale_gap={self.max_wh_scale_gap}, '
                    f'random_crop={self.random_crop}, '
                    f'num_fixed_crops={self.num_fixed_crops}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class RatioPreservingCrop(object):
    def __init__(self, input_size, scale_limits=(1.0, 0.8), targets=None, interpolation='bilinear',
                 use_kpts=False, sigma_scale=0.1):
        assert isinstance(scale_limits, (tuple, list))
        assert len(scale_limits) == 2
        self.scale_limits = float(min(scale_limits)), float(max(scale_limits))
        assert 0.0 < self.scale_limits[0] < self.scale_limits[1] <= 1.0
        self.use_kpts = use_kpts
        self.sigma_scale = float(sigma_scale)
        assert self.sigma_scale > 0.0

        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.output_size = self.input_size[1], self.input_size[0]

        self.targets = ['imgs']
        if targets is not None:
            if not isinstance(targets, (tuple, list)):
                targets = [targets]

            for target in targets:
                assert isinstance(target, str)

                if target not in self.targets:
                    self.targets.append(target)

        self.interpolation = interpolation
        if isinstance(self.interpolation, (tuple, list)):
            assert len(self.interpolation) == len(self.targets)
        else:
            self.interpolation = [self.interpolation] * len(self.targets)

    @staticmethod
    def _estimate_object_bbox(kpts, frame_idx, img_size, sigma_scale):
        valid_kpts = []
        for kpt_data in kpts.values():
            for frame_id in frame_idx:
                if frame_id in kpt_data:
                    valid_kpts.append([float(value) for value in kpt_data[frame_id][:2]])

        if len(valid_kpts) == 0:
            return None

        image_h, image_w = img_size[0], img_size[1]
        sigma = (sigma_scale * np.sqrt(image_h ** 2 + image_w ** 2))

        valid_kpts = np.array(valid_kpts, dtype=np.float32).reshape([-1, 2])
        x_min = int(np.round(np.maximum(0.0, valid_kpts[:, 0] - 3 * sigma).min()))
        y_min = int(np.round(np.maximum(0.0, valid_kpts[:, 1] - 3 * sigma).min()))
        x_max = int(np.round(np.minimum(image_w, valid_kpts[:, 0] + 3 * sigma).max()))
        y_max = int(np.round(np.minimum(image_h, valid_kpts[:, 1] + 3 * sigma).max()))

        object_bbox = None
        if x_min < x_max and y_min < y_max:
            object_bbox = x_min, y_min, x_max, y_max

        return object_bbox

    @staticmethod
    def _sample_crop_bbox(scale_limits, image_size, trg_size, object_bbox=None):
        image_h, image_w = image_size[0], image_size[1]

        scale = np.random.uniform(low=scale_limits[0], high=scale_limits[1])

        src_ar = float(image_h) / float(image_w)
        trg_ar = float(trg_size[0]) / float(trg_size[1])
        if src_ar < trg_ar:
            crop_h = scale * image_h
            crop_w = crop_h / trg_ar
        else:
            crop_w = scale * image_w
            crop_h = crop_w * trg_ar

        crop_w = min(image_w, int(crop_w))
        crop_h = min(image_h, int(crop_h))

        if object_bbox is not None:
            obj_x_min, obj_y_min, obj_x_max, obj_y_max = object_bbox

            w_offset = 0
            w_offset_min, w_offset_max = max(0, obj_x_max - crop_w), min(obj_x_min, image_w - crop_w)
            if crop_w < image_w and w_offset_min < w_offset_max:
                w_offset = random.randint(w_offset_min, w_offset_max)

            h_offset = 0
            h_offset_min, h_offset_max = max(0, obj_y_max - crop_h), min(obj_y_min, image_h - crop_h)
            if crop_h < image_h and h_offset_min < h_offset_max:
                h_offset = random.randint(h_offset_min, h_offset_max)
        else:
            w_offset = random.randint(0, image_w - crop_w) if crop_w < image_w else 0
            h_offset = random.randint(0, image_h - crop_h) if crop_h < image_h else 0

        return crop_w, crop_h, w_offset, h_offset

    def __call__(self, results):
        img_data = results['imgs']
        num_clips = results['num_clips']
        clip_len = results['clip_len']

        img_size = img_data[0].shape[:2]

        if self.use_kpts and 'kpts' in results:
            frame_inds = results['frame_inds'].flatten().reshape([num_clips, -1])
            object_bboxes = [
                self._estimate_object_bbox(results['kpts'], frame_inds[clip_id], img_size, self.sigma_scale)
                for clip_id in range(num_clips)
            ]
        else:
            object_bboxes = [None] * num_clips

        rand_boxes = []
        for clip_id in range(num_clips):
            crop_w, crop_h, offset_w, offset_h = self._sample_crop_bbox(
                self.scale_limits, img_size, self.input_size, object_bboxes[clip_id]
            )
            rand_boxes.append(np.array([offset_w, offset_h, offset_w + crop_w - 1, offset_h + crop_h - 1]))

        for trg_name, interpolation in zip(self.targets, self.interpolation):
            trg_data = results[trg_name]
            assert len(trg_data) == num_clips * clip_len

            processed_data = []
            for clip_id in range(num_clips):
                for i in range(clip_len):
                    cropped_img = mmcv.imcrop(trg_data[clip_id * clip_len + i], rand_boxes[clip_id])
                    resized_img = mmcv.imresize(cropped_img, self.output_size, interpolation=interpolation)
                    processed_data.append(resized_img)

            results[trg_name] = processed_data

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'input_size={self.input_size}, '
                    f'scale_limits={self.scale_limits}, '
                    f'use_kpts={self.use_kpts})')
        return repr_str


@PIPELINES.register_module()
class Resize(object):
    """Resize images to a specific size.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "lazy",
    "resize_size". Required keys in "lazy" is None, added or modified key is
    "interpolation".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, scale, keep_ratio=True, lazy=False, targets=None, interpolation='bilinear'):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(f'Scale must be float or tuple of int, but got {type(scale)}')

        self.scale = scale
        self.keep_ratio = keep_ratio
        self.lazy = lazy

        self.targets = ['imgs']
        if targets is not None:
            if not isinstance(targets, (tuple, list)):
                targets = [targets]

            for target in targets:
                assert isinstance(target, str)

                if target not in self.targets:
                    self.targets.append(target)

        self.interpolation = interpolation
        if isinstance(self.interpolation, (tuple, list)):
            assert len(self.interpolation) == len(self.targets)
        else:
            self.interpolation = [self.interpolation] * len(self.targets)

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        _init_lazy_if_proper(results, self.lazy)

        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h], dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if not self.lazy:
            for trg_name, interpolation in zip(self.targets, self.interpolation):
                results[trg_name] = [
                    mmcv.imresize(img, (new_w, new_h), interpolation=interpolation)
                    for img in results[trg_name]
                ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')
            lazyop['interpolation'] = self.interpolation

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class Flip(object):
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.
    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "lazy" and "flip_direction". Required keys in "lazy" is
    None, added or modified key are "flip" and "flip_direction".

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self, flip_ratio=0.5, direction='horizontal', lazy=False, targets=None):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.lazy = lazy

        self.targets = ['imgs']
        if targets is not None:
            if not isinstance(targets, (tuple, list)):
                targets = [targets]

            for target in targets:
                assert isinstance(target, str)

                if target not in self.targets:
                    self.targets.append(target)

    def __call__(self, results):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        _init_lazy_if_proper(results, self.lazy)

        modality = results['modality']
        if modality == 'Flow':
            assert self.direction == 'horizontal'

        flip = np.random.rand() < self.flip_ratio
        results['flip'] = flip
        results['flip_direction'] = self.direction

        if not self.lazy:
            if flip:
                for trg_name in self.targets:
                    data = results[trg_name]
                    for img in data:
                        mmcv.imflip_(img, self.direction)

                if modality == 'Flow':
                    lt = len(results['imgs'])
                    for i in range(0, lt, 2):
                        # flow with even indexes are x_flow, which need to be
                        # inverted when doing horizontal flip
                        results['imgs'][i] = mmcv.iminvert(results['imgs'][i])
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Use one Flip please')
            lazyop['flip'] = flip
            lazyop['flip_direction'] = self.direction

        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'flip_ratio={self.flip_ratio}, direction={self.direction}, '
            f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class Normalize(object):
    """Normalize images with the given mean and std value.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs" and "img_norm_cfg". If modality is 'Flow', additional
    keys "scale_factor" is required

    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
        adjust_magnitude (bool): Indicate whether to adjust the flow magnitude
            on 'scale_factor' when modality is 'Flow'. Default: False.
    """

    def __init__(self, mean, std, to_bgr=False, adjust_magnitude=False):
        if not isinstance(mean, Sequence):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}'
            )

        if not isinstance(std, Sequence):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_bgr = to_bgr
        self.adjust_magnitude = adjust_magnitude

    def __call__(self, results):
        modality = results['modality']

        if modality == 'RGB':
            imgs = np.array(results['imgs'], dtype=np.float32)
            for img in imgs:
                mmcv.imnormalize_(img, self.mean, self.std, self.to_bgr)

            results['imgs'] = imgs
            results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_bgr=self.to_bgr)

            return results
        elif modality == 'Flow':
            num_imgs = len(results['imgs'])
            assert num_imgs % 2 == 0
            assert self.mean.shape[0] == 2
            assert self.std.shape[0] == 2
            n = num_imgs // 2
            h, w = results['imgs'][0].shape
            x_flow = np.empty((n, h, w), dtype=np.float32)
            y_flow = np.empty((n, h, w), dtype=np.float32)
            for i in range(n):
                x_flow[i] = results['imgs'][2 * i]
                y_flow[i] = results['imgs'][2 * i + 1]
            x_flow = (x_flow - self.mean[0]) / self.std[0]
            y_flow = (y_flow - self.mean[1]) / self.std[1]
            if self.adjust_magnitude:
                x_flow = x_flow * results['scale_factor'][0]
                y_flow = y_flow * results['scale_factor'][1]
            imgs = np.stack([x_flow, y_flow], axis=-1)
            results['imgs'] = imgs
            args = dict(
                mean=self.mean,
                std=self.std,
                to_bgr=self.to_bgr,
                adjust_magnitude=self.adjust_magnitude)
            results['img_norm_cfg'] = args

            return results
        else:
            raise NotImplementedError

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'mean={self.mean}, '
                    f'std={self.std}, '
                    f'to_bgr={self.to_bgr}, '
                    f'adjust_magnitude={self.adjust_magnitude})')
        return repr_str


@PIPELINES.register_module()
class CenterCrop(object):
    """Crop the center area from images.

    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox", "lazy" and "img_shape". Required keys in "lazy" is
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        crop_size (int | tuple[int]): (w, h) of crop size.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, crop_size, lazy=False):
        self.crop_size = _pair(crop_size)
        self.lazy = lazy
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the CenterCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        img_h, img_w = results['img_shape']
        crop_w, crop_h = self.crop_size

        left = (img_w - crop_w) // 2
        top = (img_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        new_h, new_w = bottom - top, right - left

        results['crop_bbox'] = np.array([left, top, right, bottom])
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            results['imgs'] = [
                img[top:bottom, left:right] for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(crop_size={self.crop_size}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class ThreeCrop(object):
    """Crop images into three crops.

    Crop the images equally into three crops with equal intervals along the
    shorter side.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the ThreeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, False)

        imgs = results['imgs']
        img_h, img_w = results['imgs'][0].shape[:2]
        crop_w, crop_h = self.crop_size
        assert crop_h == img_h or crop_w == img_w

        if crop_h == img_h:
            w_step = (img_w - crop_w) // 2
            offsets = [
                (0, 0),  # left
                (2 * w_step, 0),  # right
                (w_step, 0),  # middle
            ]
        elif crop_w == img_w:
            h_step = (img_h - crop_h) // 2
            offsets = [
                (0, 0),  # top
                (0, 2 * h_step),  # down
                (0, h_step),  # middle
            ]

        cropped = []
        crop_bboxes = []
        for x_offset, y_offset in offsets:
            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            crop = [
                img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            cropped.extend(crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs))])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = cropped
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@PIPELINES.register_module()
class TenCrop(object):
    """Crop the images into 10 crops (corner + center + flip).

    Crop the four corners and the center part of the image with the same
    given crop_size, and flip it horizontally.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the TenCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, False)

        imgs = results['imgs']

        img_h, img_w = results['imgs'][0].shape[:2]
        crop_w, crop_h = self.crop_size

        w_step = (img_w - crop_w) // 4
        h_step = (img_h - crop_h) // 4

        offsets = [
            (0, 0),  # upper left
            (4 * w_step, 0),  # upper right
            (0, 4 * h_step),  # lower left
            (4 * w_step, 4 * h_step),  # lower right
            (2 * w_step, 2 * h_step),  # center
        ]

        img_crops = list()
        crop_bboxes = list()
        for x_offset, y_offsets in offsets:
            crop = [
                img[y_offsets:y_offsets + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            flip_crop = [np.flip(c, axis=1).copy() for c in crop]
            bbox = [x_offset, y_offsets, x_offset + crop_w, y_offsets + crop_h]
            img_crops.extend(crop)
            img_crops.extend(flip_crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs) * 2)])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = img_crops
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@PIPELINES.register_module()
class MultiGroupCrop(object):
    """Randomly crop the images into several groups.

    Crop the random region with the same given crop_size and bounding box
    into several groups.
    Required keys are "imgs", added or modified keys are "imgs", "crop_bbox"
    and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
        groups(int): Number of groups.
    """

    def __init__(self, crop_size, groups):
        self.crop_size = _pair(crop_size)
        self.groups = groups
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(
                'Crop size must be int or tuple of int, but got {}'.format(
                    type(crop_size)))

        if not isinstance(groups, int):
            raise TypeError(f'Groups must be int, but got {type(groups)}.')

        if groups <= 0:
            raise ValueError('Groups must be positive.')

    def __call__(self, results):
        """Performs the MultiGroupCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        imgs = results['imgs']
        img_h, img_w = imgs[0].shape[:2]
        crop_w, crop_h = self.crop_size

        img_crops = []
        crop_bboxes = []
        for _ in range(self.groups):
            x_offset = random.randint(0, img_w - crop_w)
            y_offset = random.randint(0, img_h - crop_h)

            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            crop = [
                img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            img_crops.extend(crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs))])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = img_crops
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}'
                    f'(crop_size={self.crop_size}, '
                    f'groups={self.groups})')
        return repr_str


@PIPELINES.register_module()
class RandomRotate(object):
    def __init__(self, delta=10.0, prob=0.5, targets=None):
        if not isinstance(delta, (int, float)):
            raise TypeError(f'Delta must be an int or float, but got {type(delta)}')
        if delta <= 0:
            raise ValueError(f'Delta must be positive, but got {delta}')
        self.delta = float(delta)

        if not isinstance(prob, (int, float)):
            raise TypeError(f'Prob must be an int or float, but got {type(prob)}')
        if prob <= 0 or prob > 1:
            raise ValueError(f'Prob must be in range (0, 1], but got {prob}')
        self.prob = prob

        self.targets = ['imgs']
        if targets is not None:
            if not isinstance(targets, (tuple, list)):
                targets = [targets]

            for target in targets:
                assert isinstance(target, str)

                if target not in self.targets:
                    self.targets.append(target)

    def __call__(self, results):
        num_clips = results['num_clips']
        clip_len = results['clip_len']

        enable_rotate = np.random.rand(num_clips) < self.prob
        rotate_delta = np.random.uniform(-self.delta, self.delta, size=num_clips)

        for trg_name in self.targets:
            trg_data = results[trg_name]
            assert len(trg_data) == num_clips * clip_len

            processed_data = []
            for clip_id in range(num_clips):
                if enable_rotate[clip_id]:
                    processed_data.extend([
                        mmcv.imrotate(trg_data[clip_id * clip_len + i], rotate_delta[clip_id])
                        for i in range(clip_len)
                    ])
                else:
                    processed_data.extend([
                        trg_data[clip_id * clip_len + i]
                        for i in range(clip_len)
                    ])

            results[trg_name] = processed_data

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(delta={self.delta}, prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class BlockDropout(object):
    def __init__(self, scale=0.2, prob=0.1):
        if not isinstance(scale, (int, float)):
            raise TypeError(f'Scale must be an int or float, but got {type(scale)}')
        if scale <= 0 or scale > 1:
            raise ValueError(f'Scale must be in range(0, 1], but got {scale}')
        self.scale = float(scale)

        if not isinstance(prob, (int, float)):
            raise TypeError(f'Prob must be an int or float, but got {type(prob)}')
        if prob <= 0 or prob > 1:
            raise ValueError(f'Prob must be in range (0, 1], but got {prob}')
        self.prob = prob

    @staticmethod
    def _generate_mask(img_size, p, scale):
        img_height, img_width = img_size[:2]

        dropout_height = min(img_height, int(1. / scale))
        dropout_weight = min(img_width, int(1. / scale))

        mask = np.random.random_sample(size=(dropout_height, dropout_weight, 1)) < p
        mask = mmcv.imresize(mask.astype(np.uint8), (img_width, img_height), interpolation='nearest')
        mask = mask.reshape((img_height, img_width, 1))

        return mask

    def __call__(self, results):
        img_data = results['imgs']
        num_clips = results['num_clips']
        clip_len = results['clip_len']

        image_size = img_data[0].shape

        processed_data = []
        for clip_id in range(num_clips):
            valid_mask = self._generate_mask(image_size, 1.0 - self.prob, self.scale)
            processed_data.extend([
                valid_mask * img_data[clip_id * clip_len + i]
                for i in range(clip_len)
            ])

        results['imgs'] = processed_data

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(scale={self.scale}, prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class MixUp(object):
    def __init__(self, root_dir, annot, imgs_root, alpha=0.2, beta=None, scale=1.0, prob=1.0):
        if not isinstance(alpha, (int, float)):
            raise TypeError(f'Alpha must be an int or float, but got {type(alpha)}')
        self.alpha = float(alpha)

        if beta is None:
            self.beta = self.alpha
        else:
            if not isinstance(beta, (int, float)):
                raise TypeError(f'Beta must be an int or float, but got {type(beta)}')

            self.beta = float(beta)

        if not isinstance(scale, (int, float)):
            raise TypeError(f'Scale must be an int or float, but got {type(scale)}')
        self.scale = float(scale)

        annot = osp.join(root_dir, annot)
        if not osp.exists(annot):
            raise ValueError(f'Annot does not exist: {annot}')

        imgs_root = osp.join(root_dir, imgs_root)
        if not osp.exists(imgs_root):
            raise ValueError(f'Annot does not exist: {imgs_root}')

        self.image_paths = self._parse_image_paths(annot, imgs_root)
        if len(self.image_paths) == 0:
            raise ValueError('Found no images for MixUp')

        self.prob = prob
        assert 0.0 <= self.prob <= 1.0

    @staticmethod
    def _parse_image_paths(annot, imgs_root):
        return [osp.join(imgs_root, x.strip().split(' ')[0]) for x in open(annot)]

    @staticmethod
    def _prepare_mixup_image(image_path, trg_size, scale=1.15):
        image = mmcv.imread(image_path)

        scale_factor = scale * float(min(trg_size)) / float(min(image.shape[:2]))
        scaled_image = mmcv.imrescale(image, scale_factor)

        h_offset = random.randint(0, scaled_image.shape[0] - trg_size[0])
        w_offset = random.randint(0, scaled_image.shape[1] - trg_size[1])
        cropped_image = scaled_image[h_offset:h_offset + trg_size[0],
                                     w_offset:w_offset + trg_size[1]]

        if np.random.randint(2):
            cropped_image = mmcv.imflip(cropped_image)

        return cropped_image

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results

        img_data = results['imgs']
        num_clips = results['num_clips']
        clip_len = results['clip_len']

        processed_data = []
        for clip_id in range(num_clips):
            mixup_image_idx = np.random.randint(len(self.image_paths))
            mixup_image_path = self.image_paths[mixup_image_idx]
            mixup_image = self._prepare_mixup_image(mixup_image_path, img_data[0].shape[:2])

            alpha = 0.5 * self.scale * np.random.beta(self.alpha, self.beta)
            scaled_mixup_image = alpha * mixup_image.astype(np.float32)

            for i in range(clip_len):
                float_img = img_data[clip_id * clip_len + i].astype(np.float32)
                mixed_image = (1.0 - alpha) * float_img + scaled_mixup_image
                processed_data.append(mixed_image.clip(0.0, 255.0).astype(np.uint8))

        results['imgs'] = processed_data

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(' \
                   f'alpha={self.alpha}, ' \
                   f'beta={self.beta}, ' \
                   f'scale={self.scale}, ' \
                   f'size={len(self.image_paths)}, ' \
                   f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class CrossNorm(object):
    def __init__(self, root_dir, mean_std_file, prob=1.0):
        mean_std_file = osp.join(root_dir, mean_std_file)
        if not osp.exists(mean_std_file):
            raise ValueError(f'mean_std_file does not exist: {mean_std_file}')

        self.cross_mean, self.cross_std = self._parse_data(mean_std_file)
        self.num_tuples = len(self.cross_mean)
        if self.num_tuples == 0:
            raise ValueError('Found no mean (or std) tuples for CrossNorm')

        self.prob = prob
        assert 0.0 <= self.prob <= 1.0

    @staticmethod
    def _parse_data(data_file):
        mean_data, std_data = [], []
        with open(data_file) as input_stream:
            for line in input_stream:
                line_parts = line.strip().split(' ')
                if len(line_parts) != 2:
                    continue

                mean_str, std_str = line_parts
                mean_value = [float(v) for v in mean_str.split(',')]
                std_value = [float(v) for v in std_str.split(',')]

                assert len(mean_value) == 3
                assert len(std_value) == 3

                mean_data.append(mean_value)
                std_data.append(std_value)

        mean_data = np.array(mean_data, dtype=np.float32)
        std_data = np.array(std_data, dtype=np.float32)

        return mean_data.reshape([-1, 1, 1, 3]), std_data.reshape([-1, 1, 1, 3])

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results

        img_data = results['imgs']
        num_clips = results['num_clips']
        clip_len = results['clip_len']

        processed_data = []
        for clip_id in range(num_clips):
            tuple_idx = np.random.randint(self.num_tuples)
            cross_mean_value = self.cross_mean[tuple_idx]
            cross_std_value = self.cross_std[tuple_idx]

            float_imgs = []
            sum_x, sum_sqr_x = np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32)
            for i in range(clip_len):
                float_img = img_data[clip_id * clip_len + i].astype(np.float32)
                float_imgs.append(float_img)

                sum_x += np.sum(float_img, axis=(0, 1))
                sum_sqr_x += np.sum(float_img ** 2, axis=(0, 1))

            num_elements = clip_len * np.prod(img_data[clip_id * clip_len].shape[:2])
            clip_mean = sum_x / float(num_elements)
            clip_sqr_mean = sum_sqr_x / float(num_elements)
            clip_std = np.sqrt((clip_sqr_mean - clip_mean ** 2).clip(0.0))

            if np.all(clip_std > 1e-3):
                for i in range(clip_len):
                    float_img = float_imgs[i]
                    norm_img = (float_img - clip_mean) / clip_std

                    cross_img = cross_std_value * norm_img + cross_mean_value
                    processed_data.append(cross_img.clip(0.0, 255.0).astype(np.uint8))
            else:
                for i in range(clip_len):
                    processed_data.append(img_data[clip_id * clip_len + i])

        results['imgs'] = processed_data

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(' \
                   f'size={self.num_tuples}, ' \
                   f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class Empty(object):
    def __call__(self, results):
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}'
        return repr_str


@PIPELINES.register_module()
class PhotometricDistortion(object):
    def __init__(self, brightness_range=None, contrast_range=None, saturation_range=None,
                 hue_delta=None, noise_sigma=None, noise_separate=True, color_scale=None):
        self.brightness_lower, self.brightness_upper = \
            brightness_range if brightness_range is not None else (None, None)
        self.contrast_lower, self.contrast_upper = \
            contrast_range if contrast_range is not None else (None, None)
        self.saturation_lower, self.saturation_upper = \
            saturation_range if saturation_range is not None else (None, None)
        self.hue_delta = hue_delta if hue_delta is not None else None
        self.noise_sigma = noise_sigma if noise_sigma is not None else None
        self.noise_separate = noise_separate
        self.color_scale_lower, self.color_scale_upper = \
            color_scale if color_scale is not None else (None, None)

    @property
    def _with_brightness(self):
        return self.brightness_lower is not None and self.brightness_upper is not None

    @property
    def _with_contrast(self):
        return self.contrast_lower is not None and self.contrast_upper is not None

    @property
    def _with_saturation(self):
        return self.saturation_lower is not None and self.saturation_upper is not None

    @property
    def _with_hue(self):
        return self.hue_delta is not None

    @property
    def _with_noise(self):
        return self.noise_sigma is not None

    @property
    def _with_color_scale(self):
        return self.color_scale_lower is not None and self.color_scale_upper is not None

    @staticmethod
    def _augment(img, brightness_delta, contrast_mode, contrast_alpha, saturation_alpha,
                 hue_delta, color_scales, noise_sigma, noise_delta):
        def _clamp_image(_img):
            _img[_img < 0.0] = 0.0
            _img[_img > 255.0] = 255.0
            return _img

        img = img.astype(np.float32)

        # random brightness
        if brightness_delta is not None:
            img += brightness_delta
            img = _clamp_image(img)

        # random contrast
        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        if contrast_mode == 1:
            if contrast_alpha is not None:
                img *= contrast_alpha
                img = _clamp_image(img)

        # convert color from BGR to HSV
        if saturation_alpha is not None or hue_delta is not None:
            img = mmcv.bgr2hsv(img / 255.)

        # random saturation
        if saturation_alpha is not None:
            img[:, :, 1] *= saturation_alpha
            img[:, :, 1][img[:, :, 1] > 1.0] = 1.0
            img[:, :, 1][img[:, :, 1] < 0.0] = 0.0

        # random hue
        if hue_delta is not None:
            img[:, :, 0] += hue_delta
            img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
            img[:, :, 0][img[:, :, 0] < 0.0] += 360.0

        # convert color from HSV to BGR
        if saturation_alpha is not None or hue_delta is not None:
            img = mmcv.hsv2bgr(img) * 255.

        # random contrast
        if contrast_mode == 0:
            if contrast_alpha is not None:
                img *= contrast_alpha
                img = _clamp_image(img)

        if color_scales is not None:
            img *= color_scales.reshape((1, 1, -1))

        # gaussian noise
        if noise_sigma is not None:
            if noise_delta is None:
                img += np.random.normal(loc=0.0, scale=noise_sigma, size=img.shape)
            else:
                img += noise_delta

        # clamp
        img = _clamp_image(img)

        return img.astype(np.uint8)

    def _process_sequence(self, img_data):
        brightness_delta = None
        if self._with_brightness and np.random.randint(2):
            images_mean_brightness = [np.mean(img) for img in img_data]
            image_brightness = np.random.choice(images_mean_brightness)

            brightness_delta_limits = [self.brightness_lower - image_brightness,
                                       self.brightness_upper - image_brightness]

            # extend the range to support the original brightness
            if image_brightness < self.brightness_lower:
                brightness_delta_limits[0] = 0.0
            elif image_brightness > self.brightness_upper:
                brightness_delta_limits[1] = 0.0

            brightness_delta = np.random.uniform(brightness_delta_limits[0], brightness_delta_limits[1])

        noise_sigma, noise_delta = None, None
        if self._with_noise and np.random.randint(2):
            image_max_brightness = max([np.max(img) for img in img_data])
            image_min_brightness = min([np.min(img) for img in img_data])
            brightness_range = image_max_brightness - image_min_brightness
            max_noise_sigma = self.noise_sigma * float(brightness_range if brightness_range > 0 else 1)
            noise_sigma = np.random.uniform(0, max_noise_sigma)

            if not self.noise_separate:
                noise_delta = np.random.normal(loc=0.0, scale=noise_sigma, size=img_data[0].shape)

        contrast_mode = np.random.randint(2)
        contrast_alpha = np.random.uniform(self.contrast_lower, self.contrast_upper) \
            if self._with_contrast and np.random.randint(2) else None

        saturation_alpha = np.random.uniform(self.saturation_lower, self.saturation_upper) \
            if self._with_saturation and np.random.randint(2) else None

        hue_delta = np.random.uniform(-self.hue_delta, self.hue_delta) \
            if self._with_hue and np.random.randint(2) else None

        color_scales = np.random.uniform(self.color_scale_lower, self.color_scale_upper, size=3) \
            if self._with_color_scale and np.random.randint(2) else None

        processed_data = [
            self._augment(img, brightness_delta, contrast_mode, contrast_alpha,
                          saturation_alpha, hue_delta, color_scales,
                          noise_sigma, noise_delta)
            for img in img_data
        ]

        return processed_data

    def __call__(self, results):
        if results['modality'] == 'Flow':
            return results

        img_data = results['imgs']
        num_clips = results['num_clips']
        clip_len = results['clip_len']

        processed_data = []
        for clip_id in range(num_clips):
            start = clip_id * clip_len
            end = start + clip_len
            processed_data.extend(self._process_sequence(img_data[start:end]))

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__} ('
        if self._with_brightness:
            repr_str += f'brightness_range=[{self.brightness_lower}, {self.brightness_upper}], '
        if self._with_color_scale:
            repr_str += f'color_scale=[{self.color_scale_lower}, {self.color_scale_upper}], '
        if self._with_contrast:
            repr_str += f'contrast_range=[{self.contrast_lower}, {self.contrast_upper}], '
        if self._with_hue:
            repr_str += f'hue_delta={self.hue_delta}, '
        if self._with_noise:
            repr_str += f'noise_sigma={self.noise_sigma}, '
        if self._with_saturation:
            repr_str += f'saturation_range=[{self.saturation_lower}, {self.saturation_upper}], '
        repr_str += ')'
        return repr_str


@PIPELINES.register_module()
class MapFlippedLabels(object):
    def __init__(self, root_dir, map_file):
        assert isinstance(map_file, dict), 'map_file is expected to be dict'

        self.labels_map = dict()
        for dataset_name, file_name in map_file.items():
            map_file = osp.join(root_dir, dataset_name, file_name)
            if not osp.exists(map_file):
                raise ValueError(f'Mapping file does not exist: {map_file}')

            self.labels_map[dataset_name] = self.load_labels_map(map_file)

    @staticmethod
    def load_labels_map(file_path):
        out_data = dict()
        with open(file_path) as input_stream:
            for line in input_stream:
                line_parts = line.strip().split(' ')
                if len(line_parts) == 2:
                    key = int(line_parts[0])
                    value = int(line_parts[1])
                    out_data[key] = value

        return out_data

    def __call__(self, results):
        if 'dataset_name' not in results:
            raise ValueError('For enabling labels map transform the record'
                             'should contain \'dataset_name\' field ')
        dataset_name = results['dataset_name']
        if results['flip'] and dataset_name in self.labels_map:
            local_labels_map = self.labels_map[dataset_name]

            old_label = results['label']
            if isinstance(old_label, (tuple, list)):
                results['label'] = [local_labels_map[ol] if ol in local_labels_map else ol
                                    for ol in old_label]
            else:
                if old_label in local_labels_map:
                    results['label'] = local_labels_map[old_label]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}'
        return repr_str
