# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
from PIL import ImageFilter
from PIL import Image
from detectron2.data.transforms.augmentation import Augmentation
from detectron2.data.transforms.transform import ResizeTransform
import numpy as np
from typing import Tuple
from fvcore.transforms.transform import (
    CropTransform,
    PadTransform,
    Transform,
    TransformList,
)

class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ResizeScale(Augmentation):
    """
    Takes target size as input and randomly scales the given target size between `min_scale`
    and `max_scale`. It then scales the input image such that it fits inside the scaled target
    box, keeping the aspect ratio constant.
    This implements the resize part of the Google's 'resize_and_crop' data augmentation:
    https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/input_utils.py#L127
    """

    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        target_height: int,
        target_width: int,
        interp: int = Image.BILINEAR,
    ):
        """
        Args:
            min_scale: minimum image scale range.
            max_scale: maximum image scale range.
            target_height: target image height.
            target_width: target image width.
            interp: image interpolation method.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> Transform:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = (self.target_height, self.target_width)
        random_scale = np.random.uniform(self.min_scale, self.max_scale)
        random_scale_size = np.multiply(output_size, random_scale)
        scale = np.minimum(
            random_scale_size[0] / input_size[0], random_scale_size[1] / input_size[1]
        )
        scaled_size = np.round(np.multiply(input_size, scale)).astype(int)
        return ResizeTransform(
            input_size[0], input_size[1], scaled_size[0], scaled_size[1], self.interp
        )


class FixedSizeCrop(Augmentation):
    """
    If `crop_size` is smaller than the input image size, then it uses a random crop of
    the crop size. If `crop_size` is larger than the input image size, then it pads
    the right and the bottom of the image to the crop size.
    """

    def __init__(self, crop_size: Tuple[int], pad_value: float = 128.0):
        """
        Args:
            crop_size: target image (height, width).
            pad_value: the padding value.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> TransformList:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add random crop if the image is scaled up.
        max_offset = np.subtract(input_size, output_size)
        max_offset = np.maximum(max_offset, 0)
        offset = np.multiply(max_offset, np.random.uniform(0.0, 1.0))
        offset = np.round(offset).astype(int)
        crop_transform = CropTransform(
            offset[1], offset[0], output_size[1], output_size[0], input_size[1], input_size[0]
        )

        # Add padding if the image is scaled down.
        pad_size = np.subtract(output_size, input_size)
        pad_size = np.maximum(pad_size, 0)
        original_size = np.minimum(input_size, output_size)
        pad_transform = PadTransform(
            0, 0, pad_size[1], pad_size[0], original_size[1], original_size[0], self.pad_value
        )

        return TransformList([crop_transform, pad_transform])