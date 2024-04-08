# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torchvision.transforms as transforms
from pteacher.data.transforms.augmentation_impl import (
    GaussianBlur,
    ResizeScale,
    FixedSizeCrop,
)
from detectron2.data import transforms as T


def build_strong_augmentation(cfg, is_train):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)
    augmentation = []
    if is_train:
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))

        randcrop_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.7, scale=(0.01, 0.023), ratio=(0.3, 3.3), value="random"
                ),
                transforms.RandomErasing(
                    p=0.5, scale=(0.01, 0.022), ratio=(0.1, 6), value="random"
                ),
                transforms.RandomErasing(
                    p=0.3, scale=(0.01, 0.03), ratio=(0.05, 8), value="random"
                ),
                transforms.ToPILImage(),
            ]
        )
        augmentation.append(randcrop_transform)
        logger.info("Augmentations used in training: " + str(augmentation))
    return transforms.Compose(augmentation)


def build_large_scale_jitter_augmentation(cfg, is_train):
    """
    self training with color jitter (autoaugmentation) and large scale jitter
    """
    logger = logging.getLogger(__name__)
    augmentation = []
    if is_train:
        # TODO: make min_size configurable
        min_scale, max_scale = cfg.INPUT.SCALE_JITTER_MIN, cfg.INPUT.SCALE_JITTER_MAX
        target_height, target_width = cfg.INPUT.SCALE_JITTER_TGT_HEIGHT, cfg.INPUT.SCALE_JITTER_TGT_WIDTH
        resize_aug = ResizeScale(min_scale, max_scale, target_height, target_width)
        fixed_size_crop_aug = FixedSizeCrop((target_height, target_width))
        hflip_aug = T.RandomFlip()
        augmentation += [resize_aug, fixed_size_crop_aug, hflip_aug]
        logger.info("Augmentations used in training: " + str(augmentation))
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
        augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
        logger.info("Augmentations used in testing: " + str(augmentation))

    return augmentation