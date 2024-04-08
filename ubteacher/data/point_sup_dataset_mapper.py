# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
from typing import List, Union
import torch

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.config import configurable

# fmt: off
from detectron2.data.detection_utils import \
    annotations_to_instances as base_annotations_to_instances
from detectron2.data.detection_utils import \
    transform_instance_annotations as base_transform_instance_annotations
from detectron2.structures import BoxMode, polygons_to_bitmask
from typing import List
from pteacher.data.detection_utils import build_strong_augmentation, build_large_scale_jitter_augmentation

from detectron2.data.dataset_mapper import DatasetMapper
from PIL import Image
# from .detection_utils import annotations_to_instances, transform_instance_annotations
import random

__all__ = [
    "PointSupDatasetMapper",
    "PointSupTwoCropSeparateDatasetMapper"

]

class PointSupDatasetMapper:
    """
    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        scale_augmentations: List[Union[T.Augmentation, T.Transform]],
        color_augmentations: List,
        image_format: str,
        # Extra data augmentation for point supervision
        sample_points: int = 0,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            sample_points: subsample points at each iteration
        """
        # fmt: off
        self.is_train               = is_train
        self.scale_augmentations          = T.AugmentationList(scale_augmentations)
        self.color_augmentations   = color_augmentations
        # self.augmentations          = augmentations
        self.image_format           = image_format
        self.sample_points          = sample_points
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Color Augmentations used in {mode}: {color_augmentations}")
        logger.info(f"[DatasetMapper] Scale Augmentations used in {mode}: {scale_augmentations}")
        logger.info(f"Point Augmentations used in {mode}: sample {sample_points} points")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        # augs = utils.build_augmentation(cfg, is_train)
        scale_augs = build_large_scale_jitter_augmentation(cfg, is_train)
        color_augs = build_strong_augmentation(cfg, is_train)
        # if cfg.INPUT.CROP.ENABLED and is_train:
        #     raise ValueError("Crop augmentation not supported to point supervision.")

        ret = {
            "is_train": is_train,
            "scale_augmentations": scale_augs,
            "color_augmentations": color_augs,
            "image_format": cfg.INPUT.FORMAT,
            "sample_points": cfg.INPUT.SAMPLE_POINTS,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format) #BGR

        # if not self.is_train:
        #     import pdb
        #     pdb.set_trace()
        #     aug_input = T.AugInput(image)
        #     transforms = self.scale_augmentations(aug_input)
        #     image = aug_input.image
        #     dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        #     dataset_dict.pop("annotations", None)
        #     return dataset_dict

        # We use torchvision augmentation, which is not compatiable with
        # detectron2, which use numpy format for images. Thus, we need to
        # convert to PIL format first.
        utils.check_image_size(dataset_dict, image)
        image_pil = Image.fromarray(image.astype("uint8"), "RGB")
        image = np.array(self.color_augmentations(image_pil))
        # image = np.array(Image.fromarray(image.astype("uint8"), "BGR"))

        aug_input = T.AugInput(image)
        transforms = self.scale_augmentations(aug_input)
        # image = aug_input.image
        image = aug_input.image[:, :, ::-1].copy() #rgb -> bgr
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if "annotations" in dataset_dict:
            # Maps points from the closed interval [0, image_size - 1] on discrete
            # image coordinates to the half-open interval [x1, x2) on continuous image
            # coordinates. We use the continuous-discrete conversion from Heckbert
            # 1990 ("What is the coordinate of a pixel?"): d = floor(c) and c = d + 0.5,
            # where d is a discrete coordinate and c is a continuous coordinate.
            for ann in dataset_dict["annotations"]:
                point_coords_wrt_image = np.array(ann["point_coords"]).astype(np.float)
                point_coords_wrt_image = point_coords_wrt_image + 0.5
                ann["point_coords"] = point_coords_wrt_image

            annos = [
                # also need to transform point coordinates
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            instances = annotations_to_instances(
                annos,
                image_shape,
                sample_points=self.sample_points,
                is_train=self.is_train
            )

            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict


class PointSupTwoCropSeparateDatasetMapper(DatasetMapper):
    """
    This customized mapper produces two augmented images from a single image
    instance. This mapper makes sure that the two augmented images have the same
    cropping and thus the same size.
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.
    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.
    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        self.augmentation = utils.build_augmentation(cfg, is_train)
        # include crop into self.augmentation
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            self.compute_tight_boxes = True
        else:
            self.compute_tight_boxes = False
        self.strong_augmentation = build_strong_augmentation(cfg, is_train)

        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image_weak_aug, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image_weak_aug.shape[:2]  # h, w

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))



        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:

            # Maps points from the closed interval [0, image_size - 1] on discrete
            # image coordinates to the half-open interval [x1, x2) on continuous image
            # coordinates. We use the continuous-discrete conversion from Heckbert
            # 1990 ("What is the coordinate of a pixel?"): d = floor(c) and c = d + 0.5,
            # where d is a discrete coordinate and c is a continuous coordinate.


            for anno in dataset_dict["annotations"]:
                point_coords_wrt_image = np.array(anno["point_coords"]).astype(np.float)
                point_coords_wrt_image = point_coords_wrt_image + 0.5
                anno["point_coords"] = point_coords_wrt_image
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    # keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # import pdb
            # pdb.set_trace()
            instances = annotations_to_instances(
                annos, image_shape, sample_points=0, is_train=True
            )

            if self.compute_tight_boxes and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            bboxes_d2_format = utils.filter_empty_instances(instances)
            dataset_dict["instances"] = bboxes_d2_format


        if self.is_train and self.load_proposals:
            # utils.transform_proposals(
            #     dataset_dict,
            #     image_shape,
            #     transforms,
            #     proposal_topk=self.proposal_topk,
            #     min_box_size=self.proposal_min_box_size,
            # )

            transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )


        # apply strong augmentation
        # We use torchvision augmentation, which is not compatiable with
        # detectron2, which use numpy format for images. Thus, we need to
        # convert to PIL format first.
        image_pil = Image.fromarray(image_weak_aug.astype("uint8"), "RGB")
        # image_pil.save('./results/vis/11_{}'.format(dataset_dict['file_name'].split('/')[-1]))
        image_strong_aug = np.array(self.strong_augmentation(image_pil))
        # result = Image.fromarray((image_strong_aug).astype(np.uint8))
        # result.save('./results/vis/{}'.format(dataset_dict['file_name'].split('/')[-1]))
        # import pdb
        # pdb.set_trace()

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
        )

        dataset_dict_key = copy.deepcopy(dataset_dict)
        dataset_dict_key["image"] = torch.as_tensor(
            np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1))
        )
        assert dataset_dict["image"].size(1) == dataset_dict_key["image"].size(1)
        assert dataset_dict["image"].size(2) == dataset_dict_key["image"].size(2)
        # return (strong aug, weak_aug)
        return (dataset_dict, dataset_dict_key)


def annotations_to_instances(annos, image_size, sample_points=0, is_train=True):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.
    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width
        sample_points (int): subsample points at each iteration, if sample_points==0, then all points are kept by default
    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_point_coords", "gt_point_labels", if they can be obtained from `annos`.
            This is the format that builtin models with point supervision expect.
    """
    target = base_annotations_to_instances(annos, image_size)
    # import pdb
    # pdb.set_trace()
    if is_train:
        if len(annos) == 0:
            # import pdb
            # pdb.set_trace()
            target.gt_point_coords = torch.empty(0, 2, 2)
            target.gt_point_labels = torch.empty(0, 2)
            return target

        assert "point_coords" in annos[0]
        assert "point_labels" in annos[0]
        # assert "segmentation" not in annos[0], "Please remove mask annotation"

    if len(annos) and "point_coords" in annos[0]:
        point_coords = []
        point_labels = []
        for i, _ in enumerate(annos):
            # Already in the image coordinate system
            if annos[i]["point_labels"][1] == 0:
                # random sampling neg points on the box
                if annos[i]["bbox_mode"] == BoxMode.XYXY_ABS:
                    x0, y0, x1, y1 = annos[i]["bbox"]
                    if random.random() < 0.5:
                        x_neg_point = random.random() * (x1 - x0) + x0
                        y_neg_point = (y0 + 0.5) if random.random() < 0.5 else (y1 - 0.5)
                        y_neg_point = y_neg_point
                    else:
                        x_neg_point = (x0 + 0.5) if random.random() < 0.5 else (x1 - 0.5)
                        x_neg_point = x_neg_point
                        y_neg_point = random.random() * (y1 - y0) + y0

                    annos[i]["point_coords"][1] = np.array([x_neg_point, y_neg_point])
                else:
                    raise NotImplementedError

            point_coords_wrt_image = np.array(annos[i]["point_coords"])
            point_labels_wrt_image = np.array(annos[i]["point_labels"])

            if sample_points > 0:
                random_indices = np.random.choice(
                    point_coords_wrt_image.shape[0],
                    sample_points,
                    replace=point_coords_wrt_image.shape[0] < sample_points,
                ).astype(int)
                point_coords_wrt_image = point_coords_wrt_image[random_indices]
                point_labels_wrt_image = point_labels_wrt_image[random_indices]
                assert point_coords_wrt_image.shape[0] == point_labels_wrt_image.size

            point_coords.append(point_coords_wrt_image)
            point_labels.append(point_labels_wrt_image)

        point_coords = torch.stack([torch.from_numpy(x) for x in point_coords])
        point_labels = torch.stack([torch.from_numpy(x) for x in point_labels])
        target.gt_point_coords = point_coords
        target.gt_point_labels = point_labels
        # import pdb
        # pdb.set_trace()

    return target


def transform_instance_annotations(annotation, transforms, image_size):
    """
    Apply transforms to box, and point annotations of a single instance.
    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for points.
    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
    Returns:
        dict:
            the same input dict with fields "bbox", "point_coords", "point_labels"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    annotation = base_transform_instance_annotations(annotation, transforms, image_size)

    # assert "segmentation" not in annotation
    assert "point_coords" in annotation
    assert "point_labels" in annotation
    point_coords = annotation["point_coords"]
    point_labels = np.array(annotation["point_labels"]).astype(np.float)
    point_coords = transforms.apply_coords(point_coords)

    # Set all out-of-boundary points to "unlabeled"
    inside = (point_coords >= np.array([0, 0])) & (point_coords <= np.array(image_size[::-1]))
    inside = inside.all(axis=1)
    point_labels[~inside] = -1

    annotation["point_coords"] = point_coords
    annotation["point_labels"] = point_labels

    return annotation



from detectron2.structures import (
    Boxes,
    BoxMode,
    Instances,
)


def transform_proposals(dataset_dict, image_shape, transforms, *, proposal_topk=10000, min_box_size=0):
    """
    Apply transformations to the proposals in dataset_dict, if any.

    Args:
        dataset_dict (dict): a dict read from the dataset, possibly
            contains fields "proposal_boxes", "proposal_objectness_logits", "proposal_bbox_mode"
        image_shape (tuple): height, width
        transforms (TransformList):
        proposal_topk (int): only keep top-K scoring proposals
        min_box_size (int): proposals with either side smaller than this
            threshold are removed

    The input dict is modified in-place, with abovementioned keys removed. A new
    key "proposals" will be added. Its value is an `Instances`
    object which contains the transformed proposals in its field
    "proposal_boxes" and "objectness_logits".
    """
    if "proposal_boxes" in dataset_dict and "instances" in dataset_dict:
        # Transform proposal boxes
        boxes = transforms.apply_box(
            BoxMode.convert(
                dataset_dict.pop("proposal_boxes"),
                dataset_dict.pop("proposal_bbox_mode"),
                BoxMode.XYXY_ABS,
            )
        )
        boxes = Boxes(boxes)
        # print("before filter {}".format(boxes.tensor.size()))
        objectness_logits = torch.as_tensor(dataset_dict.pop("proposal_objectness_logits").astype("float32"))

        if hasattr(dataset_dict['instances'], "gt_point_coords"):
            gt_point_coords = dataset_dict['instances'].gt_point_coords[:, 0, :]
            candidate_idxs = []
            for p in gt_point_coords:
                _idxs_p = ((p[0] >= boxes.tensor[:, 0]) & (p[0] <= boxes.tensor[:, 2]) & \
                           (p[1] >= boxes.tensor[:, 1]) & (p[1] <= boxes.tensor[:, 3])).nonzero().reshape(-1)
                candidate_idxs.append(_idxs_p)
            candidate_idxs = torch.cat(candidate_idxs).cpu().numpy().tolist()
            candidate_idxs = list(set(candidate_idxs))
            boxes = boxes[candidate_idxs]
            # print("after filter {}".format(boxes.tensor.size()))
            objectness_logits = objectness_logits[candidate_idxs]

        boxes.clip(image_shape)
        keep = boxes.nonempty(threshold=min_box_size)
        boxes = boxes[keep]
        objectness_logits = objectness_logits[keep]

        proposals = Instances(image_shape)
        proposals.proposal_boxes = boxes[:proposal_topk]
        proposals.objectness_logits = objectness_logits[:proposal_topk]
        dataset_dict["proposals"] = proposals