# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import contextlib
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.datasets.builtin import _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager
import io
import logging
from .pascal_voc_w_point import register_pascal_voc_w_point

logger = logging.getLogger(__name__)

JSON_ANNOTATIONS_DIR = ""
_SPLITS_COCO_FORMAT = {}
_SPLITS_COCO_FORMAT["coco"] = {
    "coco_2017_unlabel": (
        "coco/unlabeled2017",
        "coco/annotations/image_info_unlabeled2017.json",
    ),
    "coco_2017_for_voc20": (
        "coco",
        "coco/annotations/google/instances_unlabeledtrainval20class.json",
    ),
}


# COCO dataset
def register_coco_instances_with_points(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance segmentation with point annotation.
    The point annotation json does not have "segmentation" field, instead,
    it has "point_coords" and "point_labels" fields.
    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_json(json_file, image_root, name, ["point_coords", "point_labels"])
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    # point annotations without masks for weakly semi-sup faster rcnn
    "coco_2017_train_points_n2_v1": (
        "coco/train2017",
        "coco/annotations/instances_train2017_n2_v1.json",
    ),
    # point annotations with masks for weakly semi-sup mask rcnn
    "coco_2017_train_points_n2_v2": (
        "coco/train2017",
        "coco/annotations/instances_train2017_n2_v2.json",
    ),

    # point annotations with masks for weakly semi-sup mask rcnn
    "coco_2017_train_points_n10_v0": (
        "coco/train2017",
        "coco/annotations/instances_train2017_n10_v0.json",
    ),

    "coco_2017_val_points_n1_v0": (
        "coco/val2017",
        "coco/annotations/instances_val2017_n1_v0.json",
    ),

}


def register_all_coco_train_points(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances_with_points(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


def register_coco_unlabel(root):
    for _, splits_per_dataset in _SPLITS_COCO_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            meta = {}
            register_coco_unlabel_instances(
                key, meta, os.path.join(root, json_file), os.path.join(root, image_root)
            )


def register_coco_unlabel_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_unlabel_json(json_file, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def load_coco_unlabel_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    id_map = None
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_ids)

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs), json_file))

    dataset_dicts = []

    for img_dict in imgs:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        dataset_dicts.append(record)

    return dataset_dicts



def register_all_pascal_voc_w_points(root):
    SPLITS = [
        ("voc_2007_trainval_w_points", "VOC2007", "trainval"),
        ("voc_2012_trainval_w_points", "VOC2012", "trainval"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc_w_point(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
        
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_coco_unlabel(_root)
register_all_coco_train_points(_root)
register_all_pascal_voc_w_points(_root)
