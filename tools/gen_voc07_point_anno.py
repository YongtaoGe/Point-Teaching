#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import json
import numpy as np
import os
import sys
import pycocotools.mask as mask_utils

from detectron2.utils.env import seed_all_rng
from detectron2.utils.file_io import PathManager
import random

def get_point_annotations(input_filename, output_filename, num_points_per_instance=1):
    """
    :param input_filename:
    :param output_filename:
    :param num_points_per_instance:
    :return: one pos point (sampled inside box) + one neg point (sampled on box border)
    """
    with PathManager.open(input_filename, "r") as f:
        coco_json = json.load(f)

    coco_annos = coco_json.pop("annotations")
    coco_points_json = copy.deepcopy(coco_json)

    imgs = {}
    for img in coco_json["images"]:
        imgs[img["id"]] = img

    new_annos = []
    from tqdm import tqdm
    for ann in tqdm(coco_annos):
        # convert mask
        t = imgs[ann["image_id"]]
        h, w = t["height"], t["width"]

        new_ann = copy.deepcopy(ann)
        # sample points in image coordinates
        box = ann["bbox"]
        # import pdb
        # pdb.set_trace()

        point_coords_wrt_image = np.random.rand(num_points_per_instance, 2)
        point_coords_wrt_image[:, 0] = point_coords_wrt_image[:, 0] * box[2]
        point_coords_wrt_image[:, 1] = point_coords_wrt_image[:, 1] * box[3]
        point_coords_wrt_image[:, 0] += box[0]
        point_coords_wrt_image[:, 1] += box[1]
        # round to integer coordinates
        point_coords_wrt_image = np.floor(point_coords_wrt_image).astype(int)
        # get labels
        all_point_coords = []
        all_point_labels = []
        for point in point_coords_wrt_image.tolist():
            all_point_coords.append(point)
            all_point_labels.append(1)

        x0, y0, x1, y1 = box[0], box[1], box[0] + box[2], box[1] + box[3]
        if random.random() < 0.5:
            x_neg_point = random.random() * (x1 - x0) + x0
            y_neg_point = (y0 + 0.5) if random.random() < 0.5 else (y1 - 0.5)
            y_neg_point = y_neg_point
        else:
            x_neg_point = (x0 + 0.5) if random.random() < 0.5 else (x1 - 0.5)
            x_neg_point = x_neg_point
            y_neg_point = random.random() * (y1 - y0) + y0

        all_point_coords.append([int(x_neg_point), int(y_neg_point)])
        all_point_labels.append(0)
        # store new annotations
        new_ann["point_coords"] = all_point_coords
        new_ann["point_labels"] = all_point_labels

        new_annos.append(new_ann)
    coco_points_json["annotations"] = new_annos

    with PathManager.open(output_filename, "w") as f:
        json.dump(coco_points_json, f)

    print("{} is modified and stored in {}.".format(input_filename, output_filename))


if __name__ == "__main__":
    """
    Generate point-based supervision for VOC dataset.
    Usage:
        python tools/prepare_voc07_point_annotations_without_masks.py \
            NUM_POINTS_PER_INSTANCE NUM_VERSIONS_WITH_DIFFERENT_SEED
    Example to generate point-based COCO dataset with 10 points per instance:
        python tools/data_converters/prepare_voc07_pointsup_annotations.py 1
    """

    # Fix random seed
    seed_all_rng(12345)

    assert len(sys.argv) >= 2, "Please provide number of points to sample per instance"
    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "voc/annotations")
    num_points_per_instance = int(sys.argv[1])
    if len(sys.argv) == 3:
        repeat = int(sys.argv[2])
    else:
        repeat = 1
    s = "voc07_trainval"

    for version in range(repeat):
        print(
            "Start sampling {} points per instance for annotations {}.".format(
                num_points_per_instance, s
            )
        )
        get_point_annotations(
            os.path.join(dataset_dir, "{}.json".format(s)),
            os.path.join(
                dataset_dir,
                "{}_n{}_v{}.json".format(s, num_points_per_instance, 0),
            ),
            num_points_per_instance,
        )

