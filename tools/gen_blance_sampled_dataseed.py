"""
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
DATASET=coco_2017_train

OMP_NUM_THREADS=1 python tools/data_converters/gen_dataseed.py \
        --output-dir "/home/data/coco/dataseed" \
        --opts \
        DATASETS.TRAIN "('${DATASET}',)" \
"""
import json
import numpy as np
import argparse
import json
import os
import time
from itertools import chain

import cv2
import numpy as np
import tqdm
from PIL import Image

from detectron2.config import get_cfg
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.utils.logger import setup_logger
from collections import defaultdict


def setup(args):
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)
    # SupPercent = [0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    SupPercent = [0.1, 0.5, 1.0, 2.0]
    result = {}
    for sup_percent in SupPercent:
        result[str(sup_percent)] = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[], '6':[],'7':[],'8':[],'9':[]}

    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    num_all = len(dataset_dicts) # 117266
    
    all_image_ids = defaultdict(set)
    for idx, dataset_dict_per_img in enumerate(dataset_dicts):
        for anno in dataset_dict_per_img['annotations']:
            all_image_ids[anno['category_id']].add(idx)

    for sup_percent in SupPercent:
        num_imgs = int(sup_percent / 100.0 * num_all)
        num_imgs_per_class = int(num_imgs / 80) + 1
        print('num_imgs_per_class', num_imgs_per_class)

        for random_data_seed in range(0, 10):
            img_idx = []
            for k, v in all_image_ids.items():
                img_idx += np.random.choice(list(v), size=num_imgs_per_class, replace=False).tolist()
            result[str(sup_percent)][str(random_data_seed)] = list(set(img_idx))

    with open(os.path.join(args.output_dir, 'COCO_supervision_{}_blance_sampled.txt'.format(num_all)), 'w') as f:
        json.dump(result, f)