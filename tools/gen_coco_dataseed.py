"""
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
DATASET=coco_2017_train

OMP_NUM_THREADS=1 python gen_coco_dataseed.py \
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

    SupPercent = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]
    result = {}
    for sup_percent in SupPercent:
        result[str(sup_percent)] = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[], '6':[],'7':[],'8':[],'9':[]}

    num_all = 118287 # keep original anno
    # coco_random_idx.keys(): ['0.01', '0.1', '0.5', '1.0', '2.0', '5.0', '10.0']
    # random sample
    for sup_percent in SupPercent:
        num_label = int(sup_percent / 100.0 * num_all)
        for random_data_seed in range(0, 10):
            labeled_idx = np.random.choice(range(num_all), size=num_label, replace=False)
            result[str(sup_percent)][str(random_data_seed)] = labeled_idx.tolist()

    with open(os.path.join(args.output_dir, 'COCO_supervision_{}.txt'.format(num_all)), 'w') as f:
        json.dump(result, f)

    num_all = 117266 # keep original anno
    # coco_random_idx.keys(): ['0.01', '0.1', '0.5', '1.0', '2.0', '5.0', '10.0']
    # random sample
    for sup_percent in SupPercent:
        num_label = int(sup_percent / 100.0 * num_all)
        for random_data_seed in range(0, 10):
            labeled_idx = np.random.choice(range(num_all), size=num_label, replace=False)
            result[str(sup_percent)][str(random_data_seed)] = labeled_idx.tolist()

    with open(os.path.join(args.output_dir, 'COCO_supervision_{}.txt'.format(num_all)), 'w') as f:
        json.dump(result, f)