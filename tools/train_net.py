#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from ubteacher import add_ubteacher_config, add_point_sup_config, add_pointrend_config, add_fcos_config, add_shapeprop_config, add_boxinst_config
from pteacher.engine.obj_det_trainer import FasterRCNNPointSupTrainer, FCOSPointSupTrainer, UBTeacherTrainer, BaselineTrainer
from pteacher.engine.ins_seg_trainer import MaskRCNNBaselineTrainer, MaskRCNNUBTeacherTrainer, MaskRCNNPointSupTrainer, LSJMaskRCNNPointSupTrainer
# hacky way to register
from pteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from pteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from pteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
from pteacher.modeling.roi_heads.mil_roi_heads import MILROIHeadsPseudoLab
from pteacher.modeling.roi_heads.cam_mil_roi_heads import CamMILROIHeadsPseudoLab
from pteacher.modeling.roi_heads.point_head import PointHeadPseudoLab
from pteacher.modeling.roi_heads.box_head import FastRCNNConvPoolingHead
from pteacher.modeling.fcos import FCOS
import pteacher.data.datasets.builtin

from pteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    add_point_sup_config(cfg)
    add_pointrend_config(cfg)
    add_fcos_config(cfg)
    add_boxinst_config(cfg)
    add_shapeprop_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "faster_rcnn_ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "mask_rcnn_ubteacher":
        Trainer = MaskRCNNUBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "faster_rcnn_point_sup":
        Trainer = FasterRCNNPointSupTrainer
    elif cfg.SEMISUPNET.Trainer == "mask_rcnn_point_sup":
        Trainer = MaskRCNNPointSupTrainer
    elif cfg.SEMISUPNET.Trainer == "lsj_mask_rcnn_point_sup":
        Trainer = LSJMaskRCNNPointSupTrainer
    elif cfg.SEMISUPNET.Trainer == "fcos_point_sup":
        Trainer = FCOSPointSupTrainer
    elif cfg.SEMISUPNET.Trainer == "faster_rcnn_baseline":
        Trainer = BaselineTrainer
    elif cfg.SEMISUPNET.Trainer == "mask_rcnn_baseline":
        Trainer = MaskRCNNBaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "faster_rcnn_ubteacher" \
                or cfg.SEMISUPNET.Trainer == "faster_rcnn_point_sup" \
                or cfg.SEMISUPNET.Trainer == "mask_rcnn_ubteacher" \
                or cfg.SEMISUPNET.Trainer == "mask_rcnn_point_sup" \
                or cfg.SEMISUPNET.Trainer == "lsj_mask_rcnn_point_sup" \
                or cfg.SEMISUPNET.Trainer == "fcos_point_sup":
            # import pdb
            # pdb.set_trace()
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.gen_pseudo_labels_offline(cfg, ensem_ts_model.modelTeacher)
            # res = Trainer.test(cfg, ensem_ts_model.modelTeacher)
            # res = Trainer.test(cfg, ensem_ts_model.modelStudent)
        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
