# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, Optional
import torch

from detectron2.structures import ImageList, Instances
from detectron2.modeling.proposal_generator import RPN
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY


@PROPOSAL_GENERATOR_REGISTRY.register()
class PseudoLabRPN(RPN):
    """
    Region Proposal Network, introduced by :paper:`Faster R-CNN`.
    """

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[Instances] = None,
        compute_loss: bool = True,
        compute_val_loss: bool = False,
    ):
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(
                x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1]
            )
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if (self.training and compute_loss) or compute_val_loss:
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
            losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        else:  # inference
            losses = {}

        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )

        return proposals, losses


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.registry import Registry

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.proposal_utils import find_top_rpn_proposals
from detectron2.modeling.proposal_generator.rpn import build_rpn_head
from pteacher.modeling.fcos.fcos import FCOS, FCOSHead
from pteacher.modeling.fcos.fcos_outputs import FCOSOutputs
from pteacher.utils.comm import compute_locations

@PROPOSAL_GENERATOR_REGISTRY.register()
class PseudoLabRPN_FCOS(RPN):
    """
    Region Proposal Network, introduced by :paper:`Faster R-CNN`.
    """
    @configurable
    def __init__(self,
                 *,
                 fcos_in_features: List[str],
                 fcos_fpn_strides: List[int],
                 fcos_yield_proposal: bool,
                 fcos_head: nn.Module,
                 fcos_outputs: nn.Module,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.fcos_in_features = fcos_in_features
        self.fcos_fpn_strides = fcos_fpn_strides
        self.fcos_yield_proposal = fcos_yield_proposal
        self.fcos_head = fcos_head
        self.fcos_outputs = fcos_outputs
        self.in_channels_to_top_module = self.fcos_head.in_channels_to_top_module

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        in_features = cfg.MODEL.RPN.IN_FEATURES
        ret = {
            "in_features": in_features,
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
            "nms_thresh": cfg.MODEL.RPN.NMS_THRESH,
            "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
            "loss_weight": {
                "loss_rpn_cls": cfg.MODEL.RPN.LOSS_WEIGHT,
                "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,
            },
            "anchor_boundary_thresh": cfg.MODEL.RPN.BOUNDARY_THRESH,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS),
            "box_reg_loss_type": cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE,
            "smooth_l1_beta": cfg.MODEL.RPN.SMOOTH_L1_BETA,
        }

        ret["pre_nms_topk"] = (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST)
        ret["post_nms_topk"] = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST)

        ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
        ret["anchor_matcher"] = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        ret["head"] = build_rpn_head(cfg, [input_shape[f] for f in in_features])

        ret["fcos_in_features"] = cfg.MODEL.FCOS.IN_FEATURES
        ret["fcos_fpn_strides"] = cfg.MODEL.FCOS.FPN_STRIDES
        ret["fcos_yield_proposal"] =cfg.MODEL.FCOS.YIELD_PROPOSAL
        ret["fcos_head"] = FCOSHead(cfg, [input_shape[f] for f in cfg.MODEL.FCOS.IN_FEATURES])
        ret["fcos_outputs"] = FCOSOutputs(cfg)
        # ret["num_classes"] = cfg.MODEL.FCOS.NUM_CLASSES

        return ret


    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[Instances] = None,
        branch="supervised",
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        top_module=None,
    ):
        rpn_features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(rpn_features)
        # fcos
        fcos_features = [features[f] for f in self.fcos_in_features]
        locations = self.compute_locations(fcos_features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(rpn_features)
        logits_pred, reg_pred, ctrness_pred, top_feats, bbox_towers = self.fcos_head(
            fcos_features, top_module, self.fcos_yield_proposal
        )


        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(
                x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1]
            )
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        results = {}
        if (self.training and compute_loss) or compute_val_loss:
            if branch == "supervised":
                gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
                losses = self.losses(
                    anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
                )
                losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

                results, fcos_losses = self.fcos_outputs.losses(
                    logits_pred, reg_pred, ctrness_pred,
                    locations, gt_instances, top_feats
                )
                losses.update(fcos_losses)

            elif branch == "unsup_data_weak":
                losses = {}
                results["proposals_fcos"] = self.fcos_outputs.predict_proposals(
                    logits_pred, reg_pred, ctrness_pred,
                    locations, images.image_sizes, top_feats
                )
            else:
                raise NotImplementedError
        else:  # inference
            losses = {}
            results["proposals_fcos"] = self.fcos_outputs.predict_proposals(
                logits_pred, reg_pred, ctrness_pred,
                locations, images.image_sizes, top_feats
            )

        results["proposals_rpn"] = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )

        return results, losses


    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(
                h, w, self.fcos_fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations