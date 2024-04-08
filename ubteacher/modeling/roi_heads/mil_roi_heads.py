# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn

from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.config import configurable
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)

# from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from pteacher.modeling.roi_heads.mil_fast_rcnn import FastRCNNMILFocaltLossOutputLayers

import numpy as np
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals

import inspect
from .point_head import build_point_head
from .shapeprop_head import build_shapeprop_head
# TODO: use cfg
MIL_IMG_FILTER_BG = False

from detectron2.modeling.roi_heads.box_head import FastRCNNConvFCHead

@ROI_HEADS_REGISTRY.register()
class MILROIHeadsPseudoLab(StandardROIHeads):

    @configurable
    def __init__(
        self,
        *,
        point_on: False,
        point_head: Optional[nn.Module] = None,
        shapeprop_on: False,
        shapeprop_head: Optional[nn.Module] = None,
        **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask head.
                None if not using mask head.
            mask_pooler (ROIPooler): pooler to extra region features for mask head
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # import pdb
        # pdb.set_trace()
        self.point_on = point_on
        self.shapeprop_on = shapeprop_on

        if self.point_on:
            self.point_head = point_head
        if self.shapeprop_on:
            self.shapeprop_head = shapeprop_head

    @classmethod
    def from_config(cls, cfg, input_shape):
        # import pdb
        # pdb.set_trace()
        ret = super().from_config(cfg, input_shape)
        if inspect.ismethod(cls._init_point_head):
            # _ = cls._init_point_head(cfg, input_shape)
            ret.update(cls._init_point_head(cfg, input_shape))
        if inspect.ismethod(cls._init_shapeprop_head):
            # _ = cls._init_point_head(cfg, input_shape)
            ret.update(cls._init_shapeprop_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )

        # if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
        #     box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        if cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = FastRCNNMILFocaltLossOutputLayers(cfg, box_head.output_shape)
        else:
            raise ValueError("Unknown ROI head loss.")

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["mask_head"] = build_mask_head(cfg, shape)
        return ret

    @classmethod
    def _init_point_head(self, cfg, input_shape):
        # fmt: off
        # self.mask_point_on                      = cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON
        # if not self.mask_point_on:
        #     return
        # assert cfg.MODEL.ROI_HEADS.NUM_CLASSES == cfg.MODEL.POINT_HEAD.NUM_CLASSES
        point_on                           = cfg.MODEL.POINT_ON
        self.point_in_features             = cfg.MODEL.POINT_HEAD.IN_FEATURES

        # in_channels = int(np.sum([input_shape[f].channels for f in self.point_in_features]))
        # point_head = build_point_head(cfg, ShapeSpec(channels=in_channels, width=1, height=1))

        # return {
        #     "point_on": point_on,
        #     "point_head": point_head,
        # }

        # in_channels = int(np.sum([input_shape[f].channels for f in self.point_in_features]))
        # point_head = build_point_head(cfg, ShapeSpec(channels=in_channels, width=1, height=1))

        return {
            "point_on": point_on,
            # "point_head": point_head,
        }


    @classmethod
    def _init_shapeprop_head(self, cfg, input_shape):
        shapeprop_on = cfg.MODEL.SHAPEPROP_ON
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        # pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        # pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        # sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # shapeprop_pooler = ROIPooler(
        #     output_size=pooler_resolution,
        #     scales=pooler_scales,
        #     sampling_ratio=sampling_ratio,
        #     pooler_type=pooler_type,
        # )
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        shapeprop_head = build_shapeprop_head(
            cfg,
            ShapeSpec(
                channels=in_channels
            ),
        )

        return {
            "shapeprop_on": shapeprop_on,
            "shapeprop_head": shapeprop_head,
        }


    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
            point_targets: Optional[List[Instances]] = None,
            ss_proposals: Optional[List[Instances]] = None,
            compute_loss=True,
            branch="",
            compute_val_loss=False,
            mil_img_filter_bg_proposal=False,
            add_ground_truth_to_point_proposals=False,
            add_ss_proposals_to_point_proposals=False,
    ):

        # self.gt_classes_img_oh = get_image_level_gt(
        #    targets, self.num_classes
        # )
        self.gt_classes_img_oh = get_image_level_gt(
            point_targets, self.num_classes
        )
        del images

        if self.training and compute_loss:  # apply if training loss
            assert targets
            # remove neg point on the box
            self.gt_point_coords = [x.gt_point_coords[:, 0, :] for x in point_targets]
            self.gt_point_classes = [x.gt_classes for x in point_targets]
            # import pdb
            # pdb.set_trace()
            # self.gt_point_coords = [x.gt_point_coords.permute(1, 0, 2).reshape(-1, 2) for x in point_targets]
            # self.gt_point_classes = [torch.cat([x.gt_classes, x.gt_classes], dim=0) for x in point_targets]

            # 1000 --> 512
            # if MIL_IMG_FILTER_BG:
            if mil_img_filter_bg_proposal:
                # point_proposals = self.label_and_sample_point_proposals(
                #    proposals, point_targets, branch=branch)
                raise ValueError()
            else:
                point_proposals = proposals
            if add_ground_truth_to_point_proposals:
                _gt_boxes = [x.gt_boxes for x in point_targets]
                point_proposals = add_ground_truth_to_proposals(_gt_boxes, point_proposals)
            if add_ss_proposals_to_point_proposals:
                # ss_proposals = self.sample_ss_proposals(ss_proposals, point_targets)
                # pdb.set_trace()
                _ss_boxes = []
                for x in ss_proposals:
                    if hasattr(x, "proposal_boxes"):
                        _ss_boxes.append(x.proposal_boxes)
                    else:
                        _ss_boxes.append(Boxes(torch.randn(0,4).to(targets[0].gt_boxes.device)))

                point_proposals = add_ground_truth_to_proposals(_ss_boxes, point_proposals)

            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch)

        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False

            # if MIL_IMG_FILTER_BG:
            if mil_img_filter_bg_proposal:
                # point_proposals = self.label_and_sample_point_proposals(
                #    proposals, point_targets, branch=branch)
                raise ValueError()
            else:
                point_proposals = proposals
            if add_ground_truth_to_point_proposals:
                _gt_boxes = [x.gt_boxes for x in point_targets]
                point_proposals = add_ground_truth_to_proposals(_gt_boxes, point_proposals)

            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch)  # do not apply target on proposals

            self.proposal_append_gt = temp_proposal_append_gt

        del targets

        if (self.training and compute_loss) or compute_val_loss:
            # import pdb
            # pdb.set_trace()
            losses, _ = self._forward_box(
                features,
                proposals,
                point_proposals,
                compute_loss,
                compute_val_loss,
                branch
            )
            # del box_features
            # if branch=="unsup_data_weak":
            #     import pdb
            #     pdb.set_trace()
            # point-supervised Implicit PointRend
            if self.shapeprop_on:
                proposals, shapeprop_loss_dict = self._forward_shapeprop(features, proposals, branch)
                losses.update(shapeprop_loss_dict)

            
            losses.update(self._forward_mask(features, proposals, branch))

            if self.point_on:
                losses.update(self._forward_point(features, point_targets, branch))
            del point_targets
            return proposals, losses
        else:
            point_proposals = None
            pred_instances, predictions = self._forward_box(
                features, proposals, point_proposals, compute_loss, compute_val_loss, branch
            )
            # import pdb
            # pdb.set_trace()
            if self.shapeprop_on:
                pred_instances, _ = self._forward_shapeprop(features, pred_instances, branch)
            # if self.mask_on and branch=='unsup_data_weak':
            if self.mask_on:
                pred_instances = self._forward_mask(features, pred_instances, branch)

            return pred_instances, predictions

    def _forward_box(
            self,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            point_proposals: List[Instances] = None,
            compute_loss: bool = True,
            compute_val_loss: bool = False,
            branch: str = "",
            # gt_classes_img_oh: torch.tensor = None
    ):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)

        del box_features

        if (self.training and compute_loss) or compute_val_loss:  # apply if training loss or val loss

            point_features = self.box_pooler(features, [x.proposal_boxes for x in point_proposals])
            point_features = self.box_head(point_features)
            point_predictions = self.box_predictor(point_features)
            del point_features

            losses = dict()
            cls_reg_losses = self.box_predictor.losses(predictions, proposals, None, None, None)
            losses.update(cls_reg_losses)
            # if (self.mil_image_loss_weight > 0) or (self.mil_inst_loss_weight > 0):
            mil_losses = self.box_predictor.losses(point_predictions, point_proposals, self.gt_classes_img_oh,
                                                self.gt_point_coords, self.gt_point_classes)
            losses.update(mil_losses)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                            proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals, branch=branch)

            return pred_instances, predictions

    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances], branch=""):
        """
        Forward logic of the mask prediction branch.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training and (branch == "supervised" or branch == "pseudo_supervised"):
            # head is only trained on positive proposals.
            if not self.shapeprop_on:
                instances, _ = select_foreground_proposals(instances, self.num_classes)
            # import pdb
            # pdb.set_trace()

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            if branch=='unsup_data_weak':
                boxes = [x.pred_boxes for x in instances]
            else:
                boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}

        return self.mask_head(features, instances, branch)


    def _forward_point(self, features: Dict[str, torch.Tensor], instances: List[Instances], branch=""):
        if self.training and (branch == "supervised" or branch == "pseudo_supervised"):
            # import pdb
            # pdb.set_trace()
            features = [features[f] for f in self.point_in_features]

            features, gt_classes = self.point_head(features, instances, branch)
            # import pdb
            # pdb.set_trace()
            features = self.box_head(features)
            predictions = self.box_predictor(features, branch='point_head')
            point_loss = self.point_head.focal_loss(predictions, gt_classes) / gt_classes.shape[0]
            return {"loss_point": point_loss}


    def _forward_shapeprop(self, features: Dict[str, torch.Tensor], instances: List[Instances], branch=""):
        """
        Forward logic of the mask prediction branch.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.shapeprop_on:
            return {} if self.training else instances

        # if self.training and (branch == "supervised" or branch == "pseudo_supervised"):
            # head is only trained on positive proposals.
            # import pdb
            # pdb.set_trace()
        features = [features[f] for f in self.mask_in_features]
        # if branch=='unsup_data_weak':
        #     boxes = [x.pred_boxes for x in instances]
        # else:
        #     boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
        # if self.mask_pooler is not None:
        #     features = self.mask_pooler(features, boxes)
        # else:
        #     features = {f: features[f] for f in self.mask_in_features}

        return self.shapeprop_head(features, instances, branch)


    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )

            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if targets_per_image.has("gt_reg_loss_weight"):
                if has_gt:
                    gt_reg_loss_weight = targets_per_image.gt_reg_loss_weight[matched_idxs]
                    proposals_per_image.gt_reg_loss_weight = gt_reg_loss_weight[sampled_idxs]
                else:
                    # import pdb
                    # pdb.set_trace()
                    # print("matched_idxs", matched_idxs.size())
                    gt_reg_loss_weight = torch.zeros_like(matched_idxs)
                    proposals_per_image.gt_reg_loss_weight = gt_reg_loss_weight[sampled_idxs]

            
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                            trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar(
            "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
        )
        storage.put_scalar(
            "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
        )

        return proposals_with_gt


    @torch.no_grad()
    def sample_ss_proposals(
            self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ):
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        point_proposals = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            proposal_boxes = proposals_per_image.proposal_boxes.tensor
            proposal_classes = []
            candidate_idxs = []
            for p, gt_class in zip(targets_per_image.gt_point_coords[:, 0, :], targets_per_image.gt_classes):
                _idxs_p = ((p[0] >= proposal_boxes[:, 0]) & (p[0] <= proposal_boxes[:, 2]) & \
                           (p[1] >= proposal_boxes[:, 1]) & (p[1] <= proposal_boxes[:, 3])).nonzero().reshape(-1)
                candidate_idxs.append(_idxs_p)
                proposal_classes.append(gt_class.repeat(len(_idxs_p)))

            candidate_idxs = torch.cat(candidate_idxs).cpu().numpy().tolist()
            proposals_per_image = proposals_per_image[candidate_idxs]
            point_proposals.append(proposals_per_image)

        return point_proposals


    @torch.no_grad()
    def label_and_sample_point_proposals(
            self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ):
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        point_proposals = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0

            # 'proposal_boxes', 'objectness_logits', 'gt_classes', 'gt_boxes', 'gt_point_coords', 'gt_point_labels'
            # step 2. choose pseudo bboxes with provised points
            # proposal_scores_per_img = proposals_per_image.scores
            # gt_bboxes = proposals_per_image.gt_boxes.tensor
            # gt_classes = proposals_per_image.gt_classes
            proposal_boxes = proposals_per_image.proposal_boxes.tensor

            # gt_point_coords = targets_per_image.gt_point_coords
            # gt_point_labels = targets_per_image.gt_point_labels
            # gt_point_classes = targets_per_image.gt_classes

            proposal_classes = []
            candidate_idxs = []
            for p, gt_class in zip(targets_per_image.gt_point_coords[:, 0, :], targets_per_image.gt_classes):
                _idxs_p = ((p[0] >= proposal_boxes[:, 0]) & (p[0] <= proposal_boxes[:, 2]) & \
                           (p[1] >= proposal_boxes[:, 1]) & (p[1] <= proposal_boxes[:, 3])).nonzero().reshape(-1)
                candidate_idxs.append(_idxs_p)
                proposal_classes.append(gt_class.repeat(len(_idxs_p)))

            candidate_idxs = torch.cat(candidate_idxs).cpu().numpy().tolist()
            proposal_classes = torch.cat(proposal_classes)

            _ious = pairwise_iou(proposals_per_image[candidate_idxs].proposal_boxes, proposals_per_image.proposal_boxes)
            candidate_idxs = (_ious.max(dim=0).values > 0.3).nonzero().reshape(-1)
            class_candiate_idxs = _ious.max(dim=0).indices[candidate_idxs]
            candidate_idxs = candidate_idxs.cpu().numpy().tolist()
            proposals_per_image = proposals_per_image[candidate_idxs]

            proposal_classes_list = []
            for idx in class_candiate_idxs:
                proposal_classes_list.append(proposal_classes[idx])

            proposal_classes = torch.stack(proposal_classes_list)
            proposals_per_image.set("proposal_classes", proposal_classes)

            point_proposals.append(proposals_per_image)

        return point_proposals





@torch.no_grad()
def get_image_level_gt(targets, num_classes):
    if targets is None:
        return None, None, None

    gt_classes_img = [torch.unique(t.gt_classes, sorted=True) for t in targets]
    gt_classes_img_int = [gt.to(torch.int64) for gt in gt_classes_img]
    # convert to one-hot
    gt_classes_img_oh = torch.cat(
        [
            torch.zeros(
                (1, num_classes), dtype=torch.float, device=gt_classes_img[0].device
            ).scatter_(1, torch.unsqueeze(gt, dim=0), 1)
            for gt in gt_classes_img_int
        ],
        dim=0,
    )

    # [num_instances, 81]
    return gt_classes_img_oh