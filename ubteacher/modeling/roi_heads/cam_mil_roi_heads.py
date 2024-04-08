# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import inspect
from .point_head import build_point_head
from .shapeprop_head import build_shapeprop_head
from typing import Dict, List, Optional, Tuple, Union

from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.config import configurable
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals

# from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

from pteacher.utils.comm import rgb_to_lab, unfold_wo_center, get_images_color_similarity
from pteacher.modeling.roi_heads.mil_fast_rcnn import FastRCNNMILFocaltLossOutputLayers

# TODO: use cfg
MIL_IMG_FILTER_BG = False


def process_cams_batch(cams_batch, instances, cam_res, is_train=True, branch=''):
    """
    Use ground-truth (during training) or predicted class labels (during evaluation) to select the
    CAM slices corresponding to the primary class in the RoI. Afterwards, the CAMs are normalized
    to the range [0, 1] and resized to the desired spatial resolution.
    """
    if is_train and branch != 'unsup_data_weak':
        classes = torch.cat([x.gt_classes for x in instances], dim=0)
    else:
        classes = torch.cat([x.pred_classes for x in instances], dim=0)
    cams_batch = cams_batch[torch.arange(cams_batch.size(0)), classes][:, None, ...]
    return normalize_and_interpolate_batch(cams_batch, cam_res)


def normalize_and_interpolate_batch(cams_batch, cam_res):
    """
    Normalize and resize CAMs.
    """
    cams_batch = normalize_batch(cams_batch)
    return F.interpolate(cams_batch, scale_factor=(cam_res / cams_batch.size(2)), mode='bilinear')


def normalize_batch(cams_batch):
    """
    Classic min-max normalization
    """
    bs = cams_batch.size(0)
    cams_batch = cams_batch + 1e-4
    cam_mins = getattr(cams_batch.view(bs, -1).min(1), 'values').view(bs, 1, 1, 1)
    cam_maxs = getattr(cams_batch.view(bs, -1).max(1), 'values').view(bs, 1, 1, 1)
    return (cams_batch - cam_mins) / (cam_maxs - cam_mins)



@ROI_HEADS_REGISTRY.register()
class CamMILROIHeadsPseudoLab(StandardROIHeads):

    @configurable
    def __init__(
        self,
        *,
        point_on: False,
        point_head: Optional[nn.Module] = None,
        shapeprop_on: False,
        shapeprop_head: Optional[nn.Module] = None,
        cam_res:int = 14,
        boxinst_enabled = False,
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
        self.cam_res = cam_res
        self.boxinst_enabled = boxinst_enabled

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

        ret['cam_res'] = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        ret["boxinst_enabled"] = cfg.MODEL.BOXINST.ENABLED
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
            tch_features: Dict[str, torch.Tensor] = None,
    ):

        # self.gt_classes_img_oh = get_image_level_gt(
        #    targets, self.num_classes
        # )
        self.gt_classes_img_oh = get_image_level_gt(
            point_targets, self.num_classes
        )
        if not self.training:
            del images
            images = None

        if self.training and compute_loss:  # apply if training loss
            assert targets
            # remove neg point on the box
            # self.gt_point_coords = [x.gt_point_coords[:, 0, :] for x in point_targets]
            # self.gt_point_classes = [x.gt_classes for x in point_targets]
            # import pdb
            # pdb.set_trace()
            self.gt_point_coords = [x.gt_point_coords.permute(1, 0, 2).reshape(-1, 2) for x in point_targets]
            self.gt_point_classes = [torch.cat([x.gt_classes, x.gt_classes], dim=0) for x in point_targets]

            # 1000 --> 512
            # if MIL_IMG_FILTER_BG:
            if mil_img_filter_bg_proposal:
                # point_proposals = self.label_and_sample_point_proposals(
                #    proposals, point_targets, branch=branch)
                raise ValueError()
            else:
                point_proposals = proposals
            
            # if branch=="pseudo_supervised":
            #     import pdb
            #     pdb.set_trace()
            # add_ground_truth_to_point_proposals: when branch=='superivised': True, otherwise False
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
            losses, _, box_features = self._forward_box(
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

            losses.update(self._forward_mask(images, features, tch_features, box_features, proposals, branch))

            if self.point_on:
                losses.update(self._forward_point(features, point_targets, branch))
            del point_targets
            return proposals, losses
        else:
            point_proposals = None
            pred_instances, predictions, box_features = self._forward_box(
                features, proposals, point_proposals, compute_loss, compute_val_loss, branch
            )
            # pred_instances, predictions, box_features = self._forward_box(features, proposals, point_proposals, compute_loss, compute_val_loss, branch)
            if self.shapeprop_on:
                pred_instances, _ = self._forward_shapeprop(features, pred_instances, branch)
            # if self.mask_on and branch=='unsup_data_weak':
            if self.mask_on:
                # if branch=="sup_data_weak":
                    # import pdb
                    # pdb.set_trace()
                pred_instances = self._forward_mask(images, features, tch_features, box_features, pred_instances, branch)

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
        box_features_conv, box_features = self.box_head(box_features)
        # box_features = self.box_head(box_features)
        # box_features_conv = None
        # pooled to 1x1
        # import pdb
        # pdb.set_trace()
        predictions = self.box_predictor(box_features)

        if (self.training and compute_loss) or compute_val_loss:  # apply if training loss or val loss

            point_features = self.box_pooler(features, [x.proposal_boxes for x in point_proposals])
            _, point_features = self.box_head(point_features)
            point_predictions = self.box_predictor(point_features)
            # del point_features
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

            return losses, predictions, box_features_conv
        else:
            if branch=="sup_data_weak":
                return proposals, None, box_features_conv
            pred_instances, indices = self.box_predictor.inference(predictions, proposals, branch=branch)
            indices = torch.cat(indices, dim=0)
            return pred_instances, predictions, box_features_conv[indices]

    def _forward_mask(self,
                    images,
                    features: Dict[str, torch.Tensor], 
                    tch_features: Dict[str, torch.Tensor], 
                    box_features: torch.Tensor, 
                    instances: List[Instances], branch=""):
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
            instances, fg_selection_masks = select_foreground_proposals(instances, self.num_classes)
            box_features = box_features[torch.cat(fg_selection_masks, dim=0)]

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            if branch=='unsup_data_weak' or branch=='sup_data_weak':
                boxes = [x.pred_boxes for x in instances]
            else:
                boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]

            # roi_images_color_similarity_list = []
            # for roi_image_rgb in roi_images:
            #     roi_image_lab = rgb_to_lab(roi_image_rgb.byte().permute(1, 2, 0))
            #     roi_image_lab = roi_image_lab.permute(2, 0, 1)
            #     roi_image_color_similarity = get_images_color_similarity(roi_image_lab[None], kernel_size=3, dilation=1)
            #     roi_images_color_similarity_list.append(roi_image_color_similarity)

            # roi_images_color_similarity = cat(roi_images_color_similarity_list, dim=0)
            features = self.mask_pooler(features, boxes)
            if tch_features is not None:
                tch_features = self.mask_pooler(tch_features, boxes)
            
            # print(features.size(), branch, self.training)
            # if branch=='unsup_data_weak' or branch=='sup_data_weak':
            #     print(features.size(), branch, self.training)

            # if self.training and features.size(0) > 0 and (branch=='supervised' or branch=='pseudo_supervised') and self.boxinst_enabled:
            #     roi_images = ROIPooler(output_size=(features.size(2)*2, features.size(3)*2), scales=[1.0], sampling_ratio=0, pooler_type="ROIAlignV2")([images.tensor.float()], boxes)
            #     roi_images_lab = []
            #     for roi_image_rgb in roi_images:
            #         roi_image_lab = rgb_to_lab(roi_image_rgb.byte().permute(1, 2, 0))
            #         roi_image_lab = roi_image_lab.permute(2, 0, 1)
            #         roi_images_lab.append(roi_image_lab)

            #     roi_images_lab = torch.stack(roi_images_lab, dim=0)
            #     roi_images_color_similarity = get_images_color_similarity(roi_images_lab, kernel_size=3, dilation=1)

                # import cv2
                # import os
                # for i in range(roi_images_color_similarity.size(0)):
                #     for j in range(8):
                #         # import pdb
                #         # pdb.set_trace()
                #         if j == 0:
                #             mask = roi_images_color_similarity[i][j].clone()
                #     mask = mask.unsqueeze(0).repeat(3, 1, 1)
                #     img_path = os.path.join('./results/vis',  "gt_{}.jpg".format(i))
                #     vis_img = roi_images[i].clone()
                #     vis_img[mask > 0.3] = 255
                #     vis_img = vis_img.byte().permute(1, 2, 0).cpu().numpy()
                #     cv2.imwrite(img_path, vis_img)

                # import pdb
                # pdb.set_trace()
            # else:
            #     roi_images_color_similarity = None

            # cam_raw_batch = self.calculate_cams(features, box_features, instances, self.training, branch)
            # features = features + cam_raw_batch
        else:
            raise NotImplementedError
            # features = {f: features[f] for f in self.mask_in_features}

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


    def calculate_cams(self, mask_features, box_features, instances, is_train, branch):
        if mask_features.size(0) == 0:
            cams_batch = torch.zeros(0, 1, self.cam_res, self.cam_res, device=mask_features.device)
        else:
            w_cls = self.box_predictor.cls_score.weight.clone().detach()
            # import pdb
            # pdb.set_trace()
            cams_batch = F.conv2d(box_features, weight=w_cls[..., None, None])
            cams_batch = process_cams_batch(cams_batch, instances, self.cam_res, is_train=is_train, branch=branch)
        return cams_batch



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
            # if hasattr(targets_per_image, 'gt_pseudo_scores'):
            #     import pdb
            #     pdb.set_trace()
                # proposals_per_image.scores = targets_per_image.scores[matched_idxs][sampled_idxs]

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

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