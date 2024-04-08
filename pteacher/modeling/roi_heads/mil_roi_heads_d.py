# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
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


@ROI_HEADS_REGISTRY.register()
class MILROIHeadsPseudoLab(StandardROIHeads):
    # @configurable
    # def __init__(self,
    #              *,
    #              point_predictor: nn.Module,
    #              # dynamic_head: nn.Module,
    #              **kwargs):
    #     super().__init__(**kwargs)
    #     # self.register_buffer("class_embed", nn.functional.normalize(torch.randn(80, 256, 7, 7), dim=1))
    #     # self.d_model = 256
    #     # roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)
    #     # self.class_embed = nn.Embedding(80, 256)
    #     self.point_predictor = point_predictor
    #     # self.dynamic_head = dynamic_head

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
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:


        del images

        if self.training and compute_loss:  # apply if training loss
            assert targets
            self.gt_classes_img_oh = get_image_level_gt(point_targets, self.num_classes)
            # remove neg point on the box
            self.gt_point_coords = [x.gt_point_coords[:, 0, :] for x in point_targets]
            self.gt_point_classes = [x.gt_classes for x in point_targets]
            # 1000 --> 512
            point_proposals = self.label_and_sample_point_proposals(
                proposals, targets, point_targets, ss_proposals, branch=branch
            )

            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )
            # import pdb
            # pdb.set_trace()

        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        if (self.training and compute_loss) or compute_val_loss:
            losses, _ = self._forward_box(
                features,
                proposals,
                point_proposals,
                compute_loss,
                compute_val_loss,
                branch
            )
            # point-supervised Implicit PointRend
            losses.update(self._forward_mask(features, proposals, branch))

            return proposals, losses
        else:
            point_proposals = None
            pred_instances, predictions = self._forward_box(
                features, proposals, point_proposals, compute_loss, compute_val_loss, branch
            )

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
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)

        del box_features

        if (
                self.training and compute_loss
        ) or compute_val_loss:  # apply if training loss or val loss

            point_features = self.box_pooler(features, [x.proposal_boxes for x in point_proposals])
            point_features = self.box_head(point_features)
            point_predictions = self.box_predictor(point_features)
            del point_features

            losses = dict()
            cls_reg_losses = self.box_predictor.losses(predictions, proposals, None, None, None)
            mil_losses = self.box_predictor.losses(point_predictions, point_proposals, self.gt_classes_img_oh, self.gt_point_coords, self.gt_point_classes)
            losses.update(cls_reg_losses)
            losses.update(mil_losses)
            # import pdb
            # pdb.set_trace()

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

            pred_instances, _ = self.box_predictor.inference(predictions, proposals, branch)
            return pred_instances, predictions

    def _forward_mask(self, features: Dict[str, torch.Tensor], proposals: List[Instances], branch=""):
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
            return {} if self.training else proposals

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(proposals, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in proposals]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, instances, branch)

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
    def label_and_sample_point_proposals(
            self, proposals: List[Instances], targets: List[Instances], point_targets: List[Instances], ss_proposals: List[Instances], branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        point_proposals = []
        for proposals_per_image, targets_per_image in zip(proposals, point_targets):
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
            # proposal_classes = torch.cat(proposal_classes)

            if len(candidate_idxs) > 0:
                _ious = pairwise_iou(proposals_per_image[candidate_idxs].proposal_boxes, proposals_per_image.proposal_boxes)
                candidate_idxs = (_ious.max(dim=0).values > 0.3).nonzero().reshape(-1)
                # class_candiate_idxs = _ious.max(dim=0).indices[candidate_idxs]
                candidate_idxs = candidate_idxs.cpu().numpy().tolist()
            proposals_per_image = proposals_per_image[candidate_idxs]

            # proposal_classes_list =[]
            # for idx in class_candiate_idxs:
            #     proposal_classes_list.append(proposal_classes[idx])
            #
            # proposal_classes = torch.stack(proposal_classes_list)
            # proposals_per_image.set("proposal_classes", proposal_classes)

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