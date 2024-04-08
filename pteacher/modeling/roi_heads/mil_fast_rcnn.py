# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    FastRCNNOutputs,
    fast_rcnn_inference
)
from detectron2.layers import Linear, cat
from detectron2.structures import Boxes, Instances
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from fvcore.nn import giou_loss, smooth_l1_loss
from torch.cuda.amp import autocast

# image-wise mil loss + instance-wise mil loss + focal loss
class FastRCNNMILFocaltLossOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super(FastRCNNMILFocaltLossOutputLayers, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # self.det_score = Linear(input_size, self.num_classes)
        # self.pss_score = Linear(input_size, 2)

        # mean = 0.0
        # std = 0.001
        # nn.init.normal_(self.det_score.weight, mean, std)
        # nn.init.normal_(self.pss_score.weight, mean, std)
        # for l in [self.det_score, self.pss_score]:
        #    nn.init.constant_(l.bias, 0)

        self.mil_image_loss_weight = cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
        self.mil_inst_loss_weight = cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT

        if self.mil_image_loss_weight > 0:
            self.det_score = Linear(input_size, self.num_classes)
            nn.init.normal_(self.det_score.weight, 0., std=0.001)
            nn.init.constant_(self.det_score.bias, 0)

        if self.mil_inst_loss_weight > 0:
            self.pss_score = Linear(input_size, 2)
            nn.init.normal_(self.pss_score.weight, 0., std=0.001)
            nn.init.constant_(self.pss_score.bias, 0)

    def forward(self, x, branch=''):
        """
        Returns:
            Tensor: shape (N,K+1), scores for each of the N box. Each row contains the scores for
                K object categories and 1 background class.
            Tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4), or (N,4)
                for class-agnostic regression.
        """

        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        # print(x.size())
        # import pdb
        # pdb.set_trace()
        cls_scores = self.cls_score(x)

        if branch=='point_head':
            return cls_scores
            
        proposal_deltas = self.bbox_pred(x)
        pss_scores = self.pss_score(x) if self.mil_inst_loss_weight > 0 else None
        if self.mil_image_loss_weight > 0:
            det_scores = self.det_score(x)
            mil_scores = F.softmax(cls_scores, dim=1)[:, :self.num_classes] * F.softmax(det_scores, dim=0)
        else:
            mil_scores = None
        return mil_scores, pss_scores, cls_scores, proposal_deltas

    def losses(self, predictions, proposals, gt_classes_img_oh, gt_point_coords, gt_point_classes):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        mil_scores, pss_scores, cls_scores, proposal_deltas = predictions
        losses = FastRCNNMILFocalLoss(
            self.box2box_transform,
            mil_scores,
            pss_scores,
            cls_scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            num_classes=self.num_classes,
            gt_classes_img_oh=gt_classes_img_oh,
            gt_point_coords=gt_point_coords,
            gt_point_classes=gt_point_classes,
            mil_image_loss_weight=self.mil_image_loss_weight,
            mil_inst_loss_weight=self.mil_inst_loss_weight
        ).losses()

        return losses

    def inference(self, predictions, proposals, branch=""):
        """
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        mil_scores, pss_scores, cls_scores, proposal_deltas = predictions
        boxes = self.predict_boxes((cls_scores, proposal_deltas), proposals)
        scores = self.predict_probs((cls_scores, proposal_deltas), proposals)
        image_shapes = [x.image_size for x in proposals]

        if branch == 'unsup_data_weak':
            if self.mil_inst_loss_weight > 0:
                num_preds_per_image = [len(p) for p in proposals]
                pss_scores = F.softmax(pss_scores, dim=-1)
                pss_scores_list = pss_scores.split(num_preds_per_image, dim=0)

                # weight cls_score with pss_score for pseudo box selection in hungarian
                scores = [(s * p[..., [1]]).sqrt() for s, p in zip(scores, pss_scores_list)]

            return fast_rcnn_inference(
                boxes,
                scores,
                image_shapes,
                0.001,
                0.9,
                1000,
            )

        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )


class FastRCNNMILFocalLoss(FastRCNNOutputs):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
            self,
            box2box_transform,
            pred_mil_logits,
            pred_pss_logits,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta=0.0,
            box_reg_loss_type="smooth_l1",
            num_classes=80,
            mean_loss=True,
            gt_classes_img_oh=None,
            gt_point_coords=None,
            gt_point_classes=None,
            mil_image_loss_weight=1.0,
            mil_inst_loss_weight=1.0
    ):
        super(FastRCNNMILFocalLoss, self).__init__(
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta,
            box_reg_loss_type,
        )
        self.num_classes = num_classes
        self.pred_mil_logits = pred_mil_logits
        self.pred_pss_logits = pred_pss_logits
        self.mean_loss = mean_loss
        self.gt_classes_img_oh = gt_classes_img_oh
        self.gt_point_coords = gt_point_coords
        self.gt_point_classes = gt_point_classes

        if proposals[0].has("proposal_classes"):
            self.proposal_class_labels = [p.proposal_classes for p in proposals]

        # if proposals[0].has("gt_reg_loss_weight"):
        #     self.gt_reg_loss_weight = cat([p.gt_reg_loss_weight for p in proposals], dim=0)
        # else:
        self.gt_reg_loss_weight = None

        self.mil_image_loss_weight = mil_image_loss_weight
        self.mil_inst_loss_weight = mil_inst_loss_weight

    def predict_probs_img(self):
        if not hasattr(self, "pred_class_img_logits"):
            if len(self.num_preds_per_image) == 1:
                self.pred_class_img_logits = torch.sum(self.pred_mil_logits, dim=0, keepdim=True)
            else:
                self.pred_class_img_logits = cat(
                    [
                        torch.sum(xx, dim=0, keepdim=True)
                        for xx in self.pred_mil_logits.split(self.num_preds_per_image, dim=0)
                    ],
                    dim=0,
                )
            self.pred_class_img_logits = torch.clamp(
                self.pred_class_img_logits, min=1e-6, max=1.0 - 1e-6
            )
        return self.pred_class_img_logits

    def binary_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        # self._log_accuracy()
        reduction = "mean" if self.mean_loss else "sum"
        return F.binary_cross_entropy(
            self.predict_probs_img(), self.gt_classes_img_oh, reduction=reduction
        ) / self.gt_classes_img_oh.size(0)

    def instance_mil_loss(self):

        pred_class_logits_list = F.log_softmax(self.pred_class_logits, dim=1).split(self.num_preds_per_image, dim=0)
        assert self.pred_pss_logits.size(1) == 2
        pred_pss_logits_list = F.log_softmax(self.pred_pss_logits, dim=1).split(self.num_preds_per_image, dim=0)
        proposal_boxes_list = self.proposals.tensor.split(self.num_preds_per_image, dim=0)

        loss_pss = []
        for gt_point_coords_per_img, gt_point_classes_per_img, proposal_boxes_per_img, pred_class_logits_per_img, pred_pss_logits_per_img in zip(
                self.gt_point_coords, self.gt_point_classes, proposal_boxes_list, pred_class_logits_list,
                pred_pss_logits_list):

            for point, point_label in zip(gt_point_coords_per_img, gt_point_classes_per_img):
                try:
                    inside_idxs_p = ((point[0] >= proposal_boxes_per_img[:, 0]) & (
                                point[0] <= proposal_boxes_per_img[:, 2]) & \
                                     (point[1] >= proposal_boxes_per_img[:, 1]) & (
                                                 point[1] <= proposal_boxes_per_img[:, 3])).nonzero().reshape(-1)

                    score_p = pred_class_logits_per_img[inside_idxs_p]
                    pss_p = pred_pss_logits_per_img[inside_idxs_p]

                    if len(inside_idxs_p) == 0:
                        print('point without proposals')
                    else:
                        log_fg_prob = pss_p[:, 1]
                        log_fg_prob += score_p[:, point_label].detach()
                        log_bg_prob = pss_p[:, 0]

                        eye = torch.eye(len(log_fg_prob), dtype=torch.float32, device=log_fg_prob.device)
                        log_prob_prod = (eye * log_fg_prob[None, :] + (1 - eye) * log_bg_prob[None, :]).sum(dim=-1)
                        max_ = log_prob_prod.max()
                        final_log_prob = torch.log(torch.exp(log_prob_prod - max_).sum()) + max_
                        loss_pss.append(-final_log_prob)
                except:
                    # print('point: {}'.format(point))
                    # print('proposal_boxes_per_img: {}'.format(proposal_boxes_per_img))
                    continue

        if len(loss_pss) == 0:
            # loss_pss.append(self.pred_pss_logits.sum() * 0.)
            loss_pss.append(torch.where(torch.isnan(self.pred_pss_logits), torch.zeros_like(self.pred_pss_logits),
                                        self.pred_pss_logits) * 0.)
        loss_pss = torch.stack(loss_pss).mean()
        return loss_pss

    def box_reg_loss(self):
        """
        Deprecated
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()

        box_dim = self.proposals.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1
        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds should produce a valid loss of zero because reduction=sum.

        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]

        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * self.gt_classes[fg_inds, None] + torch.arange(
                box_dim, device=device
            )

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="none",
            )
            if self.gt_reg_loss_weight is not None:
                loss_box_reg = (loss_box_reg * self.gt_reg_loss_weight[fg_inds].unsqueeze(1)).sum()
            else:
                loss_box_reg = loss_box_reg.sum()
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg


    def losses(self):
        if (self.mil_image_loss_weight == 0) and (self.mil_inst_loss_weight == 0) and self.gt_classes_img_oh is not None:
            return {}
        elif self.gt_classes_img_oh is not None and ((self.mil_image_loss_weight > 0) or (self.mil_inst_loss_weight > 0)):
            loss = {}
            if self.mil_image_loss_weight > 0:
                with autocast(enabled=False):
                    loss['loss_img_mil'] = self.binary_cross_entropy_loss()
            if self.mil_inst_loss_weight > 0:
                loss['loss_ins_mil'] = self.instance_mil_loss()

            # return {
            #    "loss_img_mil": self.binary_cross_entropy_loss(),
            #    "loss_ins_mil": self.instance_mil_loss(),
            # }
            return loss
        else:
            return {
                "loss_cls": self.comput_focal_loss(),
                "loss_box_reg": self.box_reg_loss(),
            }

    def comput_focal_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            FC_loss = FocalLoss(
                gamma=1.5,
                num_classes=self.num_classes,
            )
            total_loss = FC_loss(input=self.pred_class_logits, target=self.gt_classes)
            total_loss = total_loss / self.gt_classes.shape[0]

            return total_loss


class FocalLoss(nn.Module):
    def __init__(
            self,
            weight=None,
            gamma=1.0,
            num_classes=80,
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

    def forward(self, input, target):
        # focal loss
        CE = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        return loss.sum()



