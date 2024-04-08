import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, NaiveSyncBatchNorm
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from pteacher.layers import DFConv2d, NaiveGroupNorm
from pteacher.utils.comm import compute_locations
from .mil_fcos_outputs_v3 import MIL_FCOSOutputs_v3
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
import copy
from detectron2.layers import cat
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.proposal_generator.rpn import build_rpn_head
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.sampling import subsample_labels
from detectron2.utils.memory import retry_if_cuda_oom
from typing import List, Tuple
from detectron2.modeling.proposal_generator.proposal_utils import find_top_rpn_proposals
from fvcore.nn import giou_loss, smooth_l1_loss
from detectron2.utils.events import get_event_storage

__all__ = ["MIL_FCOS_v3"]

INF = 100000000


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


@PROPOSAL_GENERATOR_REGISTRY.register()
class MIL_FCOS_v3(nn.Module):
    """
    Implement FCOS (https://arxiv.org/abs/1904.01355).
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.yield_proposal = cfg.MODEL.FCOS.YIELD_PROPOSAL

        self.fcos_head = MIL_FCOSHead(cfg, [input_shape[f] for f in self.in_features])
        self.in_channels_to_top_module = self.fcos_head.in_channels_to_top_module

        self.fcos_outputs = MIL_FCOSOutputs_v3(cfg)
        self.anchor_scale = 3
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES

        # rpn
        self.rpn_in_features = cfg.MODEL.RPN.IN_FEATURES
        self.rpn_anchor_generator = build_anchor_generator(cfg, [input_shape[f] for f in self.rpn_in_features])
        self.rpn_anchor_matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        self.rpn_head = build_rpn_head(cfg, [input_shape[f] for f in self.rpn_in_features])

        self.rpn_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
        self.rpn_nms_thresh = cfg.MODEL.RPN.NMS_THRESH
        self.rpn_batch_size_per_image = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.rpn_positive_fraction = cfg.MODEL.RPN.POSITIVE_FRACTION
        self.rpn_loss_weight = {
            "loss_rpn_cls": cfg.MODEL.RPN.LOSS_WEIGHT,
            "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,
        }
        self.rpn_anchor_boundary_thresh = cfg.MODEL.RPN.BOUNDARY_THRESH
        self.rpn_box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.rpn_box_reg_loss_type = cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE
        self.rpn_smooth_l1_beta = cfg.MODEL.RPN.SMOOTH_L1_BETA

        self.rpn_pre_nms_topk = (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST)
        self.rpn_post_nms_topk = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST)


        # MIL
        self.img_mil_loss_weight = cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
        self.ins_mil_loss_weight = cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT

    def forward_fcos_head(self, features, top_module=None):
        features = [features[f] for f in self.in_features]
        pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers = self.fcos_head(
            features, top_module, self.yield_proposal)
        return pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers

    def forward(self, images, features, gt_instances=None, branch="supervised",
                top_module=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        features = [features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        # anchors = self.compute_anchors(features)

        logits_pred, reg_pred, ctrness_pred, top_feats, bbox_towers = self.fcos_head(
            features, top_module, self.yield_proposal
        )


        anchors = self.rpn_anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.rpn_anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        results = {}

        if self.training:
            if branch == "supervised":

                gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
                losses = self.rpn_losses(
                    anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
                )
                rpn_losses = {k: v * self.rpn_loss_weight.get(k, 1.0) for k, v in losses.items()}



                results, losses = self.fcos_outputs.losses(
                    logits_pred, reg_pred, ctrness_pred,
                    locations, gt_instances, top_feats
                )

                losses.update(rpn_losses)

                results["proposals_rpn"] = self.predict_rpn_proposals(
                anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
            )

                return results, losses
            elif branch == "unsup_data_weak":
                results["proposals"] = self.fcos_outputs.predict_proposals(
                    logits_pred, reg_pred, ctrness_pred,
                    locations, images.image_sizes, top_feats, branch
                )
                results["proposals_rpn"] = self.predict_rpn_proposals(
                anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
            )
                return results, []
            else:
                raise NotImplementedError
        else:
            # if branch == "unsup_data_weak":
            results = self.fcos_outputs.predict_proposals(
                logits_pred, reg_pred, ctrness_pred,
                locations, images.image_sizes, top_feats, branch
            )

            proposals_rpn = self.predict_rpn_proposals(
                anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
            )

            # proposals_rpn = self.fcos_outputs.predict_rpn_proposals(locations, ctrness_pred, reg_pred, images.image_sizes)

            return results, proposals_rpn

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def _subsample_labels(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """

        pos_idx, neg_idx = subsample_labels(
            label, self.rpn_batch_size_per_image, self.rpn_positive_fraction, 0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.no_grad()
    def label_and_sample_anchors(self, anchors: List[Boxes], gt_instances: List[Instances]):
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
        """
        anchors = Boxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """

            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.rpn_anchor_matcher)(match_quality_matrix)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.rpn_anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1

            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes

    def rpn_losses(
        self,
        anchors,
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
    ):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        if self.rpn_box_reg_loss_type == "smooth_l1":
            anchors = type(anchors[0]).cat(anchors).tensor  # Ax(4 or 5)
            gt_anchor_deltas = [self.rpn_box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
            gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, sum(Hi*Wi*Ai), 4 or 5)
            localization_loss = smooth_l1_loss(
                cat(pred_anchor_deltas, dim=1)[pos_mask],
                gt_anchor_deltas[pos_mask],
                self.rpn_smooth_l1_beta,
                reduction="sum",
            )
        elif self.rpn_box_reg_loss_type == "giou":
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            pred_proposals = cat(pred_proposals, dim=1)
            pred_proposals = pred_proposals.view(-1, pred_proposals.shape[-1])
            pos_mask = pos_mask.view(-1)
            localization_loss = giou_loss(
                pred_proposals[pos_mask], cat(gt_boxes)[pos_mask], reduction="sum"
            )
        else:
            raise ValueError(f"Invalid rpn box reg loss type '{self.box_reg_loss_type}'")

        valid_mask = gt_labels >= 0
        objectness_loss = F.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction="sum",
        )
        normalizer = self.rpn_batch_size_per_image * num_images
        return {
            "loss_rpn_cls": objectness_loss / normalizer,
            "loss_rpn_loc": localization_loss / normalizer,
        }


    @torch.no_grad()
    def predict_rpn_proposals(
        self,
        anchors,
        pred_objectness_logits: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.

        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
                stores post_nms_topk object proposals for image i, sorted by their
                objectness score in descending order.
        """
        # The proposals are treated as fixed for approximate joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
        # are also network responses, so is approximate.
        pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
        return find_top_rpn_proposals(
            pred_proposals,
            pred_objectness_logits,
            image_sizes,
            self.rpn_nms_thresh,
            self.rpn_pre_nms_topk[self.training],
            self.rpn_post_nms_topk[self.training],
            self.rpn_min_box_size,
            self.training,
        )

    def _decode_proposals(self, anchors, pred_anchor_deltas: List[torch.Tensor]):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B)
        """
        N = pred_anchor_deltas[0].shape[0]
        proposals = []
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.tensor.size(1)
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            # Expand anchors to shape (N*Hi*Wi*A, B)
            anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            proposals_i = self.rpn_box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals




class MIL_FCOSHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: Implement the sigmoid version first.
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        head_configs = {"cls": (cfg.MODEL.FCOS.NUM_CLS_CONVS,
                                cfg.MODEL.FCOS.USE_DEFORMABLE),
                        "bbox": (cfg.MODEL.FCOS.NUM_BOX_CONVS,
                                 cfg.MODEL.FCOS.USE_DEFORMABLE),
                        "share": (cfg.MODEL.FCOS.NUM_SHARE_CONVS,
                                  False),
                        }

        self.img_mil_loss_weight = cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
        self.ins_mil_loss_weight = cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT


        norm = None if cfg.MODEL.FCOS.NORM == "none" else cfg.MODEL.FCOS.NORM
        self.num_levels = len(input_shape)

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        self.in_channels_to_top_module = in_channels

        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            for i in range(num_convs):
                if use_deformable and i == num_convs - 1:
                    conv_func = DFConv2d
                else:
                    conv_func = nn.Conv2d
                tower.append(conv_func(
                    in_channels, in_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=True
                ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, in_channels))
                elif norm == "NaiveGN":
                    tower.append(NaiveGroupNorm(32, in_channels))
                elif norm == "BN":
                    tower.append(ModuleListDial([
                        nn.BatchNorm2d(in_channels) for _ in range(self.num_levels)
                    ]))
                elif norm == "SyncBN":
                    tower.append(ModuleListDial([
                        NaiveSyncBatchNorm(in_channels) for _ in range(self.num_levels)
                    ]))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes,
            kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3,
            stride=1, padding=1
        )
        self.ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3,
            stride=1, padding=1
        )

        if cfg.MODEL.FCOS.USE_SCALE:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(self.num_levels)])
        else:
            self.scales = None

        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower, self.cls_logits,
            self.bbox_pred, self.ctrness
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x, top_module=None, yield_bbox_towers=False):
        logits = []
        bbox_reg = []
        ctrness = []
        top_feats = []
        bbox_towers = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)

            logits.append(self.cls_logits(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)

            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            bbox_reg.append(F.relu(reg))
            if top_module is not None:
                top_feats.append(top_module(bbox_tower))


        return logits, bbox_reg, ctrness, top_feats, bbox_towers


