import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, NaiveSyncBatchNorm
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from pteacher.layers import DFConv2d, NaiveGroupNorm
from pteacher.utils.comm import compute_locations
from .mil_fcos_outputs import MIL_FCOSOutputs
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
import copy
from detectron2.layers import cat

__all__ = ["MIL_FCOS"]

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
class MIL_FCOS(nn.Module):
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

        self.fcos_outputs = MIL_FCOSOutputs(cfg)
        self.anchor_scale = 3
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.img_mil_loss_weight = cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
        self.ins_mil_loss_weight = cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT

    def forward_head(self, features, top_module=None):
        features = [features[f] for f in self.in_features]
        pred_img_mil_logits, pred_ins_mil_logits, pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers = self.fcos_head(
            features, top_module, self.yield_proposal)
        return pred_img_mil_logits, pred_ins_mil_logits, pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers

    def forward(self, images, features, gt_instances=None, gt_point_instances=None, branch="supervised",
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
        anchors = self.compute_anchors(features)

        img_mil_logits_pred, ins_mil_logits_pred, logits_pred, reg_pred, ctrness_pred, top_feats, bbox_towers = self.fcos_head(
            features, top_module, self.yield_proposal
        )

        results = {}
        if self.yield_proposal:
            results["features"] = {
                f: b for f, b in zip(self.in_features, bbox_towers)
            }

        if self.training:
            if gt_point_instances is not None:
                self.gt_classes_img_oh = get_image_level_gt(gt_point_instances, self.num_classes)
                # remove neg point on the box
                self.gt_point_coords = [x.gt_point_coords[:, 0, :] for x in gt_point_instances]
                self.gt_point_classes = [x.gt_classes for x in gt_point_instances]

            results, losses = self.fcos_outputs.losses(
                ins_mil_logits_pred, logits_pred, reg_pred, ctrness_pred,
                locations, gt_instances, top_feats
            )

            if self.yield_proposal and ((self.img_mil_loss_weight > 0) or (self.ins_mil_loss_weight > 0)):
                N = logits_pred[0].size(0)
                all_proposals = []
                logits_pred = cat([
                    # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
                    x.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for x in logits_pred
                ], dim=0).reshape(N, -1, self.num_classes)

                reg_pred = cat([
                    # Reshape: (N, 4, Hi, Wi) -> (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                    x.permute(0, 2, 3, 1).reshape(-1, 4) * self.fpn_strides[level] for level, x in enumerate(reg_pred)
                ], dim=0).reshape(N, -1, 4)

                ctrness_pred = cat([
                    # Reshape: (N, 4, Hi, Wi) -> (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                    x.permute(0, 2, 3, 1).reshape(-1, 1) for level, x in enumerate(ctrness_pred)
                ], dim=0).reshape(N, -1, 1)

                locations = cat(locations, dim=0)

                if self.img_mil_loss_weight > 0:
                    img_mil_logits_pred = cat([
                        # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
                        x.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for x in img_mil_logits_pred
                    ], dim=0).reshape(N, -1, self.num_classes)
                else:
                    img_mil_logits_pred = None

                if self.ins_mil_loss_weight > 0:
                    ins_mil_logits_pred = cat([
                        # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, 2)
                        x.permute(0, 2, 3, 1).reshape(-1, 2) for x in ins_mil_logits_pred
                    ], dim=0).reshape(N, -1, 2)
                else:
                    ins_mil_logits_pred = None


                for i in range(N):

                    pre_nms_thresh = 0.05
                    pre_nms_topk = 2500
                    per_candidate_inds = logits_pred[i].sigmoid() > pre_nms_thresh
                    per_box_cls = logits_pred[i][per_candidate_inds]

                    per_pre_nms_top_n = per_candidate_inds.reshape(-1).sum(0)
                    per_pre_nms_top_n = per_pre_nms_top_n.clamp(max=pre_nms_topk)

                    per_candidate_nonzeros = per_candidate_inds.nonzero()
                    per_box_loc = per_candidate_nonzeros[:, 0]
                    per_class = per_candidate_nonzeros[:, 1]
                    per_ctr = ctrness_pred[i][per_box_loc].reshape(-1)
                    per_logits_pred = logits_pred[i][per_box_loc]
                    per_box_regression = reg_pred[i][per_box_loc]
                    per_locations = locations[per_box_loc]
                    per_anchors = anchors[per_box_loc]
                    if img_mil_logits_pred is not None:
                        per_img_mil_logits_pred = img_mil_logits_pred[i][per_box_loc]
                    if ins_mil_logits_pred is not None:
                        per_ins_mil_logits_pred = ins_mil_logits_pred[i][per_box_loc]

                    if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                        per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                        per_ctr = per_ctr[top_k_indices]
                        per_class = per_class[top_k_indices]
                        per_logits_pred = per_logits_pred[top_k_indices]
                        per_box_regression = per_box_regression[top_k_indices]
                        per_locations = per_locations[top_k_indices]
                        per_anchors = per_anchors[top_k_indices]
                        if img_mil_logits_pred is not None:
                            per_img_mil_logits_pred = per_img_mil_logits_pred[top_k_indices]
                        if ins_mil_logits_pred is not None:
                            per_ins_mil_logits_pred = per_ins_mil_logits_pred[top_k_indices]

                    with torch.no_grad():
                        detections = torch.stack([
                            per_locations[:, 0] - per_box_regression[:, 0],
                            per_locations[:, 1] - per_box_regression[:, 1],
                            per_locations[:, 0] + per_box_regression[:, 2],
                            per_locations[:, 1] + per_box_regression[:, 3],
                        ], dim=1)

                    instance = Instances((0, 0))
                    instance.logits_pred = per_logits_pred
                    if img_mil_logits_pred is not None:
                        instance.img_mil_logits_pred = per_img_mil_logits_pred
                    if ins_mil_logits_pred is not None:
                        instance.ins_mil_logits_pred = per_ins_mil_logits_pred
                    instance.anchors = per_anchors
                    instance.proposal_boxes = Boxes(detections)
                    instance.score = torch.sqrt((per_box_cls.sigmoid() * per_ctr.sigmoid()))
                    instance.pred_class = per_class

                    all_proposals.append(instance)

                if self.ins_mil_loss_weight > 0:
                    point_proposals = self.label_and_sample_point_proposals(all_proposals, gt_instances, gt_point_instances)
                else:
                    point_proposals = []

                mil_losses = self.fcos_outputs.mil_losses(all_proposals, point_proposals, self.gt_classes_img_oh,
                                                          self.gt_point_coords, self.gt_point_classes)
                losses.update(mil_losses)
            return results, losses
        else:
            if branch == "unsup_data_weak":
                results = self.fcos_outputs.predict_proposals(
                    logits_pred, reg_pred, ctrness_pred,
                    locations, images.image_sizes, top_feats, ins_mil_logits_pred, branch
                )

            else:
                results = self.fcos_outputs.predict_proposals(
                    logits_pred, reg_pred, ctrness_pred,
                    locations, images.image_sizes, top_feats, [], branch
                )

            return results, {}

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

    def compute_anchors(self, features):
        anchors = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(
                h, w, self.fpn_strides[level],
                feature.device
            )

            anchors.append(
                torch.cat(
                    (
                        locations_per_level - self.fpn_strides[level] * self.anchor_scale / 2,
                        locations_per_level + self.fpn_strides[level] * self.anchor_scale / 2,
                    ),
                    dim=1)
            )
        return Boxes(torch.cat(anchors, dim=0))

    @torch.no_grad()
    def label_and_sample_point_proposals(
            self, proposals: List[Instances], gt_instances: List[Instances], gt_point_instances: List[Instances]
    ) -> List[Instances]:
        # box_proposals = []
        point_proposals = []
        gt_boxes = [x.gt_boxes for x in gt_instances]

        for proposals_per_image, box_targets_per_image, point_targets_per_image in zip(proposals, gt_instances, gt_point_instances):
            # 1. box proposals
            proposal_boxes = proposals_per_image.proposal_boxes
            anchors = proposals_per_image.anchors
            gt_boxes = box_targets_per_image.gt_boxes
            # gt_boxes = point_targets_per_image.gt_boxes

            _ious = pairwise_iou(gt_boxes, proposal_boxes)
            # print(gt_boxes.tensor.size(), proposal_boxes.tensor.size(), _ious.size())
            if _ious.size(0) == 0 or _ious.size(1) == 0:
                box_candidate_idxs = []
            else:
                values, indices = _ious.max(dim=0)
                mask = (values > 0.5)
                box_candidate_idxs = mask.nonzero().reshape(-1)
                box_candidate_idxs = box_candidate_idxs.cpu().numpy().tolist()

                tmp_proposal_boxes = proposals_per_image.proposal_boxes.tensor
                tmp_proposal_boxes[box_candidate_idxs] = gt_boxes[indices[mask]].tensor
                proposals_per_image.remove("proposal_boxes")
                proposals_per_image.set("proposal_boxes", Boxes(tmp_proposal_boxes))
                # import pdb
                # pdb.set_trace()

            # 2. anchors
            _ious = pairwise_iou(gt_boxes, anchors)
            # print(gt_boxes.tensor.size(), proposal_boxes.tensor.size(), _ious.size())
            if _ious.size(0) == 0 or _ious.size(1) == 0:
                anchor_candidate_idxs = []
            else:
                values, indices = _ious.max(dim=0)
                mask = (values > 0.5)
                anchor_candidate_idxs = mask.nonzero().reshape(-1)
                anchor_candidate_idxs = anchor_candidate_idxs.cpu().numpy().tolist()

                tmp_proposal_boxes = proposals_per_image.proposal_boxes.tensor
                tmp_proposal_boxes[anchor_candidate_idxs] = gt_boxes[indices[mask]].tensor
                proposals_per_image.remove("proposal_boxes")
                proposals_per_image.set("proposal_boxes", Boxes(tmp_proposal_boxes))
                # proposals_per_image.proposal_boxes[anchor_candidate_idxs].tensor = gt_boxes[indices[mask]].tensor
                # import pdb
                # pdb.set_trace()

            # 3. point proposals
            has_gt = len(point_targets_per_image) > 0
            proposal_boxes = proposals_per_image.proposal_boxes.tensor

            proposal_classes = []
            point_candidate_idxs = []
            for p, gt_class in zip(point_targets_per_image.gt_point_coords[:, 0, :], point_targets_per_image.gt_classes):
                # _idxs_a = ((p[0] >= anchors[:, 0]) & (p[0] <= anchors[:, 2]) & \
                #            (p[1] >= anchors[:, 1]) & (p[1] <= anchors[:, 3])).nonzero().reshape(-1)

                _idxs_p = ((p[0] >= proposal_boxes[:, 0]) & (p[0] <= proposal_boxes[:, 2]) & \
                           (p[1] >= proposal_boxes[:, 1]) & (p[1] <= proposal_boxes[:, 3])).nonzero().reshape(-1)

                # _idxs_p = (((p[0] >= anchors[:, 0]) & (p[0] <= anchors[:, 2]) & \
                #             (p[1] >= anchors[:, 1]) & (p[1] <= anchors[:, 3])) | \
                #            ((p[0] >= proposal_boxes[:, 0]) & (p[0] <= proposal_boxes[:, 2]) & \
                #             (p[1] >= proposal_boxes[:, 1]) & (p[1] <= proposal_boxes[:, 3])) \
                #            ).nonzero().reshape(-1)

                point_candidate_idxs.append(_idxs_p)
                proposal_classes.append(gt_class.repeat(len(_idxs_p)))

            if point_candidate_idxs != []:
                point_candidate_idxs = torch.cat(point_candidate_idxs).cpu().numpy().tolist()
                if len(point_candidate_idxs) > 0:
                    _ious = pairwise_iou(proposals_per_image[point_candidate_idxs].anchors, proposals_per_image.anchors)
                    point_candidate_idxs = (_ious.max(dim=0).values > 0.7).nonzero().reshape(-1)
                    point_candidate_idxs = point_candidate_idxs.cpu().numpy().tolist()

            # candidate_idxs = list(set(box_candidate_idxs + point_candidate_idxs))
            # candidate_idxs = box_candidate_idxs
            proposals_per_image = proposals_per_image[point_candidate_idxs]
            # print(len(point_candidate_idxs))
            point_proposals.append(proposals_per_image)

        return point_proposals


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
                        "ins_mil": (cfg.MODEL.FCOS.NUM_MIL_CONVS,
                                  False),
                        }

        self.img_mil_loss_weight = cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
        self.ins_mil_loss_weight = cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT

        if self.ins_mil_loss_weight == 0:
            head_configs.pop("ins_mil")

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

        if self.img_mil_loss_weight > 0:
            self.img_mil_logits = nn.Conv2d(
                in_channels, self.num_classes,
                kernel_size=3, stride=1,
                padding=1
            )
            for modules in [self.img_mil_logits]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)

        if self.ins_mil_loss_weight > 0:
            self.ins_mil_logits = nn.Conv2d(
                in_channels, 2,
                kernel_size=3, stride=1,
                padding=1
            )
            for modules in [self.ins_mil_logits, self.ins_mil_tower]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)


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
        img_mil_logits = []
        ins_mil_logits = []
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

            # add img-mil logits and ins-mil logits
            if self.img_mil_loss_weight > 0:
                img_mil_logits.append(self.img_mil_logits(cls_tower))

            if self.ins_mil_loss_weight > 0:
                ins_mil_feat = bbox_tower.detach()
                ins_mil_score_attention = True
                if ins_mil_score_attention:
                    score_attention = (logits[l].sigmoid() * ctrness[l].sigmoid()).sqrt().detach()
                    score_attention = torch.max(score_attention, dim=1, keepdim=True)[0]
                    ins_mil_feat = ins_mil_feat * (score_attention + 1.0)

                ins_mil_tower = self.ins_mil_tower(ins_mil_feat)
                ins_mil_logits.append(self.ins_mil_logits(ins_mil_tower))

        return img_mil_logits, ins_mil_logits, logits, bbox_reg, ctrness, top_feats, bbox_towers


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