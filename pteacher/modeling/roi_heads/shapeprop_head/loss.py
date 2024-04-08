import torch
from torch.nn import functional as F

# from shapeprop.layers import smooth_l1_loss
# from detectron2.modeling.matcher import Matcher

from .matcher import Matcher
from detectron2.structures.boxes import matched_boxlist_iou
# from shapeprop.structures.boxlist_ops import boxlist_iou
from detectron2.layers import cat
from detectron2.structures.masks import PolygonMasks, BitMasks

# def project_masks_on_boxes(proposals, discretization_size):
#     masks = []
#     M = discretization_size
#     # import pdb
#     # pdb.set_trace()
#     device = proposals.proposal_boxes.device
#     segmentation_masks = proposals.gt_masks
#     #.convert("xyxy")
#     proposals = proposals.proposal_boxes
#     assert segmentation_masks.size == proposals.size, "{}, {}".format(
#         segmentation_masks, proposals
#     )

#     proposals = proposals.bbox.to(torch.device("cpu"))
#     for segmentation_mask, proposal in zip(segmentation_masks, proposals):
#         cropped_mask = segmentation_mask.crop(proposal)
#         scaled_mask = cropped_mask.resize((M, M))
#         mask = scaled_mask.get_mask_tensor()
#         masks.append(mask)
#     if len(masks) == 0:
#         return torch.empty(0, dtype=torch.float32, device=device)
#     return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class PropLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        self.bg_label = 80 #hack coco

    # def match_targets_to_proposals(self, proposal, target):
    #     match_quality_matrix = matched_boxlist_iou(target, proposal)
    #     matched_idxs = self.proposal_matcher(match_quality_matrix)
    #     target = target.copy_with_fields(["labels", "masks", "valid_masks"])
    #     matched_targets = target[matched_idxs.clamp(min=0)]
    #     matched_targets.add_field("matched_idxs", matched_idxs)
    #     return matched_targets

    # def prepare_targets(self, proposals):
    #     labels = []
    #     masks = []
    #     valid_inds = []
    #     for proposals_per_image in proposals:
        # for proposals_per_image, proposals_per_image in zip(proposals, proposals):
            # matched_targets = self.match_targets_to_proposals(
            #     proposals_per_image, proposals_per_image
            # )
            # matched_idxs = matched_targets.get_field("matched_idxs")
            # neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            # labels_per_image[neg_inds] = 0

            # valid_masks = (labels_per_image > 0) & matched_targets.get_field("valid_masks").to(dtype=torch.uint8)
            # positive_inds = torch.nonzero(valid_masks).squeeze(1)

            # segmentation_masks = matched_targets.get_field("masks")
            # segmentation_masks = segmentation_masks[positive_inds]

            # positive_proposals = proposals_per_image[positive_inds]


            # labels_per_image = matched_targets.get_field("labels")
            # labels_per_image = labels_per_image.to(dtype=torch.int64)

            # masks_per_image = project_masks_on_boxes(
            #     proposals_per_image, self.discretization_size
            # )

            # labels.append(labels_per_image)
            # masks.append(masks_per_image)
            # valid_inds.append(valid_masks)

        # return labels, masks, valid_inds

    def __call__(self, proposals, pred_mask_logits, branch=''):
        # _, prop_targets, valid_inds = self.prepare_targets(proposals)
        
        # prop_targets = cat(prop_targets, dim=0)
        # valid_inds = cat(valid_inds, dim=0)
        
        # focus on instances that have mask annotation
        # positive_inds = torch.nonzero(valid_inds).squeeze(1)
        # if branch=='pseudo_supervised':
        #     import pdb
        #     pdb.set_trace()
        if branch=='pseudo_supervised':
            return pred_mask_logits.sum() * 0

        gt_masks = []
        mask_side_len = pred_mask_logits.size(2)
        for proposals_per_image in proposals:
            if len(proposals_per_image) == 0:
                continue
            if isinstance(proposals_per_image.gt_masks, PolygonMasks): 
                gt_masks_per_image = proposals_per_image.gt_masks.crop_and_resize(
                    proposals_per_image.proposal_boxes.tensor, mask_side_len
                ).to(device=pred_mask_logits.device)
            elif isinstance(proposals_per_image.gt_masks, BitMasks):
                # import pdb
                # pdb.set_trace()
                # print(instances_per_image.gt_masks.tensor.size())
                gt_masks_per_image = proposals_per_image.gt_masks.crop_and_resize(
                    proposals_per_image.proposal_boxes.tensor, mask_side_len
                )
            else:
                raise NotImplementedError

            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_masks.append(gt_masks_per_image)

        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0

        gt_masks = cat(gt_masks, dim=0).to(dtype=torch.float32)
        prop_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
        return prop_loss


def make_propagating_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0],
        cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0],
        allow_low_quality_matches=False,
    )

    loss_evaluator = PropLossComputation(
        matcher, cfg.MODEL.ROI_SHAPEPROP_HEAD.POOLER_RESOLUTION
    )

    return loss_evaluator
