# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import List, Tuple, Union,Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import color
import math

from detectron2.layers import cat
from detectron2.structures import Boxes, Instances
from detectron2.config import configurable
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import ProposalNetwork, GeneralizedRCNN
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.structures import ImageList, Instances
from detectron2.structures.masks import PolygonMasks, BitMasks
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from pteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
from pteacher.modeling.roi_heads.mil_roi_heads import MILROIHeadsPseudoLab
from pteacher.modeling.roi_heads.cam_mil_roi_heads import CamMILROIHeadsPseudoLab

def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x


def get_images_color_similarity(images, image_masks, kernel_size, dilation):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    unfolded_weights = unfold_wo_center(
        image_masks[None, None], kernel_size=kernel_size,
        dilation=dilation
    )
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]

    return similarity * unfolded_weights


def add_ground_truth_to_proposals_single_image(
    gt: Instances
) -> Instances:
    """
    Augment `proposals` with `gt`.
    Args:
        Same as `add_ground_truth_to_proposals`, but with gt and proposals
        per image.
    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    gt_boxes = gt.gt_boxes
    device = gt.gt_boxes.device
    # Assign all ground-truth boxes an objectness logit corresponding to
    # P(object) = sigmoid(logit) =~ 1.
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)

    # Concatenating gt_boxes with proposals requires them to have the same fields
    gt_proposal = Instances(gt.image_size)
    gt_proposal.pred_boxes = gt_boxes
    gt_proposal.proposal_boxes = gt_boxes
    gt_proposal.objectness_logits = gt_logits
    # gt_proposal.gt_image_color_similarity = gt.gt_image_color_similarity
    # import pdb
    # pdb.set_trace()
    gt_proposal.scores = torch.ones(len(gt_boxes), device=device)
    gt_proposal.gt_classes = gt.gt_classes
    gt_proposal.pred_classes = gt.gt_classes
    gt_proposal.gt_bitmasks_full, gt_proposal.gt_point_coords, gt_proposal.gt_point_labels = gt.gt_bitmasks_full, gt.gt_point_coords, gt.gt_point_labels
    # gt_proposal.gt_image_color_similarity = gt.gt_image_color_similarity
    # gt_image_color_similarity, git_bitmask_full, gt_point_coords, gt_point_labels
    return gt_proposal

    
@META_ARCH_REGISTRY.register()
class PseudoLabProposalNetwork(ProposalNetwork):
    def forward(self, batched_inputs, branch="supervised"):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None


        if "point_instances" in batched_inputs[0]:
            gt_point_instances = [x["point_instances"].to(self.device) for x in batched_inputs]
        else:
            gt_point_instances = None

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances, gt_point_instances, branch)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):

    @configurable
    def __init__(
        self,
        *,
        boxinst_enabled,
        bottom_pixels_removed,
        pairwise_size,
        pairwise_dilation,
        pairwise_color_thresh,
        mask_out_stride,
        **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__(**kwargs)
        self.boxinst_enabled = boxinst_enabled
        self.bottom_pixels_removed = bottom_pixels_removed
        self.pairwise_size = pairwise_size
        self.pairwise_dilation = pairwise_dilation
        self.pairwise_color_thresh = pairwise_color_thresh
        self.mask_out_stride = mask_out_stride

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)

        # boxinst configs
        ret["boxinst_enabled"] = cfg.MODEL.BOXINST.ENABLED
        ret["bottom_pixels_removed"] = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        ret["pairwise_size"] = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        ret["pairwise_dilation"] = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        ret["pairwise_color_thresh"] = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        #  final mask's resolution is 1/4 of the input image
        ret["mask_out_stride"] = cfg.MODEL.BOXINST.MASK_OUT_STRIDE
        # ret["mask_out_stride"] = 1
        return ret


    def add_bitmasks_from_boxes(self, instances, images, im_h, im_w, branch=''):
        stride = self.mask_out_stride
        start = int(stride // 2)

        assert images.size(2) % stride == 0
        assert images.size(3) % stride == 0

        # downsampled_images = F.avg_pool2d(
        #     images.float(), kernel_size=stride,
        #     stride=stride, padding=0
        # )[:, [2, 1, 0]]
        # image_masks = image_masks[:, start::stride, start::stride]

        for im_i, per_im_gt_inst in enumerate(instances):
            if im_i > 0 and im_i == len(instances) / 2 - 1 and branch=='supervised':
                break
            # images_lab = color.rgb2lab(downsampled_images[im_i].byte().permute(1, 2, 0).cpu().numpy())
            # images_lab = torch.as_tensor(images_lab, device=downsampled_images.device, dtype=torch.float32)
            # images_lab = images_lab.permute(2, 0, 1)[None]
            # images_color_similarity = get_images_color_similarity(
            #     images_lab, image_masks[im_i],
            #     self.pairwise_size, self.pairwise_dilation
            # )

            per_im_boxes = per_im_gt_inst.gt_boxes.tensor
            # per_im_bitmasks = []
            per_im_bitmasks_full = []
            for per_box in per_im_boxes:
                bitmask_full = torch.zeros((im_h, im_w)).to(self.device).float()
                bitmask_full[int(per_box[1]):int(per_box[3] + 1), int(per_box[0]):int(per_box[2] + 1)] = 1.0

                # bitmask = bitmask_full[start::stride, start::stride]

                # assert bitmask.size(0) * stride == im_h
                # assert bitmask.size(1) * stride == im_w

                # per_im_bitmasks.append(bitmask)
                per_im_bitmasks_full.append(bitmask_full)

            # per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
            if len(per_im_bitmasks_full) == 0:
                per_im_gt_inst.gt_bitmasks_full = torch.empty(0, im_h, im_w).to(self.device).float()
                per_im_gt_inst.gt_image_color_similarity = torch.empty(0, 8, int(im_h / self.mask_out_stride), int(im_w / self.mask_out_stride)).to(self.device).float()
            else:
                # per_im_gt_inst.gt_bitmasks_full = BitMasks(torch.stack(per_im_bitmasks_full, dim=0))
                per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
                # per_im_gt_inst.gt_image_color_similarity = torch.cat([
                #     images_color_similarity for _ in range(len(per_im_gt_inst))
                # ], dim=0)

            if branch=='supervised':
                instances[2 * im_i + 1].gt_bitmasks_full = per_im_gt_inst.gt_bitmasks_full
                instances[2 * im_i + 1].gt_image_color_similarity = per_im_gt_inst.gt_image_color_similarity 


    def forward(
            self,
            batched_inputs,
            branch="supervised",
            features=None,
            given_proposals=None,
            val_mode=False,
            mil_img_filter_bg_proposal=False,
            add_ground_truth_to_point_proposals=False,
            add_ss_proposals_to_point_proposals=False,
            tch_features=None, #dict
    ):
        if (not self.training) and (not val_mode):
            # import pdb
            # pdb.set_trace()
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        # import pdb
        # pdb.set_trace()
        # print(branch, images.tensor.size(), images.image_sizes)

        if "instances" in batched_inputs[0]:
            # import pdb
            # pdb.set_trace()
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            if self.boxinst_enabled:
                # if branch=='supervised':
                #     n_imgs = len(batched_inputs)

                # original_images = [x["image_weak"].to(self.device) if "image_weak" in x else x["image"].to(self.device) for x in batched_inputs]
                # original_images = ImageList.from_tensors(original_images, self.backbone.size_divisibility)

                # original_image_masks = [torch.ones_like(x[0], dtype=torch.float32) for x in original_images]

                # mask out the bottom area where the COCO dataset probably has wrong annotations
                # for i in range(len(original_image_masks)):
                #     im_h = batched_inputs[i]["height"]
                #     pixels_removed = int(
                #         self.bottom_pixels_removed *
                #         float(original_images[i].size(1)) / float(im_h)
                #     )
                #     if pixels_removed > 0:
                #         original_image_masks[i][-pixels_removed:, :] = 0
                # original_image_masks = ImageList.from_tensors(
                #     original_image_masks, self.backbone.size_divisibility, pad_value=0.0
                # )

                # if branch != 'unsup_data_weak' and (not hasattr(gt_instances[0], "gt_image_color_similarity") or not hasattr(gt_instances[0], "gt_bitmasks_full")):
                if branch != 'unsup_data_weak' and (not hasattr(gt_instances[0], "gt_bitmasks_full")):
                    # print("branch: ", branch)
                    self.add_bitmasks_from_boxes(
                        gt_instances, images.tensor,
                        images.tensor.size(-2), images.tensor.size(-1)
                    )
        else:
            gt_instances = None

        gt_point_instances = gt_instances

        if  "proposals" in batched_inputs[0]:
            ss_proposals = [x["proposals"].to(self.device) for x in batched_inputs]
        else:
            ss_proposals = None

        if features is None:
            features = self.backbone(images.tensor)

        # if self.boxinst_enabled:
        #     images = original_images 

        if branch == "extract_feat":
            return features, {}, {}, {}
        elif branch == "supervised" or branch=="pseudo_supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
            # # roi_head lower branch
            if isinstance(self.roi_heads, StandardROIHeadsPseudoLab):
                _, detector_losses = self.roi_heads(
                    images, features, proposals_rpn, gt_instances, branch=branch
                )            
            elif isinstance(self.roi_heads,MILROIHeadsPseudoLab) or isinstance(self.roi_heads, CamMILROIHeadsPseudoLab):
                _, detector_losses = self.roi_heads(
                    images, features, proposals_rpn, gt_instances, gt_point_instances, ss_proposals,
                    branch=branch,
                    mil_img_filter_bg_proposal=mil_img_filter_bg_proposal,
                    add_ground_truth_to_point_proposals=add_ground_truth_to_point_proposals,
                    add_ss_proposals_to_point_proposals=add_ss_proposals_to_point_proposals,
                    # tch_features=tch_features,
                )
            else:
                raise NotImplementedError

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak" or "sup_data_weak":
            if branch == "unsup_data_weak":
                # Region proposal network
                proposals_rpn, _ = self.proposal_generator(
                    images, features, None, compute_loss=False
                )
            elif branch == "sup_data_weak":
                # proposals_rpn, _ = self.proposal_generator(images, features, None, compute_loss=False)
                proposals_rpn = [add_ground_truth_to_proposals_single_image(x) for x in gt_instances]
            else:
                raise NotImplementedError

            if isinstance(self.roi_heads, StandardROIHeadsPseudoLab):
                proposals_roih, ROI_predictions = self.roi_heads(
                    images,
                    features,
                    proposals_rpn,
                    targets=None,
                    compute_loss=False,
                    branch=branch,
                )
            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            elif isinstance(self.roi_heads, MILROIHeadsPseudoLab) or isinstance(self.roi_heads, CamMILROIHeadsPseudoLab):
                proposals_roih, ROI_predictions = self.roi_heads(
                    images,
                    features,
                    proposals_rpn,
                    targets=None,
                    point_targets=gt_point_instances,
                    ss_proposals=ss_proposals,
                    compute_loss=False,
                    branch=branch,
                )
            else:
                raise NotImplementedError

            return features, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            if isinstance(self.roi_heads, StandardROIHeadsPseudoLab):
                _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
                )
            elif isinstance(self.roi_heads,MILROIHeadsPseudoLab):
                # roi_head lower branch
                _, detector_losses = self.roi_heads(
                    images,
                    features,
                    proposals_rpn,
                    targets=gt_instances,
                    point_targets=gt_point_instances,
                    ss_proposals=ss_proposals,
                    branch=branch,
                    compute_val_loss=True,
                )
            else:
                raise NotImplementedError

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

    # def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
    #     """
    #     Run inference on the given inputs.
    #     Args:
    #         batched_inputs (list[dict]): same as in :meth:`forward`
    #         detected_instances (None or list[Instances]): if not None, it
    #             contains an `Instances` object per image. The `Instances`
    #             object contains "pred_boxes" and "pred_classes" which are
    #             known boxes in the image.
    #             The inference will then skip the detection of bounding boxes,
    #             and only predict other per-ROI outputs.
    #         do_postprocess (bool): whether to apply post-processing on the outputs.
    #     Returns:
    #         When do_postprocess=True, same as in :meth:`forward`.
    #         Otherwise, a list[Instances] containing raw network outputs.
    #     """
    #     assert not self.training

    #     images = self.preprocess_image(batched_inputs)
    #     features = self.backbone(images.tensor)

    #     if detected_instances is None:
    #         if self.proposal_generator is not None:
    #             proposals, _ = self.proposal_generator(images, features, None)
    #         else:
    #             assert "proposals" in batched_inputs[0]
    #             proposals = [x["proposals"].to(self.device) for x in batched_inputs]

    #         results, _ = self.roi_heads(images, features, proposals, None)
    #     else:
    #         detected_instances = [x.to(self.device) for x in detected_instances]
    #         results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

    #     if do_postprocess:
    #         return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
    #     else:
    #         return results


@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabFCOSRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            # return self.inference(batched_inputs, branch)
            return self.inference_roih(batched_inputs, branch)


        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if "point_instances" in batched_inputs[0]:
            gt_point_instances = [x["point_instances"].to(self.device) for x in batched_inputs]
        else:
            gt_point_instances = None


        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            results, proposal_losses = self.proposal_generator(
                images, features, gt_instances, branch
            )
            # # roi_head lower branch
            _, detector_losses = self.roi_heads(images, features, results["proposals_rpn"], gt_instances, gt_point_instances, branch=branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            results, _ = self.proposal_generator(images, features, None, branch)
            proposals = results["proposals_fcos"]
            proposals_rpn = results["proposals_rpn"]
            processed_proposals_rpn = []

            for results_per_image, input_per_image, image_size in zip(
                    proposals, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_proposals_rpn.append({"proposals": r})

            processed_proposals_rpn = [{"instances": r["proposals"]} for r in processed_proposals_rpn]
            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                point_targets=gt_point_instances,
                compute_loss=False,
                branch=branch,
            )

            return {}, processed_proposals_rpn, proposals_roih, ROI_predictions


    def inference(self, batched_inputs, branch="supervised", detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        results, _ = self.proposal_generator(images, features, None, branch)
        proposals_rpn = results["proposals_rpn"]
        proposals_fcos = results["proposals_fcos"]

        processed_proposals_rpn = []

        for results_per_image, input_per_image, image_size in zip(
            proposals_fcos, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_proposals_rpn.append({"proposals": r})
        processed_proposals_rpn = [{"instances": r["proposals"]} for r in processed_proposals_rpn]

        return processed_proposals_rpn

    def inference_roih(self, batched_inputs, branch="supervised", detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        results, _ = self.proposal_generator(images, features, None, branch)

        results, ROI_predictions = self.roi_heads(
            images,
            features,
            results["proposals_rpn"],
            targets=None,
            point_targets=None,
            compute_loss=False,
            branch=branch,
        )
        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results