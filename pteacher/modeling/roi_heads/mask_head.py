from detectron2.modeling.roi_heads.mask_head import MaskRCNNConvUpsampleHead, ROI_MASK_HEAD_REGISTRY
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers.roi_align import ROIAlign
from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.structures.masks import PolygonMasks, BitMasks
from .shapeprop_head.make_layers import make_conv1x1, make_conv3x3, group_norm
from pteacher.modeling.meta_arch.rcnn import unfold_wo_center
from .point_utils import get_point_coords_from_point_annotation, roi_mask_point_loss, point_sample
# from pteacher.utils.comm import compute_project_term

def crop_and_resize_bitmasks(bit_masks: torch.Tensor, boxes: torch.Tensor, mask_size: int, spatial_scale = 1.0) -> torch.Tensor:
    """
    Crop each bitmask by the given box, and resize results to (mask_size, mask_size).
    This can be used to prepare training targets for Mask R-CNN.
    It has less reconstruction error compared to rasterization with polygons.
    However we observe no difference in accuracy,
    but BitMasks requires more memory to store all the masks.
    Args:
        boxes (Tensor): Nx4 tensor storing the boxes for each mask
        mask_size (int): the size of the rasterized mask.
    Returns:
        Tensor:
            A bool tensor of shape (N, mask_size, mask_size), where
            N is the number of predicted boxes for this image.
    """
    assert len(boxes) == len(bit_masks), "{} != {}".format(len(boxes), len(bit_masks))
    device = bit_masks.device

    batch_inds = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]
    rois = torch.cat([batch_inds, boxes], dim=1)  # Nx5

    bit_masks = bit_masks.to(dtype=torch.float32)
    rois = rois.to(device=device)
    output = (
        ROIAlign((mask_size, mask_size), spatial_scale, 0, aligned=True)
            .forward(bit_masks[:, None, :, :], rois)
            .squeeze(1)
    )
    # output = output >= 0.5
    return output


def crop_and_resize_image_color_similarity(image_color_similarity: torch.Tensor, boxes: torch.Tensor, mask_size: int, spatial_scale = 0.25) -> torch.Tensor:
    """
    Crop each bitmask by the given box, and resize results to (mask_size, mask_size).
    This can be used to prepare training targets for Mask R-CNN.
    It has less reconstruction error compared to rasterization with polygons.
    However we observe no difference in accuracy,
    but BitMasks requires more memory to store all the masks.
    Args:
        image_color_similarity (Tensor): Nx8xHxW tensor storing neighbour color similarity
        boxes (Tensor): Nx4 tensor storing the boxes for each mask
        mask_size (int): the size of the rasterized mask.
    Returns:
        Tensor:
            A bool tensor of shape (N, mask_size, mask_size), where
            N is the number of predicted boxes for this image.
    """
    assert len(boxes) == len(image_color_similarity), "{} != {}".format(len(boxes), len(image_color_similarity))
    device = image_color_similarity.device

    batch_inds = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]
    rois = torch.cat([batch_inds, boxes], dim=1)  # Nx5

    image_color_similarity = image_color_similarity.to(dtype=torch.float32)
    rois = rois.to(device=device)
    
    output = (
        ROIAlign((mask_size, mask_size), spatial_scale, 0, aligned=True)
            .forward(image_color_similarity, rois)
            .squeeze(1)
    )
    # output = output >= 0.5
    return output


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def compute_project_term(mask_scores, gt_bitmasks):
    if mask_scores.size(0) == 0:
        return mask_scores.sum() * 0.0
        
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


@torch.jit.unused
def mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances: List[Instances], vis_period: int = 0, branch='', boxinst_enabled=False):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.
    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    if boxinst_enabled and (branch=="supervised" or branch=="pseudo_supervised"):
    # import pdb
    # pdb.set_trace()
    # if boxinst_enabled and branch=="supervised":
        return pred_mask_logits.sum() * 0

    # print(branch)

    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []

    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        if isinstance(instances_per_image.gt_masks, PolygonMasks): 
            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
        elif isinstance(instances_per_image.gt_masks, BitMasks):
            # import pdb
            # pdb.set_trace()
            # print(instances_per_image.gt_masks.tensor.size())
            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            )
        else:
            raise NotImplementedError

        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)
    # import pdb
    # pdb.set_trace()
    # if boxinst_enabled and branch=="supervised":
    #     # mil loss for mask
    #     mask_loss = mil_loss(F.binary_cross_entropy_with_logits, pred_mask_logits, gt_masks)
    #     # return pred_mask_logits.sum() * 0
    # else:
    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    return mask_loss


def mil_loss(loss_func, input, target):
    row_labels = target.max(1)[0]
    column_labels = target.max(2)[0]
    
    row_input = input.max(1)[0]
    column_input = input.max(2)[0]

    loss = loss_func(column_input, column_labels) +\
           loss_func(row_input, row_labels)

    return loss


@ROI_MASK_HEAD_REGISTRY.register()
class PointSupMaskRCNNConvUpsampleHead(MaskRCNNConvUpsampleHead):
    @configurable
    def __init__(self, 
        input_shape: ShapeSpec, *, 
        shapeprop_on=False, 
        boxinst_enabled=False, 
        bottom_pixels_removed=10, 
        pairwise_size=3,
        pairwise_dilation=1,
        pairwise_color_thresh=0.3,
        _warmup_iters=10000,
        mask_out_stride=4,
        **kwargs):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of foreground classes (i.e. background is not
                included). 1 if using class agnostic prediction.
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__(input_shape, **kwargs)
        # import pdb
        # pdb.set_trace()
        self.eps = 1e-5
        self.shapeprop_on = shapeprop_on
        if self.shapeprop_on:
            input_size = 256
            dilation = 1
            self.encoder = nn.Sequential(
                make_conv3x3(1, input_size, dilation=dilation, stride=1, use_gn=False),
                nn.ReLU(True),
                make_conv3x3(input_size, input_size, dilation=dilation, stride=1, use_gn=False),
                # nn.ReLU(True),
                # make_conv3x3(input_size, input_size, dilation=dilation, stride=1, use_gn=False),
                # nn.ReLU(True),
                # make_conv3x3(input_size, input_size, dilation=dilation, stride=1, use_gn=False)
            )
            self.projector = nn.Sequential(
                make_conv1x1(512, input_size, use_gn=False),
                nn.ReLU(True),
            )
        # boxinst configs
        self.boxinst_enabled = boxinst_enabled
        self.bottom_pixels_removed = bottom_pixels_removed
        self.pairwise_size = pairwise_size
        self.pairwise_dilation = pairwise_dilation
        self.pairwise_color_thresh = pairwise_color_thresh
        self._warmup_iters = _warmup_iters
        self.mask_out_stride = mask_out_stride
        self.register_buffer("_iter", torch.zeros([1]))

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        shapeprop_on = cfg.MODEL.SHAPEPROP_ON
        ret.update(
            shapeprop_on=shapeprop_on
        )
        ret["boxinst_enabled"] = cfg.MODEL.BOXINST.ENABLED
        ret["bottom_pixels_removed"] = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        ret["pairwise_size"] = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        ret["pairwise_dilation"] = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        ret["pairwise_color_thresh"] = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        ret["_warmup_iters"] = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS
        ret["mask_out_stride"] = cfg.MODEL.BOXINST.MASK_OUT_STRIDE
        return ret

    def forward(self, x, instances: List[Instances], branch=''):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.
        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        # import pdb
        # pdb.set_trace()
        if self.shapeprop_on:
            shape_activation = cat([v.get_fields()['shape_activation'] for v in instances]).unsqueeze(1)
            # normalize
            shape_activation = F.relu(shape_activation)
            if shape_activation.size(0) > 0:
                norm_factors, _ = shape_activation.view(shape_activation.shape[0], 1, -1).max(2)
                shape_activation /= (norm_factors.unsqueeze(2).unsqueeze(3) + self.eps)
            shape_activation = self.encoder(shape_activation)
            # fuse shape_activation into input
            shape_activation = F.interpolate(shape_activation, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = cat([x, shape_activation], 1)
            x = self.projector(x)

        # x = self.layers(x)
        x = self.mask_fcn1(x)
        x = self.mask_fcn2(x)
        x = self.mask_fcn3(x)
        x = self.mask_fcn4(x)
        x = self.deconv(x)
        x = self.deconv_relu(x)
        x = self.predictor(x)

        if self.training and (branch=='supervised' or branch=='pseudo_supervised'): 
            # if self.boxinst_enabled and branch=='supervised':
            loss = dict()
            if self.boxinst_enabled:
                # if branch=='pseudo_supervised':
                #     import pdb
                #     pdb.set_trace()
                # print(x.size())
                loss.update(self.project_loss(x, instances, branch))
                # Training with point supervision
                # for instances_per_image in instances:
                #     if not hasattr(instances_per_image, 'gt_point_coords') and len(instances_per_image) > 0:
                # point_coords, point_labels, point_classes = get_point_coords_from_point_annotation(instances)

                # if len(point_coords) > 0:
                #     mask_logits = point_sample(
                #         x,
                #         point_coords,
                #         align_corners=False,
                #     )
                #     loss["loss_point"] = roi_mask_point_loss(mask_logits, instances, point_labels)
                # else:
                #     loss["loss_point"] = x.sum() * 0
                # loss["loss_mask"] = mask_rcnn_loss(x, instances, self.vis_period, branch, self.boxinst_enabled) * self.loss_weight
            else:
                loss["loss_mask"] = mask_rcnn_loss(x, instances, self.vis_period, branch, self.boxinst_enabled) * self.loss_weight
            return loss
        else:
            mask_rcnn_inference(x, instances)

            if branch == "keep_pred_mask_logits":
                num_boxes_per_image = [len(i) for i in instances]
                pred_mask_logits = x.split(num_boxes_per_image, dim=0)
                for pred_mask_logits_per_img, instance in zip(pred_mask_logits, instances):
                    instance.pred_mask_logits = pred_mask_logits_per_img

            return instances

    @torch.jit.unused
    def project_loss(self, pred_mask_logits: torch.Tensor, instances: List[Instances], branch=''):

        if pred_mask_logits.size(0) == 0:
            return {
                "loss_project": pred_mask_logits.sum() * 0,
                # "loss_pairwise": pred_mask_logits.sum() * 0,
            }
        cls_agnostic_mask = pred_mask_logits.size(1) == 1
        total_num_masks = pred_mask_logits.size(0)
        mask_side_len = pred_mask_logits.size(2)
        assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

        gt_classes = []
        gt_bitmasks = []
        gt_pseudo_scores = []
        # gt_image_color_similarity = []

        for im_i, instances_per_image in enumerate(instances):
            if len(instances_per_image) == 0:
                continue
            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)

            gt_bitmasks_per_image = crop_and_resize_bitmasks(
                instances_per_image.gt_bitmasks_full, 
                instances_per_image.proposal_boxes.tensor, 
                mask_side_len)
            # roi align
            gt_bitmasks_per_image[gt_bitmasks_per_image < 1.0] = 0.0

            # gt_image_color_similarity_per_image = crop_and_resize_image_color_similarity(
            #     (instances_per_image.gt_image_color_similarity >= self.pairwise_color_thresh),
            #     instances_per_image.proposal_boxes.tensor, 
            #     mask_side_len,
            #     1.0 / self.mask_out_stride)
            gt_bitmasks.append(gt_bitmasks_per_image)

            if hasattr(instances_per_image, 'gt_pseudo_scores'):
                gt_pseudo_scores.append(instances_per_image.gt_pseudo_scores)

        if len(gt_bitmasks) == 0:
            return pred_mask_logits.sum() * 0

        gt_bitmasks = cat(gt_bitmasks, dim=0)
        # gt_image_color_similarity = cat(gt_image_color_similarity, dim=0)

        if cls_agnostic_mask:
            pred_mask_logits = pred_mask_logits[:, 0]
        else:
            indices = torch.arange(total_num_masks)
            gt_classes = cat(gt_classes, dim=0)
            pred_mask_logits = pred_mask_logits[indices, gt_classes].unsqueeze(1)

        gt_bitmasks = gt_bitmasks.to(dtype=torch.float32).unsqueeze(dim=1).float()

        # print("proposal", instances[0].proposal_boxes.tensor)
        # print("gt", instances[0].gt_boxes.tensor)
        # print("gt_bitmasks", gt_bitmasks.min())
        if branch=='pseudo_supervised' and hasattr(instances[0], 'gt_pseudo_scores'):
            gt_pseudo_scores = cat(gt_pseudo_scores, dim=0)
            mask = gt_pseudo_scores >= 0.6 # score threshold for project loss
            pred_mask_logits = pred_mask_logits[mask]
            gt_bitmasks = gt_bitmasks[mask]

        loss_project = compute_project_term(pred_mask_logits, gt_bitmasks)

        # box-supervised BoxInst losses
        # pairwise_losses = compute_pairwise_term(
        #     pred_mask_logits, self.pairwise_size,
        #     self.pairwise_dilation
        # )
        # weights = (roi_images_color_similarity > self.pairwise_color_thresh).float() * gt_bitmasks
        # loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

        # print(branch, loss_project, loss_pairwise)
        return {
                "loss_project": loss_project,
                # "loss_pairwise": loss_pairwise,
            }


    @torch.jit.unused
    def pairwise_loss(self, pred_mask_logits: torch.Tensor, instances: List[Instances], branch=''):

        if pred_mask_logits.size(0) == 0:
            return {
                # "loss_project": pred_mask_logits.sum() * 0,
                "loss_pairwise": pred_mask_logits.sum() * 0,
            }
        cls_agnostic_mask = pred_mask_logits.size(1) == 1
        total_num_masks = pred_mask_logits.size(0)
        mask_side_len = pred_mask_logits.size(2)
        assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

        gt_classes = []
        gt_bitmasks = []
        gt_pseudo_scores = []
        gt_image_color_similarity = []

        for im_i, instances_per_image in enumerate(instances):
            if len(instances_per_image) == 0:
                continue
            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)

            gt_bitmasks_per_image = crop_and_resize_bitmasks(
                instances_per_image.gt_bitmasks_full, 
                instances_per_image.proposal_boxes.tensor, 
                mask_side_len)
            # roi align
            gt_bitmasks_per_image[gt_bitmasks_per_image < 1.0] = 0.0

            gt_image_color_similarity_per_image = crop_and_resize_image_color_similarity(
                (instances_per_image.gt_image_color_similarity >= self.pairwise_color_thresh),
                instances_per_image.proposal_boxes.tensor, 
                mask_side_len,
                1.0 / self.mask_out_stride)
            gt_bitmasks.append(gt_bitmasks_per_image)

            if hasattr(instances_per_image, 'gt_pseudo_scores'):
                gt_pseudo_scores.append(instances_per_image.gt_pseudo_scores)

        # if len(gt_bitmasks) == 0:
        #     return pred_mask_logits.sum() * 0

        gt_bitmasks = cat(gt_bitmasks, dim=0)
        # gt_image_color_similarity = cat(gt_image_color_similarity, dim=0)

        if cls_agnostic_mask:
            pred_mask_logits = pred_mask_logits[:, 0]
        else:
            indices = torch.arange(total_num_masks)
            gt_classes = cat(gt_classes, dim=0)
            pred_mask_logits = pred_mask_logits[indices, gt_classes].unsqueeze(1)

        gt_bitmasks = gt_bitmasks.to(dtype=torch.float32).unsqueeze(dim=1).float()

        # print("proposal", instances[0].proposal_boxes.tensor)
        # print("gt", instances[0].gt_boxes.tensor)
        # print("gt_bitmasks", gt_bitmasks.min())
        if branch=='pseudo_supervised' and hasattr(instances[0], 'gt_pseudo_scores'):
            gt_pseudo_scores = cat(gt_pseudo_scores, dim=0)
            mask = gt_pseudo_scores >= 0.6 # score threshold for project loss
            pred_mask_logits = pred_mask_logits[mask]
            gt_bitmasks = gt_bitmasks[mask]

        # loss_project = compute_project_term(pred_mask_logits, gt_bitmasks)

        # box-supervised BoxInst losses
        pairwise_losses = compute_pairwise_term(
            pred_mask_logits, self.pairwise_size,
            self.pairwise_dilation
        )
        weights = (roi_images_color_similarity > self.pairwise_color_thresh).float() * gt_bitmasks
        loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

        # print(branch, loss_project, loss_pairwise)
        return {
                # "loss_project": loss_project,
                "loss_pairwise": loss_pairwise,
            }