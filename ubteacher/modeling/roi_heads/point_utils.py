# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat
from detectron2.layers import cat
from detectron2.utils.events import get_event_storage

def get_point_coords_from_point_annotation(instances):
    """
    Load point coords and their corresponding labels from point annotation.
    Args:
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
        point_labels (Tensor): A tensor of shape (N, P) that contains the labels of P
            sampled points. `point_labels` takes 3 possible values:
            - 0: the point belongs to background
            - 1: the point belongs to the object
            - -1: the point is ignored during training
    """
    point_coords_list = []
    point_labels_list = []
    point_classes_list = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        
        point_coords = instances_per_image.gt_point_coords.to(torch.float32)
        point_labels = instances_per_image.gt_point_labels.to(torch.float32).clone()
        point_classes = instances_per_image.gt_classes.to(torch.float32).clone()
        proposal_boxes_per_image = instances_per_image.proposal_boxes.tensor

        # Convert point coordinate system, ground truth points are in image coord.
        point_coords_wrt_box = get_point_coords_wrt_box(proposal_boxes_per_image, point_coords)

        # Ignore points that are outside predicted boxes.
        point_ignores = (
            (point_coords_wrt_box[:, :, 0] < 0)
            | (point_coords_wrt_box[:, :, 0] > 1)
            | (point_coords_wrt_box[:, :, 1] < 0)
            | (point_coords_wrt_box[:, :, 1] > 1)
        )
        point_labels[point_ignores] = -1

        point_coords_list.append(point_coords_wrt_box)
        point_labels_list.append(point_labels)
        point_classes_list.append(point_classes)
        
    if len(point_coords_list) > 0:
        return (
            cat(point_coords_list, dim=0),
            cat(point_labels_list, dim=0),
            cat(point_classes_list, dim=0),
        )
    else:
        return point_coords_list, point_labels_list, point_classes_list


def get_point_coords_wrt_box(boxes_coords, point_coords):
    """
    Convert image-level absolute coordinates to box-normalized [0, 1] x [0, 1] point cooordinates.
    Args:
        boxes_coords (Tensor): A tensor of shape (R, 4) that contains bounding boxes.
            coordinates.
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            image-normalized coordinates of P sampled points.
    Returns:
        point_coords_wrt_box (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.
    """
    with torch.no_grad():
        point_coords_wrt_box = point_coords.clone()
        point_coords_wrt_box[:, :, 0] -= boxes_coords[:, None, 0]
        point_coords_wrt_box[:, :, 1] -= boxes_coords[:, None, 1]
        point_coords_wrt_box[:, :, 0] = point_coords_wrt_box[:, :, 0] / (
            boxes_coords[:, None, 2] - boxes_coords[:, None, 0]
        )
        point_coords_wrt_box[:, :, 1] = point_coords_wrt_box[:, :, 1] / (
            boxes_coords[:, None, 3] - boxes_coords[:, None, 1]
        )
    return point_coords_wrt_box


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output



def roi_mask_point_loss(mask_logits, instances, point_labels):
    """
    Compute the point-based loss for instance segmentation mask predictions
    given point-wise mask prediction and its corresponding point-wise labels.
    Args:
        mask_logits (Tensor): A tensor of shape (R, C, P) or (R, 1, P) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images, C is the
            number of foreground classes, and P is the number of points sampled for each mask.
            The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1 correspondence with the `mask_logits`. So, i_th
            elememt of the list contains R_i objects and R_1 + ... + R_N is equal to R.
            The ground-truth labels (class, box, mask, ...) associated with each instance are stored
            in fields.
        point_labels (Tensor): A tensor of shape (R, P), where R is the total number of
            predicted masks and P is the number of points for each mask.
            Labels with value of -1 will be ignored.
    Returns:
        point_loss (Tensor): A scalar tensor containing the loss.
    """
    with torch.no_grad():
        cls_agnostic_mask = mask_logits.size(1) == 1
        total_num_masks = mask_logits.size(0)

        gt_classes = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue

            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)

    gt_mask_logits = point_labels
    point_ignores = point_labels == -1
    if gt_mask_logits.shape[0] == 0:
        return mask_logits.sum() * 0

    assert gt_mask_logits.numel() > 0, gt_mask_logits.shape

    if cls_agnostic_mask:
        mask_logits = mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        mask_logits = mask_logits[indices, gt_classes]

    # Log the training accuracy (using gt classes and 0.0 threshold for the logits)
    mask_accurate = (mask_logits > 0.0) == gt_mask_logits.to(dtype=torch.uint8)
    mask_accurate = mask_accurate[~point_ignores]
    mask_accuracy = mask_accurate.nonzero().size(0) / max(mask_accurate.numel(), 1.0)
    get_event_storage().put_scalar("point/accuracy", mask_accuracy)

    point_loss = F.binary_cross_entropy_with_logits(
        mask_logits, gt_mask_logits.to(dtype=torch.float32), weight=~point_ignores, reduction="mean"
    )
    # import pdb
    # pdb.set_trace()
    # print(point_loss)
    return point_loss