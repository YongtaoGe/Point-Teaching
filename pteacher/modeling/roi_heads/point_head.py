# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from typing import Dict, List
from detectron2.structures import Boxes, ImageList, Instances
# from pteacher.layers import MSDeformAttn
from pteacher.modeling.deformable_transformer import (
    PeseudoLabDeformableTransformerEncoderLayer,
    DeformableTransformerEncoderLayer, 
    DeformableTransformerDecoderLayer, 
    DeformableTransformerEncoder,
    DeformableTransformerDecoder, 
    MSDeformAttn
    )
from pteacher.modeling.roi_heads.mil_fast_rcnn import FocalLoss
from torch.nn.init import xavier_uniform_, constant_, normal_
from .position_encoding import build_position_encoding

POINT_HEAD_REGISTRY = Registry("POINT_HEAD")
POINT_HEAD_REGISTRY.__doc__ = """
Registry for point heads, which makes prediction for a given set of per-point features.
The registered object will be called with `obj(cfg, input_shape)`.
"""

def _as_tensor(x):
    """
    An equivalent of `torch.as_tensor`, but works under tracing.
    """
    if isinstance(x, (list, tuple)) and all([isinstance(t, torch.Tensor) for t in x]):
        return torch.stack(x)
    return torch.as_tensor(x)


def cross_entropy(input, target, *, reduction="mean", **kwargs):
    """
    Same as `torch.nn.functional.cross_entropy`, but returns 0 (instead of nan)
    for empty inputs.
    """
    if target.numel() == 0 and reduction == "mean":
        return input.sum() * 0.0  # connect the gradient
    return F.cross_entropy(input, target, **kwargs)


@POINT_HEAD_REGISTRY.register()
class PointHeadPseudoLab(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained and coarse prediction features as its input.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        super(PointHeadPseudoLab, self).__init__()
        # fmt: off
        num_classes                 = cfg.MODEL.POINT_HEAD.NUM_CLASSES
        
        num_fc                      = cfg.MODEL.POINT_HEAD.NUM_FC
        self.use_deform_attn        = cfg.MODEL.POINT_HEAD.USE_DEFORM_ATTN
        self.num_feature_levels     = cfg.MODEL.POINT_HEAD.NUM_FEATURE_LEVELS
        # input_channels              = input_shape.channels
        # fmt: on
        # self.fc_dim_in = input_channels
        # if self.use_deform_attn:
        #     fc_dim = 2 * self.fc_dim_in
        # else:
        #     fc_dim = cfg.MODEL.POINT_HEAD.FC_DIM
        
        fc_dim = 12544
        self.fc = nn.Conv1d(256, fc_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.focal_loss = FocalLoss(gamma=1.5, num_classes=num_classes)

        self.use_deform_attn = True
        if self.use_deform_attn:
            d_model=256
            dim_feedforward=1024
            dropout=0.1
            activation='relu'
            nhead=8
            enc_n_points=4
            num_encoder_layers=1
            if self.num_feature_levels==4:
                in_channels=[256, 512, 1024, 2048]
            elif self.num_feature_levels==3:
                in_channels=[512, 1024, 2048]
            else:
                raise NotImplementedError
            
            input_proj_list = []
            for i in range(self.num_feature_levels):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels[i], d_model, kernel_size=1),
                        nn.GroupNorm(32, d_model),
                    )
                )
            self.input_proj = nn.ModuleList(input_proj_list)

            self.position_encoding = build_position_encoding(d_model)
            self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, d_model))
            encoder_layer = PeseudoLabDeformableTransformerEncoderLayer(
                d_model,
                dim_feedforward,
                dropout,
                activation,
                self.num_feature_levels,
                nhead,
                enc_n_points,
            )
            self.point_encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        weight_init.c2_msra_fill(self.fc)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        normal_(self.level_embed)

    def sample_pos_points_per_img(self, sample_points_per_instance: int, instances: Instances, branch=""):
        gt_pos_point_coords_list, gt_pos_classes_list = [], []

        gt_pos_point_coords = instances.gt_point_coords[:,0,:]
        gt_pos_classes = instances.gt_classes
        # if branch=="pseudo_supervised":
        if branch=="supervised":
            gt_pos_point_coords_list.append(gt_pos_point_coords)
            gt_pos_classes_list.append(gt_pos_classes)
            gt_boxes = instances.gt_boxes.tensor
            for gt_box, gt_pos_class in zip(gt_boxes, gt_pos_classes):
                    point_coords_wrt_image = np.random.rand(sample_points_per_instance, 2)
                    point_coords_wrt_image = torch.from_numpy(point_coords_wrt_image).to(gt_pos_point_coords.device)
                    point_coords_wrt_image[:, 0] = point_coords_wrt_image[:, 0] * (gt_box[2] - gt_box[0])
                    point_coords_wrt_image[:, 1] = point_coords_wrt_image[:, 1] * (gt_box[3] - gt_box[1])
                    point_coords_wrt_image[:, 0] += gt_box[0]
                    point_coords_wrt_image[:, 1] += gt_box[1]
                    gt_pos_point_coords_list.append(point_coords_wrt_image)
                    gt_pos_classes_list.append(torch.full((sample_points_per_instance,), gt_pos_class).to(gt_pos_point_coords.device))

            gt_pos_point_coords = torch.cat(gt_pos_point_coords_list, dim=0)
            gt_pos_classes = torch.cat(gt_pos_classes_list, dim=0)

        import pdb
        pdb.set_trace()
        return gt_pos_point_coords, gt_pos_classes
    
    
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        # shape (bs,)
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        # shape (bs, 2)
        return valid_ratio

    def forward(self, features, instances: List[Instances], branch=""):
        # features = [features[f] for f in self.point_in_features]
        N, C, H, W = features[0].size()
        if len(features)==4:
            sample_mask = torch.ones((N, 4*H, 4*W), dtype=torch.bool, device=features[0].device)
        elif len(features)==3:
            sample_mask = torch.ones((N, 8*H, 8*W), dtype=torch.bool, device=features[0].device)
        else:
            raise NotImplementedError

        # sample_mask = torch.zeros((N, 4*H, 4*W), dtype=torch.bool, device=features[0].device)
        for idx in range(N):
            image_size = instances[idx].image_size
            h, w = image_size
            sample_mask[idx, :h, :w] = False
        # sample_mask shape (1, N, H, W)
        sample_mask = sample_mask[None].float()

        # srcs is a list of num_levels tensor. Each one has shape (B, C, H_l, W_l)
        srcs = []
        # masks is a list of num_levels tensor. Each one has shape (B, H_l, W_l)
        masks = []
        pos_embeds = []
        for l in range(len(features)):
            src = self.input_proj[l](features[l])
            b, _, h, w = src.size()
            mask = F.interpolate(sample_mask, size=src.shape[-2:]).to(torch.bool)[0]
            pos_l = self.position_encoding(src, mask).to(src.dtype)
            srcs.append(src)
            masks.append(mask)
            pos_embeds.append(pos_l)

        # import pdb
        # pdb.set_trace()

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # src shape (bs, h_l*w_l, c)
            src = src.flatten(2).transpose(1, 2)
            # mask shape (bs, h_l*w_l)
            mask = mask.flatten(1)
            # pos_embed shape (bs, h_l*w_l, c)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            # lvl_pos_embed shape (bs, h_l*w_l, c)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        # src_flatten shape: (bs, K, c) where K = \sum_l H_l * w_l
        src_flatten = torch.cat(src_flatten, 1)
        # mask_flatten shape: (bs, K)
        mask_flatten = torch.cat(mask_flatten, 1)
        # mask_flatten shape: (bs, K, c)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # spatial_shapes shape: (num_levels, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        # level_start_index shape: (num_levels)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        # spatial_shapes shape: (bs, num_levels, 2)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        # memory shape (bs, K, C) where K = \sum_l H_l * w_l
        memory = self.point_encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
        )

        # prepare input for decoder
        bs, _, c = memory.shape
        memory = memory.permute(0,2,1)[:,:,:H*W].reshape(bs,c,H,W)

        point_features = []
        gt_classes = []
        sample_points_per_instance = 10
        for img_id in range(len(instances)):
            img_h, img_w = instances[img_id].image_size


            # gt_pos_point_coords, gt_pos_classes_per_img = self.sample_pos_points_per_img(sample_points_per_instance, instances[img_id], branch)
            # gt_point_coords = gt_pos_point_coords
            # point_coords_scaled = gt_point_coords / _as_tensor([img_w, img_h]).to(gt_pos_point_coords.device)
            # gt_classes_per_img = gt_pos_classes_per_img
            # import pdb
            # pdb.set_trace()
            gt_classes_per_img = instances[img_id].gt_classes.unsqueeze(1).expand_as(instances[img_id].gt_point_labels).clone()
            gt_classes_per_img[instances[img_id].gt_point_labels==0] = 80
            gt_classes_per_img = gt_classes_per_img.reshape(-1)
            gt_point_coords = instances[img_id].gt_point_coords.reshape(-1, 2)
            point_coords_scaled = gt_point_coords / _as_tensor([img_w, img_h]).to(gt_point_coords.device)

            # gt_pos_point_coords = instances[img_id].gt_point_coords[:,0,:]
            # gt_pos_classes_per_img = instances[img_id].gt_classes

            # gt_neg_point_coords = instances[img_id].gt_point_coords[:,1,:]
            # gt_neg_classes_per_img = torch.ones(gt_neg_point_coords.size(0)).long().to(gt_pos_point_coords.device) * 80
            
            # gt_point_coords = torch.cat([gt_pos_point_coords, gt_neg_point_coords],dim=0)
            # point_coords_scaled = gt_point_coords / _as_tensor([img_w, img_h]).to(gt_pos_point_coords.device)
            # gt_classes_per_img = torch.cat([gt_pos_classes_per_img, gt_neg_classes_per_img], dim=0)
            # import pdb
            # pdb.set_trace()

            gt_classes.append(gt_classes_per_img)
            point_features_per_image = point_sample(
                    memory[img_id].unsqueeze(0),
                    point_coords_scaled.unsqueeze(0),
                    align_corners=False,
                ).squeeze(0).transpose(1, 0)
            point_features.append(point_features_per_image)
         
        point_features = cat(point_features, dim=0)
        point_features = point_features.permute(1,0).unsqueeze(0)
        gt_classes =cat(gt_classes, dim=0)
        # import pdb
        # pdb.set_trace()  
        point_features = F.relu(self.fc(point_features))

        # attention
        # if self.use_deform_attn:
        #     tgt2 = self.cross_attn(
        #         self.with_pos_embed(tgt, query_pos),
        #         reference_points,
        #         src,
        #         src_spatial_shapes,
        #         level_start_index,
        #         src_padding_mask,
        #     )

        # point_features = self.predictor(point_features)

        point_features = point_features.squeeze(0).permute(1,0)

        return point_features, gt_classes
        # if self.training and branch=="supervised":
        #     # print(point_features.size(), gt_classes.size())
        #     # point_loss = F.cross_entropy(point_features, gt_classes, reduction="mean")
        #     point_loss = self.focal_loss(point_features, gt_classes) / gt_classes.shape[0]
        #     return {"loss_point": point_loss}


def build_point_head(cfg, input_channels):
    """
    Build a point head defined by `cfg.MODEL.POINT_HEAD.NAME`.
    """
    head_name = cfg.MODEL.POINT_HEAD.NAME
    return POINT_HEAD_REGISTRY.get(head_name)(cfg, input_channels)




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
        point_coords = point_coords.unsqueeze(2).float()
    
    # import pdb
    # pdb.set_trace()
    #[0,1] -> [0,2] -> [-1,1]
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def point_sample_fine_grained_features(features_list, feature_scales, num_points_per_img, point_coords_wrt_image):
    """
    Get features from feature maps in `features_list` that correspond to specific point coordinates
        inside each bounding box from `boxes`.
    Args:
        features_list (list[Tensor]): A list of feature map tensors to get features from.
        feature_scales (list[float]): A list of scales for tensors in `features_list`.
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.
    Returns:
        point_features (Tensor): A tensor of shape (R, C, P) that contains features sampled
            from all features maps in feature_list for P sampled points for all R boxes in `boxes`.
        point_coords_wrt_image (Tensor): A tensor of shape (R, P, 2) that contains image-level
            coordinates of P points.
    """
    split_point_coords_wrt_image = torch.split(point_coords_wrt_image, num_points_per_img)

    point_features = []
    for idx_img, point_coords_wrt_image_per_image in enumerate(split_point_coords_wrt_image):
        point_features_per_image = []
        for idx_feature, feature_map in enumerate(features_list):
            h, w = feature_map.shape[-2:]
            scale = _as_tensor([w, h]) / feature_scales[idx_feature]
            point_coords_scaled = point_coords_wrt_image_per_image / scale.to(feature_map.device)
            point_features_per_image.append(
                point_sample(
                    feature_map[idx_img].unsqueeze(0),
                    point_coords_scaled.unsqueeze(0),
                    align_corners=False,
                )
                .squeeze(0)
                .transpose(1, 0)
            )
        point_features.append(cat(point_features_per_image, dim=1))

    return cat(point_features, dim=0), point_coords_wrt_image

