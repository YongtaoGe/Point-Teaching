import torch
import torch.nn.functional as F
from torch import nn
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals
from detectron2.modeling.poolers import ROIPooler
from .make_layers import make_conv1x1, make_conv3x3, group_norm
from detectron2.layers import cat
from .loss import make_propagating_loss_evaluator 

from detectron2.utils.registry import Registry
ROI_SHAPEPROP_HEAD_REGISTRY = Registry("ROI_SHAPEPROP_HEAD")
ROI_SHAPEPROP_HEAD_REGISTRY.__doc__ = """
Registry for shapeprop heads, which make shapeprop predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

class ShapePropFeatureExtractor(nn.Module):

    def __init__(self, cfg, in_channels):
        super(ShapePropFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_SHAPEPROP_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_SHAPEPROP_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_SHAPEPROP_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_SHAPEPROP_HEAD.POOLER_TYPE

        self.pooler = ROIPooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        input_size = in_channels

        use_gn = cfg.MODEL.ROI_SHAPEPROP_HEAD.USE_GN
        layers = cfg.MODEL.ROI_SHAPEPROP_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_SHAPEPROP_HEAD.DILATION

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "shapeprop_feature{}".format(layer_idx)

            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn
            )
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def forward(self, x, proposals, branch=''):
        # import pdb
        # pdb.set_trace()
        if branch=='unsup_data_weak':
            boxes = [x.pred_boxes for x in proposals]
        else:
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in proposals]
        x = self.pooler(x, boxes)
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))
        # torch.Size([num_insts, 256, 14, 14])
        return x



class ShapePropPredictor(nn.Module):

    def __init__(self, cfg, in_channels):
        super(ShapePropPredictor, self).__init__()
        use_gn = cfg.MODEL.ROI_SHAPEPROP_HEAD.USE_GN
        self.cls = make_conv1x1(
            in_channels, 
            cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1, #81
            use_gn)

    def forward(self, x):
        logits = self.cls(x)
        return logits



class ShapePropWeightRegressor(nn.Module):

    def __init__(self, cfg, in_channels):
        super(ShapePropWeightRegressor, self).__init__()
        use_gn = cfg.MODEL.ROI_SHAPEPROP_HEAD.USE_GN
        self.reg = make_conv1x1(
            in_channels, 
            (1 if cfg.MODEL.ROI_SHAPEPROP_HEAD.CHANNEL_AGNOSTIC else cfg.MODEL.ROI_SHAPEPROP_HEAD.LATENT_DIM) * 9,
            use_gn)

    def forward(self, x):
        weights = self.reg(x)
        return torch.sigmoid(weights)



class MessagePassing(nn.Module):

    def __init__(self, k=3, max_step=-1, sym_norm=False):
        super(MessagePassing, self).__init__()
        self.k = k
        self.size = k * k
        self.max_step = max_step
        self.sym_norm = sym_norm

    def forward(self, input, weight, branch=''):
        if input.size(0)==0:
            return input

        eps = 1e-5
        n, c, h, w = input.size()
        wc = weight.shape[1] // self.size
        weight = weight.view(n, wc, self.size, h * w)
        if self.sym_norm:
            # symmetric normalization D^(-1/2)AD^(-1/2)
            D = torch.pow(torch.sum(weight, dim=2) + eps, -1/2).view(n, wc, h, w)
            D = F.unfold(D, kernel_size=self.k, padding=self.padding).view(n, wc, self.window, h * w) * D.view(n, wc, 1, h * w)
            norm_weight = D * weight
        else:
            # random walk normalization D^(-1)A
            norm_weight = weight / (torch.sum(weight, dim=2).unsqueeze(2) + eps)
        x = input
        # if branch=='pseudo_supervised':
        #     import pdb
        #     pdb.set_trace()

        for i in range(max(h, w) if self.max_step < 0 else self.max_step):
            x = F.unfold(x, kernel_size=self.k, padding=1).view(n, c, self.size, h * w)
            x = (x * norm_weight).sum(2).view(n, c, h, w)
        return x



class ShapePropEncoder(nn.Module):

    def __init__(self, cfg, in_channels):
        super(ShapePropEncoder, self).__init__()
        use_gn = cfg.MODEL.ROI_SHAPEPROP_HEAD.USE_GN
        latent_dim = cfg.MODEL.ROI_SHAPEPROP_HEAD.LATENT_DIM
        dilation = cfg.MODEL.ROI_SHAPEPROP_HEAD.DILATION
        self.encoder = nn.Sequential(
            make_conv3x3(in_channels, latent_dim, dilation=dilation, stride=1, use_gn=use_gn),
            nn.ReLU(True),
            make_conv3x3(latent_dim, latent_dim, dilation=dilation, stride=1, use_gn=use_gn),
            # nn.ReLU(True),
            # make_conv3x3(latent_dim, latent_dim, dilation=dilation, stride=1, use_gn=use_gn)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding



class ShapePropDecoder(nn.Module):

    def __init__(self, cfg, out_channels):
        super(ShapePropDecoder, self).__init__()
        use_gn = cfg.MODEL.ROI_SHAPEPROP_HEAD.USE_GN
        latent_dim = cfg.MODEL.ROI_SHAPEPROP_HEAD.LATENT_DIM
        dilation = cfg.MODEL.ROI_SHAPEPROP_HEAD.DILATION
        self.decoder = nn.Sequential(
            make_conv3x3(latent_dim, latent_dim, dilation=dilation, stride=1, use_gn=use_gn),
            nn.ReLU(True),
            make_conv3x3(latent_dim, latent_dim, dilation=dilation, stride=1, use_gn=use_gn),
            nn.ReLU(True),
            make_conv3x3(latent_dim, out_channels, dilation=dilation, stride=1, use_gn=use_gn)
        )

    def forward(self, embedding):
        x = self.decoder(embedding)
        return x



@ROI_SHAPEPROP_HEAD_REGISTRY.register()
class ShapePropHead(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        super(ShapePropHead, self).__init__()
        self.cfg = cfg.clone()
        # activating saliency
        self.feature_extractor_activating = ShapePropFeatureExtractor(self.cfg, in_channels)
        self.predictor = ShapePropPredictor(self.cfg, self.feature_extractor_activating.out_channels)
        # propagating saliency
        self.feature_extractor_propagating = (self.feature_extractor_activating
            if self.cfg.MODEL.ROI_SHAPEPROP_HEAD.SHARE_FEATURE
            else ShapePropFeatureExtractor(self.cfg, in_channels))
        self.propagation_weight_regressor = ShapePropWeightRegressor(self.cfg, self.feature_extractor_propagating.out_channels)
        self.encoder = ShapePropEncoder(self.cfg, 1)
        self.message_passing = MessagePassing(sym_norm=self.cfg.MODEL.ROI_SHAPEPROP_HEAD.USE_SYMMETRIC_NORM)
        self.decoder = ShapePropDecoder(self.cfg, 1)
        self.propagating_loss_evaluator = make_propagating_loss_evaluator(cfg)
        self.bg_label = 80

    def activating_saliency(self, features, proposals, branch=''):
        x = self.feature_extractor_activating(features, proposals, branch)
        # [num_insts, 80, 14, 14]
        saliency = self.predictor(x)
        labels = []

        for saliency_per_image, proposals_per_image in zip(saliency.split([len(v) for v in proposals], 0), proposals):
            if self.training and branch!="unsup_data_weak":
                # print(branch, proposals_per_image.get_fields().keys())
                labels_per_image = proposals_per_image.get_fields()['gt_classes']
            else:
                labels_per_image = proposals_per_image.get_fields()['pred_classes']
            labels.append(labels_per_image)
            if len(labels_per_image) > 0:
                saliency_per_image = torch.stack([v[l] for v, l in zip(saliency_per_image, labels_per_image)])
            else:
                # torch.Size([0, 14, 14])
                saliency_per_image = saliency_per_image[:,0,:,:]
            # if branch=='pseudo_supervised':
            #     import pdb
            #     pdb.set_trace()
            # print(branch, "hhhh")
            proposals_per_image.set('saliency', saliency_per_image)
        # inference mode
        if not self.training:
            return x, proposals, saliency, None
        # compute loss
        labels = cat(labels, dim=0) 
        num_batch, num_channel, h, w = saliency.shape
        class_logits = saliency.view(num_batch, num_channel, h * w).mean(2)
        # if branch=='pseudo_supervised':

        if labels.size(0) > 0:
            loss_activating = F.cross_entropy(class_logits, labels)
        else:
            loss_activating = class_logits.sum() * 0

        return x, proposals, saliency, loss_activating

    def propagating_saliency(self, features, proposals, branch=''):
        if self.training and branch!="unsup_data_weak":
            proposals, fg_selection_masks = select_foreground_proposals(proposals, self.bg_label)
            positive_inds = cat(fg_selection_masks, dim=0)

        if self.cfg.MODEL.ROI_SHAPEPROP_HEAD.SHARE_FEATURE:
            x = features
        else:
            x = self.feature_extractor_propagating(features, proposals, branch)
        # torch.Size([num_insts, 256, 14, 14]) -> torch.Size([num_insts, 216, 14, 14])
        weights = self.propagation_weight_regressor(x) 
        # torch.Size([num_insts, 1, 14, 14])
        # if branch=='pseudo_supervised':
        #     import pdb
        #     pdb.set_trace()
            # for v in proposals:
            #     print(v.get_fields()['saliency'].size())
        saliency = cat([v.get_fields()['saliency'] for v in proposals]).unsqueeze(1)
        embedding = self.encoder(saliency)
        # torch.Size([num_insts, 24, 14, 14])
        embedding = self.message_passing(embedding, weights, branch) 
        shape_activation = self.decoder(embedding).squeeze(1)
        for proposal_per_image, shape_activation_per_image in zip(proposals, shape_activation.split([len(v) for v in proposals], 0)):
            proposal_per_image.set('shape_activation', shape_activation_per_image)
        # inference mode
        if not self.training or branch == "unsup_data_weak":
            return x, proposals, shape_activation, None
        # compute loss
        # import pdb
        # pdb.set_trace()
        # shape_activation = shape_activation[positive_inds]

        loss_propagating = self.propagating_loss_evaluator(proposals, shape_activation, branch)
        return x, proposals, shape_activation, loss_propagating

    def forward(self, features, proposals, branch=''):
        x, proposals, saliency, loss_activating = self.activating_saliency(
            features, proposals, branch)
        x, proposals, shape_activation, loss_propagating = self.propagating_saliency(
            x if self.cfg.MODEL.ROI_SHAPEPROP_HEAD.SHARE_FEATURE else features, proposals, branch)

        if not self.training:
            return proposals, {}
        return proposals, dict(loss_activating=loss_activating, loss_propagating=loss_propagating)



def build_shapeprop_head(cfg, input_shape):
    """
    Build a shapeprop head defined by `cfg.MODEL.ROI_SHAPEPROP_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_SHAPEPROP_HEAD.NAME
    return ROI_SHAPEPROP_HEAD_REGISTRY.get(name)(cfg, input_shape.channels)


