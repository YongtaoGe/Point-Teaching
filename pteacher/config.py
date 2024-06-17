# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_pteacher_config(cfg):
    """
    Add config for semisupnet.
    """
    _C = cfg
    _C.TEST.VAL_LOSS = True

    _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
    _C.MODEL.RPN.LOSS = "CrossEntropy"
    _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"

    _C.SOLVER.IMS_PER_BATCH_LABEL = 1
    _C.SOLVER.IMS_PER_BATCH_UNLABEL = 1
    _C.SOLVER.FACTOR_LIST = (1,)

    _C.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
    _C.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
    _C.DATASETS.CROSS_DATASET = False
    _C.TEST.EVALUATOR = "COCOeval"

    _C.SEMISUPNET = CN()

    # Output dimension of the MLP projector after `res5` block
    _C.SEMISUPNET.MLP_DIM = 128

    # Semi-supervised training
    _C.SEMISUPNET.Trainer = "pteacher"
    _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
    _C.SEMISUPNET.COPY_PASTE_THRESHOLD = 0.05
    _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
    _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
    _C.SEMISUPNET.BURN_UP_STEP = 12000
    _C.SEMISUPNET.EMA_KEEP_RATE = 0.0
    _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
    _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
    _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"

    # point-sup training
    # _C.SEMISUPNET.MASK_POINT_LOSS_WEIGHT = 0.0
    _C.SEMISUPNET.PRJ_LOSS_WEIGHT = 1.0
    _C.SEMISUPNET.PAIRWISE_LOSS_WEIGHT = 1.0
    _C.SEMISUPNET.IMG_MIL_LOSS_WEIGHT = 1.0
    _C.SEMISUPNET.INS_MIL_LOSS_WEIGHT = 0.05
    _C.SEMISUPNET.POINT_LOSS_WEIGHT = 0.05
    _C.SEMISUPNET.CORR_LOSS_WEIGHT = 0.1
    _C.SEMISUPNET.POINT_SUP = False
    _C.SEMISUPNET.IMG_MIL_FILTER_BG = False
    _C.SEMISUPNET.USE_SS_PROPOSALS = False
    _C.SEMISUPNET.USE_POINT_GUIDED_CP = False
    # dataloader
    # supervision level
    _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
    _C.DATALOADER.RANDOM_DATA_SEED = 0  # random seed to read data
    _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"

    _C.EMAMODEL = CN()
    _C.EMAMODEL.SUP_CONSIST = True


def add_point_sup_config(cfg):
    """
    Add config for point supervision.
    """
    # Use point annotation
    cfg.MODEL.POINT_ON = False
    # cfg.INPUT.POINT_SUP = False
    # Sample only part of points in each iteration.
    # Default: 0, use all available points.
    cfg.INPUT.SAMPLE_POINTS = 0


def add_pointrend_config(cfg):
    """
    Add config for PointRend.
    """
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Color augmentatition from SSD paper for semantic segmentation model during training.
    cfg.INPUT.COLOR_AUG_SSD = False

    # Names of the input feature maps to be used by a coarse mask head.
    cfg.MODEL.ROI_MASK_HEAD.IN_FEATURES = ("p2",)
    cfg.MODEL.ROI_MASK_HEAD.FC_DIM = 1024
    cfg.MODEL.ROI_MASK_HEAD.NUM_FC = 2
    # The side size of a coarse mask head prediction.
    cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION = 7
    # True if point head is used.
    cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON = False

    cfg.MODEL.POINT_HEAD = CN()
    cfg.MODEL.POINT_HEAD.USE_DEFORM_ATTN = False
    cfg.MODEL.POINT_HEAD.NUM_FEATURE_LEVELS = 3
    cfg.MODEL.POINT_HEAD.NAME = "StandardPointHead"
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 81
    # Names of the input feature maps to be used by a mask point head.
    cfg.MODEL.POINT_HEAD.IN_FEATURES = ("res3","res4","res5",)
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS = 14 * 14
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO = 3
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO = 0.75
    # Number of subdivision steps during inference.
    cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS = 5
    # Maximum number of points selected at each subdivision step (N).
    cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS = 28 * 28
    cfg.MODEL.POINT_HEAD.FC_DIM = 256
    cfg.MODEL.POINT_HEAD.NUM_FC = 3
    cfg.MODEL.POINT_HEAD.CLS_AGNOSTIC_MASK = False
    # If True, then coarse prediction features are used as inout for each layer in PointRend's MLP.
    cfg.MODEL.POINT_HEAD.COARSE_PRED_EACH_LAYER = True
    cfg.MODEL.POINT_HEAD.COARSE_SEM_SEG_HEAD_NAME = "SemSegFPNHead"

    """
    Add config for Implicit PointRend.
    """
    cfg.MODEL.IMPLICIT_POINTREND = CN()

    cfg.MODEL.IMPLICIT_POINTREND.IMAGE_FEATURE_ENABLED = True
    cfg.MODEL.IMPLICIT_POINTREND.POS_ENC_ENABLED = True

    cfg.MODEL.IMPLICIT_POINTREND.PARAMS_L2_REGULARIZER = 0.00001


def add_shapeprop_config(cfg):
    # ShapeProp
    _C = cfg
    _C.MODEL.SHAPEPROP_ON = False
    _C.MODEL.ROI_SHAPEPROP_HEAD = CN()
    _C.MODEL.ROI_SHAPEPROP_HEAD.NAME = "ShapePropHead"
    _C.MODEL.ROI_SHAPEPROP_HEAD.POOLER_TYPE = "ROIAlignV2"
    _C.MODEL.ROI_SHAPEPROP_HEAD.POOLER_RESOLUTION = 14
    _C.MODEL.ROI_SHAPEPROP_HEAD.POOLER_SAMPLING_RATIO = 2
    _C.MODEL.ROI_SHAPEPROP_HEAD.POOLER_SCALES = (0.25, 0.125, 0.0625, 0.03125)
    # _C.MODEL.ROI_SHAPEPROP_HEAD.CONV_LAYERS = (256, 256, 256, 256)
    # _C.MODEL.ROI_SHAPEPROP_HEAD.CONV_LAYERS = (256, 256)
    _C.MODEL.ROI_SHAPEPROP_HEAD.CONV_LAYERS = (256,)
    _C.MODEL.ROI_SHAPEPROP_HEAD.DILATION = 1
    _C.MODEL.ROI_SHAPEPROP_HEAD.LATENT_DIM = 24
    _C.MODEL.ROI_SHAPEPROP_HEAD.USE_GN = False
    _C.MODEL.ROI_SHAPEPROP_HEAD.CHANNEL_AGNOSTIC = False
    _C.MODEL.ROI_SHAPEPROP_HEAD.USE_SYMMETRIC_NORM = False
    _C.MODEL.ROI_SHAPEPROP_HEAD.SHARE_FEATURE = False


def add_fcos_config(cfg):
    _C = cfg

    _C.INPUT.SCALE_JITTER_MIN = 0.3
    _C.INPUT.SCALE_JITTER_MAX = 1.9
    _C.INPUT.SCALE_JITTER_TGT_HEIGHT = 960
    _C.INPUT.SCALE_JITTER_TGT_WIDTH = 960


    _C.MODEL.SHIFT_GENERATOR = CN()
    _C.MODEL.SHIFT_GENERATOR.NUM_SHIFTS = 1
    _C.MODEL.SHIFT_GENERATOR.OFFSET = 0.0


    _C.MODEL.BACKBONE.ANTI_ALIAS = False
    _C.MODEL.RESNETS.DEFORM_INTERVAL = 1
    _C.MODEL.MOBILENET = False
    # ---------------------------------------------------------------------------- #
    # FCOS Head
    # ---------------------------------------------------------------------------- #
    _C.MODEL.FCOS = CN()

    # This is the number of foreground classes.
    _C.MODEL.FCOS.NUM_CLASSES = 80
    _C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    _C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
    _C.MODEL.FCOS.PRIOR_PROB = 0.01
    _C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
    _C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
    _C.MODEL.FCOS.NMS_TH = 0.6
    _C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
    _C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
    _C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
    _C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
    _C.MODEL.FCOS.TOP_LEVELS = 2
    _C.MODEL.FCOS.NORM = "GN"  # Support GN or none
    _C.MODEL.FCOS.USE_SCALE = True

    # The options for the quality of box prediction
    # It can be "ctrness" (as described in FCOS paper) or "iou"
    # Using "iou" here generally has ~0.4 better AP on COCO
    # Note that for compatibility, we still use the term "ctrness" in the code
    _C.MODEL.FCOS.BOX_QUALITY = "ctrness"

    # Multiply centerness before threshold
    # This will affect the final performance by about 0.05 AP but save some time
    _C.MODEL.FCOS.THRESH_WITH_CTR = False

    # Focal loss parameters
    _C.MODEL.FCOS.LOSS_ALPHA = 0.25
    _C.MODEL.FCOS.LOSS_GAMMA = 2.0

    # The normalizer of the classification loss
    # The normalizer can be "fg" (normalized by the number of the foreground samples),
    # "moving_fg" (normalized by the MOVING number of the foreground samples),
    # or "all" (normalized by the number of all samples)
    _C.MODEL.FCOS.LOSS_NORMALIZER_CLS = "fg"
    _C.MODEL.FCOS.LOSS_WEIGHT_CLS = 1.0

    _C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
    _C.MODEL.FCOS.USE_RELU = True
    _C.MODEL.FCOS.USE_DEFORMABLE = False

    # the number of convolutions used in the cls and bbox tower
    _C.MODEL.FCOS.NUM_MIL_CONVS = 2
    _C.MODEL.FCOS.NUM_CLS_CONVS = 4
    _C.MODEL.FCOS.NUM_BOX_CONVS = 4
    _C.MODEL.FCOS.NUM_SHARE_CONVS = 0
    _C.MODEL.FCOS.CENTER_SAMPLE = True
    _C.MODEL.FCOS.POS_RADIUS = 1.5
    _C.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
    _C.MODEL.FCOS.YIELD_PROPOSAL = False


def add_boxinst_config(cfg):
    _C = cfg
    # The options for BoxInst, which can train the instance segmentation model with box annotations only
    # Please refer to the paper https://arxiv.org/abs/2012.02310
    _C.MODEL.BOXINST = CN()
    # Whether to enable BoxInst
    _C.MODEL.BOXINST.ENABLED = False
    _C.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED = 10

    _C.MODEL.BOXINST.PAIRWISE = CN()
    _C.MODEL.BOXINST.PAIRWISE.SIZE = 3
    _C.MODEL.BOXINST.PAIRWISE.DILATION = 2
    _C.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS = 10000
    _C.MODEL.BOXINST.PAIRWISE.COLOR_THRESH = 0.3
    _C.MODEL.BOXINST.MASK_OUT_STRIDE = 4