import torch
import torch.nn.functional as F
import torch.distributed as dist

from detectron2.utils.comm import get_world_size


def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def reduce_mean(tensor):
    num_gpus = get_world_size()
    total = reduce_sum(tensor)
    return total.float() / num_gpus


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations






def compute_ious(pred, target):
    """
    Args:
        pred: Nx4 predicted bounding boxes
        target: Nx4 target bounding boxes
        Both are in the form of FCOS prediction (l, t, r, b)
    """
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_aera = (target_left + target_right) * \
                  (target_top + target_bottom)
    pred_aera = (pred_left + pred_right) * \
                (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + \
                  torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + \
                  torch.min(pred_top, target_top)

    g_w_intersect = torch.max(pred_left, target_left) + \
                    torch.max(pred_right, target_right)
    g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                    torch.max(pred_top, target_top)
    ac_uion = g_w_intersect * g_h_intersect

    area_intersect = w_intersect * h_intersect
    area_union = target_aera + pred_aera - area_intersect

    ious = (area_intersect + 1.0) / (area_union + 1.0)
    gious = ious - (ac_uion - area_union) / ac_uion

    return ious, gious


def cube_root(x):
    return torch.where(x >= 0, x ** (1. / 3), -x.abs() ** (1. / 3))


def rgb2xyz(rgb):
    arr = rgb.clone()

    # From sRGB specification
    xyz_from_rgb = rgb.new_tensor([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ], dtype=torch.double)

    mask = arr > 0.04045
    arr[mask] = torch.pow((arr[mask] + 0.055) / 1.055, 2.4)
    arr[~mask] /= 12.92
    return arr @ xyz_from_rgb.t()


def xyz2lab(xyz, illuminant="D65", observer="2"):
    illuminants = \
        {"A": {'2': (1.098466069456375, 1, 0.3558228003436005),
               '10': (1.111420406956693, 1, 0.3519978321919493)},
         "D50": {'2': (0.9642119944211994, 1, 0.8251882845188288),
                 '10': (0.9672062750333777, 1, 0.8142801513128616)},
         "D55": {'2': (0.956797052643698, 1, 0.9214805860173273),
                 '10': (0.9579665682254781, 1, 0.9092525159847462)},
         "D65": {'2': (0.95047, 1., 1.08883),  # This was: `lab_ref_white`
                 '10': (0.94809667673716, 1, 1.0730513595166162)},
         "D75": {'2': (0.9497220898840717, 1, 1.226393520724154),
                 '10': (0.9441713925645873, 1, 1.2064272211720228)},
         "E": {'2': (1.0, 1.0, 1.0),
               '10': (1.0, 1.0, 1.0)}}

    xyz_ref_white = illuminants[illuminant.upper()][observer]

    # scale by CIE XYZ tristimulus values of the reference white point
    arr = xyz / xyz.new_tensor(xyz_ref_white)

    # Nonlinear distortion and linear transformation
    mask = arr > 0.008856
    arr[mask] = cube_root(arr[mask])
    arr[~mask] = 7.787 * arr[~mask] + 16. / 116.

    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]

    # Vector scaling
    L = (116. * y) - 16.
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    return torch.cat([x[..., None] for x in [L, a, b]], dim=-1)


def rgb_to_lab(rgb, illuminant="D65", observer="2"):
    assert rgb.dim() == 3 and rgb.size(-1) == 3

    if rgb.dtype == torch.uint8:
        arr = rgb.double() / 255.0
    else:
        arr = rgb.double()

    return xyz2lab(rgb2xyz(arr), illuminant, observer).float()


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


def get_images_color_similarity(images, kernel_size, dilation):
    assert images.dim() == 4
    # assert images.size(0) == 1

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )
    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    return similarity

def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]