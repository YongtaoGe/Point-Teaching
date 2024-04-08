import os
from collections import defaultdict
import numpy as np
from numpy import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

def visualize(image, rbboxes, color=(0, 0, 255), base_name=None):
    for i, (x1, y1, x2, y2, x3, y3, x4, y4) in enumerate(rbboxes):
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.line(image, (int(x2), int(y2)), (int(x3), int(y3)), color, 2)
        cv2.line(image, (int(x3), int(y3)), (int(x4), int(y4)), color, 2)
        cv2.line(image, (int(x4), int(y4)), (int(x1), int(y1)), color, 2)

    if base_name is None:
        out_file = 'vis_{}.png'.format(np.random.randint(1000, 2000))
    else:
        out_file = base_name
    out_file = os.path.join(os.path.dirname(__file__), out_file)
    cv2.imwrite(out_file, image)
    print('write: {}'.format(os.path.abspath(out_file)))
    return image


class ClassAwareInstanceBank:
    def __init__(self, prob=0.5, paste_per_cls=2, bank_length=500, bank_update_num=5):
        self.prob = prob
        self.paste_per_cls = paste_per_cls
        self.bank_length = bank_length
        self.bank_update_num = bank_update_num
        self.inst_bank = defaultdict(list)

    def _update_bank(self, label_data):


        list_instances = []
        num_proposal_output = 0.0
        gt_point_coords_list = []
        gt_point_labels_list = []
        gt_bbox_classes_list = []
        for i in range(len(label_data)):
            gt_point_coords = unlabel_data_k[i]['instances'].gt_point_coords
            # center point of gt box
            gt_point_coords[:, 1, 0] = (unlabel_data_k[i]['instances'].gt_boxes.tensor[:,0] + unlabel_data_k[i]['instances'].gt_boxes.tensor[:,2]) * 0.5
            gt_point_coords[:, 1, 1] = (unlabel_data_k[i]['instances'].gt_boxes.tensor[:,1] + unlabel_data_k[i]['instances'].gt_boxes.tensor[:,3]) * 0.5

            gt_point_coords_list.append(gt_point_coords)
            gt_point_labels_list.append(unlabel_data_k[i]['instances'].gt_point_labels)
            gt_bbox_classes_list.append(unlabel_data_k[i]['instances'].gt_classes)


        DEBUG = False
        for labeled_data in labeled_data_list:

            labeled_data["instance"].gt_classes


        h, w = img.shape[:2]
        unique_labels = list(set(gt_labels.tolist()))
        for l in unique_labels:
            candidate_bboxes = gt_bboxes[gt_labels == l]

            num = 0
            inds = list(range(len(candidate_bboxes)))
            np.random.shuffle(inds)
            for i in inds:
                if num >= self.bank_update_num:
                    break

                bbox = candidate_bboxes[i].reshape(8)
                x1, x2 = bbox[0::2].min(), bbox[0::2].max()
                y1, y2 = bbox[1::2].min(), bbox[1::2].max()
                if (x2 - x1 < 12) or (y2 - y1) < 12:
                    continue

                num += 1

                crop_x1 = int(max(0, x1 - 10))
                crop_y1 = int(max(0, y1 - 10))
                crop_x2 = int(min(x2 + 10, w))
                crop_y2 = int(min(y2 + 10, h))

                crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2, :].copy()
                crop_bbox = bbox - np.tile(np.array([crop_x1, crop_y1]), 4)

                # enlarge
                ch, cw = crop_img.shape[:2]
                s = max(ch, cw) * 2
                patch_img = np.zeros((s, s, 3))
                x1 = (s - cw) // 2
                y1 = (s - ch) // 2
                patch_img[y1:(y1 + ch), x1:(x1 + cw), :] = crop_img
                patch_bbox = crop_bbox + np.tile(np.array([x1, y1]), 4)
                patch_mask = np.zeros((s, s, 3), dtype=np.uint8)
                cv2.fillPoly(patch_mask, [patch_bbox.reshape(4, 2).astype(np.int)], (255, 255, 255))

                # roate and scale aug
                jit_angle = np.random.choice(range(0, 360, 30))
                center = np.array((s / 2, s / 2))
                M = cv2.getRotationMatrix2D(tuple(center), jit_angle, 1)
                patch_img = cv2.warpAffine(patch_img, M, (s, s), flags=cv2.INTER_LINEAR)
                patch_mask = cv2.warpAffine(patch_mask, M, (s, s), flags=cv2.INTER_LINEAR)

                jit_scale = np.random.uniform(0.5, 1.5)
                patch_img = cv2.resize(patch_img, (0, 0), fx=jit_scale, fy=jit_scale, interpolation=cv2.INTER_AREA)
                patch_mask = cv2.resize(patch_mask, (0, 0), fx=jit_scale, fy=jit_scale, interpolation=cv2.INTER_AREA)

                patch_binary = (patch_mask.mean(axis=2) > 127).astype(np.uint8)
                *_, contours, hierarchy = cv2.findContours(patch_binary, 1, 2)
                cnt = contours[0]
                rect = cv2.minAreaRect(cnt)
                patch_bbox = cv2.boxPoints(rect).reshape(-1)

                assert patch_img.shape == patch_mask.shape

                x1, x2 = patch_bbox[0::2].min(), patch_bbox[0::2].max()
                y1, y2 = patch_bbox[1::2].min(), patch_bbox[1::2].max()
                crop_x1 = int(max(0, x1 - 1))
                crop_y1 = int(max(0, y1 - 1))
                crop_x2 = int(x2 + 1)
                crop_y2 = int(y2 + 1)
                patch_img = patch_img[crop_y1:crop_y2, crop_x1:crop_x2, :]
                patch_binary = patch_binary[crop_y1:crop_y2, crop_x1:crop_x2]
                patch_bbox = patch_bbox - np.tile(np.array([crop_x1, crop_y1]), 4)

                assert patch_img.shape[:2] == patch_binary.shape

                if len(self.inst_bank[l]) < self.bank_length:
                    self.inst_bank[l].append((patch_img, patch_binary, patch_bbox))
                else:
                    p_i = np.random.choice(range(self.bank_length))
                    self.inst_bank[l][p_i] = (patch_img, patch_binary, patch_bbox)

                if DEBUG:
                    _rand = np.random.randint(1000, 2000)
                    cv2.polylines(patch_img, patch_bbox.reshape(1, 4, 2).astype(np.int32), True, (0, 0, 255), 1)
                    cv2.imwrite('vis_{}_patch_img.png'.format(_rand), patch_img)
                    cv2.imwrite('vis_{}_patch_mask.png'.format(_rand), patch_mask)
                    print('write: vis_{}_patch_img.png'.format(_rand))
                    print('write: vis_{}_patch_mask.png'.format(_rand))

    def __call__(self, unlabeled_data):
        # import pdb; pdb.set_trace()
        DEBUG = False

        gt_bboxes = results['gt_bboxes']
        gt_labels = results['gt_labels']
        h, w = results['img_shape'][:2]

        assert gt_bboxes.shape[1] == 8

        self._update_bank(results['img'], gt_bboxes, gt_labels)

        if random.uniform() < self.prob:
            if DEBUG:
                _rand = np.random.randint(1e4, 4e4)
                visualize(results['img'].copy(), results['gt_bboxes'], base_name='{}_origin.png'.format(_rand))

            img = results['img']
            for label in self.inst_bank:
                inds = np.random.choice(range(len(self.inst_bank[label])), self.paste_per_cls)
                for i in inds:
                    patch_img, patch_binary, patch_bbox = self.inst_bank[label][i]
                    p_h, p_w = patch_img.shape[:2]
                    if p_h >= h or p_w >= w: continue

                    for _ in range(2):  # try time
                        p_x1 = np.random.randint(0, w - p_w)
                        p_y1 = np.random.randint(0, h - p_h)

                        patch_bbox_p = patch_bbox + np.tile(np.array([p_x1, p_y1]), 4)
                        patch_bbox_p = patch_bbox_p.reshape(1, 8)

                        ious = bbox_ious(rbbox_to_hbbox(patch_bbox_p), rbbox_to_hbbox(gt_bboxes))
                        if ious.max() < 1e-6:
                            # print('paste')
                            img[p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w), :][patch_binary > 0.5] = patch_img[
                                patch_binary > 0.5]
                            gt_bboxes = np.concatenate((gt_bboxes, patch_bbox_p), axis=0)
                            gt_labels = np.concatenate((gt_labels, np.array([label])), axis=0)
                            break

            results['img'] = img
            results['gt_bboxes'] = gt_bboxes.astype(np.float32)
            results['gt_labels'] = gt_labels.astype(np.int64)

            if DEBUG:
                assert len(results['gt_bboxes']) == len(results['gt_labels'])
                visualize(results['img'].copy(), results['gt_bboxes'], base_name='{}_paste.png'.format(_rand))

            return results

        return results



def relu_and_l2_norm_feat(feat, dim=1):
    feat = F.relu(feat, inplace=True)
    feat_norm = ((feat ** 2).sum(dim=dim, keepdim=True) + 1e-6) ** 0.5
    feat = feat / (feat_norm + 1e-6)
    return feat



class ObjectElements:

    #@autocast(enabled=False)
    def __init__(self, size=100, img_size=56, feat_size=28, mask_size=56, n_channel=256, device='cpu', category=None):
        self.mask = torch.zeros(size, mask_size, mask_size).to(device).to(device)
        self.feature = torch.zeros(size, n_channel, feat_size, feat_size).to(device)
        self.box = torch.zeros(size, 4).to(device)
        self.img = torch.zeros(size, 3, img_size, img_size).to(device)
        self.category = int(category)
        self.ptr = 0

    #@autocast(enabled=False)
    def get_box_area(self):
        box = self.box
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    #@autocast(enabled=False)
    def get_category(self):
        return self.category

    def get_feature(self):
        return self.feature

    def get_mask(self):
        return self.mask

    def get_ratio(self):
        box = self.box
        return (box[:, 2] - box[:, 0]) / (box[:, 3] - box[:, 1] + 1e-5)

    def get_img(self):
        return self.img

    def get_box(self):
        return self.box

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        if isinstance(idx, slice) or torch.is_tensor(idx) or isinstance(idx, list):
            if torch.is_tensor(idx):
                idx = idx.to(self.mask).long()  # self.mask might be in cpu
            if self.img is not None:
                img = self.img[idx]
            else:
                img = None
            mask = self.mask[idx]
            feature = self.feature[idx]
            box = self.box[idx]
            category = self.category
        elif isinstance(idx, int):
            if self.img is not None:
                img = self.img[idx:idx + 1]
            else:
                img = None
            mask = self.mask[idx:idx + 1]
            feature = self.feature[idx:idx + 1]
            box = self.box[idx:idx + 1]
            category = self.category
        else:
            raise NotImplementedError("type: {}".format(type(idx)))
        return dict(img=img, mask=mask, feature=feature, box=box, category=category)


class ObjectFactory:

    @staticmethod
    def create_one(mask, feature, box, img, category):
        if img is not None:
            img_size = img.shape[2]
        else:
            img_size = 0
        object_elements = ObjectElements(size=1,
                                         img_size=img_size,
                                         feat_size=feature.shape[2],
                                         mask_size=mask.shape[1],
                                         n_channel=feature.shape[1],
                                         device=mask.device,
                                         category=category)
        object_elements.mask[...] = mask
        object_elements.feature[...] = feature
        object_elements.feature[...] = relu_and_l2_norm_feat(object_elements.feature[0:1])
        object_elements.box[...] = box
        if img is not None:
            object_elements.img[...] = img
        return object_elements

    @staticmethod
    def create_queue_by_one(len_queue, category, idx, feature, mask, box, img=None, device='cpu'):
        if img is not None:
            img_size = img.shape[2]
        else:
            img_size = 0
        if category == 1:
            device = mask.device

        object_elements = ObjectElements(size=len_queue,
                                         img_size=img_size,
                                         feat_size=feature.shape[2],
                                         mask_size=mask.shape[2],
                                         n_channel=feature.shape[1],
                                         device=device,
                                         category=category)
        # import pdb
        # pdb.set_trace()
        object_elements.mask[0:1] = mask[idx]
        object_elements.feature[0:1] = feature[idx:idx + 1]
        object_elements.box[0:1] = box[idx:idx + 1]
        if img is not None:
            object_elements.img[0:1] = img[idx:idx + 1]
        return object_elements


class ObjectQueues:
    #@autocast(enabled=False)
    def __init__(self, num_class, len_queue, fg_iou_thresh, bg_iou_thresh, ratio_range, appear_thresh,
                 max_retrieval_objs):
        self.num_class = num_class
        self.queues = [None for i in range(self.num_class)]
        self.len_queue = len_queue
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.appear_thresh = appear_thresh
        self.ratio_range = ratio_range
        self.max_retrieval_objs = max_retrieval_objs

    #@autocast(enabled=False)
    def append(self, class_idx, idx, feature, mask, box, img=None, device='cpu'):
        with torch.no_grad():
            if self.queues[class_idx] is None:
                self.queues[class_idx] = \
                    ObjectFactory.create_queue_by_one(
                        len_queue=self.len_queue,
                        category=class_idx,
                        idx=idx,
                        feature=feature,
                        mask=mask,
                        box=box,
                        img=img,
                        device=device
                    )
                create_new_gpu_bank = True
                self.queues[class_idx].ptr += 1
                self.queues[class_idx].ptr = self.queues[class_idx].ptr % self.len_queue
            else:
                # print('line 355:', class_idx, self.queues[class_idx].ptr)
                ptr = self.queues[class_idx].ptr
                self.queues[class_idx].feature[ptr:ptr + 1] = feature[idx:idx + 1]
                self.queues[class_idx].mask[ptr:ptr + 1] = mask[idx:idx + 1]
                self.queues[class_idx].box[ptr:ptr + 1] = box[idx:idx + 1]
                if img is not None:
                    self.queues[class_idx].img[ptr:ptr + 1] = img[idx:idx + 1]
                self.queues[class_idx].ptr = (ptr + 1) % self.len_queue
                create_new_gpu_bank = False
            return create_new_gpu_bank

    #@autocast(enabled=False)
    def cal_fg_iou(self, qobjs, kobjs):
        # return the min value of
        # foreground IoU and background IoU
        maskA, maskB = qobjs.get_mask(), kobjs.get_mask()
        maskB = maskB.to(maskA)  # might be in cpu
        fiou = (maskA * maskB).sum([1, 2]) / ((maskA + maskB) >= 1).float().sum([1, 2])
        return fiou

    #@autocast(enabled=False)
    def cal_bg_iou(self, qobjs, kobjs):
        maskA, maskB = qobjs.get_mask(), kobjs.get_mask()
        maskB = maskB.to(maskA)
        biou = ((1 - maskA) * (1 - maskB)).sum([1, 2]) / ((2 - maskA - maskB) >= 1).float().sum([1, 2])
        return biou

    #@autocast(enabled=False)
    def cal_appear_identity_sim(self, qobjs, kobjs):
        f0 = qobjs.get_feature()
        f1 = kobjs.get_feature()
        f1 = f1.to(f0)  # might be in cpu
        mask0 = qobjs.get_mask()
        mask1 = kobjs.get_mask()
        mask1 = mask1.to(mask0)  # might be in cpu
        mask0 = F.interpolate(mask0.unsqueeze(1), (f0.shape[2], f0.shape[3]), mode='bilinear',
                              align_corners=False).squeeze(1)
        mask1 = F.interpolate(mask1.unsqueeze(1), (f1.shape[2], f1.shape[3]), mode='bilinear',
                              align_corners=False).squeeze(1)
        sim = (f0 * f1 * mask0.unsqueeze(1) * mask1.unsqueeze(1)).sum([1, 2, 3]) / ((mask0 * mask1).sum([1, 2]) + 1e-6)
        return sim

    #@autocast(enabled=False)
    def cal_shape_ratio(self, qobj, kobjs):
        ratio0 = qobj.get_ratio().unsqueeze(1)
        ratio1 = kobjs.get_ratio().unsqueeze(0)
        ratio1 = ratio1.to(ratio0)  # might be in cpu
        return ratio0 / ratio1

    #@autocast(enabled=False)
    def get_similar_obj(self, qobj: ObjectElements):
        with torch.no_grad():
            category = qobj.get_category()
            if self.queues[category] is not None:
                kobjs = self.queues[qobj.category]
                fg_ious = self.cal_fg_iou(qobj, kobjs)
                bg_ious = self.cal_bg_iou(qobj, kobjs)
                appear_sim = self.cal_appear_identity_sim(qobj, kobjs)
                ratio = self.cal_shape_ratio(qobj, kobjs).squeeze(0)
                seg_masking = ((fg_ious > self.fg_iou_thresh).float() * (bg_ious > self.bg_iou_thresh).float()).to(fg_ious)
                sim_masking = (appear_sim > self.appear_thresh).float().to(fg_ious)
                ratio_masking = ((ratio >= self.ratio_range[0]).float() * (ratio <= self.ratio_range[1]).float()).to(
                    fg_ious)
                masking = torch.where((seg_masking * sim_masking * ratio_masking).bool())[0][
                          :self.max_retrieval_objs].long()
                ret_objs = kobjs[masking]
                return ret_objs
            else:
                return None
            # ObjectElements(torch.zeros([0, qmask.shape[1], qmask.shape[2]]).to(device), torch.zeros([0, qfeature.shape[1], qfeature.shape[2], qfeature.shape[3]]))


class SemanticCorrSolver:

    #@autocast(enabled=False)
    def __init__(self, exp, eps, gaussian_filter_size, low_score, num_iter, num_smooth_iter, dist_kernel):
        self.exp = exp
        self.eps = eps
        self.gaussian_filter_size = gaussian_filter_size
        self.low_score = low_score
        self.hsfilter = self.generate_gaussian_filter(gaussian_filter_size)
        self.num_iter = num_iter
        self.num_smooth_iter = num_smooth_iter
        self.count = None
        self.pairwise = None
        self.dist_kernel = dist_kernel
        self.ncells = 8192

    #@autocast(enabled=False)
    def generate_gaussian_filter(self, size=3):
        r"""Returns 2-dimensional gaussian filter"""
        dim = [size, size]

        siz = torch.LongTensor(dim)
        sig_sq = (siz.float() / 2 / 2.354).pow(2)
        siz2 = (siz - 1) / 2

        x_axis = torch.arange(-siz2[0], siz2[0] + 1).unsqueeze(0).expand(dim).float()
        y_axis = torch.arange(-siz2[1], siz2[1] + 1).unsqueeze(1).expand(dim).float()

        gaussian = torch.exp(-(x_axis.pow(2) / 2 / sig_sq[0] + y_axis.pow(2) / 2 / sig_sq[1]))
        gaussian = gaussian / gaussian.sum()

        return gaussian

    #@autocast(enabled=False)
    def perform_sinkhorn(self, a, b, M, reg, stopThr=1e-3, numItermax=100):
        # init data
        dim_a = a.shape[1]
        dim_b = b.shape[1]

        batch_size = b.shape[0]

        u = torch.ones((batch_size, dim_a), requires_grad=False).cuda() / dim_a
        v = torch.ones((batch_size, dim_b), requires_grad=False).cuda() / dim_b
        K = torch.exp(-M / reg)

        Kp = (1 / a).unsqueeze(2) * K
        cpt = 0
        err = 1
        KtransposeU = (K * u.unsqueeze(2)).sum(dim=1)  # has shape K.shape[1]

        while err > stopThr and cpt < numItermax:
            KtransposeU[...] = (K * u.unsqueeze(2)).sum(dim=1)  # has shape K.shape[1]
            v[...] = b / KtransposeU
            u[...] = 1. / (Kp * v.unsqueeze(1)).sum(dim=2)
            cpt = cpt + 1

        T = u.unsqueeze(2) * K * v.unsqueeze(1)
        # del u, K, v
        return T

    #@autocast(enabled=False)
    def appearance_similarityOT(self, m0, m1, sim):
        r"""Semantic Appearance Similarity"""

        pow_sim = torch.pow(torch.clamp(sim, min=0.3, max=0.7), 1.0)
        cost = 1 - pow_sim

        b, n1, n2 = sim.shape[0], sim.shape[1], sim.shape[2]
        m0, m1 = torch.clamp(m0, min=self.low_score, max=1 - self.low_score), torch.clamp(m1, min=self.low_score,
                                                                                          max=1 - self.low_score)
        mu = m0 / m0.sum(1, keepdim=True)
        nu = m1 / m1.sum(1, keepdim=True)
        with torch.no_grad():
            epsilon = self.eps
            cnt = 0
            while epsilon < 5:
                PI = self.perform_sinkhorn(mu, nu, cost, epsilon)
                if not torch.isnan(PI).any():
                    if cnt > 0:
                        print(cnt)
                    break
                else:
                    epsilon *= 2.0
                    cnt += 1
                    print(cnt, epsilon)

        if torch.isnan(PI).any():
            from IPython import embed
            embed()

        PI = n1 * PI  # re-scale PI
        exp = self.exp
        PI = torch.pow(torch.clamp(PI, min=0), exp)

        return PI

    #@autocast(enabled=False)
    def build_hspace(self, src_imsize, trg_imsize, ncells):
        r"""Build Hough space where voting is done"""
        hs_width = src_imsize[0] + trg_imsize[0]
        hs_height = src_imsize[1] + trg_imsize[1]
        hs_cellsize = math.sqrt((hs_width * hs_height) / ncells)
        nbins_x = int(hs_width / hs_cellsize) + 1
        nbins_y = int(hs_height / hs_cellsize) + 1

        return nbins_x, nbins_y, hs_cellsize

    #@autocast(enabled=False)
    def receptive_fields(self, rfsz, feat_size):
        r"""Returns a set of receptive fields (N, 4)"""
        width = feat_size[3]
        height = feat_size[2]

        feat_ids = torch.tensor(list(range(width))).repeat(1, height).t().repeat(1, 2).to(rfsz.device)
        feat_ids[:, 0] = torch.tensor(list(range(height))).unsqueeze(1).repeat(1, width).view(-1).to(rfsz.device)

        box = torch.zeros(feat_ids.size()[0], 4).to(rfsz.device)
        box[:, 0] = feat_ids[:, 1] - rfsz // 2
        box[:, 1] = feat_ids[:, 0] - rfsz // 2
        box[:, 2] = feat_ids[:, 1] + rfsz // 2
        box[:, 3] = feat_ids[:, 0] + rfsz // 2
        box = box.unsqueeze(0)

        return box

    #@autocast(enabled=False)
    def pass_message(self, T, shape):
        T = T.view(T.shape[0], shape[0], shape[1], shape[0], shape[1])
        pairwise = torch.zeros_like(T).to(T)
        count = torch.zeros_like(T).to(T)
        dxs, dys = [-1, 0, 1], [-1, 0, 1]
        for dx in dxs:
            for dy in dys:
                count[:, max(0, dy): min(shape[0] + dy, shape[0]), max(0, dx): min(shape[1] + dx, shape[1]),
                max(0, dy): min(shape[0] + dy, shape[0]), max(0, dx): min(shape[1] + dx, shape[1])] += 1
                pairwise[:, max(0, dy): min(shape[0] + dy, shape[0]), max(0, dx): min(shape[1] + dx, shape[1]),
                max(0, dy): min(shape[0] + dy, shape[0]), max(0, dx): min(shape[1] + dx, shape[1])] += \
                    T[:, max(0, -dy): min(shape[0] - dy, shape[0]), max(0, -dx): min(shape[1] - dx, shape[1]),
                    max(0, -dy): min(shape[0] - dy, shape[0]), max(0, -dx): min(shape[1] - dx, shape[1])]

        T[...] = pairwise / count
        T = T.view(T.shape[0], shape[0] * shape[1], shape[0] * shape[1])
        # del pairwise, count

        return T

    #@autocast(enabled=False)
    def solve(self, qobjs, kobjs, f0):
        r"""Regularized Hough matching"""
        # Unpack hyperpixels
        m0 = qobjs.mask.float()
        f0 = f0.float()
        f1 = kobjs['feature'].to(m0).float()
        m1 = kobjs['mask'].to(m0).float()
       
        fg_mask = m0.reshape(m0.shape[0], -1, 1) * m1.reshape(m1.shape[0], 1, -1)
        bg_mask = (1 - m0).reshape(m0.shape[0], -1, 1) * (1 - m1).reshape(m1.shape[0], 1, -1)
        
        m0 = F.interpolate(m0.unsqueeze(1), (f0.shape[2], f0.shape[3]), mode='bilinear', align_corners=False).squeeze(1)
        m1 = F.interpolate(m1.unsqueeze(1), (f1.shape[2], f1.shape[3]), mode='bilinear', align_corners=False).squeeze(1)
        shape = f0.shape[2], f0.shape[3]

        m0 = m0.reshape(m0.shape[0], -1)
        m1 = m1.reshape(m1.shape[0], -1)
        f0 = f0.reshape(f0.shape[0], f0.shape[1], -1).transpose(2, 1)
        f1 = f1.reshape(f1.shape[0], f1.shape[1], -1)

        f0_norm = torch.norm(f0, p=2, dim=2, keepdim=True) + 1e-4
        f1_norm = torch.norm(f1, p=2, dim=1, keepdim=True) + 1e-4
        with autocast(enabled=False):
            Cu = torch.matmul((f0 / f0_norm), (f1 / f1_norm))

        eye = torch.eye(shape[0] * shape[1]).to(f0).reshape(1, -1, shape[0], shape[1])
        dist_mask = F.max_pool2d(eye, kernel_size=self.dist_kernel, stride=1, padding=self.dist_kernel//2).reshape(1, shape[0] * shape[1],
                                                                                    shape[0] * shape[1]).transpose(2, 1)
        with torch.no_grad():
            C = Cu.clone() * dist_mask

        for i in range(self.num_iter):
            pairwise_votes = C.clone()
            for _ in range(self.num_smooth_iter):
                pairwise_votes = self.pass_message(pairwise_votes, (shape[0], shape[1]))
                pairwise_votes = pairwise_votes / (pairwise_votes.sum(2, keepdim=True) + 1e-4)

            max_val, _ = pairwise_votes.max(2, keepdim=True)

            C = Cu + pairwise_votes
            C = C / (C.sum(2, keepdim=True) + 1e-4)

        return Cu, C, fg_mask, bg_mask


class MeanField(nn.Module):

    # feature map (RGB)
    # B = #num of object
    # shape of [N 3 H W]

    #@autocast(enabled=False)
    def __init__(self, feature_map, kernel_size=3, require_grad=False, theta0=0.5, theta1=30, theta2=10, alpha0=3,
                 iter=20, base=0.45, gamma=0.01):
        super(MeanField, self).__init__()
        self.require_grad = require_grad
        self.kernel_size = kernel_size
        with torch.no_grad():
            self.unfold = torch.nn.Unfold(kernel_size, stride=1, padding=kernel_size//2)
            feature_map = feature_map + 10
            unfold_feature_map = self.unfold(feature_map).view(feature_map.size(0), feature_map.size(1), kernel_size**2, -1)
            self.feature_map = feature_map
            self.theta0 = theta0
            self.theta1 = theta1
            self.theta2 = theta2
            self.alpha0 = alpha0
            self.gamma = gamma
            self.base = base
            self.spatial = torch.tensor((np.arange(kernel_size**2)//kernel_size - kernel_size//2) ** 2 +\
                                        (np.arange(kernel_size**2) % kernel_size - kernel_size//2) ** 2).to(feature_map.device).float()

            self.kernel = alpha0 * torch.exp((-(unfold_feature_map - feature_map.view(feature_map.size(0), feature_map.size(1), 1, -1)) ** 2).sum(1) / (2 * self.theta0 ** 2) + (-(self.spatial.view(1, -1, 1) / (2 * self.theta1 ** 2))))
            self.kernel = self.kernel.unsqueeze(1)

            self.iter = iter

    # input x
    # shape of [N H W]
    #@autocast(enabled=False)
    def forward(self, x, targets, inter_img_mask=None):
        with torch.no_grad():
            x = x * targets
            x = (x > 0.5).float() * (1 - self.base*2) + self.base
            U = torch.cat([1-x, x], 1)
            U = U.view(-1, 1, U.size(2), U.size(3))
            if inter_img_mask is not None:
                inter_img_mask.reshape(-1, 1, inter_img_mask.shape[2], inter_img_mask.shape[3])
            ret = U
            for _ in range(self.iter):
                nret = self.simple_forward(ret, targets, inter_img_mask)
                ret = nret
            ret = ret.view(-1, 2, ret.size(2), ret.size(3))
            ret = ret[:,1:]
            ret = (ret > 0.5).float()
            count = ret.reshape(ret.shape[0], -1).sum(1)
            valid = (count >= ret.shape[2] * ret.shape[3] * 0.05) * (count <= ret.shape[2] * ret.shape[3] * 0.95)
            valid = valid.float()
        return ret, valid

    #@autocast(enabled=False)
    def simple_forward(self, x, targets, inter_img_mask):
        h, w = x.size(2), x.size(3)
        unfold_x = self.unfold(-torch.log(x)).view(x.size(0)//2, 2, self.kernel_size**2, -1)
        aggre = (unfold_x * self.kernel).sum(2)
        aggre = aggre.view(-1, 1, h, w)
        f = torch.exp(-aggre)
        f = f.view(-1, 2, h, w)
        if inter_img_mask is not None:
            f += inter_img_mask * self.gamma
        f[:, 1:] *= targets
        f = f + 1e-6
        f = f / f.sum(1, keepdim=True)
        f = (f > 0.5).float() * (1 - self.base*2) + self.base
        f = f.view(-1, 1, h, w)

        return f