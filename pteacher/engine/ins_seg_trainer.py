# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import random
import numpy as np
from collections import OrderedDict, defaultdict
from typing import List
import copy

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures import Instances, Boxes, BitMasks, ROIMasks, BoxMode, polygons_to_bitmask, PolygonMasks, pairwise_iou
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import cat
from detectron2.utils.comm import get_world_size

from pteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)

from pteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from pteacher.data.point_sup_dataset_mapper import PointSupDatasetMapper, PointSupTwoCropSeparateDatasetMapper
from pteacher.engine.hooks import LossEvalHook
from pteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from pteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from pteacher.solver.build import build_lr_scheduler
from scipy.optimize import linear_sum_assignment
from detectron2.structures import ImageList
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from pteacher.utils.events import PSSODMetricPrinter
# from .inst_bank import ObjectFactory, ObjectQueues
from pteacher.engine.inst_bank import ObjectFactory, ObjectQueues, SemanticCorrSolver, MeanField
from pteacher.utils.comm import rgb_to_lab, unfold_wo_center, get_images_color_similarity, compute_pairwise_term
from torch.cuda.amp import autocast
import pdb

# Supervised-only Trainer
class MaskRCNNBaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)


# Unbiased Teacher Trainer without pointsup
class MaskRCNNpteacherTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        assert cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE == "thresholding", 'Hungarian is not supported in Mask RCNN Unbiased Teacher Trainer!'

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        # dict_keys(['model', 'optimizer', 'scheduler', 'iteration'])
        # import pdb
        # pdb.set_trace()
        # aa=torch.load("results/semi_weak_sup/mrcnn/r50_coco_10_point_match_ins_mil_d/model_0049999_old.pth")
        # checkpoint = self.checkpointer.resume_or_load("results/semi_weak_sup/mrcnn/r50_coco_10_point_match_ins_mil_d/model_0049999_old.pth", resume=False)
        # aa['model'] = self.checkpointer.model.state_dict()
        # aa['optimizer'] = self.checkpointer.checkpointables['optimizer'].state_dict()
        # torch.save(aa, "results/semi_weak_sup/mrcnn/r50_coco_10_point_match_ins_mil/model_0049999.pth")
        # import pdb
        # pdb.set_trace()
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if self.checkpointer.has_checkpoint():
            if resume:
                self.start_iter = checkpoint.get("iteration", -1) + 1
                self.scheduler.milestones = self.cfg.SOLVER.STEPS
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        if cfg.TEST.EVALUATOR == "COCOeval":
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            raise ValueError("Unknown test evaluator.")

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    def build_writers(self):
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            PSSODMetricPrinter(self.max_iter),
            # CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih", mask_thres=0.5):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

            if isinstance(proposal_bbox_inst.pred_masks, ROIMasks):
                roi_masks = proposal_bbox_inst.pred_masks[valid_map]
            else:
                # pred_masks is a tensor of shape (N, 1, M, M)
                roi_masks = ROIMasks(proposal_bbox_inst.pred_masks[valid_map][:, 0, :, :])

            output_height, output_width = image_shape
            # print("line 813", image_shape)
            # import pdb
            # pdb.set_trace()
            new_proposal_inst.gt_masks = roi_masks.to_bitmasks(new_proposal_inst.gt_boxes, output_height, output_width,
                                                               mask_thres)
        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[pteacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            print(self.iter)
            record_dict, _, _, _ = self.model(label_data_q, branch="supervised")
            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)

            elif (
                self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}
            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            joint_proposal_dict = {}
            joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
            (
                pesudo_proposals_rpn_unsup_k,
                nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                proposals_rpn_unsup_k, cur_threshold, "rpn", self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE
            )
            joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, "roih", self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE
            )
            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

            num_unlabel_gt = np.mean([len(a['instances']) for a in unlabel_data_q])
            #  add pseudo-label to unlabeled data
            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)

            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            )
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            )

            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q

            num_unlabel_pseudo = np.mean([len(a['instances']) for a in all_unlabel_data])

            record_dict['num_unlabel_gt'] = num_unlabel_gt
            record_dict['num_unlabel_pseudo'] = num_unlabel_pseudo

            record_all_label_data, _, _, _ = self.model(
                all_label_data, branch="supervised"
            )
            record_dict.update(record_all_label_data)
            record_all_unlabel_data, _, _, _ = self.model(
                all_unlabel_data, branch="supervised"
            )
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif "loss_point" in key:
                        loss_dict[key] = record_dict[key] * 0.05
                        record_dict[key] = record_dict[key] * 0.05
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                            record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_teacher))

        if cfg.TEST.VAL_LOSS:  # default is True # save training time if not applied
            ret.append(
                LossEvalHook(
                    cfg.TEST.EVAL_PERIOD,
                    self.model,
                    build_detection_test_loader(
                        self.cfg,
                        self.cfg.DATASETS.TEST[0],
                        DatasetMapper(self.cfg, True),
                    ),
                    model_output="loss_proposal",
                    model_name="student",
                )
            )

            ret.append(
                LossEvalHook(
                    cfg.TEST.EVAL_PERIOD,
                    self.model_teacher,
                    build_detection_test_loader(
                        self.cfg,
                        self.cfg.DATASETS.TEST[0],
                        DatasetMapper(self.cfg, True),
                    ),
                    model_output="loss_proposal",
                    model_name="",
                )
            )

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


# PointSup Trainer based on Unbiased Teacher Trainer for mask-rcnn
class MaskRCNNPointSupTrainer(MaskRCNNpteacherTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        # boxinst utils
        self.amp_enabled = cfg.SOLVER.AMP.ENABLED
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        self.boxinst_pairwise_warmup_iters = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS
            
        # init inst_bank which stores cropped instances
        self.inst_bank = defaultdict(list)
        self.num_gts_per_cls = defaultdict(int)
        self.num_pseudos_per_cls = defaultdict(int)
        self.pseudo_recall = defaultdict(float)
        self.sampling_freq = defaultdict(float)

        for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES):
            self.inst_bank[i] = []
            self.num_gts_per_cls[i] = 1
            self.num_pseudos_per_cls[i] = 1
            self.pseudo_recall[i] = 1
            self.sampling_freq[i] = 1.0 / 80

        self.inst_bank = [self.inst_bank[i] for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES)]

        self.bank_update_num = 20
        self.bank_length = 500

        # if self.boxinst_enabled:
        if True:
            # used for correspondence visualization 
            img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

            loss_corr = dict(
                type='InfoNCE',
                loss_weight=1.0,
                corr_exp=1.0,
                corr_eps=0.05,
                gaussian_filter_size=3,
                low_score=0.3,
                corr_num_iter=10,
                corr_num_smooth_iter=1,
                save_corr_img=False,
                dist_kernel=9,
                obj_bank=dict(
                    img_norm_cfg=img_norm_cfg,
                    len_object_queues=100,
                    fg_iou_thresh=0.7,
                    bg_iou_thresh=0.7,
                    ratio_range=[0.9, 1.2],
                    appear_thresh=0.7,
                    min_retrieval_objs=2,
                    max_retrieval_objs=5,
                    feat_height=7,
                    feat_width=7,
                    mask_height=28,
                    mask_width=28,
                    img_height=200,
                    img_width=200,
                    min_size=32,
                    num_gpu_bank=20,
                )
            )
            self.semantic_corr_solver = SemanticCorrSolver(loss_corr['corr_exp'],
                                                           loss_corr['corr_eps'],
                                                           loss_corr['gaussian_filter_size'],
                                                           loss_corr['low_score'],
                                                           loss_corr['corr_num_iter'],
                                                           loss_corr['corr_num_smooth_iter'],
                                                           dist_kernel=loss_corr['dist_kernel'])

            obj_bank = loss_corr['obj_bank']
            self.obj_bank = obj_bank
            self.vis_cnt = 0
            self.corr_loss_weight = loss_corr['loss_weight']
            self.object_queues = ObjectQueues(num_class=cfg.MODEL.ROI_HEADS.NUM_CLASSES, len_queue=obj_bank['len_object_queues'],
                                              fg_iou_thresh=obj_bank['fg_iou_thresh'],
                                              bg_iou_thresh=obj_bank['bg_iou_thresh'],
                                              ratio_range=obj_bank['ratio_range'],
                                              appear_thresh=obj_bank['appear_thresh'],
                                              max_retrieval_objs=obj_bank['max_retrieval_objs'])

            self.img_norm_cfg = obj_bank['img_norm_cfg']
            self.corr_feat_height, self.corr_feat_width = obj_bank['feat_height'], obj_bank['feat_width']
            self.corr_mask_height, self.corr_mask_width = obj_bank['mask_height'], obj_bank['mask_width']
            self.objbank_min_size = obj_bank['min_size']
            self.save_corr_img = loss_corr['save_corr_img']
            self.qobj = None
            self.num_created_gpu_bank = 0
            self.num_gpu_bank = obj_bank['num_gpu_bank']
            self.color_panel = np.array([ (i * 32, j * 32, k * 32) for i in range(8)
                                 for j in range(8) for k in range(8) ])
            np.random.shuffle(self.color_panel)

            # self.loss_corr = nn.CrossEntropyLoss()


        self.resume = False
        self.register_hooks(self.build_hooks())


    def resume_or_load(self, resume=True):
        self.resume = resume
        super().resume_or_load(resume=resume)


    def rename_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                # will not used in training, except the points
                label_datum["point_instances"] = label_datum["instances"]
                del label_datum["instances"]
                # remove gt_boxes for unlabeled images
                label_datum["point_instances"].remove('gt_boxes')
        return label_data


    def state_dict(self):
        ret = super().state_dict()
        if self.amp_enabled:
            ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret


    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.amp_enabled:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])


    @classmethod
    def build_train_loader(cls, cfg):
        mapper = PointSupTwoCropSeparateDatasetMapper(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)


    def build_writers(self):
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            PSSODMetricPrinter(self.max_iter),
            # CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih", mask_thres=0.5):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]

            new_proposal_inst.pred_boxes = new_proposal_inst.gt_boxes
            new_proposal_inst.pred_classes = new_proposal_inst.gt_classes
            new_proposal_inst.gt_pseudo_scores = proposal_bbox_inst.scores[valid_map]
            # import pdb
            # pdb.set_trace()
            if hasattr(proposal_bbox_inst, "gt_image_color_similarity"):
                new_proposal_inst.gt_image_color_similarity = proposal_bbox_inst.gt_image_color_similarity
            if hasattr(proposal_bbox_inst, "gt_bitmasks_full"):
                new_proposal_inst.gt_bitmasks_full = proposal_bbox_inst.gt_bitmasks_full
            if hasattr(proposal_bbox_inst, "gt_point_coords"):
                new_proposal_inst.gt_point_coords = proposal_bbox_inst.gt_point_coords
            if hasattr(proposal_bbox_inst, "gt_point_labels"):
                new_proposal_inst.gt_point_labels = proposal_bbox_inst.gt_point_labels

            if isinstance(proposal_bbox_inst.pred_masks, ROIMasks):
                roi_masks = proposal_bbox_inst.pred_masks[valid_map]
            else:
                # pred_masks is a tensor of shape (N, 1, M, M)
                roi_masks = ROIMasks(proposal_bbox_inst.pred_masks[valid_map][:, 0, :, :])
        
            output_height, output_width = image_shape
            # print("line 813", image_shape)
            # import pdb
            # pdb.set_trace()
            new_proposal_inst.gt_masks = roi_masks.to_bitmasks(new_proposal_inst.gt_boxes, output_height, output_width, mask_thres)
            # print("line 815", new_proposal_inst.gt_masks.size())
        return new_proposal_inst

    def inst_bank_has_empty_classes(self):
        for i, inst_bank_per_class in enumerate(self.inst_bank):
            if len(inst_bank_per_class) == 0:
                print("Class {} doesn't have any cropped instances!".format(i))
                return True
        return False


    def rasterize_polygons_within_box(
        self, polygons: List[np.ndarray], box: np.ndarray
    ) -> torch.Tensor:
        """
        Rasterize the polygons into a mask image and
        crop the mask content in the given box.
        The cropped mask is resized to (mask_size, mask_size).

        This function is used when generating training targets for mask head in Mask R-CNN.
        Given original ground-truth masks for an image, new ground-truth mask
        training targets in the size of `mask_size x mask_size`
        must be provided for each predicted box. This function will be called to
        produce such targets.

        Args:
            polygons (list[ndarray[float]]): a list of polygons, which represents an instance.
            box: 4-element numpy array
            mask_size (int):

        Returns:
            Tensor: BoolTensor of shape (mask_size, mask_size)
        """
        # 1. Shift the polygons w.r.t the boxes
        w, h = box[2] - box[0], box[3] - box[1]
        # polygons_origin = copy.deepcopy(polygons)
        polygons = copy.deepcopy(polygons)
        for p in polygons:
            p[0::2] = p[0::2] - box[0]
            p[1::2] = p[1::2] - box[1]

        # 2. Rescale the polygons to the new box size
        # max() to avoid division by small number
        ratio_h = h / max(h, 0.1)
        ratio_w = w / max(w, 0.1)

        if ratio_h == ratio_w:
            for p in polygons:
                p *= ratio_h
        else:
            for p in polygons:
                p[0::2] *= ratio_w
                p[1::2] *= ratio_h

        # if len(polygons) <= 0:
        #     print(polygons)

        # 3. Rasterize the polygons with coco api
        mask = polygons_to_bitmask(polygons, h, w)
        mask = torch.from_numpy(mask)
        return mask


    def update_inst_bank(self, label_data):
        # num_proposal_output = 0.0
        # label_data = random.choice([label_data_q, label_data_k])
        # DEBUG = False
        for label_data_per_img in label_data:

            gt_labels = label_data_per_img["instances"].gt_classes
            gt_bboxes = label_data_per_img["instances"].gt_boxes.tensor
            gt_polymasks = label_data_per_img["instances"].gt_masks

            # mask_side_len = 28
            # gt_masks = label_data_per_img["instances"].gt_masks.crop_and_resize(gt_bboxes, mask_side_len)

            # # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            # gt_masks.append(gt_masks_per_image)

            img = label_data_per_img["image"]
            c, h, w = img.shape
            unique_labels = list(set(gt_labels.tolist()))
            for l in unique_labels:
                candidate_bboxes = gt_bboxes[gt_labels == l]
                candidate_polymasks = gt_polymasks[gt_labels == l]
                num = 0
                inds = list(range(len(candidate_bboxes)))
                np.random.shuffle(inds)
                for i in inds:
                    if num >= self.bank_update_num:
                        break
                    bbox = candidate_bboxes[i]
                    # x1, x2 = bbox[0::2].min(), bbox[0::2].max()
                    # y1, y2 = bbox[1::2].min(), bbox[1::2].max()

                    x1, y1, x2, y2 = candidate_bboxes[i]
                    if (x2 - x1 < 10) or (y2 - y1) < 10:
                        continue
                    num += 1

                    crop_x1 = int(x1)
                    crop_y1 = int(y1)
                    crop_x2 = int(x2)
                    crop_y2 = int(y2)

                    bbox = np.array([crop_x1, crop_y1, crop_x2, crop_y2])
                    gt_polymask = copy.deepcopy(candidate_polymasks.polygons[i])
                    # [crop_h, crop_w]
                    gt_bitmask = self.rasterize_polygons_within_box(gt_polymask, bbox)
                    # 1. Shift the polygons w.r.t the boxes
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    
                    for p in gt_polymask:
                        p[0::2] = p[0::2] - bbox[0]
                        p[1::2] = p[1::2] - bbox[1]
                    # import pdb
                    # pdb.set_trace()

                    crop_img = img[:, crop_y1:crop_y2, crop_x1:crop_x2].clone()

                    if len(self.inst_bank[l]) < self.bank_length:
                        self.inst_bank[l].append((crop_img, gt_bitmask, gt_polymask))
                    else:
                        p_i = np.random.choice(range(self.bank_length))
                        self.inst_bank[l][p_i] = (crop_img, gt_bitmask, gt_polymask)

    
    def paste_inst_bank_to_unlabel_data_v2(self, unlabel_image, pseudo_instance, paste_classes, paste_positions, num_paste_objs=2):
        _, img_h, img_w = unlabel_image.size()
        for _ in range(num_paste_objs):
            # step 1: random location paste
            for _ in range(4): # try times
                paste_cls = random.randint(0, 79) # random sample cls index between 0-79 for coco
                paste_cls = torch.tensor(paste_cls).to(paste_classes.device)
                if len(self.inst_bank[int(paste_cls)]) == 0:
                    paste_img = None
                    continue
                paste_img, paste_bitmask, paste_polymask = random.choice(self.inst_bank[int(paste_cls)])
                p_h, p_w = paste_img.shape[1:]
                if img_w - p_w < 1 or img_h - p_h < 1:
                    paste_img = None
                    continue
                break

            if paste_img is not None:
                p_x1 = np.random.randint(0, img_w - p_w)
                p_y1 = np.random.randint(0, img_h - p_h)
                for p in paste_polymask:
                        p[0::2] = p[0::2] + p_x1
                        p[1::2] = p[1::2] + p_y1

                paste_img = paste_img * paste_bitmask + unlabel_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)] * paste_bitmask.bitwise_not()
                unlabel_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)] = paste_img
                paste_box = Boxes(torch.tensor([[p_x1, p_y1, p_x1 + p_w, p_y1 + p_h]]).to(paste_classes.device))

                paste_instance = Instances(pseudo_instance.image_size)
                paste_instance.gt_classes = paste_cls[None]
                paste_instance.scores = torch.ones_like(paste_cls[None])
                paste_instance.gt_reg_loss_weight = torch.ones_like(paste_cls[None])
                paste_instance.gt_boxes = paste_box
                paste_instance.gt_masks = BitMasks.from_polygon_masks([paste_polymask], height=img_h, width=img_w).to(paste_cls.device)

                if len(pseudo_instance) > 0:
                    ious = pairwise_iou(paste_box, pseudo_instance.gt_boxes)
                    # print(ious[0])
                    for ind, iou in enumerate(ious[0]):
                        # print(ind, iou)
                        if iou > 0.05:
                            # import pdb
                            # pdb.set_trace()
                            ins = pseudo_instance[ind]
                            ins.gt_masks.tensor[torch.bitwise_and(paste_instance.gt_masks.tensor == ins.gt_masks.tensor, ins.gt_masks.tensor==True)] = False
                        
                    # recompute bounding boxes based on modified masks
                    # import pdb
                    # pdb.set_trace()
                    pseudo_instance.gt_boxes = pseudo_instance.gt_masks.get_bounding_boxes().to(pseudo_instance.gt_masks.device)
                    # remove instance whose box area smaller than xxx
                    non_empty = pseudo_instance.gt_masks.nonempty()
                    pseudo_instance = pseudo_instance[non_empty]
                    # import pdb
                    # pdb.set_trace()
                pseudo_instance = Instances.cat([paste_instance, pseudo_instance])

        for paste_pos, paste_cls in zip(paste_positions, paste_classes):
            if len(self.inst_bank[int(paste_cls)]) > 0:
                paste_img, paste_bitmask, paste_polymask = random.choice(self.inst_bank[int(paste_cls)])
                p_h, p_w = paste_img.shape[1:]

                for _ in range(5):  # try time

                    if img_w - p_w < 1 or img_h - p_h < 1:
                        break

                    p_x1 = np.random.randint(0, img_w - p_w)
                    p_y1 = np.random.randint(0, img_h - p_h)

                    paste_box = Boxes(torch.tensor([[p_x1, p_y1, p_x1 + p_w, p_y1 + p_h]]).to(paste_cls.device))
                    ious = torch.zeros_like(paste_cls)
                    if len(pseudo_instance) > 0:
                        ious = pairwise_iou(paste_box, pseudo_instance.gt_boxes)

                    if ious.max() < 1e-2:
                        # print('paste')
                        paste_img = paste_img * paste_bitmask + unlabel_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)] * paste_bitmask.bitwise_not()

                        for p in paste_polymask:
                            p[0::2] = p[0::2] + p_x1
                            p[1::2] = p[1::2] + p_y1

                        unlabel_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)] = paste_img

                        paste_instance = Instances(pseudo_instance.image_size)
                        paste_instance.gt_classes = paste_cls[None]
                        paste_instance.scores = torch.ones_like(paste_cls[None])
                        paste_instance.gt_reg_loss_weight = torch.ones_like(paste_cls[None])
                        paste_instance.gt_boxes = paste_box
                        paste_instance.gt_masks = BitMasks.from_polygon_masks([paste_polymask], height=img_h, width=img_w).to(paste_cls.device)
                        # paste_instance.gt_masks = PolygonMasks([paste_polymask])
                        pseudo_instance = Instances.cat([paste_instance, pseudo_instance])
                        break

        return unlabel_image, pseudo_instance


    def paste_inst_bank_to_unlabel_data_v3(self, unlabel_image, pseudo_instance, paste_classes, paste_positions, num_paste_objs=3, mixup_lambda=0.65):
        _, img_h, img_w = unlabel_image.size()
        for _ in range(num_paste_objs):
            # step 1: random location paste
            for _ in range(4): # try times
                # paste_cls = random.randint(0, 79) # random sample cls index between 0-79 for coco
                p = np.array([v for v in self.sampling_freq.values()])
                paste_cls = np.random.choice(list(range(0, 80)), p=p.ravel())
                paste_cls = torch.tensor(paste_cls).to(paste_classes.device)
                if len(self.inst_bank[int(paste_cls)]) == 0:
                    paste_img = None
                    continue
                paste_img, paste_bitmask, paste_polymask = random.choice(self.inst_bank[int(paste_cls)])
                p_h, p_w = paste_img.shape[1:]
                if img_w - p_w < 1 or img_h - p_h < 1:
                    paste_img = None
                    continue
                break

            if paste_img is not None:
                p_x1 = np.random.randint(0, img_w - p_w)
                p_y1 = np.random.randint(0, img_h - p_h)
                for p in paste_polymask:
                        p[0::2] = p[0::2] + p_x1
                        p[1::2] = p[1::2] + p_y1

                paste_img = paste_img * paste_bitmask + unlabel_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)] * paste_bitmask.bitwise_not()
                # unlabel_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)] = paste_img


                                # mixup
                unlabel_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)] = mixup_lambda * paste_img + \
                (1.0 - mixup_lambda) * unlabel_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)]

                paste_box = Boxes(torch.tensor([[p_x1, p_y1, p_x1 + p_w, p_y1 + p_h]]).to(paste_classes.device))

                paste_instance = Instances(pseudo_instance.image_size)
                paste_instance.gt_classes = paste_cls[None]
                paste_instance.scores = torch.ones_like(paste_cls[None])
                paste_instance.gt_reg_loss_weight = torch.ones_like(paste_cls[None])
                paste_instance.gt_boxes = paste_box
                paste_instance.gt_masks = BitMasks.from_polygon_masks([paste_polymask], height=img_h, width=img_w).to(paste_cls.device)

                if len(pseudo_instance) > 0:                        
                    # recompute bounding boxes based on modified masks
                    pseudo_instance.gt_boxes = pseudo_instance.gt_masks.get_bounding_boxes().to(pseudo_instance.gt_masks.device)
                    # remove instance whose box area smaller than xxx
                    non_empty = pseudo_instance.gt_masks.nonempty()
                    pseudo_instance = pseudo_instance[non_empty]

                pseudo_instance = Instances.cat([paste_instance, pseudo_instance])

        for paste_pos, paste_cls in zip(paste_positions, paste_classes):
            if len(self.inst_bank[int(paste_cls)]) > 0:
                paste_img, paste_bitmask, paste_polymask = random.choice(self.inst_bank[int(paste_cls)])
                p_h, p_w = paste_img.shape[1:]

                for _ in range(5):  # try time

                    if img_w - p_w < 1 or img_h - p_h < 1:
                        break

                    p_x1 = np.random.randint(0, img_w - p_w)
                    p_y1 = np.random.randint(0, img_h - p_h)

                    paste_box = Boxes(torch.tensor([[p_x1, p_y1, p_x1 + p_w, p_y1 + p_h]]).to(paste_cls.device))
                    ious = torch.zeros_like(paste_cls)
                    if len(pseudo_instance) > 0:
                        ious = pairwise_iou(paste_box, pseudo_instance.gt_boxes)

                    if ious.max() < 1e-2:
                        # print('paste')
                        paste_img = paste_img * paste_bitmask + unlabel_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)] * paste_bitmask.bitwise_not()

                        for p in paste_polymask:
                            p[0::2] = p[0::2] + p_x1
                            p[1::2] = p[1::2] + p_y1

                        unlabel_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)] = paste_img

                        paste_instance = Instances(pseudo_instance.image_size)
                        paste_instance.gt_classes = paste_cls[None]
                        paste_instance.scores = torch.ones_like(paste_cls[None])
                        paste_instance.gt_reg_loss_weight = torch.ones_like(paste_cls[None])
                        paste_instance.gt_boxes = paste_box
                        paste_instance.gt_masks = BitMasks.from_polygon_masks([paste_polymask], height=img_h, width=img_w).to(paste_cls.device)
                        # paste_instance.gt_masks = PolygonMasks([paste_polymask])
                        pseudo_instance = Instances.cat([paste_instance, pseudo_instance])
                        break

        return unlabel_image, pseudo_instance


    def paste_inst_bank_to_label_data(self, label_image, src_instance, num_paste_objs=3, mixup_lambda=0.65):
        _, img_h, img_w = label_image.size()
        for _ in range(num_paste_objs):
            # step 1: random location paste
            for _ in range(4): # try times
                # paste_cls = random.randint(0, 79) # random sample cls index between 0-79 for coco
                p = np.array([v for v in self.sampling_freq.values()])
                paste_cls = np.random.choice(list(range(0, 80)), p=p.ravel())
                paste_cls = torch.tensor(paste_cls).to(src_instance.gt_classes.device)
                if len(self.inst_bank[int(paste_cls)]) == 0:
                    paste_img = None
                    continue
                # paste_img = random.choice(self.inst_bank[int(paste_cls)])
                paste_img, paste_bitmask, paste_polymask = random.choice(self.inst_bank[int(paste_cls)])
                p_h, p_w = paste_img.shape[1:]
                if img_w - p_w < 1 or img_h - p_h < 1:
                    paste_img = None
                    continue
                break

            if paste_img is not None:
                p_x1 = np.random.randint(0, img_w - p_w)
                p_y1 = np.random.randint(0, img_h - p_h)
                paste_box = Boxes(torch.tensor([[p_x1, p_y1, p_x1 + p_w, p_y1 + p_h]]).to(src_instance.gt_classes.device))

                for p in paste_polymask:
                    p[0::2] = p[0::2] + p_x1
                    p[1::2] = p[1::2] + p_y1
                # mixup
                # label_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)] = mixup_lambda * paste_img + \
                # (1.0 - mixup_lambda) * label_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)]

                paste_img = paste_img * paste_bitmask + label_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)] * paste_bitmask.bitwise_not()
                label_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)] = paste_img

                paste_instance = Instances(src_instance.image_size)
                paste_instance.gt_classes = paste_cls[None]
                paste_instance.gt_boxes = paste_box
                # paste_instance.gt_masks = BitMasks.from_polygon_masks([paste_polymask], height=img_h, width=img_w).to(paste_cls.device)
                paste_instance.gt_masks = PolygonMasks([paste_polymask])
                #  random.uniform(0, 1)
                # np.random.uniform(low=0.5, high=13.3, size=(50,))
                pos_coord = torch.DoubleTensor([[[np.random.uniform(p_x1, p_x1 + p_w), np.random.uniform(p_y1, p_y1 + p_h)]]])
                neg_coord = torch.DoubleTensor([[[p_x1, p_y1]]])
                gt_point_coords = torch.cat([pos_coord, neg_coord], dim=1)
                
                paste_instance.gt_point_coords = gt_point_coords.to(src_instance.gt_classes.device)
                paste_instance.gt_point_labels = torch.DoubleTensor([[1,0]]).to(src_instance.gt_classes.device)
                src_instance = Instances.cat([paste_instance, src_instance])
            
        return label_image, src_instance


    def process_pseudo_label_with_point_anno(
            self, unlabel_data_k, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method="",
    ):
        list_instances = []
        num_proposal_output = 0.0
        gt_image_list = []
        gt_point_coords_list = []
        gt_point_labels_list = []
        gt_bbox_classes_list = []
        for i in range(len(unlabel_data_k)):
            gt_point_coords = unlabel_data_k[i]['instances'].gt_point_coords

            if psedo_label_method == "hungarian_with_center_point":
                # center point of gt box
                gt_point_coords[:, 1, 0] = (unlabel_data_k[i]['instances'].gt_boxes.tensor[:,0] + unlabel_data_k[i]['instances'].gt_boxes.tensor[:,2]) * 0.5
                gt_point_coords[:, 1, 1] = (unlabel_data_k[i]['instances'].gt_boxes.tensor[:,1] + unlabel_data_k[i]['instances'].gt_boxes.tensor[:,3]) * 0.5

            gt_image_list.append(unlabel_data_k[i]['image'])
            gt_point_coords_list.append(gt_point_coords)
            gt_point_labels_list.append(unlabel_data_k[i]['instances'].gt_point_labels)
            gt_bbox_classes_list.append(unlabel_data_k[i]['instances'].gt_classes)
        # per img iter
        for ind, (point_coord_inst, point_class_inst, point_label_inst, proposal_bbox_inst) in enumerate(zip(
                                                                                            # gt_image_list,
                                                                                            gt_point_coords_list,
                                                                                            gt_bbox_classes_list,
                                                                                            gt_point_labels_list,
                                                                                            proposals_rpn_unsup_k)):
            # step 1. thresholding
            # import pdb
            # pdb.set_trace()
            pos_point_coord_inst = point_coord_inst[:, 0, :]
            if psedo_label_method == "thresholding":
                # Instances(num_instances=0, image_height=1105, image_width=736,
                # fields=[gt_boxes: Boxes(tensor([], device='cuda:0', size=(0, 4))),
                # objectness_logits: tensor([], device='cuda:0')])
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            elif psedo_label_method == 'top-1':
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
                
                _scores = proposal_bbox_inst.scores
                _bboxes = proposal_bbox_inst.gt_boxes.tensor
                _labels = proposal_bbox_inst.gt_classes

                _points = pos_point_coord_inst.to(_scores.device)
                _point_classes = point_class_inst.to(_scores.device)
                _point_labels = point_label_inst.to(_scores.device)
          
                # 0 for point inside box, and 1 for outside box
                cost_inside_box = 1.0 - (_points[:, 0][None, :] > _bboxes[:, 0][:, None]) * \
                                        (_points[:, 0][None, :] < _bboxes[:, 2][:, None]) * \
                                        (_points[:, 1][None, :] > _bboxes[:, 1][:, None]) * \
                                        (_points[:, 1][None, :] < _bboxes[:, 3][:, None]) * 1.0

                # when point and box has same class label, cost is (1 - score), elsewise cost is 1.0
                cost_prob = 1.0 - (_labels[:, None] == _point_classes[None, :]) * _scores[:, None]

                cost = cost_inside_box * 1.0 + cost_prob * 1.0                  # (#proposals, #points)
                if len(proposal_bbox_inst) > 0:
                    top1_value, top1_index = cost.min(axis=0)
                    keep = top1_value < 1.0
                    proposal_bbox_inst = proposal_bbox_inst[top1_index][keep]
                
                    gt_point_coords = point_coord_inst[top1_index][keep]
                    proposal_bbox_inst.set("gt_point_coords", gt_point_coords)                
                    gt_point_labels = point_label_inst[top1_index][keep]
                    proposal_bbox_inst.set("gt_point_labels", gt_point_labels)
                    proposal_bbox_inst.gt_reg_loss_weight = torch.zeros_like(proposal_bbox_inst.gt_classes)

            elif psedo_label_method == "hungarian":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
                # step 2. choose pseudo bboxes with provised points
                _scores = proposal_bbox_inst.gt_pseudo_scores
                _bboxes = proposal_bbox_inst.gt_boxes.tensor
                _labels = proposal_bbox_inst.gt_classes

                _points = pos_point_coord_inst.to(_scores.device)
                _point_classes = point_class_inst.to(_scores.device)
                _point_labels = point_label_inst.to(_scores.device)
                # inside = (point_coords >= np.array([0, 0])) & (point_coords <= np.array(image_size[::-1]))
                # inside = inside.all(axis=1)

                # 0 for point inside box, and 1 for outside box
                cost_inside_box = 1.0 - (_points[:, 0][None, :] > _bboxes[:, 0][:, None]) * (
                            _points[:, 0][None, :] < _bboxes[:, 2][:, None]) * \
                                  (_points[:, 1][None, :] > _bboxes[:, 1][:, None]) * (
                                              _points[:, 1][None, :] < _bboxes[:, 3][:, None]) * 1.0

                # when point and box has same class label, cost is (1 - score), elsewise cost is 1.0
                cost_prob = 1.0 - (_labels[:, None] == _point_classes[None, :]) * _scores[:, None]

                cost = cost_inside_box * 1.0 + cost_prob * 1.0
                cost = cost.detach().cpu()
                matched_row_inds, matched_col_inds = linear_sum_assignment(cost)

                # only preserve indise box and has the same predicted class

                keep = (cost_inside_box[matched_row_inds, matched_col_inds] < 0.5) & (
                            _labels[matched_row_inds] == _point_classes[matched_col_inds])

                proposal_bbox_inst = proposal_bbox_inst[matched_row_inds][keep]
                gt_point_coords = point_coord_inst[matched_col_inds][keep]
                proposal_bbox_inst.set("gt_point_coords", gt_point_coords)                
                gt_point_labels = point_label_inst[matched_col_inds][keep]
                proposal_bbox_inst.set("gt_point_labels", gt_point_labels)
                proposal_bbox_inst.gt_reg_loss_weight = torch.zeros_like(proposal_bbox_inst.gt_classes)

                # gt_labels = proposal_bbox_inst.gt_classes
                # unique_labels = list(set(gt_labels.tolist()))
                # for label in unique_labels:
                #     self.num_pseudos_per_cls[int(label)] += int((gt_labels==label).sum())

                # paste label gt to proposal_bbox_inst based on point annotations which failed to match any proposals.
                paste_classes = None
                paste_positions = None
                # if len(matched_col_inds) > 0 and keep.sum() < keep.size(0):
                #     paste_classes = _point_classes[matched_col_inds][keep==False]
                #     paste_positions = _points[matched_col_inds][keep==False]

                # elif len(matched_col_inds) == 0:
                #     paste_classes = _point_classes
                #     paste_positions = _points

                if paste_classes is not None and paste_positions is not None:
                    # print("need paste1")
                    unlabel_data_k[ind]['image'], proposal_bbox_inst = \
                    self.paste_inst_bank_to_unlabel_data_v2(unlabel_data_k[ind]['image'], 
                                                            proposal_bbox_inst, 
                                                            paste_classes, 
                                                            paste_positions,
                                                            num_paste_objs=3)
                    # unlabel_data_k[ind]['image'], proposal_bbox_inst = \
                    # self.paste_inst_bank_to_unlabel_data_v3(unlabel_data_k[ind]['image'], 
                    #                                         proposal_bbox_inst, 
                    #                                         paste_classes, 
                    #                                         paste_positions,
                    #                                         num_paste_objs=3, 
                    #                                         mixup_lambda=0.65)

                    debug=False
                    if debug:
                        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
                        scale = 1.0
                        img = unlabel_data_k[ind]["image"].permute(1, 2, 0).numpy().copy()
                        img_id = unlabel_data_k[ind]['file_name'].split('/')[-1]
                        visualizer = Visualizer(img, metadata=metadata, scale=scale)
                        # target_fields = unlabel_data_k[ind]['instances'].get_fields()
                        target_fields = proposal_bbox_inst.get_fields()
                        labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
                        vis = visualizer.overlay_instances(
                            labels=labels,
                            boxes=target_fields.get("gt_boxes", None).to('cpu'),
                            masks=target_fields.get("gt_masks", None).to('cpu'),
                            keypoints=None,
                        )
                        dirname = "./results/vis_v3"
                        fname = img_id[:-4] + "_" + str(img.shape[0]) + "x" + str(img.shape[1]) + "_after.jpg"
                        filepath = os.path.join(dirname, fname)
                        print("Saving to {} ...".format(filepath))
                        vis.save(filepath)
            else:
                raise ValueError("Unkown pseudo label boxes methods")

            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)

        return list_instances, num_proposal_output

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[pteacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data

        if self.boxinst_enabled:
            for l_q, l_k, unl_q, unl_k in zip(label_data_q, label_data_k, unlabel_data_q, unlabel_data_k):
                l_q['image_weak'] = l_k['image']
                unl_q['image_weak'] = unl_k['image']

        for l in label_data_q:
            # print('before', len(l['instances']))
            gt_labels = l['instances'].gt_classes
            unique_labels = list(set(gt_labels.tolist()))

            for label in unique_labels:
                self.num_gts_per_cls[int(label)] += int((gt_labels==label).sum())

            # for gt_class in l['instances'].gt_classes:
            #     self.num_gts_per_cls[int(gt_class)] += 1

            l['image'], l['instances'] = self.paste_inst_bank_to_label_data(l['image'], l['instances'], num_paste_objs=3, mixup_lambda=0.5)

        # if self.resume and self.inst_bank_has_empty_classes():
        #     while self.inst_bank_has_empty_classes():
        #         data = next(self._trainer._data_loader_iter)
        #         label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        #         self.update_inst_bank(label_data_q)

                # for l in unique_labels:
                #     self.num_gts_per_cls[int(l)] += int((gt_labels==l).sum())


        # self.update_inst_bank(label_data_q)
        data_time = time.perf_counter() - start

        # burn-in stage (supervised training with labeled data)
        if self.boxinst_enabled:
            # boxinst_pairwise_loss_warmup_factor = max(min((self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP) / float(self.boxinst_pairwise_warmup_iters), 1.0), 0.0)
            boxinst_pairwise_loss_warmup_factor = max(min((self.iter) / float(self.boxinst_pairwise_warmup_iters), 1.0), 0.0)
            # print("boxinst_pairwise_loss_warmup_factor", boxinst_pairwise_loss_warmup_factor)

        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            # input both strong and weak supervised data into model
            # label_data_q.extend(label_data_k)
            for l in label_data_q:    # clone for labeled images
                l['point_instances'] = l['instances']
            with autocast(enabled=self.amp_enabled):
                record_dict, _, _, _ = self.model(label_data_q,
                                                branch="supervised",
                                                mil_img_filter_bg_proposal=self.cfg.SEMISUPNET.IMG_MIL_FILTER_BG,
                                                add_ground_truth_to_point_proposals=True,
                                                add_ss_proposals_to_point_proposals=self.cfg.SEMISUPNET.USE_SS_PROPOSALS)

                pairwise_loss = self.loss_pairwise(label_data_k, branch="supervised")
                record_dict.update(pairwise_loss)

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    if key == "loss_point":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.POINT_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.POINT_LOSS_WEIGHT
                    elif key == "loss_img_mil":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
                    elif key == "loss_ins_mil":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT
                    elif key == "loss_project":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.PRJ_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.PRJ_LOSS_WEIGHT
                    elif key == "loss_pairwise":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.PAIRWISE_LOSS_WEIGHT * boxinst_pairwise_loss_warmup_factor
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.PAIRWISE_LOSS_WEIGHT * boxinst_pairwise_loss_warmup_factor
                    # elif key == "loss_mask":
                    #     loss_dict[key] = record_dict[key] * 0
                    #     record_dict[key] = record_dict[key] * 0
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)

            # elif (
            #         self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            # ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
            #     self._update_teacher_model(keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}
            # generate pseudo masks
            # if self.boxinst_enabled:
            corr_loss = self.loss_corr(label_data_k, label_data_q)
            record_dict.update(corr_loss)

            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well
            with torch.no_grad():
                # self.branch = "unlabel_data"
                (
                    unlabed_data_k_feature,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")
            
            # import pdb
            # pdb.set_trace()
            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            #TODO: tmp, to check
            # warmup_threshold_step = 500.0 # float(self.cfg.SEMISUPNET.BURN_UP_STEP)
            # cur_threshold = 0.9 - (0.9 - cur_threshold) * \
            #         np.clip((self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP)  / warmup_threshold_step, 0., 1.0)

            joint_proposal_dict = {}
            pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label_with_point_anno(
                unlabel_data_k, proposals_roih_unsup_k, cur_threshold, "roih", self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE, # "thresholding" or "hungarian"
            )
            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

            unlabel_data_q = self.rename_label(unlabel_data_q)

            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            )
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            )
            # all_label_data = label_data_q + label_data_k
            all_label_data = label_data_q
            for l in all_label_data:    # clone for labeled images
                l['point_instances'] = l['instances']
            all_unlabel_data = unlabel_data_q

            num_unlabel_gt = np.mean([len(a['point_instances']) for a in unlabel_data_q])
            num_unlabel_pseudo = np.mean([len(a['instances']) for a in all_unlabel_data])

            record_dict['num_unlabel_gt'] = num_unlabel_gt
            record_dict['num_unlabel_pseudo'] = num_unlabel_pseudo
            # for instances_per_image in all_unlabel_data:
            #     if not hasattr(instances_per_image['instances'], 'gt_point_coords') and len(instances_per_image) > 0:
            with autocast(enabled=self.amp_enabled):
                record_all_label_data, _, _, _ = self.model(
                    all_label_data,
                    branch="pseudo_supervised",
                    mil_img_filter_bg_proposal=self.cfg.SEMISUPNET.IMG_MIL_FILTER_BG,
                    add_ground_truth_to_point_proposals=True,
                    add_ss_proposals_to_point_proposals=self.cfg.SEMISUPNET.USE_SS_PROPOSALS)
                record_dict.update(record_all_label_data)

                # self.branch = "unlabel_data"
                record_all_unlabel_data, _, _, _ = self.model(
                    all_unlabel_data,
                    branch="pseudo_supervised",
                    mil_img_filter_bg_proposal=self.cfg.SEMISUPNET.IMG_MIL_FILTER_BG,
                    add_ground_truth_to_point_proposals=False,
                    add_ss_proposals_to_point_proposals=self.cfg.SEMISUPNET.USE_SS_PROPOSALS)

                corr_loss = self.loss_corr(unlabel_data_k, unlabel_data_q)
                record_all_unlabel_data.update(corr_loss)

            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

            num_unlabel_pairwise_pseudo = np.mean([len(a['instances']) for a in all_unlabel_data])
            record_dict['num_unlabel_pairwise_pseudo'] = num_unlabel_pairwise_pseudo

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":

                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo" or key == "loss_fcos_loc_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0

                    elif key == "loss_fcos_cls_pseudo" or key == "loss_fcos_ctr_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 1

                    elif key == "loss_point" or key == "loss_point_pseudo":
                        # if self.iter < 100:
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.POINT_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.POINT_LOSS_WEIGHT
                    elif key == "loss_img_mil" or key == "loss_img_mil_pseudo":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
                    elif key == "loss_ins_mil" or key == "loss_ins_mil_pseudo":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT
                    elif key == "loss_mask_pseudo":
                        # print("loss_mask_pseudo")
                        loss_dict[key] = record_dict[key] * 1.0
                        record_dict[key] = record_dict[key] * 1.0
                        # record_dict[key] = record_dict[key] * 4.0
                    elif key == "loss_project":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.PRJ_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.PRJ_LOSS_WEIGHT
                    elif key == "loss_project_pseudo":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.PRJ_LOSS_WEIGHT * boxinst_pairwise_loss_warmup_factor
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.PRJ_LOSS_WEIGHT * boxinst_pairwise_loss_warmup_factor
                    elif key == "loss_pairwise" or key == "loss_pairwise_pseudo":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.PAIRWISE_LOSS_WEIGHT * boxinst_pairwise_loss_warmup_factor
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.PAIRWISE_LOSS_WEIGHT * boxinst_pairwise_loss_warmup_factor
                    elif key == "loss_corr" or key == "loss_corr_pseudo":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.CORR_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.CORR_LOSS_WEIGHT
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                                record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        # print(metrics_dict)
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        
        pseudo_recall_sum = 0
        for key in self.num_gts_per_cls.keys():
            self.pseudo_recall[key] = self.num_pseudos_per_cls[key] / (self.cfg.DATALOADER.SUP_PERCENT * self.num_gts_per_cls[key])
            pseudo_recall_sum += self.pseudo_recall[key]

        sorted_pseudo_recall = sorted(self.pseudo_recall.items(), key=lambda kv: kv[1], reverse=True)
        for ind, sorted_pseudo_recall_i in enumerate(sorted_pseudo_recall):
            k = sorted_pseudo_recall[79 - ind][0]
            v =  sorted_pseudo_recall_i[1] / pseudo_recall_sum
            self.sampling_freq[k] = v

        # print("self.num_gts_per_cls", self.num_gts_per_cls)
        # print("self.num_pseudos_per_cls", self.num_pseudos_per_cls)
        # print("self.sampling_freq", self.sampling_freq)
        # self.sampling_freq = None

    def superres_T(self, T):
        # bilinear interpolation
        T_bchw = T.reshape(-1, self.corr_feat_height * self.corr_feat_width, self.corr_feat_height, self.corr_feat_width)
        T_bchw = F.interpolate(T_bchw, (self.corr_mask_height, self.corr_mask_width),
                               mode='bilinear', align_corners=False)
        T_b1hwc = T_bchw.reshape(-1, 1, self.corr_feat_height, self.corr_feat_width,
                                 self.corr_mask_height * self.corr_mask_width)
        T_b1hwc = F.interpolate(T_b1hwc,
                                (self.corr_mask_height, self.corr_mask_width, self.corr_mask_height * self.corr_mask_width),
                                mode='trilinear', align_corners=False)
        T_superres = T_b1hwc.reshape(-1, self.corr_mask_height * self.corr_mask_width,
                                     self.corr_mask_height * self.corr_mask_width) * \
                     (1.0 * self.corr_feat_height * self.corr_feat_width / self.corr_mask_height / self.corr_mask_width)

        return T_superres

    def loss_pairwise(
        self, 
        label_data_k, 
        # label_data_q,
        branch="supervised",
        ):        
        loss_dict = {}
        with autocast(enabled=self.amp_enabled):
            (
                label_data_k_feature_s,
                _,
                _,
                _,
            ) = self.model(label_data_k, branch="extract_feat")

        # feat_t = [label_data_k_feature[f] for f in ["p2", "p3", "p4", "p5"]]
        feat_s = [label_data_k_feature_s[f] for f in ["p2", "p3", "p4", "p5"]]
        device = feat_s[0].device
        # loss_corr = torch.tensor(0).to(device).float()
        loss_pairwise = torch.tensor(0).to(device).float()
        # import pdb
        # pdb.set_trace()

        boxes = []
        instances = []
        for i in range(len(label_data_k)):
            instance = label_data_k[i]['instances']
            instance.pred_boxes = instance.gt_boxes
            instance.pred_classes = instance.gt_classes
            instance =instance.to(device)
            boxes.append(instance.gt_boxes)
            instances.append(instance)

        with autocast(enabled=self.amp_enabled):
            roi_feat_s = ROIPooler(
                output_size=(int(self.obj_bank['mask_height']/2), int(self.obj_bank['mask_width']/2)), 
                scales=[0.25, 0.125, 0.0625, 0.03125], 
                sampling_ratio=0, 
                pooler_type="ROIAlignV2")(feat_s, boxes)

        # roi_images_color_similarity = None
        # with torch.no_grad() and autocast(enabled=self.amp_enabled):
        #     instances = self.model_teacher.roi_heads.mask_head(roi_feat_t, instances=instances, branch="sup_data_weak")
        
        # for i in range(len(instances)):
        #     instances[i].pred_masks_t = instances[i].pred_masks

        with autocast(enabled=self.amp_enabled):
            if comm.get_world_size() > 1:
                instances = self.model.module.roi_heads.mask_head(roi_feat_s, instances=instances, branch="keep_pred_mask_logits")
            else:
                instances = self.model.roi_heads.mask_head(roi_feat_s, instances=instances, branch="keep_pred_mask_logits")
            pred_mask_logits = cat([instance.pred_mask_logits for instance in instances], dim=0)

        # instances[0].pred_masks.size() = [bs, 1, 56, 56]
        num_boxes_per_image = [len(i) for i in instances]
        # roi_feat_t_list = roi_feat_t.split(num_boxes_per_image, dim=0)
        roi_feat_s_list = roi_feat_s.split(num_boxes_per_image, dim=0)

        if self.cfg.SEMISUPNET.PAIRWISE_LOSS_WEIGHT > 0:
            images = [x["image_weak"].to(device) if "image_weak" in x else x["image"].to(device) for x in label_data_k]
            images = ImageList.from_tensors(images, self.model_teacher.backbone.size_divisibility)
            # print(boxes[0], boxes[1])
            roi_images = ROIPooler(
                output_size=(int(self.obj_bank['mask_height']), int(self.obj_bank['mask_width'])),
                            scales=[1.0], 
                            sampling_ratio=0, 
                            pooler_type="ROIAlignV2")([images.tensor.float()], boxes)

            if roi_images.size(0) > 0:
                roi_images_lab = []
                for roi_image_rgb in roi_images:
                    roi_image_lab = rgb_to_lab(roi_image_rgb.byte().permute(1, 2, 0))
                    roi_image_lab = roi_image_lab.permute(2, 0, 1)
                    roi_images_lab.append(roi_image_lab)

                roi_images_lab = torch.stack(roi_images_lab, dim=0)
                roi_images_color_similarity = get_images_color_similarity(roi_images_lab, kernel_size=3, dilation=1)

                # box-supervised BoxInst losses
                pairwise_losses = compute_pairwise_term(
                    pred_mask_logits, self.pairwise_size,
                    self.pairwise_dilation
                )

                weights = (roi_images_color_similarity > self.pairwise_color_thresh).float()
                loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

            loss_dict.update({"loss_pairwise": loss_pairwise})
        
        # print("loss corr: {}, loss pairwise: {}".format(loss_corr.item(), loss_pairwise.item()))
        return loss_dict


    def loss_corr(
        self, 
        label_data_k, 
        label_data_q,
        branch="supervised", # or "pseudo_supervised"
        # mean_fields,
        ):
        # generate pseudo mask for sup data, e.g. 10% box annotated data
        # color_feats = F.interpolate(img, (s_ins_pred.shape[2], s_ins_pred.shape[3]), mode='bilinear', align_corners=True)
        # mean_fields = [MeanField(color_feat.unsqueeze(0), alpha0=self.alpha0,
        #                          theta0=self.theta0, theta1=self.theta1, theta2=self.theta2,
        #                          iter=self.crf_max_iter, kernel_size=self.mkernel, base=self.crf_base) \
        #                for color_feat in color_feats]
        # if branch=="pseudo_supervised":
        #     import pdb
        #     pdb.set_trace()
        
        if hasattr(label_data_q[0]['instances'], 'gt_pseudo_scores'):
            for k, q in zip(label_data_k, label_data_q):
                mask = q['instances'].gt_pseudo_scores >= 0.5
                k['instances'], q['instances'] = k['instances'][mask], q['instances'][mask]

        loss_dict = {}
        with torch.no_grad() and autocast(enabled=self.amp_enabled):
            (
                label_data_k_feature,
                _,
                _,
                _,
            ) = self.model_teacher(label_data_k, branch="extract_feat")

        with autocast(enabled=self.amp_enabled):
            (
                label_data_k_feature_s,
                _,
                _,
                _,
            ) = self.model(label_data_k, branch="extract_feat")

        feat_t = [label_data_k_feature[f] for f in ["p2", "p3", "p4", "p5"]]
        feat_s = [label_data_k_feature_s[f] for f in ["p2", "p3", "p4", "p5"]]
        device = feat_t[0].device
        loss_corr = torch.tensor(0).to(device).float()
        loss_pairwise = torch.tensor(0).to(device).float()

        boxes = []
        instances = []
        for i in range(len(label_data_k)):
            instance = label_data_k[i]['instances']
            instance.pred_boxes = instance.gt_boxes
            instance.pred_classes = instance.gt_classes
            instance =instance.to(device)
            boxes.append(instance.gt_boxes)
            instances.append(instance)
        
        with torch.no_grad() and autocast(enabled=self.amp_enabled):
            roi_feat_t = ROIPooler(
                output_size=(int(self.obj_bank['mask_height']/2), int(self.obj_bank['mask_width']/2)), 
                scales=[0.25, 0.125, 0.0625, 0.03125], 
                sampling_ratio=0, 
                pooler_type="ROIAlignV2")(feat_t, boxes)

        with autocast(enabled=self.amp_enabled):
            roi_feat_s = ROIPooler(
                output_size=(int(self.obj_bank['mask_height']/2), int(self.obj_bank['mask_width']/2)), 
                scales=[0.25, 0.125, 0.0625, 0.03125], 
                sampling_ratio=0, 
                pooler_type="ROIAlignV2")(feat_s, boxes)

        # roi_images_color_similarity = None
        with torch.no_grad() and autocast(enabled=self.amp_enabled):
            instances = self.model_teacher.roi_heads.mask_head(roi_feat_t, instances=instances, branch="sup_data_weak")
        
        for i in range(len(instances)):
            instances[i].pred_masks_t = instances[i].pred_masks

        with autocast(enabled=self.amp_enabled):
            if comm.get_world_size() > 1:
                instances = self.model.module.roi_heads.mask_head(roi_feat_s, instances=instances, branch="keep_pred_mask_logits")
            else:
                instances = self.model.roi_heads.mask_head(roi_feat_s, instances=instances, branch="keep_pred_mask_logits")
            pred_mask_logits = cat([instance.pred_mask_logits for instance in instances], dim=0)

        # instances[0].pred_masks.size() = [bs, 1, 56, 56]
        num_boxes_per_image = [len(i) for i in instances]
        roi_feat_t_list = roi_feat_t.split(num_boxes_per_image, dim=0)
        roi_feat_s_list = roi_feat_s.split(num_boxes_per_image, dim=0)

        if self.save_corr_img:
            images = [x["image_weak"].to(device) if "image_weak" in x else x["image"].to(device) for x in label_data_k]
            images = ImageList.from_tensors(images, self.model_teacher.backbone.size_divisibility)
            roi_images = ROIPooler(
                output_size=(self.obj_bank['img_height'], self.obj_bank['img_width']), 
                scales=[1.0], 
                sampling_ratio=0, 
                pooler_type="ROIAlignV2")([images.tensor.float()], boxes)
            roi_images_list = roi_images.split(num_boxes_per_image, dim=0)

        loss_corr_normalizer = 0.0
        for i in range(len(instances)):
            if len(instances[i]) > 0:
                instances[i].roi_feat_t = roi_feat_t_list[i]
                instances[i].roi_feat_s = roi_feat_s_list[i]
                if self.save_corr_img:
                    instances[i].roi_images = roi_images_list[i]
    
                queue_area_mask = instances[i].gt_boxes.area() > self.objbank_min_size * self.objbank_min_size

                for idx in torch.arange(len(instances[i])):
                    roi_t_mask = instances[i].pred_masks_t
                    roi_t_feat = instances[i].roi_feat_t
                    roi_s_mask = instances[i].pred_masks
                    roi_s_feat = instances[i].roi_feat_s
                    gt_boxes_per_img = instances[i].gt_boxes.tensor
                    kernel_labels = instances[i].gt_classes
                    if self.save_corr_img:
                        roi_img = instances[i].roi_images

                    if self.qobj is None:
                        self.qobj = ObjectFactory.create_one(mask=roi_s_mask[idx].detach(),
                                                            feature=roi_s_feat[idx:idx+1].detach(),
                                                            box=gt_boxes_per_img[idx:idx+1].detach(),
                                                            category=kernel_labels[idx],
                                                            img=roi_img[idx] if self.save_corr_img else None)
                    else:
                        self.qobj.mask[...] = roi_s_mask[idx].detach()
                        self.qobj.feature[...] = roi_s_feat[idx:idx+1].detach()
                        self.qobj.box[...] = gt_boxes_per_img[idx:idx+1].detach()
                        self.qobj.category = kernel_labels[idx]
                        if self.save_corr_img:
                            self.qobj.img[...] = roi_img[idx:idx + 1]

                    kobjs = self.object_queues.get_similar_obj(self.qobj)

                    if kobjs is not None and kobjs['mask'].shape[0] >= 5:
                        # pdb.set_trace()
                        loss_corr_normalizer += 1
                        Cu, T, fg_mask, bg_mask = self.semantic_corr_solver.solve(self.qobj, kobjs, roi_s_feat[idx:idx + 1])

                        if self.save_corr_img:
                            self.vis_corr(self.qobj.img, kobjs['img'][0:1], T[0], self.qobj.mask, kobjs['mask'][0:1], **self.img_norm_cfg)
                        nce_loss = nn.CrossEntropyLoss()
                        assignment = T.argmax(2).reshape(-1)
                        Cu = Cu.float()
                        Cu = F.softmax(Cu, 2).reshape(-1, Cu.shape[2])
                        loss_corr += nce_loss(Cu, assignment)
                        # num_ins += 1
                        # pdb.set_trace()

                        # with torch.no_grad():
                        #     T = T * Cu.reshape(T.shape)

                        # T = T / (T.sum(2, keepdim=True) + 1e-5)

                        # T_superres = self.superres_T(T)

                        # fg_ci = torch.matmul(T_superres * (fg_mask > 0.5).float(), torch.clamp(kobjs['mask'], min=0.1, max=0.9).reshape(T_superres.shape[0], T_superres.shape[2], 1).to(Cu)).mean(0).reshape(roi_s_mask.shape[1:])
                        # bg_ci = torch.matmul(T_superres * (bg_mask > 0.5).float(), torch.clamp(1-kobjs['mask'], min=0.1, max=0.9).reshape(T_superres.shape[0], T_superres.shape[2], 1).to(Cu)).mean(0).reshape(roi_s_mask.shape[1:])

                        # fg_ci = F.interpolate(fg_ci.reshape(1, 1, fg_ci.shape[0], fg_ci.shape[1]),
                        #                       (int(boxes[idx][3] - boxes[idx][1]), int(boxes[idx][2] - boxes[idx][0])),
                        #                       mode='bilinear', align_corners=False).squeeze()
                        # bg_ci = F.interpolate(bg_ci.reshape(1, 1, bg_ci.shape[0], bg_ci.shape[1]),
                        #                       (int(boxes[idx][3] - boxes[idx][1]), int(boxes[idx][2] - boxes[idx][0])),
                        #                       mode='bilinear', align_corners=False).squeeze()
                        # iiu[idx*2, int(boxes[idx, 1]):int(boxes[idx, 3]),
                        #             int(boxes[idx, 0]):int(boxes[idx, 2])] = bg_ci
                        # iiu[idx*2+1, int(boxes[idx, 1]):int(boxes[idx, 3]),
                        #             int(boxes[idx, 0]):int(boxes[idx, 2])] = fg_ci

                        # if self.save_corr_img:
                        #     self.cnt += 1
                        #     vis_seg(self.qobj.img, ci, self.img_norm_cfg, save_dir='work_dirs/corr_vis', data_id=self.cnt)
                        #     self.cnt += 1
                        #     vis_seg(self.qobj.img, roi_s_mask[idx], self.img_norm_cfg, save_dir='work_dirs/corr_vis', data_id=self.cnt)

                    if queue_area_mask[idx]:
                        created_gpu_bank = self.object_queues.append(int(kernel_labels[idx]),
                                                                    idx,
                                                                    roi_t_feat,
                                                                    roi_t_mask,
                                                                    gt_boxes_per_img.detach(),
                                                                    roi_img if self.save_corr_img else None,
                                                                    device=device if self.num_created_gpu_bank >= self.num_gpu_bank else 'cpu')
                        self.num_created_gpu_bank += created_gpu_bank

        # if loss_corr > 0:
        #     print("loss corr: {}".format(loss_corr.item() / loss_corr_normalizer))
        loss_dict = {"loss_corr": loss_corr / max(1.0, loss_corr_normalizer)}

        # if self.boxinst_enabled:
        if self.cfg.SEMISUPNET.PAIRWISE_LOSS_WEIGHT > 0:
            images = [x["image_weak"].to(device) if "image_weak" in x else x["image"].to(device) for x in label_data_k]
            images = ImageList.from_tensors(images, self.model_teacher.backbone.size_divisibility)
            image_masks = [torch.ones_like(x[0], dtype=torch.float32) for x in images]
            # mask out the bottom area where the COCO dataset probably has wrong annotations
            for i in range(len(image_masks)):
                im_h = label_data_k[i]["height"]
                pixels_removed = int(
                    self.bottom_pixels_removed *
                    float(images[i].size(1)) / float(im_h)
                )
                if pixels_removed > 0:
                    image_masks[i][-pixels_removed:, :] = 0
            image_masks = ImageList.from_tensors(
                image_masks, self.backbone.size_divisibility, pad_value=0.0
            )
            # print(boxes[0], boxes[1])
            roi_images = ROIPooler(
                            output_size=(int(self.obj_bank['mask_height']), int(self.obj_bank['mask_width'])),
                            scales=[1.0], 
                            sampling_ratio=0, 
                            pooler_type="ROIAlignV2")([images.tensor.float()], boxes)

            if roi_images.size(0) > 0:
                roi_images_lab = []
                for roi_image_rgb in roi_images:
                    roi_image_lab = rgb_to_lab(roi_image_rgb.byte().permute(1, 2, 0))
                    roi_image_lab = roi_image_lab.permute(2, 0, 1)
                    roi_images_lab.append(roi_image_lab)

                roi_images_lab = torch.stack(roi_images_lab, dim=0)
                roi_images_color_similarity = get_images_color_similarity(roi_images_lab, kernel_size=3, dilation=1)

                # box-supervised BoxInst losses
                pairwise_losses = compute_pairwise_term(
                    pred_mask_logits, self.pairwise_size,
                    self.pairwise_dilation
                )

                weights = (roi_images_color_similarity > self.pairwise_color_thresh).float()
                loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

            loss_dict.update({"loss_pairwise": loss_pairwise})
        
        # print("loss corr: {}, loss pairwise: {}".format(loss_corr.item(), loss_pairwise.item()))
        return loss_dict


            # iiu = iiu.reshape(iiu.shape[0] // 2, 2, iiu.shape[1], iiu.shape[2])
            # for img_idx in range(len(mean_fields)):
            #     obj_inds = (img_inds == img_idx)
            #     enlarged_target = F.max_pool2d(target.float().unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1).byte()
            #     if obj_inds.sum() > 0:
            #         pseudo_label, valid = mean_fields[int(img_idx)](
            #             (t_input[obj_inds].unsqueeze(1) + s_input[obj_inds].unsqueeze(1)) / 2,
            #             target[obj_inds].unsqueeze(1), iiu[obj_inds])
            #         cropped_s_input = s_input[obj_inds] * enlarged_target[obj_inds]
            #         cropped_s_input = cropped_s_input * mean_fields[int(img_idx)].gamma + cropped_s_input.detach() * (1 - mean_fields[int(img_idx)].gamma)
            #         loss_ts.append(dice_loss(cropped_s_input, pseudo_label))

        # pdb.set_trace()
        # proposals_roih_sup_k = [self.threshold_bbox(x, thres=0.7, proposal_type="roih", mask_thres=0.5) for x in proposals_roih_sup_k]

        # label_data_k = self.add_label(
        #     label_data_k, proposals_roih_sup_k
        # )                    
        # label_data_q = self.add_label(
        #     label_data_q, proposals_roih_sup_k
        # )
        # return label_data_k, label_data_q



# Large Scale Jitter
class LSJMaskRCNNPointSupTrainer(MaskRCNNPointSupTrainer):

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        # data_loader = self.build_train_loader(cfg)
        lsj_data_loader, data_loader = self.build_train_loader(cfg)
        self._lsj_data_loader_iter = iter(lsj_data_loader)
        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())


    @classmethod
    def build_train_loader(cls, cfg):
        lsj_mapper = PointSupDatasetMapper(cfg, is_train=True)
        mapper = PointSupTwoCropSeparateDatasetMapper(cfg, is_train=True)
        return build_detection_semisup_train_loader(cfg, lsj_mapper), \
               build_detection_semisup_train_loader_two_crops(cfg, mapper)


    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[pteacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data

        if self.iter >= self.cfg.SEMISUPNET.BURN_UP_STEP:
            lsj_label_data = next(self._lsj_data_loader_iter)

        data_time = time.perf_counter() - start

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            for l in label_data_q:    # clone for labeled images
                l['point_instances'] = l['instances']
            record_dict, _, _, _ = self.model(label_data_q,
                                              branch="supervised",
                                              mil_img_filter_bg_proposal=self.cfg.SEMISUPNET.IMG_MIL_FILTER_BG,
                                              add_ground_truth_to_point_proposals=True,
                                              add_ss_proposals_to_point_proposals=self.cfg.SEMISUPNET.USE_SS_PROPOSALS)
            # weight losses
            loss_dict = {}

            for key in record_dict.keys():
                if key[:4] == "loss":
                    if key == "loss_point":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.POINT_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.POINT_LOSS_WEIGHT
                    elif key == "loss_img_mil":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
                    elif key == "loss_ins_mil":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)

            elif (
                    self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}
            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well
            with torch.no_grad():
                # self.branch = "unlabel_data"
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            #TODO: tmp, to check
            warmup_threshold_step = 500.0 # float(self.cfg.SEMISUPNET.BURN_UP_STEP)
            cur_threshold = 0.9 - (0.9 - cur_threshold) * \
                    np.clip((self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP)  / warmup_threshold_step, 0., 1.0)


            joint_proposal_dict = {}
            # joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
            # (
            #     pesudo_proposals_rpn_unsup_k,
            #     nun_pseudo_bbox_rpn,
            # ) = self.process_pseudo_label_with_point_anno(
            #     proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding", gt_point_coords, gt_point_labels
            # )
            # joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)

            pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label_with_point_anno(
                unlabel_data_k, proposals_roih_unsup_k, cur_threshold, "roih", self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE, # "thresholding" or "hungarian"
            )
            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

            #  add pseudo-label to unlabeled data
            # unlabel_data_q = self.remove_label(unlabel_data_q)
            # unlabel_data_k = self.remove_label(unlabel_data_k)
            unlabel_data_q = self.rename_label(unlabel_data_q)

            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            )
            # unlabel_data_k = self.add_label(
            #     unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            # )
            # import pdb
            # pdb.set_trace()
            # all_label_data = label_data_q + label_data_k
            all_label_data = lsj_label_data + label_data_k
            # all_label_data = label_data_q
            for l in all_label_data:    # clone for labeled images
                l['point_instances'] = l['instances']

            all_unlabel_data = unlabel_data_q
            record_all_label_data, _, _, _ = self.model(
                all_label_data,
                branch="supervised",
                mil_img_filter_bg_proposal=self.cfg.SEMISUPNET.IMG_MIL_FILTER_BG,
                add_ground_truth_to_point_proposals=True,
                add_ss_proposals_to_point_proposals=self.cfg.SEMISUPNET.USE_SS_PROPOSALS)
            record_dict.update(record_all_label_data)

            # self.branch = "unlabel_data"
            record_all_unlabel_data, _, _, _ = self.model(
                all_unlabel_data,
                branch="supervised",
                mil_img_filter_bg_proposal=self.cfg.SEMISUPNET.IMG_MIL_FILTER_BG,
                add_ground_truth_to_point_proposals=False,
                add_ss_proposals_to_point_proposals=self.cfg.SEMISUPNET.USE_SS_PROPOSALS)

            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":

                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo" or key == "loss_fcos_loc_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0

                    elif key == "loss_fcos_cls_pseudo" or key == "loss_fcos_ctr_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 1

                    elif key == "loss_point" or key == "loss_point_pseudo":
                        # if self.iter < 100:
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.POINT_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.POINT_LOSS_WEIGHT
                    elif key == "loss_img_mil" or key == "loss_img_mil_pseudo":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
                    elif key == "loss_mask_pseudo":
                        loss_dict[key] = record_dict[key] * 1.0
                        record_dict[key] = record_dict[key] * 1.0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                                record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

            num_unlabel_gt = np.mean([len(a['point_instances']) for a in unlabel_data_q])
            num_unlabel_pseudo = np.mean([len(a['instances']) for a in all_unlabel_data])

            record_dict['num_unlabel_gt'] = num_unlabel_gt
            record_dict['num_unlabel_pseudo'] = num_unlabel_pseudo


        metrics_dict = record_dict
        # print(metrics_dict)
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        if self.cfg.SOLVER.AMP.ENABLED:
            self.grad_scaler.scale(losses).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            self.optimizer.step()