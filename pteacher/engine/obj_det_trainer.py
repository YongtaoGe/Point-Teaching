# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict, defaultdict
import random
import cv2

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import COCOEvaluator, verify_results, PascalVOCDetectionEvaluator, DatasetEvaluators
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes, pairwise_iou
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import inference_context

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
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from pteacher.utils.events import PSSODMetricPrinter
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.evaluation.testing import print_csv_format
from contextlib import ExitStack, contextmanager
import torch.nn as nn
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
)
# Supervised-only Trainer
class BaselineTrainer(DefaultTrainer):
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
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

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


# Unbiased Teacher Trainer
class pteacherTrainer(DefaultTrainer):
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
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
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
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

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
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
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

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):  
        # import pdb
        # pdb.set_trace()
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    self, proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
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
        # import pdb
        # pdb.set_trace()
        data_time = time.perf_counter() - start

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            unlabel_data_q.extend(unlabel_data_k)
            label_data_q.extend(unlabel_data_q)
            
            record_dict = self.model(label_data_q)
            # record_dict, _, _, _ = self.model(label_data_q, branch="supervised")

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
                proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

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

    def build_gen_pseudo_loader(cfg, dataset_name):
        mapper = DatasetMapper(cfg, True)
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
        # from . import transforms as T
        import detectron2.data.transforms as T
        augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
        augmentation = T.AugmentationList(augmentation)
        # import pdb
        # pdb.set_trace()
        mapper.augmentations = augmentation


        return build_detection_test_loader(cfg, dataset_name, mapper)

    @classmethod
    def gen_pseudo_labels_offline(cls, cfg, model):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        from detectron2.modeling.postprocessing import detector_postprocess
        logger = logging.getLogger(__name__)
        results = OrderedDict()

        with ExitStack() as stack:
            # if isinstance(model, nn.Module):
            #     stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
                # processed_results = []
                data_loader = cls.build_gen_pseudo_loader(cfg, dataset_name)

                evaluator = cls.build_evaluator(cfg, dataset_name)
                # results_i = inference_on_dataset(model, data_loader, evaluator)

                evaluator.reset()

                for idx, inputs in enumerate(data_loader):
                    if comm.is_main_process() and idx % 25 == 0:
                        print("processed {} images".format(idx))

                    # with torch.no_grad():
                    (
                        _,
                        proposals_rpn_unsup_k,
                        proposals_roih_unsup_k,
                        _,
                    ) = model(inputs, branch="unsup_data_weak")

                    # proposals_roih_unsup_k_new = model(inputs, branch="unsup_data_weak")
                    # evaluator.process(inputs, proposals_roih_unsup_k_new)
                    # import pdb
                    # pdb.set_trace()

                    #  Pseudo-labeling
                    cur_threshold = cfg.SEMISUPNET.BBOX_THRESHOLD
                    # cur_threshold = 0.1
                    
                    # Pseudo_labeling for ROI head (bbox location/objectness)
                    pesudo_proposals_roih_unsup_k, _ = cls.process_pseudo_label(
                        cls, proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
                    )

                    height = inputs[0]["height"]
                    width = inputs[0]["width"]
                    results_per_image = pesudo_proposals_roih_unsup_k[0].to("cpu")
                    results_per_image.pred_boxes = results_per_image.gt_boxes
                    results_per_image.pred_classes = results_per_image.gt_classes

                    r = detector_postprocess(results_per_image, height, width)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    evaluator.process(inputs, [{"instances": r}])
                    
                results_i = evaluator.evaluate()
                # An evaluator may return None when not in main process.
                # Replace it by an empty dict instead to make it easier for downstream code to handle
                if results_i is None:
                    results_i = {}

                if comm.is_main_process():
                    assert isinstance(
                        results_i, dict
                    ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                        results_i
                    )
                    logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                    print_csv_format(results_i)

                results[dataset_name] = results_i
        
        return results


# PointSup Trainer based on Unbiased Teacher Trainer for faster-rcnn
class FasterRCNNPointSupTrainer(pteacherTrainer):
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
        self.resume = False
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.use_point_guided_cp = cfg.SEMISUPNET.USE_POINT_GUIDED_CP
        if self.use_point_guided_cp:
            # init inst_bank which stores cropped instances
            self.inst_bank = defaultdict(list)
            self.num_gts_per_cls = defaultdict(int)
            self.num_pseudos_per_cls = defaultdict(int)
            self.pseudo_recall = defaultdict(float)
            self.sampling_freq = defaultdict(float)

            self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            # print("self.num_classes", self.num_classes)
            for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES):
                self.inst_bank[i] = []
                self.num_gts_per_cls[i] = 1
                self.num_pseudos_per_cls[i] = 0
                self.pseudo_recall[i] = 1
                self.sampling_freq[i] = 1.0 / cfg.MODEL.ROI_HEADS.NUM_CLASSES

            self.inst_bank = [self.inst_bank[i] for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES)]

            self.bank_update_num = 20
            self.bank_length = 600

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

    def inst_bank_has_empty_classes(self):
        for i, inst_bank_per_class in enumerate(self.inst_bank):
            if len(inst_bank_per_class) == 0:
                print("Class {} doesn't have any cropped instances!".format(i))
                return True
        return False

    def update_inst_bank(self, label_data):
        # num_proposal_output = 0.0
        # label_data = random.choice([label_data_q, label_data_k])
        # DEBUG = False
        for label_data_per_img in label_data:

            gt_labels = label_data_per_img["instances"].gt_classes
            gt_bboxes = label_data_per_img["instances"].gt_boxes.tensor
            img = label_data_per_img["image"]
            c, h, w = img.shape
            unique_labels = list(set(gt_labels.tolist()))
            for l in unique_labels:
                candidate_bboxes = gt_bboxes[gt_labels == l]
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
                    
                    crop_img = img[:, crop_y1:crop_y2, crop_x1:crop_x2].clone()

                    if len(self.inst_bank[l]) < self.bank_length:
                        self.inst_bank[l].append(crop_img)
                    else:
                        p_i = np.random.choice(range(self.bank_length))
                        self.inst_bank[l][p_i] = crop_img

    def update_inst_bank_v2(self, label_data, branch="supervised", score_thres=0.7):
        # num_proposal_output = 0.0
        # label_data = random.choice([label_data_q, label_data_k])
        # DEBUG = False
        for label_data_per_img in label_data:
            if branch == "pseudo_supervised":
                scores = label_data_per_img["instances"].scores
                tmp_instances = label_data_per_img["instances"][torch.bitwise_and(scores > score_thres, scores < 1)]
            else:
                tmp_instances = label_data_per_img["instances"]

            if len(tmp_instances) > 0:
                gt_labels = tmp_instances.gt_classes
                gt_bboxes = tmp_instances.gt_boxes.tensor
                img = label_data_per_img["image"]
                c, h, w = img.shape
                unique_labels = list(set(gt_labels.tolist()))
                for l in unique_labels:
                    candidate_bboxes = gt_bboxes[gt_labels == l]
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
                        
                        crop_img = img[:, crop_y1:crop_y2, crop_x1:crop_x2].clone()

                        if len(self.inst_bank[l]) < self.bank_length:
                            self.inst_bank[l].append(crop_img)
                        else:
                            p_i = np.random.choice(range(self.bank_length))
                            self.inst_bank[l][p_i] = crop_img

    def paste_inst_bank_to_unlabel_data_v1(self, unlabel_image, pseudo_instance, paste_classes, paste_positions):
        _, img_h, img_w = unlabel_image.size()

        for paste_pos, paste_cls in zip(paste_positions, paste_classes):
            if len(self.inst_bank[int(paste_cls)]) > 0:
                paste_img = random.choice(self.inst_bank[int(paste_cls)])
                box_h, box_w = paste_img.shape[1:]
                box_xc, box_yc = paste_pos
                box_xc, box_yc = int(box_xc), int(box_yc)
                x1, x2 = int(max(0, box_xc - box_w / 2)), int(min(img_w, box_xc + box_w / 2))
                y1, y2 = int(max(0, box_yc - box_h / 2)), int(min(img_h, box_yc + box_h / 2))
                
                crop_w, crop_h = int(x2 - x1), int(y2 - y1)
                assert crop_w > 0 and crop_h > 0, "crop width and crop height are not valid."
                box_x1 = int(random.uniform(0, 1) * (box_w - crop_w)) if crop_w < box_w else 0
                box_y1 = int(random.uniform(0, 1) * (box_h - crop_h)) if crop_h < box_h else 0
                box_x2 = box_x1 + crop_w
                box_y2 = box_y1 + crop_h

                unlabel_image[:, y1:y2, x1:x2] = paste_img[:, box_y1:box_y2, box_x1:box_x2]

                paste_instance = Instances(pseudo_instance.image_size)
                paste_instance.gt_classes = paste_cls[None]
                paste_instance.scores = torch.ones_like(paste_cls[None])
                paste_instance.gt_boxes = Boxes(torch.tensor([[x1, y1, x2, y2]]).to(paste_cls.device))
                pseudo_instance = Instances.cat([paste_instance, pseudo_instance])

        return unlabel_image, pseudo_instance

    def paste_inst_bank_to_unlabel_data_v2(self, unlabel_image, pseudo_instance, paste_classes, paste_positions):
        _, img_h, img_w = unlabel_image.size()

        for paste_pos, paste_cls in zip(paste_positions, paste_classes):
            if len(self.inst_bank[int(paste_cls)]) > 0:
                # step 1: random location paste
                paste_img = random.choice(self.inst_bank[int(paste_cls)])
                p_h, p_w = paste_img.shape[1:]

                for _ in range(5): # try time
                    
                    if img_w - p_w < 1 or img_h - p_h < 1:
                        break

                    p_x1 = np.random.randint(0, img_w - p_w)
                    p_y1 = np.random.randint(0, img_h - p_h)

                    paste_box = Boxes(torch.tensor([[p_x1, p_y1, p_x1 + p_w, p_y1 + p_h]]).to(paste_cls.device))
                    ious = torch.zeros_like(paste_cls)
                    if len(pseudo_instance) > 0:
                        ious = pairwise_iou(paste_box, pseudo_instance.gt_boxes)
                    if ious.max() < 1e-2:
                        #print('paste')
                        unlabel_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)] = paste_img
                        paste_instance = Instances(pseudo_instance.image_size)
                        paste_instance.gt_classes = paste_cls[None]
                        paste_instance.scores = torch.ones_like(paste_cls[None])
                        paste_instance.gt_reg_loss_weight = torch.ones_like(paste_cls[None])
                        paste_instance.gt_boxes = paste_box
                        pseudo_instance = Instances.cat([paste_instance, pseudo_instance])
                        break

        return unlabel_image, pseudo_instance

    def paste_inst_bank_to_unlabel_data_v3(self, unlabel_image, pseudo_instance, paste_classes, paste_positions):
        _, img_h, img_w = unlabel_image.size()

        # step 1: random location paste
        paste_cls = random.randint(0, 79)
        while len(self.inst_bank[int(paste_cls)]) == 0:
            paste_cls = random.randint(0, 79)
        # import pdb
        # pdb.set_trace()
        paste_cls = torch.tensor(paste_cls).to(paste_classes.device)
        paste_img = random.choice(self.inst_bank[int(paste_cls)])
        p_h, p_w = paste_img.shape[1:]

        for _ in range(5):  # try time

            if img_w - p_w < 1 or img_h - p_h < 1:
                break

            p_x1 = np.random.randint(0, img_w - p_w)
            p_y1 = np.random.randint(0, img_h - p_h)

            paste_box = Boxes(torch.tensor([[p_x1, p_y1, p_x1 + p_w, p_y1 + p_h]]).to(paste_classes.device))
            ious = torch.zeros_like(paste_classes[0])
            if len(pseudo_instance) > 0:
                ious = pairwise_iou(paste_box, pseudo_instance.gt_boxes)
            if ious.max() < 2e-1:
                # print('paste')
                # unlabel_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)] = paste_img
                # mix up
                mixup_ratio = 0.6
                unlabel_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)] = mixup_ratio * paste_img + \
                (1.0 - mixup_ratio) * unlabel_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)]
                paste_instance = Instances(pseudo_instance.image_size)
                paste_instance.gt_classes = paste_cls[None]
                paste_instance.scores = torch.ones_like(paste_cls[None])
                paste_instance.gt_reg_loss_weight = torch.ones_like(paste_cls[None])
                paste_instance.gt_boxes = paste_box
                pseudo_instance = Instances.cat([paste_instance, pseudo_instance])
                break

        for paste_pos, paste_cls in zip(paste_positions, paste_classes):
            if len(self.inst_bank[int(paste_cls)]) > 0:
                # step 2: paste based on point location
                paste_img = random.choice(self.inst_bank[int(paste_cls)])
                p_h, p_w = paste_img.shape[1:]
                box_xc, box_yc = paste_pos
                box_xc, box_yc = int(box_xc), int(box_yc)
                x1, x2 = int(max(0, box_xc - p_w / 2)), int(min(img_w, box_xc + p_w / 2))
                y1, y2 = int(max(0, box_yc - p_h / 2)), int(min(img_h, box_yc + p_h / 2))

                crop_w, crop_h = int(x2 - x1), int(y2 - y1)
                if crop_w > 0 and crop_h > 0:
                    box_x1 = int(random.uniform(0, 1) * (p_w - crop_w)) if crop_w < p_w else 0
                    box_y1 = int(random.uniform(0, 1) * (p_h - crop_h)) if crop_h < p_h else 0
                    box_x2 = box_x1 + crop_w
                    box_y2 = box_y1 + crop_h

                    unlabel_image[:, y1:y2, x1:x2] = paste_img[:, box_y1:box_y2, box_x1:box_x2]

                    paste_instance = Instances(pseudo_instance.image_size)
                    paste_instance.gt_classes = paste_cls[None]
                    paste_instance.scores = torch.ones_like(paste_cls[None])
                    paste_instance.gt_reg_loss_weight = torch.ones_like(paste_cls[None])
                    paste_instance.gt_boxes = Boxes(torch.tensor([[x1, y1, x2, y2]]).to(paste_cls.device))
                    pseudo_instance = Instances.cat([paste_instance, pseudo_instance])

        return unlabel_image, pseudo_instance

    def paste_inst_bank_to_unlabel_data_v4(self, unlabel_image, pseudo_instance, paste_classes, paste_positions, num_paste_objs=2, mixup_lambda=0.65):
        _, img_h, img_w = unlabel_image.size()

        for _ in range(num_paste_objs):
            # step 1: random location paste
            for _ in range(4): # try times
                paste_cls = random.randint(0, self.num_classes-1) # random sample cls index between 0-79 for coco
                # p = np.array([v for v in self.sampling_freq.values()])
                # paste_cls = np.random.choice(list(range(0, 80)), p=p.ravel())
                paste_cls = torch.tensor(paste_cls).to(paste_classes.device)
                if len(self.inst_bank[int(paste_cls)]) == 0:
                    paste_img = None
                    continue
                paste_img = random.choice(self.inst_bank[int(paste_cls)])
                p_h, p_w = paste_img.shape[1:]
                if img_w - p_w < 1 or img_h - p_h < 1:
                    paste_img = None
                    continue
                break

            if paste_img is not None:
                p_x1 = np.random.randint(0, img_w - p_w)
                p_y1 = np.random.randint(0, img_h - p_h)
                paste_box = Boxes(torch.tensor([[p_x1, p_y1, p_x1 + p_w, p_y1 + p_h]]).to(paste_classes.device))
                # mixup
                unlabel_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)] = mixup_lambda * paste_img + \
                (1.0 - mixup_lambda) * unlabel_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)]
                paste_instance = Instances(pseudo_instance.image_size)
                paste_instance.gt_classes = paste_cls[None]
                paste_instance.scores = torch.ones_like(paste_cls[None])
                paste_instance.gt_reg_loss_weight = torch.ones_like(paste_cls[None])
                paste_instance.gt_boxes = paste_box

                pos_point_coord = [p_x1 + np.random.random_sample() * p_w, p_y1 + np.random.random_sample() * p_h]
                neg_point_coord = [p_x1, p_y1]
                # import pdb
                # pdb.set_trace()
                paste_instance.gt_point_coords = torch.DoubleTensor([[pos_point_coord, neg_point_coord]]).to(paste_classes.device)
                paste_instance.gt_point_labels = torch.DoubleTensor([[1, 0]]).to(paste_classes.device)

                pseudo_instance = Instances.cat([paste_instance, pseudo_instance])

        for paste_pos, paste_cls in zip(paste_positions, paste_classes):
            if len(self.inst_bank[int(paste_cls)]) > 0:
                # step 2: paste based on point location
                paste_img = random.choice(self.inst_bank[int(paste_cls)])
                
                # import pdb
                # pdb.set_trace()
                p_h, p_w = paste_img.shape[1:]
                box_xc, box_yc = paste_pos
                box_xc, box_yc = int(box_xc), int(box_yc)
                x1, x2 = int(max(0, box_xc - p_w / 2)), int(min(img_w, box_xc + p_w / 2))
                y1, y2 = int(max(0, box_yc - p_h / 2)), int(min(img_h, box_yc + p_h / 2))

                crop_w, crop_h = int(x2 - x1), int(y2 - y1)
                if crop_w > 0 and crop_h > 0:
                    box_x1 = int(random.uniform(0, 1) * (p_w - crop_w)) if crop_w < p_w else 0
                    box_y1 = int(random.uniform(0, 1) * (p_h - crop_h)) if crop_h < p_h else 0
                    box_x2 = box_x1 + crop_w
                    box_y2 = box_y1 + crop_h

                    unlabel_image[:, y1:y2, x1:x2] = mixup_lambda * paste_img[:, box_y1:box_y2, box_x1:box_x2] + \
                                                    (1.0 - mixup_lambda) * unlabel_image[:, y1:y2, x1:x2]

                    paste_instance = Instances(pseudo_instance.image_size)
                    paste_instance.gt_classes = paste_cls[None]
                    paste_instance.scores = torch.ones_like(paste_cls[None])
                    paste_instance.gt_reg_loss_weight = torch.ones_like(paste_cls[None])
                    paste_instance.gt_boxes = Boxes(torch.tensor([[x1, y1, x2, y2]]).to(paste_cls.device))

                    pos_point_coord = [x1 + np.random.random_sample() * crop_w, y1 + np.random.random_sample() * crop_h]
                    neg_point_coord = [x1, y1]
                    paste_instance.gt_point_coords = torch.DoubleTensor([[pos_point_coord, neg_point_coord]]).to(paste_classes.device)
                    paste_instance.gt_point_labels = torch.DoubleTensor([[1, 0]]).to(paste_classes.device)

                    pseudo_instance = Instances.cat([paste_instance, pseudo_instance])

        return unlabel_image, pseudo_instance

    def paste_inst_bank_to_label_data(self, label_image, src_instance, num_paste_objs=3, mixup_lambda=0.65):
        _, img_h, img_w = label_image.size()
        for _ in range(num_paste_objs):
            # step 1: random location paste
            for _ in range(4): # try times
                # paste_cls = random.randint(0, 79) # random sample cls index between 0-79 for coco
                p = np.array([v for v in self.sampling_freq.values()])
                paste_cls = np.random.choice(list(range(0, )), p=p.ravel())
                paste_cls = torch.tensor(paste_cls).to(src_instance.gt_classes.device)
                if len(self.inst_bank[int(paste_cls)]) == 0:
                    paste_img = None
                    continue
                paste_img = random.choice(self.inst_bank[int(paste_cls)])
                p_h, p_w = paste_img.shape[1:]
                if img_w - p_w < 1 or img_h - p_h < 1:
                    paste_img = None
                    continue
                break

            if paste_img is not None:
                p_x1 = np.random.randint(0, img_w - p_w)
                p_y1 = np.random.randint(0, img_h - p_h)
                paste_box = Boxes(torch.tensor([[p_x1, p_y1, p_x1 + p_w, p_y1 + p_h]]).to(src_instance.gt_classes.device))
                # mixup
                label_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)] = mixup_lambda * paste_img + \
                (1.0 - mixup_lambda) * label_image[:, p_y1:(p_y1 + p_h), p_x1:(p_x1 + p_w)]
                
                paste_instance = Instances(src_instance.image_size)
                paste_instance.gt_classes = paste_cls[None]
                paste_instance.gt_boxes = paste_box
                
                #  random.uniform(0, 1)
                # np.random.uniform(low=0.5, high=13.3, size=(50,))
                # import pdb
                # pdb.set_trace()
                pos_coord = torch.DoubleTensor([[[np.random.uniform(p_x1, p_x1 + p_w), np.random.uniform(p_y1, p_y1 + p_h)]]])
                neg_coord = torch.DoubleTensor([[[p_x1, p_y1]]])
                gt_point_coords = torch.cat([pos_coord, neg_coord], dim=1)
                
                paste_instance.gt_point_coords = gt_point_coords.to(src_instance.gt_classes.device)
                paste_instance.gt_point_labels = torch.DoubleTensor([[1, 0]]).to(src_instance.gt_classes.device)
                src_instance = Instances.cat([paste_instance, src_instance])
            
        return label_image, src_instance

    def process_pseudo_label_with_point_anno(
            self, unlabel_data_k, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method="", copy_paste_threshold=0.5
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
            pos_point_coord_inst = point_coord_inst[:, 0, :]
            ctr_point_coord_inst = point_coord_inst[:, 1, :]
            if psedo_label_method == "thresholding":
                # Instances(num_instances=0, image_height=1105, image_width=736,
                # fields=[gt_boxes: Boxes(tensor([], device='cuda:0', size=(0, 4))),
                # objectness_logits: tensor([], device='cuda:0')])
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            elif psedo_label_method == "hungarian_with_center_point":
                

                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
                # step 2. choose pseudo bboxes with provised points
                _scores = proposal_bbox_inst.scores
                _bboxes = proposal_bbox_inst.gt_boxes.tensor
                _labels = proposal_bbox_inst.gt_classes

                _points = pos_point_coord_inst.to(_scores.device)
                _ctr_points = ctr_point_coord_inst.to(_scores.device)
                _point_classes = point_class_inst.to(_scores.device)
                _point_labels = point_label_inst.to(_scores.device)
                # inside = (point_coords >= np.array([0, 0])) & (point_coords <= np.array(image_size[::-1]))
                # inside = inside.all(axis=1)

                # 0 for point inside box, and 1 for outside box
                # 1. ctrness cost [num_pred_boxes, num_ctr_points]

                inside_mask = (_ctr_points[:, 0][None, :] > _bboxes[:, 0][:, None]) * (_ctr_points[:, 0][None, :] < _bboxes[:, 2][:, None]) * \
                                (_ctr_points[:, 1][None, :] > _bboxes[:, 1][:, None]) * (_ctr_points[:, 1][None, :] < _bboxes[:, 3][:, None])

                ####################################################################
                left = (_ctr_points[:, 0][None, :] - _bboxes[:, 0][:, None]).abs()
                right = (_ctr_points[:, 0][None, :] - _bboxes[:, 2][:, None]).abs()
                top = (_ctr_points[:, 1][None, :] - _bboxes[:, 1][:, None]).abs()
                down = (_ctr_points[:, 1][None, :] - _bboxes[:, 3][:, None]).abs()
                left_right = torch.min(left, right) / torch.max(left, right)
                top_down = torch.min(top, down) / torch.max(top, down)
                ctrness = left_right * top_down
                cost_ctrness = 1.0 - inside_mask * ctrness
                # if _bboxes.shape[0] > 0:
                #     print(cost_ctrness.sum(), cost_ctrness.min(), cost_ctrness.max())
                ####################################################################

                # 2. inside cost
                cost_inside_box = 1.0 - (_points[:, 0][None, :] > _bboxes[:, 0][:, None]) * (_points[:, 0][None, :] < _bboxes[:, 2][:, None]) * \
                                  (_points[:, 1][None, :] > _bboxes[:, 1][:, None]) * (_points[:, 1][None, :] < _bboxes[:, 3][:, None]) * 1.0

                # 3. when point and box has same class label, cost is (1 - score), elsewise cost is 1.0
                cost_prob = 1.0 - (_labels[:, None] == _point_classes[None, :]) * _scores[:, None]

                # cost = cost_ctrness * 1.0 + cost_inside_box * 1.0 + cost_prob * 1.0
                cost = cost_inside_box * 1.0 + cost_prob * 1.0
                cost = cost.detach().cpu()
                matched_row_inds, matched_col_inds = linear_sum_assignment(cost)

                # only preserve indise box and has the same predicted class
                keep = ((cost_ctrness[matched_row_inds, matched_col_inds] < 0.5) | (
                    cost_inside_box[matched_row_inds, matched_col_inds] < 0.5)) & (
                    _labels[matched_row_inds] == _point_classes[matched_col_inds])

                proposal_bbox_inst = proposal_bbox_inst[matched_row_inds][keep]

            elif psedo_label_method == "hungarian":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
                # step 2. choose pseudo bboxes with provised points
                _scores = proposal_bbox_inst.scores
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
                proposal_bbox_inst.gt_reg_loss_weight = torch.zeros_like(proposal_bbox_inst.gt_classes)
                # import pdb
                # pdb.set_trace()
                proposal_bbox_inst.gt_point_coords = point_coord_inst[matched_col_inds][keep].to(_bboxes.device)
                proposal_bbox_inst.gt_point_labels = _point_labels[matched_col_inds][keep].to(_bboxes.device)

                # gt_labels = proposal_bbox_inst.gt_classes
                # unique_labels = list(set(gt_labels.tolist()))
                # for label in unique_labels:
                #     self.num_pseudos_per_cls[int(label)] += int((gt_labels==label).sum())

                keep_copy_paste = (cost_inside_box[matched_row_inds, matched_col_inds] < 0.5) & (
                            _labels[matched_row_inds] == _point_classes[matched_col_inds]) & (_scores[matched_row_inds] >= copy_paste_threshold)
                
                if self.use_point_guided_cp:
                    # paste label gt to proposal_bbox_inst based on point annotations which failed to match any proposals.
                    paste_classes = None
                    paste_positions = None
                    if len(matched_col_inds) > 0 and keep_copy_paste.sum() < keep_copy_paste.size(0):
                        paste_classes = _point_classes[matched_col_inds][keep_copy_paste==False]
                        paste_positions = _points[matched_col_inds][keep_copy_paste==False]

                    elif len(matched_col_inds) == 0:
                        paste_classes = _point_classes
                        paste_positions = _points

                    if paste_classes is not None and paste_positions is not None:
                        # print("need paste")
                        # unlabel_data_k[ind]['image'], proposal_bbox_inst = \
                        # self.paste_inst_bank_to_unlabel_data_v3(unlabel_data_k[ind]['image'],
                        #                                         proposal_bbox_inst, 
                        #                                         paste_classes, 
                        #                                         paste_positions)

                        unlabel_data_k[ind]['image'], proposal_bbox_inst = \
                        self.paste_inst_bank_to_unlabel_data_v4(unlabel_data_k[ind]['image'],
                                                                proposal_bbox_inst, 
                                                                paste_classes, 
                                                                paste_positions, 
                                                                num_paste_objs=0, 
                                                                mixup_lambda=0.7)
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
                            masks=None,
                            keypoints=None,
                        )
                        dirname = "./results/vis"
                        fname = img_id[:-4] + "_" + str(img.shape[0]) + "x" + str(img.shape[1]) + "_after.jpg"
                        filepath = os.path.join(dirname, fname)
                        print("Saving to {} ...".format(filepath))
                        vis.save(filepath)
                        # import pdb
                        # pdb.set_trace()
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return unlabel_data_k, list_instances, num_proposal_output


    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[pteacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        if self.resume and self.use_point_guided_cp:
            if self.inst_bank_has_empty_classes():
                while self.inst_bank_has_empty_classes():
                    data = next(self._trainer._data_loader_iter)
                    label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
                    self.update_inst_bank(label_data_k)

        # [print(len(self.inst_bank[i])) for i in range(80)]
        # import pdb
        # pdb.set_trace()
        data = next(self._trainer._data_loader_iter)
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data

        if self.use_point_guided_cp:
            for l in label_data_q:
                gt_labels = l['instances'].gt_classes
                unique_labels = list(set(gt_labels.tolist()))
                for label in unique_labels:
                    self.num_gts_per_cls[int(label)] += int((gt_labels==label).sum())
            # print('before', len(l['instances']))
            # l['image'], l['instances'] = self.paste_inst_bank_to_label_data(l['image'], l['instances'], num_paste_objs=3, mixup_lambda=0.5)
            # print('after', len(l['instances']))

        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak

        data_time = time.perf_counter() - start
        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            if self.use_point_guided_cp:
                self.update_inst_bank(label_data_q)
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
                    if key == "loss_mask_point":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.MASK_POINT_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.MASK_POINT_LOSS_WEIGHT
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
            if self.iter % 5 == 0 and self.use_point_guided_cp:
                self.update_inst_bank(label_data_q)

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
            # warmup_threshold_step = 2500.0 # float(self.cfg.SEMISUPNET.BURN_UP_STEP)
            # cur_threshold = 0.9 - (0.9 - cur_threshold) * \
            #         np.clip((self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP)  / warmup_threshold_step, 0., 1.0)

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
            # unlabel_data_q = self.rename_label(unlabel_data_q)

            for l in unlabel_data_q:    # clone for labeled images
                l['point_instances'] = l['instances']

            debug = False
            if debug:
                metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
                scale = 1.0
                for ind in range(len(unlabel_data_q)):
                    img = unlabel_data_q[ind]["image"].permute(1, 2, 0).numpy().copy()
                    img_id = unlabel_data_q[ind]['file_name'].split('/')[-1]
                    visualizer = Visualizer(img, metadata=metadata, scale=scale)
                    # import pdb
                    # pdb.set_trace()
                    target_fields = unlabel_data_q[ind]['instances'].get_fields()
                    labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
                    vis = visualizer.overlay_instances(
                        labels=labels,
                        boxes=target_fields.get("gt_boxes", None).to('cpu'),
                        masks=None,
                        keypoints=None,
                    )
                    dirname = "./results/vis"
                    fname = img_id[:-4] + "_" + str(img.shape[0]) + "x" + str(img.shape[1]) + "_before.jpg"
                    filepath = os.path.join(dirname, fname)
                    print("Saving to {} ...".format(filepath))
                    vis.save(filepath)


            unlabel_data_q, pesudo_proposals_roih_unsup_q, _ = self.process_pseudo_label_with_point_anno(
                unlabel_data_q,
                proposals_roih_unsup_k,
                cur_threshold,
                "roih",
                self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE,
                self.cfg.SEMISUPNET.COPY_PASTE_THRESHOLD,
            )
            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_q

            #  add pseudo-label to unlabeled data
            # unlabel_data_q = self.remove_label(unlabel_data_q)
            # unlabel_data_k = self.remove_label(unlabel_data_k)

            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            )
            if self.iter % 3 == 0 and self.use_point_guided_cp:
                self.update_inst_bank_v2(unlabel_data_q, branch="pseudo_supervised", score_thres=0.6)

            debug = False
            if debug:
                metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
                scale = 1.0
                for ind in range(len(unlabel_data_q)):
                    img = unlabel_data_q[ind]["image"].permute(1, 2, 0).numpy().copy()
                    img_id = unlabel_data_q[ind]['file_name'].split('/')[-1]
                    visualizer = Visualizer(img, metadata=metadata, scale=scale)
                    # import pdb
                    # pdb.set_trace()
                    target_fields = unlabel_data_q[ind]['instances'].get_fields()
                    labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
                    vis = visualizer.overlay_instances(
                        labels=labels,
                        boxes=target_fields.get("gt_boxes", None).to('cpu'),
                        masks=None,
                        keypoints=None,
                    )
                    dirname = "./results/vis"
                    fname = img_id[:-4] + "_" + str(img.shape[0]) + "x" + str(img.shape[1]) + "_after.jpg"
                    filepath = os.path.join(dirname, fname)
                    print("Saving to {} ...".format(filepath))
                    vis.save(filepath)
                    # import pdb
                    # pdb.set_trace()
                    # img_out = visualizer.draw_instance_predictions(unlabel_data_q[ind]['instances'].to("cpu")).get_image()
                    # cv2.imwrite(os.path.join(dirname, fname), img_out)

            # unlabel_data_k = self.add_label(
            #     unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            # )
            all_label_data = label_data_q + label_data_k
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
                    # if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo" or key == "loss_fcos_loc_pseudo":
                    if key == "loss_rpn_loc_pseudo" or key == "loss_fcos_loc_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                        record_dict[key] = record_dict[key] * 0
                    elif key == "loss_fcos_cls_pseudo" or key == "loss_fcos_ctr_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 1
                    elif key == "loss_mask_point" or key == "loss_mask_point_pseudo":
                        # if self.iter < 100:
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.MASK_POINT_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.MASK_POINT_LOSS_WEIGHT
                    elif key == "loss_img_mil" or key == "loss_img_mil_pseudo":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
                    elif key == "loss_ins_mil" or key == "loss_ins_mil_pseudo":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                                record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

            num_unlabel_gt = np.mean([len(a['point_instances']) for a in all_unlabel_data])
            num_unlabel_pseudo = np.mean([len(a['instances']) for a in all_unlabel_data])

            record_dict['num_unlabel_gt'] = num_unlabel_gt
            record_dict['num_unlabel_pseudo'] = num_unlabel_pseudo


        metrics_dict = record_dict
        # print(metrics_dict)
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        # pseudo_recall_sum = 0
        # for key in self.num_gts_per_cls.keys():
        #     self.pseudo_recall[key] = self.num_pseudos_per_cls[key] / (self.cfg.DATALOADER.SUP_PERCENT * self.num_gts_per_cls[key])
        #     pseudo_recall_sum += self.pseudo_recall[key]

        # sorted_pseudo_recall = sorted(self.pseudo_recall.items(), key=lambda kv: kv[1], reverse=True)
        # for ind, sorted_pseudo_recall_i in enumerate(sorted_pseudo_recall):
        #     k = sorted_pseudo_recall[79 - ind][0]
        #     v =  sorted_pseudo_recall[ind][1] / pseudo_recall_sum
        #     self.sampling_freq[k] = v

    @classmethod
    def build_gen_pseudo_loader(cls, cfg, dataset_name):
        # mapper = DatasetMapper(cfg, True)
        mapper = PointSupDatasetMapper(cfg, True)        
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
        # from . import transforms as T
        import detectron2.data.transforms as T
        augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
        augmentation = T.AugmentationList(augmentation)
        # import pdb
        # pdb.set_trace()
        mapper.scale_augmentations = augmentation

        return build_detection_test_loader(cfg, dataset_name, mapper)
        # return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def gen_pseudo_labels_offline(cls, cfg, model):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        self = cls(cfg)
        from detectron2.modeling.postprocessing import detector_postprocess
        logger = logging.getLogger(__name__)
        results = OrderedDict()

        with ExitStack() as stack:
            # if isinstance(model, nn.Module):
            #     stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
                # processed_results = []
                data_loader = cls.build_gen_pseudo_loader(cfg, dataset_name)
                self_data_loader = iter(self.build_gen_pseudo_loader(cfg, dataset_name))
                # import pdb
                # pdb.set_trace()

                if self.inst_bank_has_empty_classes():
                    while self.inst_bank_has_empty_classes():
                        data = next(self_data_loader)
                        self.update_inst_bank(data)
                # import pdb
                # pdb.set_trace()

                evaluator = cls.build_evaluator(cfg, dataset_name)
                # results_i = inference_on_dataset(model, data_loader, evaluator)

                evaluator.reset()

                for idx, inputs in enumerate(data_loader):
                    self.update_inst_bank(inputs)
                    if comm.is_main_process() and idx % 25 == 0:
                        print("processed {} images".format(idx))
                    
                    # with torch.no_grad():
                    (
                        _,
                        proposals_rpn_unsup_k,
                        proposals_roih_unsup_k,
                        _,
                    ) = model(inputs, branch="unsup_data_weak")

                    # proposals_roih_unsup_k_new = model(inputs, branch="unsup_data_weak")
                    # evaluator.process(inputs, proposals_roih_unsup_k_new)
                    # import pdb
                    # pdb.set_trace()

                    #  Pseudo-labeling
                    cur_threshold = cfg.SEMISUPNET.BBOX_THRESHOLD
                    # cur_threshold = 0.1
                    
                    # Pseudo_labeling for ROI head (bbox location/objectness)
                    # pesudo_proposals_roih_unsup_k, _ = cls.process_pseudo_label(
                    #     cls, proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
                    # )

                    _, pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label_with_point_anno(
                        inputs,
                        proposals_roih_unsup_k,
                        cur_threshold,
                        "roih",
                        self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE,
                        self.cfg.SEMISUPNET.COPY_PASTE_THRESHOLD,
                    )

                    # import pdb
                    # pdb.set_trace()

                    height = inputs[0]["height"]
                    width = inputs[0]["width"]
                    results_per_image = pesudo_proposals_roih_unsup_k[0].to("cpu")
                    results_per_image.pred_boxes = results_per_image.gt_boxes
                    results_per_image.pred_classes = results_per_image.gt_classes

                    r = detector_postprocess(results_per_image, height, width)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    evaluator.process(inputs, [{"instances": r}])
                    
                results_i = evaluator.evaluate()
                # An evaluator may return None when not in main process.
                # Replace it by an empty dict instead to make it easier for downstream code to handle
                if results_i is None:
                    results_i = {}

                if comm.is_main_process():
                    assert isinstance(
                        results_i, dict
                    ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                        results_i
                    )
                    logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                    print_csv_format(results_i)

                results[dataset_name] = results_i
        
        return results

# PointSup Trainer based on Unbiased Teacher Trainer for fcos
class OneStageFCOSPointSupTrainer(pteacherTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        burn_up_data_loader, data_loader = self.build_train_loader(cfg)
        self._burn_up_data_loader_iter = iter(burn_up_data_loader)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher
        self.model_teacher.eval()

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self._trainer = SimpleTrainer(
            model, data_loader, self.optimizer
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
        burn_up_mapper = PointSupDatasetMapper(cfg, is_train=True)
        mapper = PointSupTwoCropSeparateDatasetMapper(cfg, is_train=True)
        return build_detection_semisup_train_loader(cfg, burn_up_mapper), build_detection_semisup_train_loader_two_crops(cfg, mapper)


    def rename_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                # will not used in training, except the points
                label_datum["point_instances"] = label_datum["instances"]
                del label_datum["instances"]
                # remove gt_boxes for unlabeled images
                label_datum["point_instances"].remove('gt_boxes')
        return label_data


    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.5):
        valid_map = proposal_bbox_inst.scores > thres
        new_proposal_inst = proposal_bbox_inst[valid_map]
        return new_proposal_inst


    def process_pseudo_label_with_point_anno_debug(
            self, processed_results_unsup_k, cur_threshold, psedo_label_method="", gt_point_coords=None,
            gt_point_labels=None, unlabel_data_k=None
    ):
        list_instances = []
        num_proposal_output = 0.0
        import cv2
        # per img iter

        for point_coord_inst, point_label_inst, proposal_bbox_inst, unlabel_img in zip(gt_point_coords, gt_point_labels,
                                                                                       processed_results_unsup_k,
                                                                                       unlabel_data_k):
            # step 1. thresholding
            if psedo_label_method == "thresholding":
                # Instances(num_instances=0, image_height=1105, image_width=736,
                # fields=[gt_boxes: Boxes(tensor([], device='cuda:0', size=(0, 4))),
                # objectness_logits: tensor([], device='cuda:0')])
                proposal_bbox_inst = self.threshold_bbox(proposal_bbox_inst, thres=cur_threshold)

                from detectron2.utils.visualizer import Visualizer
                from detectron2.data import MetadataCatalog
                metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
                scale = 1.0
                img = unlabel_img["image"].permute(1, 2, 0).numpy().copy()
                img_id = unlabel_img['file_name'].split('/')[-1]
                for point in point_coord_inst:
                    point = (int(point[0]), int(point[1]))
                    # Radius of circle
                    radius = 2
                    # Red color in BGR
                    color = (0, 0, 255)
                    # Line thickness of 2 px
                    thickness = 2
                    # Using cv2.circle() method
                    # Draw a circle with blue line borders of thickness of 2 px
                    img = cv2.circle(img, point, radius, color, thickness)

                if self.cfg.SEMISUPNET.POINT_SUP:
                    # step 2. choose pseudo bboxes with provised points

                    _scores = proposal_bbox_inst.scores
                    _bboxes = proposal_bbox_inst.gt_boxes.tensor
                    _labels = proposal_bbox_inst.gt_classes

                    _points = point_coord_inst.to(_scores.device)
                    _point_labels = point_label_inst.to(_scores.device)
                    # inside = (point_coords >= np.array([0, 0])) & (point_coords <= np.array(image_size[::-1]))
                    # inside = inside.all(axis=1)
                    # 0 for point inside box, and 1 for outside box
                    cost_inside_box = 1.0 - (_points[:, 0][None, :] > _bboxes[:, 0][:, None]) * (
                            _points[:, 0][None, :] < _bboxes[:, 2][:, None]) * \
                                      (_points[:, 1][None, :] > _bboxes[:, 1][:, None]) * (
                                              _points[:, 1][None, :] < _bboxes[:, 3][:, None]) * 1.0

                    # when point and box has same class label, cost is (1 - score), elsewise cost is 1.0
                    cost_prob = 1.0 - (_labels[:, None] == _point_labels[None, :]) * _scores[:, None]

                    cost = cost_inside_box * 1.0 + cost_prob * 1.0
                    cost = cost.detach().cpu()
                    matched_row_inds, matched_col_inds = linear_sum_assignment(cost)

                    # only preserve indise box and has the same predicted class
                    keep = (cost_inside_box[matched_row_inds, matched_col_inds] < 0.5) & (
                            _labels[matched_row_inds] == _point_labels[matched_col_inds])
                    # print("before", proposal_bbox_inst)
                    proposal_bbox_inst = proposal_bbox_inst[matched_row_inds][keep]
                    # print("after", proposal_bbox_inst)
                    # _pseudo_bboxes = _bboxes[matched_row_inds][keep]
                    # _pseudo_labels = _point_labels[matched_col_inds][keep]

                    visualizer = Visualizer(img[..., ::-1].copy(), metadata=metadata, scale=scale)
                    target_fields = unlabel_img['instances'].get_fields()
                    # target_fields = proposal_bbox_inst.get_fields()
                    labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
                    vis = visualizer.overlay_instances(
                        labels=labels,
                        boxes=target_fields.get("gt_boxes", None).to('cpu'),
                        masks=None,
                        keypoints=None,
                    )
                    dirname = "/home/zhengyi/code/semi-det/vis/"
                    fname = img_id[:-4] + "_" + str(img.shape[0]) + "x" + str(img.shape[1]) + "_gt.jpg"
                    filepath = os.path.join(dirname, fname)
                    print("Saving to {} ...".format(filepath))
                    vis.save(filepath)

                    for label, score, bbox in zip(proposal_bbox_inst.gt_classes, proposal_bbox_inst.scores,
                                                  proposal_bbox_inst.gt_boxes.tensor):

                        label = metadata.thing_classes[label]

                        x1, y1, x2, y2 = bbox
                        color = (0, 0, 255)
                        thickness = 3

                        # Write some Text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerOfText = (int(x1), int(y1))
                        fontScale = 1
                        fontColor = (255, 0, 0)
                        lineType = 2

                        cv2.putText(img, str(score),
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

                    # target_fields = pseudo_data[ind]['instances'].get_fields()
                    fname = img_id[:-4] + "_" + str(img.shape[0]) + "x" + str(img.shape[1]) + "_pred.jpg"
                    filepath = os.path.join(dirname, fname)
                    print("Saving to {} ...".format(filepath))
                    cv2.imwrite(filepath, img)
                    pdb.set_trace()


            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(processed_results_unsup_k)
        return list_instances, num_proposal_output




    def process_pseudo_label_with_point_anno(
            self, unlabel_data_k, unlabel_data_q, processed_results_unsup_k, cur_threshold,  psedo_label_method="",
    ):
        list_instances = []
        num_proposal_output = 0.0
        gt_point_coords = []
        gt_point_labels = []
        gt_bbox_classes = []
        for i in range(len(unlabel_data_k)):
            import pdb;pdb.set_trace()
            gt_point_coords.append(unlabel_data_k[i]['instances'].gt_point_coords)
            gt_point_labels.append(unlabel_data_k[i]['instances'].gt_point_labels)
            gt_bbox_classes.append(unlabel_data_k[i]['instances'].gt_classes)

            origin_img_h, origin_img_w = unlabel_data_k[i]['height'], unlabel_data_k[i]['width']
            auged_img_h, auged_img_w = unlabel_data_k[i]['instances'].image_size
            scale = auged_img_h / origin_img_h

            processed_results_unsup_k[i].gt_boxes = Boxes(processed_results_unsup_k[i].pred_boxes.tensor * scale)
            processed_results_unsup_k[i].gt_classes = processed_results_unsup_k[i].pred_classes
            processed_results_unsup_k[i].remove('pred_boxes')
            processed_results_unsup_k[i].remove('pred_classes')
            processed_results_unsup_k[i].remove('locations')
            processed_results_unsup_k[i].remove('fpn_levels')

            mask = torch.ones_like(unlabel_data_k[i]["image"], device=unlabel_data_k[i]["image"].device)
            for box_ind in reversed(range(0, len(processed_results_unsup_k[i].scores))):
                score = processed_results_unsup_k[i].scores[box_ind]
                xmin, ymin, xmax, ymax = processed_results_unsup_k[i].gt_boxes[box_ind].tensor.long().squeeze(0)
                if score < self.cfg.SEMISUPNET.BBOX_THRESHOLD:
                    mask[:, ymin:ymax, xmin:xmax] = 0
                else:
                    mask[:, ymin:ymax, xmin:xmax] = 1

            unlabel_data_k[i]["image"][mask == 0] = 128
            unlabel_data_q[i]["image"][mask == 0] = 128


        # per img iter
        for point_coord_inst, point_class_inst, point_label_inst, proposal_bbox_inst in zip(gt_point_coords,
                                                                                            gt_bbox_classes,
                                                                                            gt_point_labels,
                                                                                            processed_results_unsup_k):
            # step 1. thresholding
            pos_point_coord_inst = point_coord_inst[:, 0, :]
            if psedo_label_method == "thresholding":
                # Instances(num_instances=0, image_height=1105, image_width=736,
                # fields=[gt_boxes: Boxes(tensor([], device='cuda:0', size=(0, 4))),
                # objectness_logits: tensor([], device='cuda:0')])
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold
                )
            elif psedo_label_method == "hungarian":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold
                )
                # step 2. choose pseudo bboxes with provised points
                _scores = proposal_bbox_inst.scores
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
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(processed_results_unsup_k)
        return list_instances, num_proposal_output




    # =====================================================
    # =================== Training Flow ===================
    # =====================================================


    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[pteacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            # label_data_q = next(self._burn_up_data_loader_iter)
            data = next(self._trainer._data_loader_iter)
            # data_q and data_k from different augmentations (q:strong, k:weak)
            # label_strong, label_weak, unlabed_strong, unlabled_weak
            label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
            label_data_q.extend(label_data_k)
            for l in label_data_q:  # clone for labeled data
                l['point_instances'] = l['instances']
            data_time = time.perf_counter() - start
            record_dict = self.model(label_data_q, branch="supervised")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    if key == "loss_img_mil":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
                    elif key == "loss_ins_mil":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT
                    else:
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        else:
            data = next(self._trainer._data_loader_iter)
            # data_q and data_k from different augmentations (q:strong, k:weak)
            # label_strong, label_weak, unlabed_strong, unlabled_weak
            label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
            data_time = time.perf_counter() - start

            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                del self._burn_up_data_loader_iter
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
                tch_forward_time_start = time.perf_counter()
                # processed_results_unsup_k = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")
                processed_results_unsup_k = self.model_teacher(unlabel_data_k)
                processed_results_unsup_k = [results_per_img['instances'] for results_per_img in
                                             processed_results_unsup_k]
                # import pdb;pdb.set_trace()
                tch_forward_time = time.perf_counter() - tch_forward_time_start

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            joint_proposal_dict = {}

            # pesudo_proposals_unsup_k, _ = self.process_pseudo_label_with_point_anno_debug(
            #     processed_results_unsup_k, cur_threshold, "thresholding", gt_point_coords, gt_point_labels, unlabel_data_k
            # )

            pesudo_proposals_unsup_k, _ = self.process_pseudo_label_with_point_anno(
                unlabel_data_k, unlabel_data_q, processed_results_unsup_k, cur_threshold, self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE
            )
            joint_proposal_dict["proposals_pseudo"] = pesudo_proposals_unsup_k

            #  add pseudo-label to unlabeled data
            unlabel_data_q = self.rename_label(unlabel_data_q)
            # unlabel_data_k = self.remove_label(unlabel_data_k)

            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo"]
            )
            # unlabel_data_k = self.add_label(
            #     unlabel_data_k, joint_proposal_dict["proposals_pseudo"]
            # )

            all_label_data = label_data_q + label_data_k
            for l in all_label_data:    # clone for labeled images
                l['point_instances'] = l['instances']
            all_unlabel_data = unlabel_data_q

            record_all_label_data = self.model(all_label_data)
            record_dict.update(record_all_label_data)
            record_all_unlabel_data = self.model(all_unlabel_data)
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
                    if key == "loss_fcos_loc_pseudo" or key == "loss_fcos_iou_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key == "loss_img_mil" or key == "loss_img_mil_pseudo":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
                    elif key == "loss_ins_mil" or key == "loss_ins_mil_pseudo":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                                record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )

                        # print(record_dict[key], self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT)
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT

                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        # metrics_dict["tch_forward_time"] = tch_forward_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()



class FCOSPointSupTrainer(pteacherTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        burn_up_data_loader, data_loader = self.build_train_loader(cfg)
        self._burn_up_data_loader_iter = iter(burn_up_data_loader)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher
        # self.model_teacher.eval()

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self._trainer = SimpleTrainer(
            model, data_loader, self.optimizer
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
        burn_up_mapper = PointSupDatasetMapper(cfg, is_train=True)
        mapper = PointSupTwoCropSeparateDatasetMapper(cfg, is_train=True)
        return build_detection_semisup_train_loader(cfg,
                                                    burn_up_mapper), build_detection_semisup_train_loader_two_crops(cfg,
                                                                                                                    mapper)

    def rename_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                # will not used in training, except the points
                label_datum["point_instances"] = label_datum["instances"]
                del label_datum["instances"]
                # remove gt_boxes for unlabeled images
                label_datum["point_instances"].remove('gt_boxes')
        return label_data

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.5):
        valid_map = proposal_bbox_inst.scores > thres
        new_proposal_inst = proposal_bbox_inst[valid_map]
        return new_proposal_inst

    def process_pseudo_label_with_point_anno_debug(
            self, processed_results_unsup_k, cur_threshold, psedo_label_method="", gt_point_coords=None,
            gt_point_labels=None, unlabel_data_k=None
    ):
        list_instances = []
        num_proposal_output = 0.0
        import cv2
        # per img iter

        for point_coord_inst, point_label_inst, proposal_bbox_inst, unlabel_img in zip(gt_point_coords, gt_point_labels,
                                                                                       processed_results_unsup_k,
                                                                                       unlabel_data_k):
            # step 1. thresholding
            if psedo_label_method == "thresholding":
                # Instances(num_instances=0, image_height=1105, image_width=736,
                # fields=[gt_boxes: Boxes(tensor([], device='cuda:0', size=(0, 4))),
                # objectness_logits: tensor([], device='cuda:0')])
                proposal_bbox_inst = self.threshold_bbox(proposal_bbox_inst, thres=cur_threshold)

                from detectron2.utils.visualizer import Visualizer
                from detectron2.data import MetadataCatalog
                metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
                scale = 1.0
                img = unlabel_img["image"].permute(1, 2, 0).numpy().copy()
                img_id = unlabel_img['file_name'].split('/')[-1]
                for point in point_coord_inst:
                    point = (int(point[0]), int(point[1]))
                    # Radius of circle
                    radius = 2
                    # Red color in BGR
                    color = (0, 0, 255)
                    # Line thickness of 2 px
                    thickness = 2
                    # Using cv2.circle() method
                    # Draw a circle with blue line borders of thickness of 2 px
                    img = cv2.circle(img, point, radius, color, thickness)

                if self.cfg.SEMISUPNET.POINT_SUP:
                    # step 2. choose pseudo bboxes with provised points

                    _scores = proposal_bbox_inst.scores
                    _bboxes = proposal_bbox_inst.gt_boxes.tensor
                    _labels = proposal_bbox_inst.gt_classes

                    _points = point_coord_inst.to(_scores.device)
                    _point_labels = point_label_inst.to(_scores.device)
                    # inside = (point_coords >= np.array([0, 0])) & (point_coords <= np.array(image_size[::-1]))
                    # inside = inside.all(axis=1)
                    # 0 for point inside box, and 1 for outside box
                    cost_inside_box = 1.0 - (_points[:, 0][None, :] > _bboxes[:, 0][:, None]) * (
                            _points[:, 0][None, :] < _bboxes[:, 2][:, None]) * \
                                      (_points[:, 1][None, :] > _bboxes[:, 1][:, None]) * (
                                              _points[:, 1][None, :] < _bboxes[:, 3][:, None]) * 1.0

                    # when point and box has same class label, cost is (1 - score), elsewise cost is 1.0
                    cost_prob = 1.0 - (_labels[:, None] == _point_labels[None, :]) * _scores[:, None]

                    cost = cost_inside_box * 1.0 + cost_prob * 1.0
                    cost = cost.detach().cpu()
                    matched_row_inds, matched_col_inds = linear_sum_assignment(cost)

                    # only preserve indise box and has the same predicted class
                    keep = (cost_inside_box[matched_row_inds, matched_col_inds] < 0.5) & (
                            _labels[matched_row_inds] == _point_labels[matched_col_inds])
                    # print("before", proposal_bbox_inst)
                    proposal_bbox_inst = proposal_bbox_inst[matched_row_inds][keep]
                    # print("after", proposal_bbox_inst)
                    # _pseudo_bboxes = _bboxes[matched_row_inds][keep]
                    # _pseudo_labels = _point_labels[matched_col_inds][keep]

                    visualizer = Visualizer(img[..., ::-1].copy(), metadata=metadata, scale=scale)
                    target_fields = unlabel_img['instances'].get_fields()
                    # target_fields = proposal_bbox_inst.get_fields()
                    labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
                    vis = visualizer.overlay_instances(
                        labels=labels,
                        boxes=target_fields.get("gt_boxes", None).to('cpu'),
                        masks=None,
                        keypoints=None,
                    )
                    dirname = "/home/zhengyi/code/semi-det/vis/"
                    fname = img_id[:-4] + "_" + str(img.shape[0]) + "x" + str(img.shape[1]) + "_gt.jpg"
                    filepath = os.path.join(dirname, fname)
                    print("Saving to {} ...".format(filepath))
                    vis.save(filepath)

                    for label, score, bbox in zip(proposal_bbox_inst.gt_classes, proposal_bbox_inst.scores,
                                                  proposal_bbox_inst.gt_boxes.tensor):
                        import pdb
                        pdb.set_trace()
                        label = metadata.thing_classes[label]

                        x1, y1, x2, y2 = bbox
                        color = (0, 0, 255)
                        thickness = 3

                        # Write some Text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerOfText = (int(x1), int(y1))
                        fontScale = 1
                        fontColor = (255, 0, 0)
                        lineType = 2

                        cv2.putText(img, str(score),
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

                    # target_fields = pseudo_data[ind]['instances'].get_fields()
                    fname = img_id[:-4] + "_" + str(img.shape[0]) + "x" + str(img.shape[1]) + "_pred.jpg"
                    filepath = os.path.join(dirname, fname)
                    print("Saving to {} ...".format(filepath))
                    cv2.imwrite(filepath, img)
                    pdb.set_trace()


            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(processed_results_unsup_k)
        return list_instances, num_proposal_output

    def process_pseudo_label_with_point_anno(
            self, unlabel_data_k, unlabel_data_q, processed_results_unsup_k, cur_threshold, proposal_type="rpn", psedo_label_method="hungarian",
    ):
        list_instances = []
        num_proposal_output = 0.0
        gt_point_coords = []
        gt_point_labels = []
        gt_bbox_classes = []
        for i in range(len(unlabel_data_k)):
            # import pdb;pdb.set_trace()
            gt_point_coords.append(unlabel_data_k[i]['instances'].gt_point_coords)
            gt_point_labels.append(unlabel_data_k[i]['instances'].gt_point_labels)
            gt_bbox_classes.append(unlabel_data_k[i]['instances'].gt_classes)

            # if proposal_type == "rpn":
            # origin_img_h, origin_img_w = unlabel_data_k[i]['height'], unlabel_data_k[i]['width']
            # auged_img_h, auged_img_w = unlabel_data_k[i]['instances'].image_size
            # scale = auged_img_h / origin_img_h
            processed_results_unsup_k[i].gt_boxes = processed_results_unsup_k[i].pred_boxes
            processed_results_unsup_k[i].gt_classes = processed_results_unsup_k[i].pred_classes
            processed_results_unsup_k[i].remove('pred_boxes')
            processed_results_unsup_k[i].remove('pred_classes')

            mask = torch.ones_like(unlabel_data_k[i]["image"], device=unlabel_data_k[i]["image"].device)
            for box_ind in reversed(range(0, len(processed_results_unsup_k[i].scores))):
                score = processed_results_unsup_k[i].scores[box_ind]
                xmin, ymin, xmax, ymax = processed_results_unsup_k[i].gt_boxes[box_ind].tensor.long().squeeze(0)
                if score < self.cfg.SEMISUPNET.BBOX_THRESHOLD:
                    mask[:, ymin:ymax, xmin:xmax] = 0
                else:
                    mask[:, ymin:ymax, xmin:xmax] = 1

            unlabel_data_k[i]["image"][mask == 0] = 128
            unlabel_data_q[i]["image"][mask == 0] = 128

        # per img iter
        for point_coord_inst, point_class_inst, point_label_inst, proposal_bbox_inst in zip(gt_point_coords,
                                                                                            gt_bbox_classes,
                                                                                            gt_point_labels,
                                                                                            processed_results_unsup_k):
            # step 1. thresholding
            pos_point_coord_inst = point_coord_inst[:, 0, :]
            if psedo_label_method == "thresholding":
                # Instances(num_instances=0, image_height=1105, image_width=736,
                # fields=[gt_boxes: Boxes(tensor([], device='cuda:0', size=(0, 4))),
                # objectness_logits: tensor([], device='cuda:0')])
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold
                )
            elif psedo_label_method == "hungarian":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold
                )
                # step 2. choose pseudo bboxes with provised points
                _scores = proposal_bbox_inst.scores
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
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(processed_results_unsup_k)
        return list_instances, num_proposal_output



    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[pteacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            # label_data_q = next(self._burn_up_data_loader_iter)
            data = next(self._trainer._data_loader_iter)
            # data_q and data_k from different augmentations (q:strong, k:weak)
            # label_strong, label_weak, unlabed_strong, unlabled_weak
            label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
            label_data_q.extend(label_data_k)
            for l in label_data_q:  # clone for labeled data
                l['point_instances'] = l['instances']
            data_time = time.perf_counter() - start
            # record_dict = self.model(label_data_q, branch="supervised")
            record_dict, _, _, _ = self.model(label_data_q, branch="supervised")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    if key == "loss_img_mil":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
                        # loss_dict[key] = record_dict[key] * 0.0
                        # record_dict[key] = record_dict[key] * 0.0
                    elif key == "loss_ins_mil":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT
                        # loss_dict[key] = record_dict[key] * 0.0
                        # record_dict[key] = record_dict[key] * 0.0
                    elif key == "loss_cls" or key == "loss_box_reg":
                        loss_dict[key] = record_dict[key] * 1.0
                        record_dict[key] = record_dict[key] * 1.0
                        # loss_dict[key] = record_dict[key] * 0.0
                        # record_dict[key] = record_dict[key] * 0.0
                    else:
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        else:
            data = next(self._trainer._data_loader_iter)
            # data_q and data_k from different augmentations (q:strong, k:weak)
            # label_strong, label_weak, unlabed_strong, unlabled_weak
            label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
            data_time = time.perf_counter() - start

            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                del self._burn_up_data_loader_iter
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
                tch_forward_time_start = time.perf_counter()
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

                proposals_rpn_unsup_k = [results_per_img['instances'] for results_per_img in
                                             proposals_rpn_unsup_k]

                # processed_results_unsup_k = []
                # for idx, (rpn_proposal, roih_proprosal) in enumerate(zip(proposals_rpn_unsup_k, proposals_roih_unsup_k)):
                #     rpn_proposal.remove('locations')
                #     rpn_proposal.remove('fpn_levels')
                #     import pdb
                #     pdb.set_trace()
                #     origin_img_h, origin_img_w = rpn_proposal.image_size
                #     auged_img_h, auged_img_w = roih_proprosal.image_size
                #     scale = auged_img_h / origin_img_h
                #     rpn_proposal._image_size = roih_proprosal._image_size
                #     rpn_proposal.set("pred_boxes", Boxes(rpn_proposal.pred_boxes.tensor * scale))
                #     processed_results_unsup_k.append(Instances.cat([rpn_proposal, roih_proprosal]))

                processed_results_unsup_k = proposals_roih_unsup_k

                tch_forward_time = time.perf_counter() - tch_forward_time_start

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            joint_proposal_dict = {}
            pesudo_proposals_unsup_k, _ = self.process_pseudo_label_with_point_anno(
                unlabel_data_k, unlabel_data_q, processed_results_unsup_k, cur_threshold, "rpn",
                self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE
            )
            joint_proposal_dict["proposals_pseudo"] = pesudo_proposals_unsup_k

            #  add pseudo-label to unlabeled data
            unlabel_data_q = self.rename_label(unlabel_data_q)
            # unlabel_data_k = self.remove_label(unlabel_data_k)

            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo"]
            )
            # unlabel_data_k = self.add_label(
            #     unlabel_data_k, joint_proposal_dict["proposals_pseudo"]
            # )

            all_label_data = label_data_q + label_data_k
            for l in all_label_data:  # clone for labeled images
                l['point_instances'] = l['instances']
            all_unlabel_data = unlabel_data_q

            record_all_label_data, _, _, _ = self.model(all_label_data)
            record_dict.update(record_all_label_data)
            record_all_unlabel_data, _, _, _ = self.model(all_unlabel_data)
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
                    # if key == "loss_fcos_loc_pseudo" or key == "loss_fcos_iou_pseudo" or key == "loss_fcos_ctr_pseudo" or key == "loss_box_reg_pseudo":
                    if key == "loss_rpn_loc_pseudo" or key == "loss_fcos_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                        record_dict[key] = record_dict[key] * 0
                    elif key == "loss_img_mil" or key == "loss_img_mil_pseudo":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.IMG_MIL_LOSS_WEIGHT
                    elif key == "loss_ins_mil" or key == "loss_ins_mil_pseudo":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_MIL_LOSS_WEIGHT
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                                record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )

                        # print(record_dict[key], self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT)
                        record_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT

                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        # metrics_dict["tch_forward_time"] = tch_forward_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
