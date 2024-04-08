import torch
import logging
import time
import datetime
from detectron2.utils.events import CommonMetricPrinter, get_event_storage, EventWriter

class PSSODMetricPrinter(CommonMetricPrinter):

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None
        # import pdb
        # pdb.set_trace()
        try:
            num_unlabel_gt = storage.history("num_unlabel_gt").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            num_unlabel_gt = None

        try:
            num_unlabel_pseudo = storage.history("num_unlabel_pseudo").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            num_unlabel_pseudo = None

        try:
            num_unlabel_pairwise_pseudo = storage.history("num_unlabel_pairwise_pseudo").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            num_unlabel_pairwise_pseudo = None

        eta_string = None
        try:
            iter_time = storage.history("time").global_avg()
            eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration)
            storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint=False)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            iter_time = None
            # estimate eta on our own - more noisy
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (
                    iteration - self._last_write[0]
                )
                eta_seconds = estimate_iter_time * (self._max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iteration, time.perf_counter())

        try:
            lr = "{:.6f}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        if num_unlabel_gt is None or num_unlabel_pseudo is None:
            self.logger.info(
                " {eta}iter: {iter}  {losses}  {time}{data_time}lr: {lr}  {memory}".format(
                    eta=f"eta: {eta_string}  " if eta_string else "",
                    iter=iteration,
                    losses="  ".join(
                        [
                            "{}: {:.3f}".format(k, v.median(20))
                            for k, v in storage.histories().items()
                            if "loss" in k
                        ]
                    ),
                    time="time: {:.4f}  ".format(iter_time) if iter_time is not None else "",
                    data_time="data_time: {:.4f}  ".format(data_time) if data_time is not None else "",
                    lr=lr,
                    memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
                )
            )
        elif num_unlabel_pairwise_pseudo is not None:
            self.logger.info(
                " {eta}iter: {iter}  {losses} {num_unlabel_gt}{num_unlabel_pseudo}{num_unlabel_pairwise_pseudo} {time}{data_time}lr: {lr}  {memory}".format(
                    eta=f"eta: {eta_string}  " if eta_string else "",
                    iter=iteration,
                    losses="  ".join(
                        [
                            "{}: {:.3f}".format(k, v.median(20))
                            for k, v in storage.histories().items()
                            if "loss" in k
                        ]
                    ),
                    num_unlabel_gt="num_unlabel_gt: {:.4f}  ".format(
                        num_unlabel_gt) if num_unlabel_gt is not None else "",
                    num_unlabel_pseudo="num_unlabel_pseudo: {:.4f}  ".format(
                        num_unlabel_pseudo) if num_unlabel_pseudo is not None else "",
                    num_unlabel_pairwise_pseudo="num_unlabel_pairwise_pseudo: {:.4f}  ".format(
                        num_unlabel_pairwise_pseudo) if num_unlabel_pairwise_pseudo is not None else "",
                    time="time: {:.4f}  ".format(iter_time) if iter_time is not None else "",
                    data_time="data_time: {:.4f}  ".format(data_time) if data_time is not None else "",
                    lr=lr,
                    memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
                )
            )
        else:
            self.logger.info(
                " {eta}iter: {iter}  {losses} {num_unlabel_gt}{num_unlabel_pseudo} {time}{data_time}lr: {lr}  {memory}".format(
                    eta=f"eta: {eta_string}  " if eta_string else "",
                    iter=iteration,
                    losses="  ".join(
                        [
                            "{}: {:.3f}".format(k, v.median(20))
                            for k, v in storage.histories().items()
                            if "loss" in k
                        ]
                    ),
                    num_unlabel_gt="num_unlabel_gt: {:.4f}  ".format(
                        num_unlabel_gt) if num_unlabel_gt is not None else "",
                    num_unlabel_pseudo="num_unlabel_pseudo: {:.4f}  ".format(
                        num_unlabel_pseudo) if num_unlabel_pseudo is not None else "",

                    time="time: {:.4f}  ".format(iter_time) if iter_time is not None else "",
                    data_time="data_time: {:.4f}  ".format(data_time) if data_time is not None else "",
                    lr=lr,
                    memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
                )
            )