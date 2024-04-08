# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from bisect import bisect_right
from typing import List
import torch
from detectron2.solver.lr_scheduler import _get_warmup_factor_at_iter


class WarmupTwoStageMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        factor_list: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
        if len(milestones) + 1 != len(factor_list):
            raise ValueError("Length of milestones should match length of factor_list.")

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.factor_list = factor_list

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:

        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )

        return [
            base_lr
            * warmup_factor
            * self.factor_list[bisect_right(self.milestones, self.last_epoch)]
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


if __name__ == "__main__":
    # test
    """
    Create configs and perform basic setups.
    """
    from adet.config import get_cfg
    from detectron2.engine import default_setup
    import torch.nn as nn
    import matplotlib.pyplot as plt

    cfg = get_cfg()
    config_file = (
        "./configs/SEMI_Faster-RCNN/faster_rcnn_R_50_FPN_2x.yaml"
    )
    cfg.SOLVER.MAX_ITER = 180000
    cfg.merge_from_file(config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    # default_setup(cfg, args)

    import torch.nn.functional as F
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    model = Net()

    optimizer = torch.optim.SGD(
        model.parameters(),
        cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        nesterov=cfg.SOLVER.NESTEROV,
    )

    lr = WarmupTwoStageMultiStepLR(optimizer=optimizer,
                                   warmup_iters=1000,
                                   milestones=[60000, 80000, 90000, 360000],
                                   factor_list=[1, 0.1, 1, 0.1, 1],
                                   )
    # lr = UnbiasedTchWarmupMultiStepLR(optimizer=optimizer, milestones=cfg.SOLVER.STEPS)

    lr_list = [[], []]
    for iter in range(0, cfg.SOLVER.MAX_ITER):
        lr.last_epoch = iter
        lr_list[0].append(iter)
        # if iter >= 90000:
        #     import pdb
        #     pdb.set_trace()
        if iter == cfg.SOLVER.MAX_ITER - 1:
            print("last lr :", lr.get_lr()[0])
        lr_list[1].append(lr.get_lr()[0])

    fig, ax = plt.subplots(figsize=(12, 4), dpi=80)
    ax.plot(lr_list[0], lr_list[1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    plt.gcf().savefig("/home/zhengyi/code/semi-det/lr.png")
    # plt.gcf().savefig("lr.pdf")
    plt.clf()