# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Learning rate policy."""

import math

from torch.optim.lr_scheduler import LambdaLR


def lr_lambda(current_step, num_training_steps, num_warmup_steps):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_lr_at_epoch(cfg, cur_epoch, global_step):
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
        global_step (int): the number of step of the current training stage
    """
    if cfg.SOLVER.LR_POLICY == "linear":
        num_warmup_steps = int(
            cfg.SOLVER.NUM_STEPS * cfg.SOLVER.WARMUP_PROPORTION
        )
        alpha = lr_lambda(global_step, cfg.SOLVER.NUM_STEPS, num_warmup_steps)
        lr = cfg.SOLVER.BASE_LR * alpha
    else:
        lr = get_lr_func(cfg.SOLVER.LR_POLICY)(cfg, cur_epoch)
        # Perform warm up.
        if cur_epoch < cfg.SOLVER.WARMUP_EPOCHS:
            lr_start = cfg.SOLVER.WARMUP_START_LR
            lr_end = get_lr_func(cfg.SOLVER.LR_POLICY)(
                cfg, cfg.SOLVER.WARMUP_EPOCHS
            )
            alpha = (lr_end - lr_start) / cfg.SOLVER.WARMUP_EPOCHS
            lr = cur_epoch * alpha + lr_start
    return lr


def lr_func_cosine(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return (
        cfg.SOLVER.BASE_LR
        * (math.cos(math.pi * cur_epoch / cfg.SOLVER.MAX_EPOCH) + 1.0)
        * 0.5
    )


def get_lr_func(lr_policy):
    """
    Given the configs, retrieve the specified lr policy function.
    Args:
        lr_policy (string): the learning rate policy to use for the job.
    """
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]
