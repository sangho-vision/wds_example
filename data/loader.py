# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import random
from functools import partial
import numpy as np
import torch
import math
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import _InfiniteConstantSampler

from data.build import build_dataset
from data.collate import COLLATE_FN
from utils import distributed as du

import webdataset as wds


def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    collate_fn = None
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS)
        batch_size = int(batch_size / du.get_world_size())
        drop_last = True
        length = cfg.TRAIN.DATASET_SIZE
        nominal = int(length / batch_size)
    elif split in ["val"]:
        dataset_name = cfg.VAL.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / du.get_world_size())
        drop_last = False
        length = cfg.VAL.DATASET_SIZE
        nominal = int(length / batch_size)
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / du.get_world_size())
        drop_last = False
        length = cfg.TEST.DATASET_SIZE
        nominal = math.ceil(length / batch_size)

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)
    if dataset_name == "KineticsSounds":
        collate_fn = COLLATE_FN["kinetics"]

    # Create a loader
    if cfg.DATA_LOADER.NUM_WORKERS > 0:
        loader = wds.MultiDataset(
            dataset,
            workers=cfg.DATA_LOADER.NUM_WORKERS,
            nominal=nominal,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        )
        if split in ["train"]:
            loader = loader.shuffle(batch_size)
        loader = loader.batched(batch_size)
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )

    return loader


def shuffle_dataset(loader, cur_epoch):
    """"
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler, _InfiniteConstantSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
