import sys
import random

import torch

import utils.multiprocessing as mpu
from config import get_cfg

from train import train
from test import test

def main():
    """
    Main function to spawn the train and test process.
    """
    cfg = get_cfg()

    if cfg.TRAIN.ENABLE:
        if cfg.NUM_GPUS > 1:
            torch.multiprocessing.spawn(
                mpu.run,
                nprocs=cfg.NUM_GPUS,
                args=(
                    cfg.NUM_GPUS,
                    train,
                    cfg.DIST_INIT_METHOD,
                    cfg.SHARD_ID,
                    cfg.NUM_SHARDS,
                    cfg.DIST_BACKEND,
                    cfg,
                ),
                daemon=False,
            )
        else:
            train(cfg=cfg)

    if cfg.TEST.ENABLE:
        if cfg.NUM_GPUS > 1:
            torch.multiprocessing.spawn(
                mpu.run,
                nprocs=cfg.NUM_GPUS,
                args=(
                    cfg.NUM_GPUS,
                    test,
                    cfg.DIST_INIT_METHOD,
                    cfg.SHARD_ID,
                    cfg.NUM_SHARDS,
                    cfg.DIST_BACKEND,
                    cfg,
                ),
                daemon=False,
            )
        else:
            test(cfg=cfg)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver")
    main()
