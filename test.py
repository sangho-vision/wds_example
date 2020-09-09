import os
import random
import math
import time
import pprint
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.precise_bn import get_bn_modules, update_bn_stats
import models.optimizer as optim
import models.losses as losses
import utils.distributed as du
import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
from data import loader
from models import build_model
from utils.meters import TrainMeter, ValMeter, TestMeter

logger = logging.get_logger(__name__)


def test(cfg):
    """
    Perform multi-view testing on the trained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
        config.py
    """
    # Set random seed from configs.
    if cfg.RNG_SEED != -1:
        random.seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.NUM_GPUS)

    # Print config.
    logger.info("Test with config:")
    logger.info(pprint.pformat(cfg))

    # Model for testing
    model = build_model(cfg)
    # Print model statistics.
    if du.is_master_proc(cfg.NUM_GPUS):
        misc.log_model_info(model, cfg, use_train_input=False)

    if cfg.TEST.CHECKPOINT_FILE_PATH:
        if os.path.isfile(cfg.TEST.CHECKPOINT_FILE_PATH):
            logger.info(
                "=> loading checkpoint '{}'".format(
                    cfg.TEST.CHECKPOINT_FILE_PATH
                )
            )
            ms = model.module if cfg.NUM_GPUS > 1 else model
            # Load the checkpoint on CPU to avoid GPU mem spike.
            checkpoint = torch.load(
                cfg.TEST.CHECKPOINT_FILE_PATH, map_location='cpu'
            )
            ms.load_state_dict(checkpoint['state_dict'])
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    cfg.TEST.CHECKPOINT_FILE_PATH,
                    checkpoint['epoch']
                )
            )
    else:
        logger.info("Test with random initialization for debugging")

    # Create video testing loaders
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    # Create meters for multi-view testing.
    test_meter = TestMeter(
        len(test_loader.dataset),
        cfg.TEST.DATASET_SIZE,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
        cfg.LOG_PERIOD,
    )

    cudnn.benchmark = True

    # # Perform multi-view test on the entire dataset.
    perform_test(test_loader, model, test_meter, cfg)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg):
    model.eval()
    test_meter.iter_tic()

    for cur_step, (inputs, labels, video_idx) in enumerate(test_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda()
        video_idx = video_idx.cuda()

        preds = model(inputs)
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather(
                [preds, labels, video_idx]
            )
        preds = preds.cpu()
        labels = labels.cpu()
        video_idx = video_idx.cpu()
        test_meter.iter_toc()
        test_meter.update_stats(
            preds.detach(), labels.detach(), video_idx.detach()
        )
        test_meter.log_iter_stats(cur_step)
        test_meter.iter_tic()

    test_meter.finalize_metrics()
    test_meter.reset()
