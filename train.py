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


def calculate_and_update_precise_bn(loader, model, num_iters=100):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader():
        for inputs, labels, video_idx in loader:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            elif isinstance(inputs, (dict, )):
                for key in inputs.keys():
                    if isinstance(inputs[key], (list, )):
                        for i in range(len(inputs[key])):
                            inputs[key][i] = inputs[key][i].cuda(non_blocking=True)
                    else:
                        inputs[key] = inputs[key].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def save_checkpoint(state, is_best=False, filename='checkpoint.pyth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pyth')


def train(cfg):
    """
    Train function.
    Args:
        cfg (CfgNode) : configs. Details can be found in
            config.py
    """
    # Set random seed from configs.
    if cfg.RNG_SEED != -1:
        random.seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.NUM_GPUS, os.path.join(cfg.LOG_DIR, "log.txt"))

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Model for training.
    model = build_model(cfg)
    # Construct te optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Print model statistics.
    if du.is_master_proc(cfg.NUM_GPUS):
        misc.log_model_info(model, cfg, use_train_input=True)

    # Create dataloaders.
    train_loader = loader.construct_loader(cfg, 'train')
    val_loader = loader.construct_loader(cfg, 'val')

    if cfg.SOLVER.MAX_EPOCH != -1:
        max_epoch = cfg.SOLVER.MAX_EPOCH * cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS
        num_steps = max_epoch * len(train_loader)
        cfg.SOLVER.NUM_STEPS = cfg.SOLVER.MAX_EPOCH * len(train_loader)
        cfg.SOLVER.WARMUP_PROPORTION = cfg.SOLVER.WARMUP_EPOCHS / cfg.SOLVER.MAX_EPOCH
    else:
        num_steps = cfg.SOLVER.NUM_STEPS * cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS
        max_epoch = math.ceil(num_steps / len(train_loader))
        cfg.SOLVER.MAX_EPOCH = cfg.SOLVER.NUM_STEPS / len(train_loader)
        cfg.SOLVER.WARMUP_EPOCHS = cfg.SOLVER.MAX_EPOCH * cfg.SOLVER.WARMUP_PROPORTION

    start_epoch = 0
    global_step = 0
    if cfg.TRAIN.CHECKPOINT_FILE_PATH:
        if os.path.isfile(cfg.TRAIN.CHECKPOINT_FILE_PATH):
            logger.info(
                "=> loading checkpoint '{}'".format(
                    cfg.TRAIN.CHECKPOINT_FILE_PATH
                )
            )
            ms = model.module if cfg.NUM_GPUS > 1 else model
            # Load the checkpoint on CPU to avoid GPU mem spike.
            checkpoint = torch.load(
                cfg.TRAIN.CHECKPOINT_FILE_PATH, map_location='cpu'
            )
            start_epoch = checkpoint['epoch']
            ms.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            global_step = checkpoint['epoch'] * len(train_loader)
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    cfg.TRAIN.CHECKPOINT_FILE_PATH,
                    checkpoint['epoch']
                )
            )
    else:
        logger.info("Training with random initialization.")

    # Create meters.
    train_meter = TrainMeter(
        len(train_loader),
        num_steps,
        max_epoch,
        cfg
    )
    val_meter = ValMeter(
        len(val_loader),
        max_epoch,
        cfg
    )

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch+1))

    cudnn.benchmark = True

    best_epoch, best_top1_err, top5_err = 0, 100.0, 100.0

    for cur_epoch in range(start_epoch, max_epoch):
        is_best_epoch = False
        # Shuffle the dataset.
        # loader.shuffle_dataset(train_loader, cur_epoch)
        # Pretrain for one epoch.
        global_step = train_epoch(
            train_loader,
            model,
            optimizer,
            train_meter,
            cur_epoch,
            global_step,
            num_steps,
            cfg
        )

        if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            calculate_and_update_precise_bn(
                train_loader, model, cfg.BN.NUM_BATCHES_PRECISE
            )

        if misc.is_eval_epoch(cfg, cur_epoch, max_epoch):
            stats = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)
            if best_top1_err > float(stats["top1_err"]):
                best_epoch = cur_epoch + 1
                best_top1_err = float(stats["top1_err"])
                top5_err = float(stats["top5_err"])
                is_best_epoch = True
            logger.info("BEST: Epoch: {}, Top1_err: {:.2f}, Top5_err: {:.2f}".format(best_epoch, best_top1_err, top5_err))

        if (cur_epoch + 1) % cfg.SAVE_EVERY_EPOCH == 0 and du.get_rank() == 0:
            sd = \
                model.module.state_dict() if cfg.NUM_GPUS > 1 else \
                model.state_dict()
            save_checkpoint(
                {
                    'epoch': cur_epoch + 1,
                    'model_arch': cfg.MODEL.DOWNSTREAM_ARCH,
                    'state_dict': sd,
                    'optimizer': optimizer.state_dict(),
                },
                filename=os.path.join(cfg.SAVE_DIR, f'epoch{cur_epoch+1}.pyth')
            )

def train_epoch(
    train_loader,
    model,
    optimizer,
    train_meter,
    cur_epoch,
    global_step,
    num_steps,
    cfg
):
    model.train()
    train_meter.iter_tic()

    data_size = len(train_loader) / cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS

    epoch_step = 0
    _global_step = global_step // cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS
    lr = optim.get_epoch_lr(cur_epoch + float(epoch_step) / data_size, _global_step, cfg)
    for cur_step, (inputs, labels, _) in enumerate(train_loader):
        global_step += 1
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda()

        preds = model(inputs)

        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction='mean')

        loss = loss_fun(preds, labels)

        if cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS

        # Check Nan Loss.
        misc.check_nan_losses(loss)

        loss.backward()

        if global_step % cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS == 0:
            epoch_step += 1
            _global_step = global_step // cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS
            lr = optim.get_epoch_lr(cur_epoch + float(epoch_step) / data_size, _global_step, cfg)
            optim.set_lr(optimizer, lr)
            if cfg.SOLVER.GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.SOLVER.MAX_GRAD_NORM
                )
            optimizer.step()
            optimizer.zero_grad()

        if cfg.DATA.MULTI_LABEL:
            if cfg.NUM_GPUS > 1:
                [loss] = du.all_reduce([loss])

            loss = loss.item()
            top1_err, top5_err = None, None
        else:
            top1_err, top5_err = metrics.topk_errors(preds, labels, (1, 5))

            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

            loss, top1_err, top5_err = (
                loss.item(),
                top1_err.item(),
                top5_err.item(),
            )

        train_meter.iter_toc()
        # Update and log stats.
        train_meter.update_stats(
            top1_err,
            top5_err,
            loss,
            lr,
            labels.size(0) * cfg.NUM_GPUS
        )

        train_meter.log_iter_stats(cur_epoch, cur_step, global_step)

        if global_step == num_steps and (cur_step + 1) != len(train_loader):
            return global_step

        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

    return global_step


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    model.eval()
    val_meter.iter_tic()

    for cur_step, (inputs, labels, _) in enumerate(val_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda()

        preds = model(inputs)
        if cfg.DATA.MULTI_LABEL:
            if cfg.NUM_GPUS > 1:
                preds, labels = du.all_gather([preds, labels])
            val_meter.iter_toc()
            val_meter.update_predictions(preds, labels)
        else:
            top1_err, top5_err = metrics.topk_errors(preds, labels, (1, 5))
            if cfg.NUM_GPUS > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])
            top1_err, top5_err = top1_err.item(), top5_err.item()

            val_meter.iter_toc()
            val_meter.update_stats(
                top1_err, top5_err, labels.size(0) * cfg.NUM_GPUS
            )
        val_meter.log_iter_stats(cur_epoch, cur_step)
        val_meter.iter_tic()

    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


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
