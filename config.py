# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import argparse
from datetime import datetime
import torch
from pathlib import Path
"""Configs."""
from fvcore.common.config import CfgNode
import warnings

project_dir = str(Path(__file__).resolve().parent)
dataset_root = os.path.join(project_dir, 'datasets')
output_root = os.path.join(project_dir, 'runs')

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()


# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = False

# Dataset.
_C.TRAIN.DATASET = "KineticsSounds"

_C.TRAIN.DATASET_SIZE = 19215

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 32

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 1

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `caffe2` or `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

# If True, perform inflation when loading checkpoint.
_C.TRAIN.CHECKPOINT_INFLATE = False

# ---------------------------------------------------------------------------- #
# Validation options
# ---------------------------------------------------------------------------- #
_C.VAL = CfgNode()

_C.VAL.DATASET = "KineticsSounds"

_C.VAL.DATASET_SIZE = 1316

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = False

# Dataset for testing.
_C.TEST.DATASET = "KineticsSounds"

_C.TEST.DATASET_SIZE = 2679

# Total mini-batch size
_C.TEST.BATCH_SIZE = 2

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.TEST.NUM_SPATIAL_CROPS = 3

# Checkpoint types include `caffe2` or `pytorch`.
_C.TEST.CHECKPOINT_TYPE = "pytorch"


# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]


# -----------------------------------------------------------------------------
# Nonlocal options
# -----------------------------------------------------------------------------
_C.NONLOCAL = CfgNode()

# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"


# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]

# -----------------------------------------------------------------------------
# MODEL options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = "slowfast"

# Model name
_C.MODEL.MODEL_NAME = "SlowFast"

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 32

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = ["c2d", "i3d", "slowonly"]

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

_C.MODEL.HEAD_ACT = "softmax"

# Normalization type
_C.MODEL.NORM_TYPE = "batch_norm"

# Normalization hyperparameter
_C.MODEL.EPSILON = 1e-5

# Normalization hyperparameter
_C.MODEL.MOMENTUM = 0.1


# -----------------------------------------------------------------------------
# SlowFast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The spatial crop size of the input clip.
_C.DATA.CROP_SIZE = 224

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 32

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 2

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]

# List of input frame channel dimensions.
_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256

# Input videos may has different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 30

# Decoding backend, options include `pyav` or `torchvision`
_C.DATA.DECODING_BACKEND = "pyav"

# if True, sample uniformly in [1 / max_scale, 1 / min_scale] and take a
# reciprocal to get the scale. If False, take a uniform sample from
# [min_scale, max_scale].
_C.DATA.INV_UNIFORM_SAMPLE = False

_C.DATA.ENSEMBLE_METHOD = "sum"

_C.DATA.MULTI_LABEL = False

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 196

# Maximal number of steps
_C.SOLVER.NUM_STEPS = 80000

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 34.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Warmup proportion
_C.SOLVER.WARMUP_PROPORTION = 0.06

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Gradient accumulation
_C.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1

# Gradient clipping.
_C.SOLVER.GRADIENT_CLIPPING = False

# Gradient clipping
_C.SOLVER.MAX_GRAD_NORM = 2.0


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use per machine (applies to both training and testing).
_C.NUM_GPUS = -1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Whether to use mixed precision or not
_C.MIXED_PRECISION = False

# Data basedir.
_C.DATASET_ROOT = ""

# Storage SAS Key
_C.STORAGE_SAS_KEY = ""

# Data dir.
_C.DATASET_DIR = ""

# Output basedir.
_C.OUTPUT_ROOT = ""

# Checkpoints dir.
_C.SAVE_DIR = ""

# Log dir.
_C.LOG_DIR = ""

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = -1

# Log period in iters.
_C.LOG_PERIOD = 50

# Save period in iters.
_C.SAVE_PERIOD = 10000

# Save period in epochs.
_C.SAVE_EVERY_EPOCH = 1

# Distributed init method.
_C.DIST_INIT_METHOD = "tcp://localhost:9999"

# Distributed backend.
_C.DIST_BACKEND = "nccl"

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False


def _assert_and_infer_cfg(cfg):
    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]

    # TEST assertions.
    assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TEST.NUM_SPATIAL_CROPS == 3

    # RESNET assertions.
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    config = _C.clone()

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--dist_init_method", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--configuration", type=str, default=None)
    parser.add_argument("--cfg_file", type=str, default=None)
    parser.add_argument("--train_checkpoint_path", type=str, default=None)
    parser.add_argument("--test_checkpoint_path", type=str, default=None)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()
    if args.cfg_file is not None:
        config.merge_from_file(args.cfg_file)
    if args.opts is not None:
        config.merge_from_list(args.opts)

    if config.NUM_GPUS == -1:
        config.NUM_GPUS = torch.cuda.device_count()

    if args.dist_init_method is not None:
        config.DIST_INIT_METHOD = args.dist_init_method

    if args.dataset_root is not None:
        config.DATASET_ROOT = args.dataset_root
    elif not config.DATASET_ROOT:
        config.DATASET_ROOT = dataset_root

    if args.output_root is not None:
        config.OUTPUT_ROOT = args.output_root
    elif not config.OUTPUT_ROOT:
        config.OUTPUT_ROOT = output_root

    if args.configuration is not None:
        configuration = args.configuration
    else:
        configuration = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    model_arch = config.MODEL.ARCH

    config.SAVE_DIR = os.path.join(
        config.OUTPUT_ROOT,
        model_arch,
        configuration,
        "checkpoints"
    )
    if not args.test:
        Path(config.SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # azureml.tensorboard only streams from /logs directory, therefore hardcoded
    config.LOG_DIR = os.path.join(
        config.OUTPUT_ROOT,
        model_arch,
        configuration,
        "logs"
    )
    if not args.test:
        Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)

    if args.train_checkpoint_path is not None:
        config.TRAIN.CHECKPOINT_FILE_PATH = args.train_checkpoint_path
    if args.test_checkpoint_path is not None:
        config.TEST.CHECKPOINT_FILE_PATH = args.test_checkpoint_path

    return _assert_and_infer_cfg(config)
