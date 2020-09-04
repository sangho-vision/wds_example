import logging
import numpy as np
import time
import torch
import random

import data.transform as transform

logger = logging.getLogger(__name__)


def get_video(
    cfg,
    frames,
    temporal_sample_index,
    spatial_sample_index,
    num_clips,
    video_fps,
    min_scale,
    max_scale,
    crop_size,
):
    _num_frames = (
        cfg.DATA.NUM_FRAMES
        * cfg.DATA.SAMPLING_RATE
        * video_fps
        / cfg.DATA.TARGET_FPS
    )
    delta = max(frames.shape[0] - _num_frames, 0)
    if temporal_sample_index == -1:
        # Random temporal sampling
        start_idx = random.uniform(0, delta)
    else:
        start_idx = int(delta * temporal_sample_index / (num_clips - 1))
    end_idx = start_idx + _num_frames - 1
    view = temporal_sampling(
        frames,
        start_idx,
        end_idx,
        cfg.DATA.NUM_FRAMES,
    )

    # Perform color normalization.
    view = view.float()
    view = view / 255.0
    view = view - torch.tensor(cfg.DATA.MEAN)
    view = view / torch.tensor(cfg.DATA.STD)

    # T H W C -> C T H W
    view = view.permute(3, 0, 1, 2)
    crop = spatial_sampling(
        view,
        spatial_sample_index,
        min_scale,
        max_scale,
        crop_size,
        cfg.DATA.INV_UNIFORM_SAMPLE
    )

    return crop


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    inverse_uniform_sampling=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale, max_scale].
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        frames = transform.random_short_side_scale_jitter(
            images=frames,
            min_size=min_scale,
            max_size=max_scale,
            inverse_uniform_sampling=inverse_uniform_sampling,
        )
        frames = transform.random_crop(frames, crop_size)
        frames = transform.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames = transform.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames = transform.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def as_binary_vector(labels, num_classes):
    """
    Construct binary label vector given a list of label indices.
    Args:
        labels (list): The input label list.
        num_classes (int): Number of classes of the label vector.
    Returns:
        labels (numpy array): the resulting binary vector.
    """
    label_arr = np.zeros((num_classes,))

    for lbl in set(labels):
        label_arr[lbl] = 1.0
    return label_arr


def pack_pathway_output(cfg, frames, dim=1):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frame_list = [frames]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            dim,
            torch.linspace(
                0, frames.shape[dim] - 1, frames.shape[dim] // cfg.SLOWFAST.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Visual model arch {} is not in {}".format(
                cfg.MODEL.ARCH,
                cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frame_list
