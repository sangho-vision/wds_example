import os
import random
import math
import json
import torch
import torch.utils.data
import torchvision
from functools import partial
from pathlib import Path

import tempfile
import webdataset as wds

import utils.logging as logging
from utils import distributed as du

import data.utils as utils
from data.build import DATASET_REGISTRY
from data.collate import COLLATE_FN


logger = logging.get_logger(__name__)


class Decoder(object):
    def __init__(self, cfg, dataset_name, mode):
        self.cfg = cfg
        # Only support pretrain mode.
        assert mode in [
            "train",
            "val",
            "test"
        ], "Split '{}' not supported for Kinetics".format(mode)

        self.mode = mode
        self.dataset_name = dataset_name
        self.cfg = cfg
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )
        logger.info(f"Constructing {dataset_name} mode {self.mode}")
        self._construct_loader()

    def _construct_loader(self):
        dir_to_files = Path(
            os.path.join(
                self.cfg.DATASET_ROOT,
                self.dataset_name
            )
        )
        path_to_file = dir_to_files.joinpath(f"{self.mode}.json")
        with open(path_to_file, "r") as f:
            dataset = json.load(f)
        self.idx2yid = sorted(list(dataset.keys()))
        self.yid2idx = {yid: idx for idx, yid in enumerate(self.idx2yid)}
        self.idx2class = sorted(
            {
                dataset[yid]['annotations']['label']
                for yid in self.idx2yid
            }
        )
        self.class2idx = {c: idx for idx, c in enumerate(self.idx2class)}

        assert (
            len(self.idx2yid) > 0
        ), "Failed to load {} mode {}".format(
            self.dataset_name, self.mode
        )
        logger.info(
            "Constructing {} dataloader (mode: {}, size: {})".format(
                self.dataset_name, self.mode, len(self.idx2yid)
            )
        )

    def jsondecode(self, data):
        anno = json.loads(data)
        label = anno['annotations']['label']
        yid = anno['url'][-11:]
        labels = torch.tensor([self.class2idx[label]] * self._num_clips, dtype=torch.long)
        indices = torch.tensor(
            list(range(self._num_clips * self.yid2idx[yid], self._num_clips * (self.yid2idx[yid] + 1))),
            dtype=torch.long,
        )
        return labels, indices

    def mp4decode(self, data):
        with tempfile.TemporaryDirectory() as dname:
            with open(dname+"/sample.mp4", "wb") as stream:
                stream.write(data)
            frames, waveform, info = \
                torchvision.io.read_video(
                    dname+"/sample.mp4",
                    pts_unit="sec",
                )

        video_fps = round(info["video_fps"])
        if self.mode in ["train", "val"]:
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            visual_clip = utils.get_video(
                self.cfg,
                frames,
                temporal_sample_index,
                spatial_sample_index,
                1,
                video_fps,
                min_scale,
                max_scale,
                crop_size
            )
            visual_input = utils.pack_pathway_output(
                self.cfg,
                torch.unsqueeze(visual_clip, dim=0),
                dim=2,
            )
        elif self.mode in ["test"]:
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            assert len({min_scale, max_scale, crop_size}) == 1
            clips_lst = []
            for temporal_sample_index in range(self.cfg.TEST.NUM_ENSEMBLE_VIEWS):
                for spatial_sample_index in range(self.cfg.TEST.NUM_SPATIAL_CROPS):
                    visual_clip = utils.get_video(
                        self.cfg,
                        frames,
                        temporal_sample_index,
                        spatial_sample_index,
                        self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                        video_fps,
                        min_scale,
                        max_scale,
                        crop_size,
                    )
                    clips_lst.append(visual_clip)
            visual_input = torch.stack(clips_lst, dim=0)
            visual_input = utils.pack_pathway_output(
                self.cfg,
                visual_input,
                dim=2,
            )
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        return visual_input


@DATASET_REGISTRY.register()
def KineticsSounds(cfg, split):
    if split == 'train':
        max_idx = 19
    elif split == 'val':
        max_idx = 1
    elif split == 'test':
        max_idx = 2
    dataset_root = cfg.DATASET_ROOT
    if dataset_root.endswith('/'):
        dataset_root = dataset_root[:-1]
    url = f"{dataset_root}/KineticsSounds/{split}/shard-{{000000..{max_idx:06d}}}.tar"
    if cfg.STORAGE_SAS_KEY:
        url += cfg.STORAGE_SAS_KEY

    _decoder = Decoder(cfg, "KineticsSounds", split)
    if split == 'train':
        batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS)
        batch_size = int(batch_size / du.get_world_size())
        length = int(cfg.TRAIN.DATASET_SIZE / du.get_world_size())
        nominal = int(length / batch_size)
    elif split == 'val':
        batch_size = int(cfg.TRAIN.BATCH_SIZE / du.get_world_size())
        length = int(cfg.VAL.DATASET_SIZE / du.get_world_size())
        nominal = int(length / batch_size)
    elif split == 'test':
        batch_size = int(cfg.TEST.BATCH_SIZE / du.get_world_size())
        length = math.ceil(cfg.TEST.DATASET_SIZE / du.get_world_size())
        nominal = math.ceil(length / batch_size)

    wds.filters.batched = wds.filters.Curried(
        partial(wds.filters.batched_, collation_fn=COLLATE_FN["kinetics"])
    )

    dataset = wds.Dataset(
        url,
        handler=wds.warn_and_continue,
        shard_selection=du.shard_selection,
        length=length,
    )
    if split == 'train':
        dataset = dataset.shuffle(100)
    dataset = (
        dataset.map_dict(
            handler=wds.warn_and_continue,
            mp4=_decoder.mp4decode,
            json=_decoder.jsondecode,
        )
    )
    if cfg.DATA_LOADER.NUM_WORKERS > 0:
        length = nominal
    dataset = wds.ResizedDataset(
        dataset,
        length=length,
        nominal=nominal,
    )
    return dataset
