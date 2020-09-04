# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import torch
import torch.nn as nn
import math

import utils.weight_init_helper as init_helper

from models import head_helper, resnet_helper, stem_helper
from models.utils import get_normalization_name, get_video_normalization
from models.build import MODEL_REGISTRY

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 34: (3, 4, 6, 3), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slowonly": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
}

_POOL1 = {
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slowonly": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
}


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_type="batch_norm",
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for normalization.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_type (str): normalization type.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.norm_name = get_normalization_name(norm_type)
        self.add_module(
            self.norm_name,
            get_video_normalization(
                norm_type,
                dim_in * fusion_conv_channel_ratio,
                eps,
                bn_mmt,
                False,
            )
        )
        # self.bn = nn.BatchNorm3d(
        #     dim_in * fusion_conv_channel_ratio, eps=eps, momentum=bn_mmt
        # )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = getattr(self, self.norm_name)(fuse)
        # fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


@MODEL_REGISTRY.register()
class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _compute_dim_in(
        self,
        idx,
        trans_func,
        width_per_group,
        out_dim_ratio,
        beta_inv
    ):
        if trans_func == 'basic_transform':
            factor = 1 if idx == 0 else 2 ** (idx - 1)
        elif trans_func == 'bottleneck_transform':
            factor = 1 if idx == 0 else 2 * (2 ** idx)
        else:
            raise NotImplementedError(
                "Does not support {} transfomration".format(trans_func)
            )

        dim_in = [
            width_per_group * factor + width_per_group * factor // out_dim_ratio,
            width_per_group * factor // beta_inv,
        ]
        return dim_in

    def _compute_dim_out(
        self,
        idx,
        trans_func,
        width_per_group,
        beta_inv
    ):
        if trans_func == 'basic_transform':
            factor = 2 ** idx
        elif trans_func == 'bottleneck_transform':
            factor = 4 * (2 ** idx)
        else:
            raise NotImplementedError(
                "Does not support {} transfomration".format(trans_func)
            )

        dim_out = [
            width_per_group * factor,
            width_per_group * factor // beta_inv,
        ]
        return dim_out

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
            norm_type=cfg.MODEL.NORM_TYPE,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
            norm_type=cfg.MODEL.NORM_TYPE,
        )

        dim_in_l = [
            self._compute_dim_in(
                i,
                cfg.RESNET.TRANS_FUNC,
                width_per_group,
                out_dim_ratio,
                cfg.SLOWFAST.BETA_INV
            )
            for i in range(4)
        ]

        dim_out_l = [
            self._compute_dim_out(
                i,
                cfg.RESNET.TRANS_FUNC,
                width_per_group,
                cfg.SLOWFAST.BETA_INV
            )
            for i in range(4)
        ]

        self.s2 = resnet_helper.ResStage(
            dim_in=dim_in_l[0],
            dim_out=dim_out_l[0],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
            norm_type=cfg.MODEL.NORM_TYPE,
        )
        self.s2_fuse = FuseFastToSlow(
            dim_out_l[0][1],
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
            norm_type=cfg.MODEL.NORM_TYPE,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=dim_in_l[1],
            dim_out=dim_out_l[1],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
            norm_type=cfg.MODEL.NORM_TYPE,
        )
        self.s3_fuse = FuseFastToSlow(
            dim_out_l[1][1],
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
            norm_type=cfg.MODEL.NORM_TYPE,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=dim_in_l[2],
            dim_out=dim_out_l[2],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
            norm_type=cfg.MODEL.NORM_TYPE,
        )
        self.s4_fuse = FuseFastToSlow(
            dim_out_l[2][1],
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
            norm_type=cfg.MODEL.NORM_TYPE,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=dim_in_l[3],
            dim_out=dim_out_l[3],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
            norm_type=cfg.MODEL.NORM_TYPE,
        )

        self.head = head_helper.ResNetBasicHead(
            dim_in=dim_out_l[3],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=[
                [
                    cfg.DATA.NUM_FRAMES
                    // cfg.SLOWFAST.ALPHA
                    // pool_size[0][0],
                    math.ceil(cfg.DATA.CROP_SIZE / 32) // pool_size[0][1],
                    math.ceil(cfg.DATA.CROP_SIZE / 32) // pool_size[0][2],
                ],
                [
                    cfg.DATA.NUM_FRAMES // pool_size[1][0],
                    math.ceil(cfg.DATA.CROP_SIZE / 32) // pool_size[1][1],
                    math.ceil(cfg.DATA.CROP_SIZE / 32) // pool_size[1][2],
                ],
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )

    def forward(self, x):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        x = self.head(x)
        return x


@MODEL_REGISTRY.register()
class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, SlowOnly).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _compute_dim_in(
        self,
        idx,
        trans_func,
        width_per_group,
    ):
        if trans_func == 'basic_transform':
            factor = 1 if idx == 0 else 2 ** (idx - 1)
        elif trans_func == 'bottleneck_transform':
            factor = 1 if idx == 0 else 2 * (2 ** idx)
        else:
            raise NotImplementedError(
                "Does not support {} transfomration".format(trans_func)
            )

        dim_in = [width_per_group * factor]
        return dim_in

    def _compute_dim_out(
        self,
        idx,
        trans_func,
        width_per_group,
    ):
        if trans_func == 'basic_transform':
            factor = 2 ** idx
        elif trans_func == 'bottleneck_transform':
            factor = 4 * (2 ** idx)
        else:
            raise NotImplementedError(
                "Does not support {} transfomration".format(trans_func)
            )

        dim_out = [width_per_group * factor]
        return dim_out

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
            norm_type=cfg.MODEL.NORM_TYPE,
        )

        dim_in_l = [
            self._compute_dim_in(
                i,
                cfg.RESNET.TRANS_FUNC,
                width_per_group
            )
            for i in range(4)
        ]

        dim_out_l = [
            self._compute_dim_out(
                i,
                cfg.RESNET.TRANS_FUNC,
                width_per_group
            )
            for i in range(4)
        ]

        self.s2 = resnet_helper.ResStage(
            dim_in=dim_in_l[0],
            dim_out=dim_out_l[0],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
            norm_type=cfg.MODEL.NORM_TYPE,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=dim_in_l[1],
            dim_out=dim_out_l[1],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
            norm_type=cfg.MODEL.NORM_TYPE,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=dim_in_l[2],
            dim_out=dim_out_l[2],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
            norm_type=cfg.MODEL.NORM_TYPE,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=dim_in_l[3],
            dim_out=dim_out_l[3],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
            norm_type=cfg.MODEL.NORM_TYPE,
        )

        self.head = head_helper.ResNetBasicHead(
            dim_in=dim_out_l[3],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=[
                [
                    cfg.DATA.NUM_FRAMES // pool_size[0][0],
                    math.ceil(cfg.DATA.CROP_SIZE / 32) // pool_size[0][1],
                    math.ceil(cfg.DATA.CROP_SIZE / 32) // pool_size[0][2],
                ]
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.head(x)
        return x
