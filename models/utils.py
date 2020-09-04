import torch
import torch.nn as nn


class TFLayerNorm(nn.Module):
    def __init__(
        self,
        dim_channel,
        norm_axes,
        param_axis,
        variance_epsilon
    ):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(TFLayerNorm, self).__init__()
        shape = [
            1 if axis != param_axis else dim_channel
            for axis in norm_axes
        ]
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.norm_axes = norm_axes
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(self.norm_axes, keepdim=True)
        s = (x - u).pow(2).mean(self.norm_axes, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


def get_normalization_name(norm_type):
    if norm_type == "batch_norm":
        name = "bn"
    elif norm_type == "layer_norm":
        name = "ln"
    else:
        raise NotImplementedError(
            "Does not support {} normalization".format(norm_type)
        )
    return name


def get_video_normalization(
    norm_type,
    dim_out,
    eps=1e-5,
    bn_mmt=0.1,
    final_bn=False,
):
    if norm_type.lower() == "batch_norm":
        norm = nn.BatchNorm3d(dim_out, eps=eps, momentum=bn_mmt)
        if final_bn:
            norm.transform_final_bn = True
    elif norm_type.lower() == "layer_norm":
        norm = TFLayerNorm(dim_out, [1, 2, 3, 4], 1, eps)
    else:
        raise NotImplementedError(
            "Does not support {} normalization".format(norm_type)
        )
    return norm
