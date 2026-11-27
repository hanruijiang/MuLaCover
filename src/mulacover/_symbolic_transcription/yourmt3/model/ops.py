# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""op.py"""


import torch
import torch.nn as nn
from einops import rearrange
from .t5_compat import T5LayerNorm as RMSNorm


def get_layer_norm(
    dim: int, layer_norm_type: str = "layer_norm", layer_norm_eps: float = 1e-5
):
    """Get layer normalization layer.
    Args:
        dim (int): Feature dimension
        layer_norm_type (str): "layer_norm" or "rms_norm"
        layer_norm_eps (float): Epsilon value for numerical stability

    Returns:
        nn.Module: Layer normalization layer
    """
    if layer_norm_type == "rms_norm":
        # T5LayerNorm is equivalent to RMSNorm. https://arxiv.org/abs/1910.07467
        return RMSNorm(hidden_size=dim, eps=layer_norm_eps)
    else:
        return nn.LayerNorm(normalized_shape=dim, eps=layer_norm_eps)


def minmax_normalize(x: torch.Tensor, eps: float = 0.008) -> torch.FloatTensor:
    """Min-max normalization:

    x_norm = (x - x_min) / (x_max - x_min + eps)

    Args:
        x (torch.Tensor): (B, T, F)
    Returns:
        torch.Tensor: (B, T, F) with output range of [0, 1]
    """
    x_max = rearrange(x, "b t f -> b (t f)").max(1, keepdim=True)[0]
    x_min = rearrange(x, "b t f -> b (f t)").min(1, keepdim=True)[0]
    x_max = x_max[:, None, :]  # (B,1,1)
    x_min = x_min[:, None, :]  # (B,1,1)
    return (x - x_min) / (x_max - x_min + eps)


def optional_compiler_disable(func):
    return torch.compiler.disable(func)
