# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""projection_layer.py"""

from typing import Tuple

from torch import nn

from einops import rearrange


class MultiChannelLinearProjection(nn.Module):

    def __init__(self, in_ch=26, in_dim=128, out_ch=13, out_dim=512):
        super().__init__()
        self.in_ch = in_ch
        self.in_dim = in_dim
        self.out_ch = out_ch
        self.out_dim = out_dim

        self.in_ch_per_group = in_ch // out_ch
        self.linear_in_ch = in_ch // self.in_ch_per_group
        self.linear_in_dim = in_dim * self.in_ch_per_group

        # Reshaped Input shape: (b, t, in_dim//in_ch_per_group, in_dim*in_ch_per_group)
        # Output shape: (b, t, out_ch, out_dim)
        if in_dim * self.in_ch_per_group == out_dim:
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(
                in_features=self.linear_in_dim, out_features=out_dim, bias=False
            )

    def forward(self, x):
        """
        Args:
            x: (B, T, C, D)

        Returns:
            x: (B, C_target, T, D_target)
        """
        x = rearrange(
            x,
            "b t (c1 c2) d -> b c1 t (c2 d)",
            c1=self.linear_in_ch,
            c2=self.in_ch_per_group,
        )
        return self.linear(x)


def get_multi_channel_projection_layer(
    input_shape: Tuple[int], output_shape: Tuple[int], proj_type: str
) -> nn.Module:
    """This function returns one of the projection layers for multi-channel models."""
    in_ch = input_shape[-2]
    in_dim = input_shape[-1]
    out_ch = output_shape[-2]
    out_dim = output_shape[-1]

    if proj_type == "mc_shared_linear":
        return MultiChannelLinearProjection(in_ch, in_dim, out_ch, out_dim)
