# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def init_layer(layer: nn.Module) -> None:
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.zero_()


def init_bn(bn: nn.Module) -> None:
    """Initialize a Batchnorm layer."""
    bn.bias.data.zero_()
    bn.weight.data.fill_(1.0)
    bn.running_mean.data.zero_()
    bn.running_var.data.fill_(1.0)


def act(x: torch.Tensor, activation: str) -> torch.Tensor:
    """Activation function."""
    funcs = {
        "relu": F.relu_,
        "leaky_relu": lambda x: F.leaky_relu_(x, 0.01),
        "swish": lambda x: x * torch.sigmoid(x),
    }
    return funcs.get(activation, lambda x: Exception("Incorrect activation!"))(x)


class Res2DAVPBlock(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size, avp_kernel_size, activation
    ):
        """Convolutional residual block adapted from bytedance/music_source_separation."""
        super().__init__()

        padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.activation = activation
        self.bn1, self.bn2 = nn.BatchNorm2d(out_channels), nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, padding=padding, bias=False
        )

        self.is_shortcut = in_channels != out_channels
        if self.is_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

        self.avp = nn.AvgPool2d(avp_kernel_size)
        self.init_weights()

    def init_weights(self):
        for m in [self.conv1, self.conv2] + (
            [self.shortcut] if self.is_shortcut else []
        ):
            init_layer(m)
        for m in [self.bn1, self.bn2]:
            init_bn(m)

    def forward(self, x):
        origin = x
        x = act(self.bn1(self.conv1(x)), self.activation)
        x = self.bn2(self.conv2(x))
        x += self.shortcut(origin) if self.is_shortcut else origin
        x = act(x, self.activation)
        return self.avp(x)


class PreEncoderBlockRes3B(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        avp_kernerl_size=(1, 2),
        activation="relu",
    ):
        """Pre-Encoder with 3 Res2DAVPBlocks."""
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Res2DAVPBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size,
                    avp_kernerl_size,
                    activation,
                )
                for i in range(3)
            ]
        )

    def forward(self, x):  # (B, T, F)
        x = rearrange(x, "b t f -> b 1 t f")
        for block in self.blocks:
            x = block(x)
        return rearrange(x, "b c t f -> b t f c")


# ====================================================================================================================
# PreEncoderBlockHFTT: hFT-Transformer-like Pre-encoder
# ====================================================================================================================
