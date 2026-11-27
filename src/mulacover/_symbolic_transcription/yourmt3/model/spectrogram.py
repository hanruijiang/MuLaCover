# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""Log-magnitude spectrogram frontend."""

import torch
import torch.nn as nn
import torchaudio
from einops import rearrange
from .ops import minmax_normalize


class Spectrogram(nn.Module):
    def __init__(self, n_fft=2048, hop_length=300, eps=1e-5):
        super().__init__()
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, window_fn=torch.hann_window, power=1.0
        )
        self.eps = eps

    def forward(self, x):
        x = self.stft(x)[:, :, 1:, :]
        x = rearrange(x, "b 1 f t -> b t f")
        x = minmax_normalize(torch.log(x + self.eps))
        return torch.nan_to_num(x)


def get_spectrogram_layer_from_audio_cfg(audio_cfg):
    layer = Spectrogram(n_fft=audio_cfg["n_fft"], hop_length=audio_cfg["hop_length"])
    with torch.no_grad():
        output_shape = layer(torch.randn(1, 1, audio_cfg["input_frames"])).shape[1:]
    return layer, output_shape
