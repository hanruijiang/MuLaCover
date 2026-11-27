# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""spectrogram.py"""

import importlib
from typing import Optional, Literal, Dict, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from mulacover._symbolic_transcription.yourmt3.model.ops import minmax_normalize, optional_compiler_disable
from mulacover._symbolic_transcription.yourmt3.config.config import (
    audio_cfg as default_audio_cfg,
)

class Spectrogram(nn.Module):

    def __init__(
        self,
        audio_backend: Literal["torchaudio"] = "torchaudio",
        n_fft: int = 2048,
        hop_length: int = 128,
        eps: float = 1e-5,
        **kwargs,
    ):
        """
        Log-Magnitude Spectrogram

        Args:
            audio_backend (str): 'torchaudio' or 'nnaudio'
            n_fft (int): FFT window size, creates n_fft // 2 + 1 freq-bins
            hop_length (int): hop length in samples
            eps (float): epsilon for numerical stability

        """
        super(Spectrogram, self).__init__()
        self.audio_backend = audio_backend.lower()

        if audio_backend.lower() == "torchaudio":
            torchaudio = importlib.import_module("torchaudio")
            self.stft = torchaudio.transforms.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                window_fn=torch.hann_window,
                power=1.0,
            )  # (B, 1, F, T), remove DC component
        else:
            raise NotImplementedError(audio_backend)
        self.eps = eps

    @optional_compiler_disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, T)
        """
        Args:
            x (torch.Tensor): (B, 1, T)

        Returns:
            torch.Tensor: (B, T, F)

        """
        if self.audio_backend == "torchaudio":
            x = self.stft(x)[:, :, 1:, :]  # (B, 1, F, T) remove DC component
            x = rearrange(x, "b 1 f t -> b t f")
            x = minmax_normalize(torch.log(x + self.eps))
            return torch.nan_to_num(
                x
            )  # some versions of torchaudio returns nan when input is all-zeros


def get_spectrogram_layer_from_audio_cfg(
    audio_cfg: Optional[Dict] = None,
) -> Tuple[nn.Module, Tuple[int]]:
    """Get mel-/spectrogram layer from config.
    - Used by 'ymt3' to create a spectrogram layer.
    - Returns output shape of spectrogram layer, which is used to determine input shape of model.

    Args:
        audio_cfg (dict): see config/config.py

    Returns:
        layer (nn.Module): mel-/spectrogram layer
        output_shape (tuple): inferred output shape of layer excluding batch dim. (T, F)
    """
    if audio_cfg is None:
        audio_cfg = default_audio_cfg

    if audio_cfg["codec"] == "spec":
        layer = Spectrogram(**audio_cfg)
    else:
        raise NotImplementedError(audio_cfg["codec"])

    # Infer output shape of the spectrogram layer
    with torch.no_grad():
        output_shape = layer(torch.randn(1, 1, audio_cfg["input_frames"])).shape[1:]
    return layer, output_shape
