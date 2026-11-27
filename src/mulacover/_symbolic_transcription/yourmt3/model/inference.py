# Copyright 2024 The YourMT3 Authors.
# Licensed under the Apache License, Version 2.0; see ../LICENSE-APACHE-2.0.
"""YourMT3 Perceiver MoE encoder and multi-channel T5 decoder."""

import torch
from torch import nn
from transformers import T5Config

from ..config.config import T5_BASE_CFG, audio_cfg, encoder_cfg, decoder_cfg
from ..utils.task_manager import TaskManager
from .conv_block import PreEncoderBlockRes3B
from .lm_head import LMHead
from .perceiver_helper import PerceiverTFConfig
from .perceiver_mod import PerceiverTFEncoder
from .projection_layer import get_multi_channel_projection_layer
from .spectrogram import get_spectrogram_layer_from_audio_cfg
from .t5mod import MultiChannelT5Decoder
from .t5mod_helper import task_cond_dec_generate


class YourMT3(nn.Module):
    """The fixed Perceiver MoE encoder and 13-channel T5 decoder."""

    def __init__(self):
        super().__init__()
        self.audio_cfg = dict(audio_cfg)
        self.task_manager = TaskManager()
        self.max_total_token_length = self.task_manager.max_total_token_length

        self.spectrogram, spec_shape = get_spectrogram_layer_from_audio_cfg(
            self.audio_cfg
        )
        self.pre_encoder = nn.Sequential(
            PreEncoderBlockRes3B(
                1, 128, kernel_size=(3, 3), avp_kernerl_size=(1, 2), activation="relu"
            )
        )
        encoder_config = dict(
            encoder_cfg,
            num_max_positions=spec_shape[0],
            vocab_size=self.task_manager.num_tokens,
        )
        decoder_config = dict(
            decoder_cfg,
            num_max_positions=max(spec_shape[0], 1024) + 10,
            vocab_size=self.task_manager.num_tokens,
        )
        self.pre_decoder = nn.Sequential(
            get_multi_channel_projection_layer(
                input_shape=(encoder_config["num_latents"], encoder_config["d_model"]),
                output_shape=(
                    decoder_config["num_channels"],
                    decoder_config["d_model"],
                ),
                proj_type="mc_shared_linear",
            )
        )
        self.lm_head = LMHead(decoder_config, 1.0, True)
        self.embed_tokens = nn.Embedding(
            decoder_config["vocab_size"], decoder_config["d_model"]
        )
        perceiver_config = PerceiverTFConfig()
        perceiver_config.update(encoder_config)
        self.encoder = PerceiverTFEncoder(perceiver_config)
        t5_config = T5Config(**T5_BASE_CFG)
        self.decoder = MultiChannelT5Decoder(decoder_config, t5_config)

    @torch.inference_mode()
    def inference(self, audio: torch.Tensor) -> torch.Tensor:
        """Decode a batch of mono 32,767-sample segments into event IDs."""
        features = self.pre_encoder(self.spectrogram(audio))
        hidden_states = self.encoder(inputs_embeds=features)["last_hidden_state"]
        hidden_states = self.pre_decoder(hidden_states)
        return task_cond_dec_generate(
            decoder=self.decoder,
            embed_tokens=self.embed_tokens,
            lm_head=self.lm_head,
            encoder_hidden_states=hidden_states,
            max_length=self.max_total_token_length,
        )
