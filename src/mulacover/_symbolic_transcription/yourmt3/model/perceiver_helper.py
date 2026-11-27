# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
from transformers.utils import ModelOutput
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel



class PerceiverTFConfig(PretrainedConfig):
    """Configuration of the Perceiver-TF transcription encoder."""

    model_type = "perceivertf"

    def __init__(
        self,
        num_latents=26,
        d_latents=128,
        d_model=128,
        kv_dim=128,
        num_blocks=3,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        num_local_transformers_per_block=2,
        num_temporal_transformers_per_block=2,
        qk_channels=128,
        v_channels=128,
        cross_attention_shape_for_attention="q",
        hidden_act="silu",
        dropout_rate=0.1,
        initializer_range=0.02,
        layer_norm_type="layer_norm",
        layer_norm_eps=1e-5,
        sca_use_query_residual=True,
        use_query_residual=True,
        position_encoding_type="rope",
        num_max_positions=330,
        vocab_size=1391,
        attention_to_channel=True,
        ff_layer_type="moe",
        ff_widening_factor=4,
        moe_num_experts=8,
        moe_topk=2,
        rope_type_sca="pixel",
        rope_type_latent="pixel",
        rope_type_temporal="lang",
        rope_apply_to_keys=False,
        rope_partial_pe=True,
        rope_trainable=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_latents = num_latents
        self.d_latents = d_latents
        self.d_model = d_model
        self.kv_dim = kv_dim
        self.qk_channels = qk_channels
        self.v_channels = v_channels

        self.num_blocks = num_blocks
        self.num_self_attention_heads = num_self_attention_heads
        self.num_cross_attention_heads = num_cross_attention_heads
        self.num_local_transformers_per_block = num_local_transformers_per_block
        self.num_temporal_transformers_per_block = num_temporal_transformers_per_block
        self.sca_use_query_residual = sca_use_query_residual
        self.use_query_residual = use_query_residual
        self.position_encoding_type = position_encoding_type
        self.num_max_positions = num_max_positions
        self.cross_attention_shape_for_attention = cross_attention_shape_for_attention
        self.attention_to_channel = attention_to_channel
        self.ff_layer_type = ff_layer_type
        self.ff_widening_factor = ff_widening_factor
        self.moe_num_experts = moe_num_experts
        self.moe_topk = moe_topk
        self.rope_type_sca = rope_type_sca
        self.rope_type_latent = rope_type_latent
        self.rope_type_temporal = rope_type_temporal
        self.rope_apply_to_keys = rope_apply_to_keys
        self.rope_partial_pe = rope_partial_pe
        self.rope_trainable = rope_trainable

        self.hidden_act = hidden_act
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.layer_norm_type = layer_norm_type
        self.layer_norm_eps = layer_norm_eps

        self.vocab_size = vocab_size


class PerceiverTFPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PerceiverTFConfig
    base_model_prefix = "perceivertf"
    main_input_name = "inputs"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif hasattr(module, "latents"):
            module.latents.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.ParameterDict):
            for modality in module.keys():
                module[modality].data.normal_(
                    mean=0.0, std=self.config.initializer_range
                )
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# Replace the 'ModelOutputWithCrossAttentions' with 'MoEModelOutputWithCrossAttentions' for MoE
@dataclass
class MoEModelOutputWithCrossAttentions(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    Plus, router_probs for Mixture of Experts models.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        router_probs (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router probabilities that are computed by MoE routers, these terms are used to compute the auxiliary
            loss and the z_loss for Mixture of Experts models.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
