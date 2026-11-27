# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""Sparse mixture-of-experts feedforward layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.activations import ACT2FN
from mulacover._symbolic_transcription.yourmt3.model.ops import (
    optional_compiler_disable,
)


class MixtralBlockSparseTop2MLP(nn.Module):
    """
    The Gated Multilayer Perceptron (GMLP) used in Mixtral of Experts (MoE).

    """

    def __init__(self, config: PretrainedConfig, input_size: int, widening_factor: int):
        super().__init__()
        self.hidden_dim = input_size
        self.ffn_dim = int(input_size * widening_factor)

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.gate = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.gate(
            hidden_states
        )
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config, input_size: int, widening_factor: int):
        super().__init__()
        self.hidden_dim = input_size
        self.widening_factor = widening_factor
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_topk

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                MixtralBlockSparseTop2MLP(config, self.hidden_dim, self.widening_factor)
                for _ in range(self.num_experts)
            ]
        )

    @optional_compiler_disable
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state)
                * routing_weights[top_x_list, idx_list, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits


def get_ff_layer(config: PretrainedConfig, input_size: int, widening_factor: int):
    return MixtralSparseMoeBlock(config, input_size, widening_factor)
