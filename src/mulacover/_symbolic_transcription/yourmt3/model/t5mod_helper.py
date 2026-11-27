# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""t5mod_helper.py"""

import torch
from torch import nn
from mulacover._symbolic_transcription.yourmt3.model.t5mod import (
    MultiChannelT5Decoder,
)
from typing import Optional


@torch.no_grad()
def task_cond_dec_generate(
    decoder: MultiChannelT5Decoder,
    embed_tokens: nn.Embedding,
    lm_head: nn.Module,
    encoder_hidden_states: torch.FloatTensor,
    max_length: int = 1024,
    stop_at_eos: bool = True,
    eos_id: Optional[int] = 1,
    pad_id: Optional[int] = 0,
    decoder_start_token_id: Optional[int] = 0,
) -> torch.LongTensor:
    """Generate event IDs with the multi-channel decoder and KV cache."""
    bsz = encoder_hidden_states.shape[0]
    device = encoder_hidden_states.device

    dec_input_shape = (bsz, decoder.num_channels, 1)
    dec_input_ids = torch.tile(
        torch.LongTensor([decoder_start_token_id]).to(device), dec_input_shape
    )
    dec_inputs_embeds = embed_tokens(dec_input_ids)  # (B, L, D) or (B, C, L, D)

    dec_hs, past_key_values = decoder(
        inputs_embeds=dec_inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        return_dict=False,
    )
    logits = lm_head(dec_hs)  # (b, T=1, vocab_size) or (b, C, T=1, vocab_size)
    pred_ids = logits.argmax(-1)  # (B, C, 1)

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(dec_input_shape, dtype=torch.long, device=device)

    # Fast generation with past_key_values for the rest of the sequence
    dec_input_ids = pred_ids[:, :, -1].unsqueeze(-1)  # (B, C, 1)
    for _ in range(max_length - 1):

        # when past_key_values is provided, we use only the last token as input_ids
        dec_inputs_embeds = embed_tokens(dec_input_ids)  # (B, 1, D) or (B, C, 1, D)
        dec_hs, _past_key_values = decoder(
            inputs_embeds=dec_inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            return_dict=False,
        )
        logits = lm_head(dec_hs)  # (b, 1, vocab_size) or (b, K, 1, vocab_size)
        _pred_ids = logits.argmax(-1)  # (B, 1) or (B, K, 1)

        # update input_ids and past_key_values for next iteration
        dec_input_ids = (
            _pred_ids.clone()
        )  # (B, 1) or (B, C, 1), deepcopy of _pred_ids because _pred_ids will be modified for finished sentences
        past_key_values = _past_key_values

        # finished sentences should have their next token be a padding token
        if eos_id is not None:
            if pad_id is None:
                raise ValueError(
                    "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                )
            _pred_ids = _pred_ids * unfinished_sequences + pad_id * (
                1 - unfinished_sequences
            )

        # update pred_ids
        pred_ids = torch.cat(
            (pred_ids, _pred_ids), dim=-1
        )  # (B, T') or (B, C, T') with increasing T'

        # update state of unfinished_sequences
        if eos_id is not None:
            unfinished_sequences = unfinished_sequences * _pred_ids.ne(eos_id).long()

            # early-stop when each sentence is finished
            if stop_at_eos is True and unfinished_sequences.max() == 0:
                break

    return pred_ids  # (B, L) or (B, C, L)
