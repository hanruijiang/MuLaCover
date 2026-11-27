"""positional_encoding.py"""

from typing import Optional, Literal
import numpy as np
import torch
from torch import nn

from mulacover._symbolic_transcription.yourmt3.model.RoPE.RoPE import RotaryEmbedding


class FixedSinusoidalPositionalEmbedding(nn.Embedding):
    """
    Sinusoidal Absolute Positional Embeddings (APE) of any length.

    Adapted from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding

    """

    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


def get_rotary_emb(
    d_by_head: int,
    freqs_for: Literal["l", "lang", "p", "pixel"],
    partial_pe: bool = False,
    learned_freq: bool = False,
):
    if partial_pe is True:
        rdim = d_by_head // 2
    else:
        rdim = d_by_head

    if freqs_for in ["l", "lang"]:
        freqs_for = "lang"
    elif freqs_for in ["p", "pixel"]:
        freqs_for = "pixel"
    else:
        raise ValueError(
            f"freqs_for must be 'l' or 'lang' or 'p' or 'pixel', but got {freqs_for}"
        )
    return RotaryEmbedding(dim=rdim, freqs_for=freqs_for, learned_freq=learned_freq)
