# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""task.py"""

from .vocabulary import program_vocab_presets, drum_vocab_presets

task_cfg = {
    "mc13_full_plus_256": {  # multi-channel decoding task of {34 classes + drums + singing & chorus} mapped to 13 channels
        "name": "mc13_full_plus_256",
        "train_program_vocab": program_vocab_presets["mt3_full_plus"],
        "train_drum_vocab": drum_vocab_presets["gm"],
        "program2channel_vocab_source": program_vocab_presets["gm_plus"],
        "num_decoding_channels": 13,
        "max_note_token_length_per_ch": 256,  # multi-channel decoding exclusive parameter
        "mask_loss_strategy": None,  # multi-channel decoding exclusive parameter
    },
}
