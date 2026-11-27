# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""Token decoding for the 13-channel transcription model."""

from ..config.task import task_cfg
from .tokenizer import NoteEventTokenizer


class TaskManager:
    def __init__(self):
        task = task_cfg["mc13_full_plus_256"]
        self.num_decoding_channels = task["num_decoding_channels"]
        self.max_total_token_length = task["max_note_token_length_per_ch"]
        self.tokenizer = NoteEventTokenizer(
            program_vocabulary=task["program_vocab"],
            drum_vocabulary=task["drum_vocab"],
        )
        self.num_tokens = self.tokenizer.num_tokens

    def detokenize_list_batches(self, list_batch_tokens, list_start_times, return_events=True):
        return self.tokenizer.decode_list_batches(
            list_batch_tokens, list_start_times, return_events
        )
