# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""Event vocabulary for transcription token decoding."""

from typing import List, Dict, Optional
from mulacover._symbolic_transcription.yourmt3.utils.note_event_dataclasses import (
    Event,
    EventRange,
)

class FastCodec:
    """Map transcription token IDs to events."""

    def __init__(
        self,
        special_tokens: List[str],
        max_shift_steps: int,
        event_ranges: List[EventRange],
        program_vocabulary: Optional[Dict] = None,
        drum_vocabulary: Optional[Dict] = None,
        extra_tokens: List[str] = [],
        name: Optional[str] = None,
    ):
        # Store the special tokens and event ranges.
        self.special_tokens = special_tokens
        self._special_token_ranges = []
        self._extra_token_ranges = []

        for token in special_tokens:
            self._special_token_ranges.append(EventRange(token, 0, 0))
        for token in extra_tokens:
            self._extra_token_ranges.append(EventRange(token, 0, 0))
        self._shift_range = EventRange(
            type="shift", min_value=0, max_value=max_shift_steps - 1
        )
        self._event_ranges = (
            self._special_token_ranges
            + [self._shift_range]
            + event_ranges
            + self._extra_token_ranges
        )
        # Ensure all event types have unique names.
        assert len(self._event_ranges) == len(
            set([er.type for er in self._event_ranges])
        )

        # Store the name of the codec, so that we can identify it in tokenizer.
        self._name = name

        # Create dictionary for decoding
        self._decode_dict = {}
        idx = 0
        for er in self._event_ranges:
            for value in range(er.min_value, er.max_value + 1):
                self._decode_dict[idx] = Event(type=er.type, value=value)
                idx += 1

        self._num_classes = idx

        # Create inverse vocabulary for instrument groups
        if program_vocabulary is not None:
            self.inverse_vocab_program = {}
            self._create_inverse_vocab_program(program_vocabulary)
        else:
            self.inverse_vocab_program = None

        # Create inverse vocabulary for drum mapping
        if drum_vocabulary is not None:
            self.inverse_vocab_drum = {}
            self._create_inverse_vocab_drum(drum_vocabulary)
        else:
            self.inverse_vocab_drum = None

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def _create_inverse_vocab_program(self, vocab):
        for key, values in vocab.items():
            for value in values:
                self.inverse_vocab_program[value] = values[0]

    def _create_inverse_vocab_drum(self, vocab):
        for key, values in vocab.items():
            for value in values:
                self.inverse_vocab_drum[value] = values[0]

    def decode_event_index(self, index: int) -> Event:
        """Decode an event index to an Event."""
        if index < 0 or index >= self.num_classes:
            raise ValueError(f"Unknown event index: {index}")
        decoded_event = self._decode_dict[index]

        # Create a new event with the same type and value
        return Event(type=decoded_event.type, value=decoded_event.value)

