# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
from dataclasses import dataclass, field
from typing import Set, Optional

@dataclass
class Note:
    is_drum: bool
    program: int  # MIDI program number (0-127)
    onset: float  # onset time in seconds
    offset: float  # offset time in seconds
    pitch: int  # MIDI note number (0-127)
    velocity: int  # (0-1) if ignore_velocity is True, otherwise (0-127)


@dataclass
class NoteEvent:
    is_drum: bool
    program: int  # [0, 127], 128 for drum but ignored in tokenizer
    time: Optional[float]  # absolute time. allow None for tie note events
    velocity: int  # currently 1 for onset, 0 for offset, drum has no offset
    pitch: int  # MIDI pitch
    activity: Optional[Set[int]] = field(default_factory=set)

@dataclass
class EventRange:
    type: str
    min_value: int
    max_value: int


@dataclass
class Event:
    type: str
    value: int
