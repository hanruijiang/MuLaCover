# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""Decode transcription token IDs into note events."""

import numpy as np
from collections import Counter
from typing import List, Union, Tuple

from .note_event_dataclasses import Event, EventRange, NoteEvent
from .event_codec import FastCodec
from .event2note import event2note_event


class NoteEventTokenizer:
    def __init__(self, program_vocabulary, drum_vocabulary):
        self.codec = FastCodec(
            special_tokens=["PAD", "EOS", "UNK"],
            max_shift_steps=206,
            event_ranges=[
                EventRange("pitch", 0, 127),
                EventRange("velocity", 0, 1),
                EventRange("tie", 0, 0),
                EventRange("program", 0, 127),
                EventRange("drum", 0, 127),
            ],
            program_vocabulary=program_vocabulary,
            drum_vocabulary=drum_vocabulary,
        )
        self.num_tokens = self.codec.num_classes
        self.tps = 100


    def decode(
        self,
        tokens: List[int],
        start_time: float = 0.0,
        return_events: bool = False,
    ) -> Union[
        Tuple[List[NoteEvent], List[NoteEvent]],
        Tuple[List[NoteEvent], List[NoteEvent], List[Tuple[int]], List[Event], int],
    ]:
        events = [self.codec.decode_event_index(idx) for idx in tokens]
        note_events, tie_note_events, last_activity, err_cnt = event2note_event(
            events, start_time, True, self.tps
        )
        if return_events:
            return note_events, tie_note_events, last_activity, events, err_cnt
        else:
            return note_events, tie_note_events, last_activity, err_cnt

    def decode_list_batches(
        self,
        list_batch_tokens: Union[List[List[List[int]]], List[np.ndarray]],
        list_start_times: Union[List[List[float]], List[float]],
        return_events: bool = True,
    ) -> Union[
        Tuple[
            List[
                List[
                    Tuple[
                        List[NoteEvent], List[NoteEvent], List[Tuple[int]], List[float]
                    ]
                ]
            ],
            Counter[str],
        ],
        Tuple[
            List[
                List[
                    Tuple[
                        List[NoteEvent], List[NoteEvent], List[Tuple[int]], List[float]
                    ]
                ]
            ],
            List[List[Event]],
            Counter[str],
        ],
    ]:
        """
        Decodes a list of variable-size batches of token array to a list of
        zipped note_events and tie_note_events.

        Args:
            list_batch_tokens: List[np.ndarray], where array shape is (batch_size, variable_length)
            list_start_times: List[float], where the length is sum of all batch_sizes.
            return_events: bool, Defaults to False.

        Returns:
            list_list_zipped_note_events_and_tie:
                List[
                    Tuple[
                        List[NoteEvent]: A list of note events.
                        List[NoteEvent]: A list of tie note events.
                        List[Tuple[int]]: A list of last activity of segment. [(program, pitch), ...]. This is useful
                            for validating notes within a batch of segments extracted from a file.
                        List[float]: A list of segment start times.
                    ]
                ]
            (Optional) list_events:
                List[List[Event]]
            total_err_cnt:
                Counter[str]: error counter.
        """
        list_tokens = []
        for arr in list_batch_tokens:
            for tokens in arr:
                list_tokens.append(tokens)
        assert len(list_tokens) == len(list_start_times)

        zipped_note_events_and_tie = []
        list_events = []
        total_err_cnt = Counter()
        for tokens, start_time in zip(list_tokens, list_start_times):
            decoded = self.decode(tokens, start_time, return_events)
            if return_events:
                note_events, tie_note_events, last_activity, events, err_cnt = decoded
            else:
                note_events, tie_note_events, last_activity, err_cnt = decoded
            zipped_note_events_and_tie.append(
                (note_events, tie_note_events, last_activity, start_time)
            )
            if return_events:
                list_events.append(events)
            total_err_cnt += err_cnt

        if return_events:
            return zipped_note_events_and_tie, list_events, total_err_cnt
        else:
            return zipped_note_events_and_tie, total_err_cnt
