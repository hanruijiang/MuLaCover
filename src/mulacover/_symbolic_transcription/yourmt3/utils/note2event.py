# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""note2event.py

Note tools:
• mix_notes(notes_to_mix, sort, trim_overlap, fix_offset)
    -> List[Note]
• validate_notes(notes, fix)
    -> List[Note]
• trim_overlapping_notes(notes, sort)
    -> List[Note]
• sort_notes(notes)
    -> List[Note]
• notes2pc_notes(notes, note_offs)
    -> List[Note]
• extract_program_from_notes(notes)
    -> Set[int]
• extract_notes_selected_by_programs(notes, programs, sort)
    -> List[Note]

Note to NoteEvent
• note2note_event(notes, sort, return_activity)
    -> List[NoteEvent]

NoteEvent tools:
• slice_note_events_and_ties(note_events, start_time, end_time, tidyup)
    -> Tuple[List[NoteEvent], List[NoteEvent], int])
• slice_multiple_note_events_and_ties_to_bundle(note_events, start_times, duration_sec, tidyup)
    -> List[List[NoteEvent], List[NoteEvent], int]] # Note implmented yet..
• mix_note_event_lists_bundle(note_events_to_mix, sort, start_time_to_zero)
    -> NoteEventListsBundle
• pitch_shift_note_events(note_events, semitone, use_deepcopy)
    -> List[NoteEvent]
• separate_by_subunit_programs_from_note_event_lists_bundle(
        source_note_event_lists_bundle,
        subunit_programs)
    -> NoteEventListsBundle:
• separate_channel_by_program_group_from_note_event_lists_bundle(
        source_note_event_lists_bundle,
        num_program_groups,
        program2channel_vocab)
    -> List[NoteEventListsBundle]:

NoteEvent to Event:
• note_event2event(note_events, tie_note_events, start_time, tps, sort)
    -> List[Event]

Event tools:
• check_event_len_from_bundle(note_events_dic_a, note_events_dic_b, max_len, fast_check)
    -> bool
"""

import warnings
from itertools import chain
from typing import Optional, Tuple, List

from mulacover._symbolic_transcription.yourmt3.utils.note_event_dataclasses import Note

DRUM_OFFSET_TIME = 0.01  # in seconds
MINIMUM_OFFSET_TIME = 0.01  # this is used to avoid zero-length notes
DRUM_PROGRAM = 128


def mix_notes(
    notes_to_mix: Tuple[List[Note]],
    sort: bool = True,
    trim_overlap: bool = True,
    fix_offset: bool = True,
) -> List[Note]:
    """
    mix_notes:
    Mixes a tuple of many lists of Note instances into a single list of Note
     instances. This processes 'notes1 + notes2 + ... + notesN' faster.
    Because Note instances use absolute timing, the Note instances in the
    same timiming will be sorted by increasing order of program and pitch.

    Args:
    - notes_to_mix (tuple[list[Note]]): A tuple of lists of Note instances.
    - sort (bool): If True, sort the Note instances by increasing order of
      onsets, and at the same timing, by increasing order of program and pitch.
      Default is True.

    Returns:
    - notes (list[Note]): A list of Note instances.
    """
    mixed_notes = list(chain(*notes_to_mix))
    if sort and len(mixed_notes) > 0:
        mixed_notes.sort(
            key=lambda note: (
                note.onset,
                note.is_drum,
                note.program,
                note.velocity,
                note.pitch,
                note.offset,
            )
        )

    # Trim overlapping notes
    if trim_overlap:
        mixed_notes = trim_overlapping_notes(mixed_notes, sort=sort)

    # fix offset >= onset the Note instances
    if fix_offset:
        mixed_notes = validate_notes(mixed_notes, fix=True)
    return mixed_notes


def validate_notes(
    notes: Tuple[List[Note]], minimum_offset: Optional[bool] = 0.01, fix: bool = True
) -> List[Note]:
    """validate and fix unrealistic notes"""
    if len(notes) > 0:
        for note in list(notes):
            if note.onset == None:
                if fix:
                    notes.remove(note)
                continue
            elif note.offset == None:
                if fix:
                    note.offset = note.onset + MINIMUM_OFFSET_TIME
            elif note.onset > note.offset:
                warnings.warn(f"📙 Note at {note} has onset > offset.")
                if fix:
                    note.offset = max(note.offset, note.onset + MINIMUM_OFFSET_TIME)
                    print(
                        f"✅\033[92m Fixed! Setting offset to onset + {MINIMUM_OFFSET_TIME}.\033[0m"
                    )
            elif note.is_drum is False and note.offset - note.onset < 0.01:
                # fix 13 Oct: too short notes issue for the dataset with non-MIDI annotations
                # warnings.warn(f'📙 Note at {note} has offset - onset < 0.01.')
                if fix:
                    note.offset = note.onset + MINIMUM_OFFSET_TIME
                    # print(f'✅\033[92m Fixed! Setting offset to onset + {MINIMUM_OFFSET_TIME}.\033[0m')

    return notes


def trim_overlapping_notes(notes: List[Note], sort: bool = True) -> List[Note]:
    """Trim overlapping notes and dropping zero-length notes.
    https://github.com/magenta/mt3/blob/3deffa260ba7de3cf03cda1ea513a4d7ba7144ca/mt3/note_sequences.py#L52

    Trimming was only applied to train set, not test set in MT3.
    """
    if len(notes) <= 1:
        return notes

    trimmed_notes = []
    channels = set((note.pitch, note.program, note.is_drum) for note in notes)

    for pitch, program, is_drum in channels:
        channel_notes = [
            note
            for note in notes
            if note.pitch == pitch
            and note.program == program
            and note.is_drum == is_drum
        ]
        sorted_notes = sorted(channel_notes, key=lambda note: note.onset)

        for i in range(1, len(sorted_notes)):
            if sorted_notes[i - 1].offset > sorted_notes[i].onset:
                sorted_notes[i - 1].offset = sorted_notes[i].onset

        # Filter out zero-length notes
        valid_notes = [note for note in sorted_notes if note.onset < note.offset]

        trimmed_notes.extend(valid_notes)

    if sort:
        trimmed_notes.sort(
            key=lambda note: (
                note.onset,
                note.is_drum,
                note.program,
                note.velocity,
                note.pitch,
            )
        )
    return trimmed_notes


def sort_notes(notes: List[Note]) -> List[Note]:
    """Sort notes by increasing order of onsets, and at the same timing, by increasing order of program and pitch."""
    if len(notes) > 0:
        notes.sort(
            key=lambda note: (
                note.onset,
                note.is_drum,
                note.program,
                note.velocity,
                note.pitch,
                note.offset,
            )
        )
    return notes
