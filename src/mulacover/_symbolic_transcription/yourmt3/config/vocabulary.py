# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""Instrument and drum vocabularies for transcription."""

import numpy as np

MT3_FULL = {
    "Acoustic Piano": [0, 1, 3, 6, 7],
    "Electric Piano": [2, 4, 5],
    "Chromatic Percussion": np.arange(8, 16),
    "Organ": np.arange(16, 24),
    "Acoustic Guitar": np.arange(24, 26),
    "Clean Electric Guitar": np.arange(26, 29),
    "Distorted Electric Guitar": np.arange(29, 32),
    "Acoustic Bass": [32, 35],
    "Electric Bass": [33, 34, 36, 37, 38, 39],
    "Violin": [40],
    "Viola": [41],
    "Cello": [42],
    "Contrabass": [43],
    "Orchestral Harp": [46],
    "Timpani": [47],
    "String Ensemble": [48, 49, 44, 45],
    "Synth Strings": [50, 51],
    "Choir and Voice": [52, 53, 54],
    "Orchestra Hit": [55],
    "Trumpet": [56, 59],
    "Trombone": [57],
    "Tuba": [58],
    "French Horn": [60],
    "Brass Section": [61, 62, 63],
    "Soprano/Alto Sax": [64, 65],
    "Tenor Sax": [66],
    "Baritone Sax": [67],
    "Oboe": [68],
    "English Horn": [69],
    "Bassoon": [70],
    "Clarinet": [71],
    "Pipe": [73, 72, 74, 75, 76, 77, 78, 79],
    "Synth Lead": np.arange(80, 88),
    "Synth Pad": np.arange(88, 96),
}

MT3_FULL_PLUS = MT3_FULL.copy()
MT3_FULL_PLUS["Singing Voice"] = [100]
MT3_FULL_PLUS["Singing Voice (chorus)"] = [101]

GM_DRUM_NOTES = {
    "Kick Drum": [36, 35],  # Listed by order of most common annotation
    "Snare X-stick": [37, 2],  # Snare X-Stick, https://youtu.be/a2KFrrKaoYU?t=80
    "Snare Drum": [38, 40],  # Snare (head) and Electric Snare
    "Closed Hi-Hat": [42, 44, 22],  # 44 is pedal hi-hat
    "Open Hi-Hat": [46, 26],
    "Cowbell": [56],
    "High Floor Tom": [43],
    "Low Floor Tom": [41],  # Lowest Tom
    "Low Tom": [45],
    "Low-Mid Tom": [47],
    "Mid Tom": [48],
    "Low Tom (Rim)": [50],
    "Mid Tom (Rim)": [58],
    "Ride": [51],
    "Ride (Bell)": [53],  # https://youtu.be/b94hZoM5s3k?t=323
    "Ride (Edge)": [59],
    "Chinese Cymbal": [52],
    "Crash Cymbal": [49, 57],
    "Splash Cymbal": [55],
}

program_vocab_presets = {
    "mt3_full_plus": MT3_FULL_PLUS,
}

drum_vocab_presets = {"gm": GM_DRUM_NOTES}
