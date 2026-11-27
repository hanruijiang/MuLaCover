# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""vocabulary.py

Vocabulary for instrument classes. Vocabulary can be used as train_vocab
or test_vocab in data_presets.py or train.py arguments.

- When it is used as train_vocab, it maps the instrument classes to the first
  program number of the class. For example, if you use 'GM_INSTR_CLASS' as
  train_vocab, then the program number of 'Piano' is [0,1,2,3,4,5,6,7]. These
  program numbers are trained as program [0] in the model.

  - When it is used as eval_vocab, any program number in the instrument class
  is considered as correct.


MUSICNET_INSTR_CLASS: 3 classes used for MusicNet benchmark
GM_INSTR_CLASS: equivalent to 'MIDI Class' defined by MT3.
GM_INSTR_CLASS_PLUS: GM_INSTR_CLASS + singing voice
GM_INSTR_FULL: 128 GM instruments, which is extended from 'MT3_FULL'
MT3_FULL: this matches the class names in Table 3 of MT3 paper
ENST_DRUM_NOTES: 20 drum notes used in ENST dataset
GM_DRUM_NOTES: 45 GM drum notes with percussions

Program 128 is reserved for 'drum' internally.
Program 129 is reserved for 'unannotated', internally.
Program 100 is reserved for 'singing voice (melody)' in GM_INSTR_CLASS_PLUS.
Program 101 is reserved for 'singing voice (chorus)' in GM_INSTR_CLASS_PLUS.


"""

# yapf: disable
import numpy as np

GM_INSTR_CLASS = {
    "Piano": np.arange(0, 8),
    "Chromatic Percussion": np.arange(8, 16),
    "Organ": np.arange(16, 24),
    "Guitar": np.arange(24, 32),
    "Bass": np.arange(32, 40),
    "Strings": np.arange(40, 56),  # Strings + Ensemble
    # "Strings": np.arange(40, 48),
    # "Ensemble": np.arange(48, 56),
    "Brass": np.arange(56, 64),
    "Reed": np.arange(64, 72),
    "Pipe": np.arange(72, 80),
    "Synth Lead": np.arange(80, 88),
    "Synth Pad": np.arange(88, 96),
}

GM_INSTR_CLASS_PLUS = GM_INSTR_CLASS.copy()
GM_INSTR_CLASS_PLUS["Singing Voice"] = [100, 101]

MT3_FULL = { # this matches the class names in Table 3 of MT3 paper 
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
    "Low Tom (Rim)": [50],  # TD-17: 47, 50, 58  
    "Mid Tom (Rim)": [58],
    # "Ride Cymbal": [51, 53, 59],
    "Ride": [51],
    "Ride (Bell)": [53],  # https://youtu.be/b94hZoM5s3k?t=323
    "Ride (Edge)": [59],
    "Chinese Cymbal": [52],
    "Crash Cymbal": [49, 57],
    "Splash Cymbal": [55],
}

program_vocab_presets = {
    "mt3_full_plus": MT3_FULL_PLUS,
    "gm_plus": GM_INSTR_CLASS_PLUS,
}

drum_vocab_presets = {"gm": GM_DRUM_NOTES}
