# MuLaCover CLI inference

MuLaCover generates cover songs from reference audio or melody/chord MIDI, with
lyrics and a text style description. This repository supports local inference
through the `mulacover` command.

## Installation

Requirements: Linux, Python 3.10, an NVIDIA GPU and a compatible driver.
Environment: Ubuntu 24.04, PyTorch 2.10.0 / CUDA 13.0 and NVIDIA B300. For audio input/output, install FFmpeg and libsndfile. Reference audio supports WAV, FLAC, OGG and MP3; other formats such as M4A/AAC depend on the available decoder.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1
git clone https://github.com/hanruijiang/MuLaCover.git
cd MuLaCover
conda create -y -n mulacover python=3.10 pip
conda activate mulacover
```

For reference-audio inference:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e '.[audio]'
mulacover --help
```

For MIDI-only inference (no `torchaudio` or `audio` extra required):

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e .
mulacover --help
```

PyTorch 2.6 or newer is required by the restricted YourMT3 checkpoint loader.
Use matching PyTorch and torchaudio versions for reference audio. On a machine with conflicting CUDA libraries
(e.g. a Deep Learning AMI), prefix the inference command with
`env -u LD_LIBRARY_PATH` if cuBLAS reports an invalid handle or a missing symbol.

## Checkpoints

Download the models into a single checkpoint root. The Hugging Face CLI
(`hf`) is provided by huggingface-hub; authenticate with `hf auth login` if
required.

```bash
hf download HeartMuLa/MuLaCover --local-dir ckpt/MuLaCover
(cd ckpt/MuLaCover && sha256sum -c SHA256SUMS)
hf download Qwen/Qwen3-Embedding-0.6B --local-dir ckpt/Qwen3-Embedding-0.6B
hf download HeartMuLa/HeartCodec-oss-20260123 --local-dir ckpt/HeartCodec-oss
```

For reference-audio inference only, download the YourMT3 checkpoint and five
ChordNet checkpoints:

```bash
mkdir -p ckpt/SymbolicTranscriptor/yourmt3 ckpt/SymbolicTranscriptor/chord
curl -fL \
  'https://huggingface.co/spaces/mimbres/YourMT3/resolve/main/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt' \
  -o ckpt/SymbolicTranscriptor/yourmt3/last.ckpt
for fold in 0 1 2 3 4; do
  chord_file="joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s${fold}.best.sdict"
  curl -fL \
    "https://raw.githubusercontent.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition/master/cache_data/$chord_file" \
    -o "ckpt/SymbolicTranscriptor/chord/$chord_file"
done
```

Expected layout:

```text
ckpt/
├── MuLaCover/                    # tokenizer.json, config.json, gen_config.json, model shards and index
├── HeartCodec-oss/
├── Qwen3-Embedding-0.6B/
└── SymbolicTranscriptor/         # reference audio only
    ├── yourmt3/last.ckpt
    └── chord/                   # five .best.sdict files
```

## Inference

Run commands from the repository root. Both `--lyrics` and `--tags` accept a
UTF-8 file path or literal text. Example inputs are
[assets/lyrics.txt](assets/lyrics.txt) and [assets/tags.txt](assets/tags.txt).
Use section markers on their own lines, one lyric per line, blank lines
between sections, and no timestamps. For style, use a single line such as
`topic:[Longing]; genre:[country]; instrument:[Strings,acoustic guitar]; mood:[hopeful]`.

### From melody and chord MIDI

```bash
mulacover \
  --model_path ./ckpt \
  --melody_midi /path/to/melody.mid --chord_midi /path/to/chord.mid \
  --lyrics assets/lyrics.txt --tags assets/tags.txt \
  --save_path cover.wav --device cuda:0 --seed 42
```

`--drum_midi` is optional. MIDI timing is read in musical beats.

### From reference audio

```bash
mulacover \
  --model_path ./ckpt --ref_audio /path/to/reference.mp3 \
  --lyrics assets/lyrics.txt --tags assets/tags.txt \
  --symbolic_save_dir transcribed --save_path cover.wav \
  --device cuda:0 --seed 42
```

`--symbolic_save_dir` optionally exports the transcribed MIDI. Use `--bpm`
to override automatic BPM estimation for reference audio. Audio and MIDI
inputs are mutually exclusive.

Lazy loading is enabled by default to reduce GPU memory use.
Use `mulacover --help` for all options, including `--max_audio_length_ms`,
`--temperature`, `--topk`, `--cfg_scale` and `--dtype`.

## Licensing and attribution

Source code is licensed under [Apache-2.0](LICENSE). Official MuLaCover
weights and generated outputs are restricted to noncommercial use under
[MODEL_LICENSE](MODEL_LICENSE); see [LICENSING.md](LICENSING.md).
Third-party components retain their own terms and notices:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Model weights: [HeartMuLa/MuLaCover](https://huggingface.co/HeartMuLa/MuLaCover).
Paper: [MuLaCover: Flexible Cover Song Generation via Symbolic Tonal Control](https://arxiv.org/abs/2610.26684).
