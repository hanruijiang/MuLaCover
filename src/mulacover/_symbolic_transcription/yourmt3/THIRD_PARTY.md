# YourMT3

The melody and percussion transcriber is adapted from [YourMT3](https://github.com/mimbres/YourMT3) (Copyright 2024 The YourMT3 Authors). Paper: https://arxiv.org/abs/2407.04822.

The adapted files carry Apache-2.0 headers; see [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0). The [upstream repository license](https://github.com/mimbres/YourMT3/blob/main/LICENSE) is GPL-3.0. The applicable source revision and authorization require verification before public distribution.

MuLaCover adapts the source into a fixed YPTF.MoE+Multi inference model with package-relative imports, local checkpoint loading and a tuple KV cache.

- `model/t5_compat.py`: adapted from Hugging Face Transformers 4.45.1, `transformers/models/t5/modeling_t5.py`, under Apache-2.0. Copyright notices are retained.
- `model/RoPE/RoPE.py`: adapted from [rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch); see [LICENSE-ROTARY-EMBEDDING](LICENSE-ROTARY-EMBEDDING).

Model weights are distributed separately under their applicable terms.
