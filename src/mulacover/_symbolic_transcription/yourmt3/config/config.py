"""YourMT3 model configuration."""

audio_cfg = {
    "hop_length": 300,
    "sample_rate": 16000,
    "input_frames": 32767,
    "n_fft": 2048,
}

encoder_cfg = {
    "num_latents": 26,
    "d_latent": 128,
    "d_model": 128,
    "num_blocks": 3,
    "num_local_transformers_per_block": 2,
    "num_temporal_transformers_per_block": 2,
    "sca_use_query_residual": True,
    "dropout_rate": 0.1,
    "position_encoding_type": "rope",
    "attention_to_channel": True,
    "layer_norm_type": "layer_norm",
    "ff_layer_type": "moe",
    "ff_widening_factor": 4,
    "moe_num_experts": 8,
    "moe_topk": 2,
    "hidden_act": "silu",
    "rope_type_sca": "pixel",
    "rope_type_latent": "pixel",
    "rope_type_temporal": "lang",
    "rope_apply_to_keys": False,
    "rope_partial_pe": True,
}

decoder_cfg = {
    "d_model": 512,
    "num_heads": 6,
    "num_layers": 8,
    "dropout_rate": 0.05,
    "position_encoding_type": "sinusoidal",
    "ff_widening_factor": 2,
    "ff_layer_type": "t5_gmlp",
    "num_channels": 13,
}

T5_BASE_CFG = {
    "architectures": ["T5ForConditionalGeneration"],
    "d_ff": 1024,
    "d_kv": 64,
    "decoder_start_token_id": 0,
    "dense_act_fn": "gelu_new",
    "eos_token_id": 1,
    "feed_forward_proj": "gated-gelu",
    "initializer_factor": 1.0,
    "is_encoder_decoder": True,
    "is_gated_act": True,
    "layer_norm_epsilon": 1e-6,
    "model_type": "t5",
    "output_past": True,
    "pad_token_id": 0,
    "relative_attention_num_buckets": 32,
    "use_cache": True,
}
