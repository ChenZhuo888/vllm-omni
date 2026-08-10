# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PiD network & sampling config registry for all supported backbones.

All backbones share the same PixDiT_T2I architecture; only the LQ-related
constructor args differ per VAE.

Backbone -> VAE characteristics:
    Qwen-Image:  16ch latent, 8x spatial compression
    Flux1:       16ch latent, 8x spatial compression
    SD3:         16ch latent, 8x spatial compression
    SDXL:         4ch latent, 8x spatial compression
    Flux2:      128ch latent, 16x spatial compression (2x2 patchify -> 32 raw ch)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Shared PixDiT_T2I backbone args (identical across all backbones)
# ---------------------------------------------------------------------------

_SHARED_BACKBONE = dict(
    in_channels=3,
    num_groups=24,
    hidden_size=1536,
    pixel_hidden_size=16,
    pixel_attn_hidden_size=1152,
    pixel_num_groups=16,
    patch_depth=14,
    pixel_depth=2,
    num_text_blocks=4,
    patch_size=16,
    txt_embed_dim=2304,
    txt_max_length=300,
    use_text_rope=True,
    text_rope_theta=10000.0,
    rope_mode="ntk_aware",
    rope_ref_h=2048,
    rope_ref_w=2048,
    repa_encoder_index=6,
    enable_ed=False,
)

# ---------------------------------------------------------------------------
# Shared PidNet SR args (same across all backbones, except LQ channels/spacing)
# ---------------------------------------------------------------------------

# v1pt5 defaults (pid_sr4x_v1pt5): replicate padding, per-token gate, PiT
# injection, aux RGB head, deeper LQ projection.
_SHARED_PID_SR = dict(
    lq_inject_mode="controlnet",
    lq_in_channels=0,
    lq_hidden_dim=1024,
    lq_num_res_blocks=4,
    lq_latent_unpatchify_factor=1,
    lq_aux_rgb_head=True,
    lq_aux_rgb_head_latent_block_idx=-1,
    lq_conv_padding_mode="replicate",
    lq_gate_type="sigma_aware_per_token",
    lq_interval=2,
    zero_init_lq=True,
    train_lq_proj_only=True,
    sr_scale=4,
    pit_lq_inject=True,
)


def _make_net_config(lq_latent_channels: int, latent_spatial_down_factor: int) -> dict:
    """Build a complete net config for a given VAE."""
    cfg = dict(_SHARED_BACKBONE)
    cfg.update(_SHARED_PID_SR)
    cfg.update(
        lq_latent_channels=lq_latent_channels,
        latent_spatial_down_factor=latent_spatial_down_factor,
    )
    return cfg


# ---------------------------------------------------------------------------
# Per-backbone net configs
# ---------------------------------------------------------------------------

QWENIMAGE_PID_NET_CONFIG = _make_net_config(lq_latent_channels=16, latent_spatial_down_factor=8)
FLUX_PID_NET_CONFIG = _make_net_config(lq_latent_channels=16, latent_spatial_down_factor=8)
SD3_PID_NET_CONFIG = _make_net_config(lq_latent_channels=16, latent_spatial_down_factor=8)
SDXL_PID_NET_CONFIG = _make_net_config(lq_latent_channels=4, latent_spatial_down_factor=8)
FLUX2_PID_NET_CONFIG = _make_net_config(lq_latent_channels=128, latent_spatial_down_factor=16)

# ---------------------------------------------------------------------------
# Sampling config (shared across all distill checkpoints)
# ---------------------------------------------------------------------------

PID_SAMPLING_CONFIG = dict(
    student_sample_steps=4,
    student_sample_type="sde",
    student_t_list=[0.999, 0.866, 0.634, 0.342, 0.0],
    student_input_mode="teacher_forcing",
    prediction_type="velocity",
    fm_timescale=1000.0,
    cfg_scale=5.0,
    dynamic_shift=dict(
        base_shift=6.0,
        base_image_size_for_shift_calc=2048,
    ),
)

# ---------------------------------------------------------------------------
# Typed config wrappers
# ---------------------------------------------------------------------------


@dataclass
class PidNetConfig:
    """Typed wrapper for a backbone-specific PidNet constructor args."""
    backbone: Literal["qwenimage", "flux", "sd3", "sdxl", "flux2"] = "qwenimage"
    net_kwargs: dict = field(default_factory=lambda: dict(QWENIMAGE_PID_NET_CONFIG))


@dataclass
class PidSamplingConfig:
    """Typed wrapper for sampling parameters."""
    sampling_kwargs: dict = field(default_factory=lambda: dict(PID_SAMPLING_CONFIG))


# ---------------------------------------------------------------------------
# Convenience getters
# ---------------------------------------------------------------------------

def get_pid_net_config(backbone: str) -> dict:
    """Return the net config dict for ``backbone``."""
    mapping = {
        "qwenimage": QWENIMAGE_PID_NET_CONFIG,
        "flux": FLUX_PID_NET_CONFIG,
        "sd3": SD3_PID_NET_CONFIG,
        "sdxl": SDXL_PID_NET_CONFIG,
        "flux2": FLUX2_PID_NET_CONFIG,
    }
    if backbone not in mapping:
        raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(mapping.keys())}.")
    return dict(mapping[backbone])


def get_pid_sampling_config() -> dict:
    """Return a copy of the shared sampling config."""
    return dict(PID_SAMPLING_CONFIG)
