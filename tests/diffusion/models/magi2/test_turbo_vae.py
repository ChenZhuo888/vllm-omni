# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm_omni.diffusion.models.magi2.turbo_vae import (
    extract_turbo_decoder_state_dict,
    turbo_unpatchify,
)


def test_turbo_unpatchify_matches_wan_channel_order():
    # Four patch channels for one output channel. MAGI's order interleaves
    # width before height when reconstructing each 2x2 patch.
    patched = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1, 1)
    output = turbo_unpatchify(patched, patch_size=2)
    expected = torch.tensor([[[[[0.0, 2.0], [1.0, 3.0]]]]])
    torch.testing.assert_close(output, expected)


def test_extract_turbo_decoder_prefers_ema_and_drops_training_heads():
    decoder_weight = torch.ones(1)
    checkpoint = {
        "state_dict": {"module.decoder.old.weight": torch.zeros(1)},
        "ema_state_dict": {
            "module.decoder.conv_in.conv.weight": decoder_weight,
            "module.aligned_feature_projection_heads.0.weight": torch.empty(1),
        },
    }
    assert extract_turbo_decoder_state_dict(checkpoint) == {"decoder.conv_in.conv.weight": decoder_weight}
