# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.magi2.preview_data_proxy import (
    Magi2DataProxy,
    Magi2PreviewDataProxyConfig,
    Modality,
    ModelInput,
    sinusoidal_embedding_1d,
)
from vllm_omni.diffusion.models.magi2.sampler_magi2 import (
    CFGConfig,
    Magi2PreviewSampler,
    build_magi2_preview_schedulers,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _tiny_model_input() -> ModelInput:
    video = torch.tensor([[[[[0.0, 1.0], [2.0, 3.0]]], [[[4.0, 5.0], [6.0, 7.0]]]]])
    audio = torch.tensor([[[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]])
    text = torch.tensor([[[20.0, 21.0, 22.0, 23.0], [24.0, 25.0, 26.0, 27.0]]])
    ref_image = torch.tensor([[[[[[40.0, 41.0]]], [[[42.0, 43.0]]]]]])
    return ModelInput(
        x_t=video,
        audio_x_t=audio,
        audio_feat_len=torch.tensor([2]),
        txt_feat=text,
        txt_feat_len=torch.tensor([1]),
        t=torch.tensor([0.5]),
        per_token_video_t=torch.full((1, 1, 1, 2, 2), 0.5),
        per_token_audio_t=torch.full((1, 3, 1), 0.5),
        ref_image_feat=ref_image,
        ref_image_feat_len=torch.tensor([[[1, 2]]]),
        ref_image_special_token_embedding=torch.tensor([[[30.0, 31.0, 32.0, 33.0]]]),
    )


def test_preview_packing_order_coordinates_and_depack():
    proxy = Magi2DataProxy(Magi2PreviewDataProxyConfig(time_channel_dim=4))
    tokens, coords, modality, varlen, time_channel = proxy.process_input(_tiny_model_input())

    expected_tokens = torch.tensor(
        [
            [0.0, 4.0, 0.0, 0.0],
            [1.0, 5.0, 0.0, 0.0],
            [2.0, 6.0, 0.0, 0.0],
            [3.0, 7.0, 0.0, 0.0],
            [10.0, 11.0, 12.0, 0.0],
            [13.0, 14.0, 15.0, 0.0],
            [20.0, 21.0, 22.0, 23.0],
            [30.0, 31.0, 32.0, 33.0],
            [40.0, 42.0, 0.0, 0.0],
            [41.0, 43.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(tokens, expected_tokens, rtol=0, atol=0)
    torch.testing.assert_close(
        modality,
        torch.tensor(
            [
                Modality.VIDEO,
                Modality.VIDEO,
                Modality.VIDEO,
                Modality.VIDEO,
                Modality.AUDIO,
                Modality.AUDIO,
                Modality.TEXT,
                Modality.TEXT,
                Modality.VIDEO,
                Modality.VIDEO,
            ],
            dtype=torch.int32,
        ),
        rtol=0,
        atol=0,
    )
    assert varlen.cu_seqlens_q.tolist() == [0, 10]
    assert varlen.cu_seqlens_k.tolist() == [0, 10]
    assert varlen.max_seqlen_q == varlen.max_seqlen_k == 10

    torch.testing.assert_close(
        coords[0],
        torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0]),
    )
    torch.testing.assert_close(
        coords[6],
        torch.tensor([-1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    torch.testing.assert_close(
        coords[7],
        torch.tensor([3.0, -1.0, -1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0]),
    )
    torch.testing.assert_close(
        coords[8],
        torch.tensor([3.0, 0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0]),
    )

    noisy_time = sinusoidal_embedding_1d(4, torch.tensor([0.5]))[0]
    clean_time = sinusoidal_embedding_1d(4, torch.tensor([0.0]))[0]
    torch.testing.assert_close(time_channel[:6], noisy_time.expand(6, -1))
    torch.testing.assert_close(time_channel[6:], clean_time.expand(4, -1))

    video, audio = proxy.process_output(tokens)
    torch.testing.assert_close(video, _tiny_model_input().x_t)
    torch.testing.assert_close(audio[:, :2], _tiny_model_input().audio_x_t[:, :2])
    torch.testing.assert_close(audio[:, 2], torch.zeros_like(audio[:, 2]))


def test_i2v_cfg_keeps_reference_in_released_configuration():
    sampler = Magi2PreviewSampler(nn.Identity())
    feature = torch.arange(8.0).reshape(1, 1, 2, 1, 2, 2)
    lengths = torch.tensor([[[2, 2]]])
    special = torch.arange(4.0).reshape(1, 1, 4)

    kept, kept_lengths, kept_special = sampler._prepare_ref_image_cfg(
        feature,
        lengths,
        special,
        CFGConfig(use_ref_for_uncond=True),
    )
    torch.testing.assert_close(kept[0], feature[0])
    torch.testing.assert_close(kept[1], feature[0])
    torch.testing.assert_close(kept_lengths, lengths.repeat(2, 1, 1))
    torch.testing.assert_close(kept_special, special.repeat(2, 1, 1))

    steered, _, steered_special = sampler._prepare_ref_image_cfg(
        feature,
        lengths,
        special,
        CFGConfig(use_ref_for_uncond=False),
    )
    torch.testing.assert_close(steered[0], feature[0])
    torch.testing.assert_close(steered[1], torch.zeros_like(feature[0]))
    # The text-derived image marker remains present in both branches.
    torch.testing.assert_close(steered_special, special.repeat(2, 1, 1))


def test_prepare_model_input_preserves_cfg_order_and_time_normalization():
    sampler = Magi2PreviewSampler(nn.Identity())
    latent = torch.arange(8.0).reshape(1, 2, 1, 2, 2)
    audio = torch.arange(6.0).reshape(1, 2, 3)
    positive = torch.ones(1, 2, 4)
    negative = torch.full((1, 3, 4), -1.0)

    model_input = sampler.prepare_model_input(
        latent,
        audio,
        positive,
        negative,
        ref_audio_feat=torch.empty(1, 0, 3),
        t=torch.tensor(750.0),
        cfg_config=CFGConfig(),
    )
    torch.testing.assert_close(model_input.x_t, latent.repeat(2, 1, 1, 1, 1))
    torch.testing.assert_close(model_input.audio_x_t, audio.repeat(2, 1, 1))
    assert model_input.txt_feat_len.tolist() == [2, 3]
    assert model_input.txt_feat.shape == (2, 3, 4)
    torch.testing.assert_close(model_input.txt_feat[0, 2], torch.zeros(4))
    torch.testing.assert_close(model_input.t, torch.tensor([0.75, 0.75]))
    torch.testing.assert_close(model_input.per_token_video_t, torch.full((2, 1, 1, 2, 2), 0.75))
    torch.testing.assert_close(model_input.per_token_audio_t, torch.full((2, 2, 1), 0.75))


def test_cfg_velocity_and_dynamic_cfg_match_released_formula():
    sampler = Magi2PreviewSampler(nn.Identity(), device="cpu")
    cond_uncond_video = torch.tensor([2.0, 1.0]).reshape(2, 1, 1, 1, 1)
    cond_uncond_audio = torch.tensor([3.0, -1.0]).reshape(2, 1, 1)
    cfg_video, cfg_audio = sampler.cfg_velocity(
        (cond_uncond_video, cond_uncond_audio),
        torch.tensor([5.0]),
        7.0,
        CFGConfig(),
    )
    torch.testing.assert_close(cfg_video, torch.tensor([6.0]).reshape(1, 1, 1, 1, 1))
    torch.testing.assert_close(cfg_audio, torch.tensor([27.0]).reshape(1, 1, 1))

    video_cfgs, audio_cfgs = sampler.precalculate_cfg(
        [torch.tensor(999.0), torch.tensor(100.0)],
        latent_length=3,
        cfg_config=CFGConfig(
            use_cfg_trick=True,
            cfg_trick_start_frame=1,
            use_dynamic_cfg=True,
            dynamic_cfg_start_t=500,
            dynamic_cfg_cutoff_value=2.5,
        ),
    )
    torch.testing.assert_close(video_cfgs[0].flatten(), torch.tensor([2.0, 5.0, 5.0]))
    torch.testing.assert_close(video_cfgs[1].flatten(), torch.tensor([2.0, 2.5, 2.5]))
    assert audio_cfgs == [7.0, 7.0]


def test_sampler_forward_does_not_move_model_for_dlo_compatibility():
    class PlacementOwnedModel(nn.Module):
        def to(self, *args, **kwargs):  # pragma: no cover - failure path assertion
            raise AssertionError("the sampler must not own model placement")

        def forward(self, tokens, *_args):
            return tokens

    sampler = Magi2PreviewSampler(
        PlacementOwnedModel(),
        Magi2DataProxy(Magi2PreviewDataProxyConfig(time_channel_dim=4)),
    )
    video, audio = sampler.forward(_tiny_model_input())
    assert video.shape == (1, 2, 1, 2, 2)
    assert audio.shape == (1, 3, 3)


def test_scheduler_builder_uses_independent_shifted_flow_unipc_instances():
    video_scheduler, audio_scheduler = build_magi2_preview_schedulers(
        2,
        device="cpu",
        shift=7.0,
    )
    assert video_scheduler is not audio_scheduler
    torch.testing.assert_close(video_scheduler.timesteps, torch.tensor([999, 874]))
    torch.testing.assert_close(audio_scheduler.timesteps, video_scheduler.timesteps)
