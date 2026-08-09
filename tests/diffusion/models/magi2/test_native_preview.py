# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from vllm_omni.diffusion.models.magi2.attention import (
    VarlenHandler,
    correct_out_lse_with_sink,
    torch_varlen_attention_with_sink,
)
from vllm_omni.diffusion.models.magi2.configuration_magi2 import (
    Magi2MHCConfig,
    Magi2MoEConfig,
    Magi2PreviewConfig,
)
from vllm_omni.diffusion.models.magi2.layers import (
    MultiModalityRMSNorm,
    sinkhorn_knopp,
)
from vllm_omni.diffusion.models.magi2.mh_moe import (
    Magi2MultiHeadMoE,
    Magi2MultiHeadMoEConfig,
    compute_topk_probs_and_indices,
)
from vllm_omni.diffusion.models.magi2.modeling_magi2 import (
    Magi2PreviewTransformer,
    Modality,
)
from vllm_omni.diffusion.models.magi2.parallel import (
    Magi2ParallelGroup,
    balanced_split_sizes,
)
from vllm_omni.diffusion.offloader.block_discovery import get_blocks_from_dit


def _tiny_config(params_dtype: torch.dtype = torch.float32) -> Magi2PreviewConfig:
    return Magi2PreviewConfig(
        num_layers=1,
        hidden_size=16,
        head_dim=8,
        num_query_groups=2,
        video_in_channels=4,
        audio_in_channels=4,
        text_in_channels=4,
        intermediate_factor=2,
        multimodal_layers=(0,),
        params_dtype=params_dtype,
        mhc=Magi2MHCConfig(num_streams=2),
        moe=Magi2MoEConfig(
            num_heads=2,
            num_experts=4,
            top_k=2,
            expert_intermediate_size=8,
            shared_expert_intermediate_size=8,
            modality_shared_expert_intermediate_size=8,
            layers=(0,),
        ),
    )


def _initialize_tiny_model(model: Magi2PreviewTransformer, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=parameter.dtype,
                )
                * 0.02
            )
        for module in model.modules():
            if isinstance(module, MultiModalityRMSNorm):
                module.weight.zero_()
            elif isinstance(module, Magi2MultiHeadMoE):
                module.router.expert_bias.zero_()
                module.router.expert_bias_ema.zero_()


def test_attention_sink_correction_adds_zero_value_softmax_mass() -> None:
    output = torch.ones(2, 3, 4)
    lse = torch.zeros(3, 2)
    sink = torch.zeros(1, 3)

    corrected, corrected_lse = correct_out_lse_with_sink(output, lse, sink)

    torch.testing.assert_close(corrected, torch.full_like(output, 0.5))
    torch.testing.assert_close(corrected_lse, torch.full_like(lse, torch.log(torch.tensor(2.0))))


def test_torch_varlen_attention_keeps_segments_isolated_and_applies_sink() -> None:
    q = torch.zeros(2, 1, 2)
    k = torch.zeros_like(q)
    v = torch.tensor([[[2.0, 4.0]], [[6.0, 8.0]]])
    cumulative = torch.tensor([0, 1, 2], dtype=torch.int32)

    output = torch_varlen_attention_with_sink(
        q,
        k,
        v,
        cu_seqlens_q=cumulative,
        cu_seqlens_k=cumulative,
        sink=torch.zeros(1, 1),
    )

    torch.testing.assert_close(output, v / 2)


def test_sinkhorn_knopp_produces_doubly_stochastic_matrices() -> None:
    logits = torch.tensor([[[0.1, 0.3], [0.2, -0.4]]], dtype=torch.float32)
    matrix = sinkhorn_knopp(logits, iterations=20, epsilon=1e-12)
    torch.testing.assert_close(matrix.sum(dim=-1), torch.ones(1, 2), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(matrix.sum(dim=-2), torch.ones(1, 2), atol=1e-5, rtol=1e-5)


def test_router_bias_changes_selection_but_not_probability_source() -> None:
    logits = torch.zeros(1, 2, 3)
    probabilities, indices = compute_topk_probs_and_indices(
        logits,
        1,
        score_func="sigmoid",
        expert_bias=torch.tensor([[0.0, 10.0, 0.0]]),
    )
    assert torch.equal(indices, torch.ones_like(indices))
    torch.testing.assert_close(probabilities, torch.ones_like(probabilities))


def test_router_topk_keeps_reference_score_order_before_normalization() -> None:
    logits = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]], dtype=torch.float32)

    probabilities, indices = compute_topk_probs_and_indices(
        logits,
        3,
        score_func="sigmoid",
    )

    assert torch.equal(indices, torch.tensor([[[3, 2, 1]]]))
    expected = torch.sigmoid(logits).gather(-1, indices)
    expected = torch.nn.functional.normalize(expected, p=1, dim=-1, eps=1e-12)
    torch.testing.assert_close(probabilities, expected)


def test_ep_slice_pads_only_the_rank_local_head_rows() -> None:
    config = Magi2MultiHeadMoEConfig(
        hidden_size=6,
        num_heads=3,
        num_experts=2,
        top_k=1,
        expert_intermediate_size=4,
        params_dtype=torch.float32,
    )
    module = Magi2MultiHeadMoE(config, ep_group=Magi2ParallelGroup(None, world_size=2, rank=1))
    checkpoint = torch.arange(6 * 2, dtype=torch.float32).view(6, 2)

    local = module.ep_slice(checkpoint)

    assert local.shape == (4, 2)
    torch.testing.assert_close(local[:2], checkpoint[4:])
    torch.testing.assert_close(local[2:], torch.zeros(2, 2))


def test_tiny_native_preview_forward_returns_video_audio_channels_only() -> None:
    model = Magi2PreviewTransformer(_tiny_config())
    _initialize_tiny_model(model, seed=7)
    packed = torch.randn(6, 4)
    coordinates = torch.ones(6, 9)
    modalities = torch.tensor(
        [
            Modality.VIDEO,
            Modality.VIDEO,
            Modality.AUDIO,
            Modality.AUDIO,
            Modality.TEXT,
            Modality.TEXT,
        ]
    )
    varlen = VarlenHandler(
        torch.tensor([0, 6], dtype=torch.int32),
        torch.tensor([0, 6], dtype=torch.int32),
        6,
        6,
    )

    with torch.no_grad():
        output = model(packed, coordinates, modalities, varlen)

    assert output.shape == (6, 4)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[4:], torch.zeros_like(output[4:]))


def test_tiny_native_preview_matches_pinned_reference_golden() -> None:
    """Full-model golden from SandAI reference f68a0f9bbccb.

    The reference's unavailable compiler/FA3/MoE CUDA boundaries were replaced
    by their eager PyTorch equations when generating this tensor.  Keeping the
    resulting golden local makes this regression independent of that runtime.
    """

    model = Magi2PreviewTransformer(_tiny_config(torch.bfloat16))
    with torch.no_grad():
        trainable_parameters = (parameter for parameter in model.parameters() if parameter.requires_grad)
        for index, parameter in enumerate(trainable_parameters):
            values = (torch.arange(parameter.numel(), dtype=torch.float32) % 17 - 8) * 0.002 + (index % 5 - 2) * 0.0001
            parameter.copy_(values.reshape(parameter.shape).to(parameter.dtype))
        moe = model.block.layers[0].mlp.moe_mlp
        moe.router.expert_bias.zero_()
        moe.router.expert_bias_ema.zero_()

    packed = torch.tensor(
        [
            [0.1, -0.2, 0.3, -0.4],
            [0.5, 0.6, -0.7, 0.8],
            [-0.9, 1.0, -0.1, 0.2],
            [0.3, -0.4, 0.5, -0.6],
            [0.7, 0.8, 0.9, 1.0],
            [-0.2, 0.4, -0.6, 0.8],
            [0.9, -0.7, 0.5, -0.3],
        ]
    )
    coordinates = torch.tensor(
        [
            [0, 0, 0, 2, 2, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 1, 2, 2, 1, 2, 2],
            [0, 1, 1, 1, 2, 2, 1, 2, 2],
            [0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    modalities = torch.tensor(
        [
            Modality.VIDEO,
            Modality.VIDEO,
            Modality.AUDIO,
            Modality.AUDIO,
            Modality.TEXT,
            Modality.TEXT,
            Modality.TIME,
        ]
    )
    cumulative = torch.tensor([0, 4, 7], dtype=torch.int32)
    time_tokens = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 0.9, 0.8],
            [0.7, 0.6, 0.5],
            [0.4, 0.3, 0.2],
            [0.1, 0.0, -0.1],
        ]
    )
    expected = torch.tensor(
        [
            [-0.093289732933, 0.133922040462, 0.149612516165, 0.069642566144],
            [-0.118302434683, 0.129068359733, 0.156465008855, 0.084789708257],
            [-0.114637844265, 0.137497439981, 0.159646511078, 0.088356778026],
            [-0.104430131614, 0.146924808621, 0.152607530355, 0.092736266553],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    varlen = VarlenHandler(cumulative, cumulative, 4, 4)

    with torch.no_grad():
        actual = model(packed, coordinates, modalities, varlen, time_tokens)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_strict_loader_covers_all_keys_and_uses_ema_router_bias() -> None:
    source = Magi2PreviewTransformer(_tiny_config())
    _initialize_tiny_model(source, seed=11)
    source_moe = source.block.layers[0].mlp.moe_mlp
    source_moe.router.expert_bias.fill_(2.0)
    source_moe.router.expert_bias_ema.fill_(3.0)
    checkpoint = [(name, value.detach().clone()) for name, value in source.state_dict().items()]

    target = Magi2PreviewTransformer(_tiny_config())
    loaded = target.load_weights(checkpoint)

    assert loaded == set(target.state_dict())
    target_moe = target.block.layers[0].mlp.moe_mlp
    torch.testing.assert_close(
        target_moe.router.expert_bias,
        torch.full_like(target_moe.router.expert_bias, 3.0),
    )
    torch.testing.assert_close(
        target_moe.router.expert_bias_ema, torch.full_like(target_moe.router.expert_bias_ema, 3.0)
    )


def test_balanced_split_sizes_supports_uneven_and_empty_rank_slices() -> None:
    assert balanced_split_sizes(10, 4) == [3, 3, 2, 2]
    assert balanced_split_sizes(2, 4) == [1, 1, 0, 0]


def test_layerwise_offload_discovers_registered_checkpoint_block() -> None:
    model = Magi2PreviewTransformer(_tiny_config())

    attr_names, blocks = get_blocks_from_dit(model)

    assert attr_names == ["block"]
    assert "block" in model._modules
    assert blocks == list(model.block.layers)
