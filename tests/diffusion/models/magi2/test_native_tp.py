# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import vllm_omni.diffusion.models.magi2.layers as layers_module
import vllm_omni.diffusion.models.magi2.mh_moe as mh_moe_module
from vllm_omni.diffusion.models.magi2.configuration_magi2 import (
    Magi2MHCConfig,
    Magi2MoEConfig,
    Magi2PreviewConfig,
)
from vllm_omni.diffusion.models.magi2.layers import Magi2GroupedLinear
from vllm_omni.diffusion.models.magi2.mh_moe import (
    Magi2MultiHeadMoE,
    Magi2MultiHeadMoEConfig,
)
from vllm_omni.diffusion.models.magi2.modeling_magi2 import Magi2PreviewTransformer
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _group(world_size: int, rank: int) -> Magi2ParallelGroup:
    return Magi2ParallelGroup(None, world_size, rank, replicated_sequence=True)


def _tiny_config() -> Magi2PreviewConfig:
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
        params_dtype=torch.float32,
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


def test_segmented_grouped_qkv_column_shard_preserves_modality_and_qkv_order() -> None:
    layer = Magi2GroupedLinear(
        2,
        12,
        num_experts=2,
        parallel_mode="column",
        qkv_splits=(4, 4, 4),
        tp_group=_group(2, 1),
    )
    checkpoint = torch.arange(24 * 2, dtype=torch.float32).view(24, 2)

    actual = layer.shard_checkpoint_weight(checkpoint)

    grouped = checkpoint.view(2, 12, 2)
    expected = torch.cat((grouped[:, 2:4], grouped[:, 6:8], grouped[:, 10:12]), dim=1).reshape(12, 2)
    torch.testing.assert_close(actual, expected)


def test_grouped_row_shards_sum_to_the_released_linear() -> None:
    checkpoint = torch.arange(2 * 6 * 4, dtype=torch.float32).view(12, 4) / 10
    inputs = (torch.arange(3 * 4, dtype=torch.float32).view(3, 4) - 4) / 7
    modality_sizes = (1, 2)
    expected_parts = [
        F.linear(part, weight)
        for part, weight in zip(
            torch.split(inputs, modality_sizes),
            checkpoint.view(2, 6, 4),
        )
    ]

    partials: list[list[torch.Tensor]] = []
    for rank in range(2):
        layer = Magi2GroupedLinear(
            4,
            6,
            num_experts=2,
            parallel_mode="row",
            tp_group=_group(2, rank),
        )
        shard = layer.shard_checkpoint_weight(checkpoint).view(2, 6, 2)
        input_shard = inputs[:, rank * 2 : (rank + 1) * 2]
        partials.append(
            [F.linear(part, weight) for part, weight in zip(torch.split(input_shard, modality_sizes), shard)]
        )

    for expert, expected in enumerate(expected_parts):
        torch.testing.assert_close(partials[0][expert] + partials[1][expert], expected)


def test_tp_loader_slices_every_native_tensor_parallel_weight(monkeypatch) -> None:
    source = Magi2PreviewTransformer(_tiny_config())
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    generator=generator,
                )
            )
    checkpoint = [(name, tensor.detach().clone()) for name, tensor in source.state_dict().items()]
    tp_group = _group(2, 1)
    monkeypatch.setattr(layers_module, "get_magi2_tp_group", lambda: tp_group)
    monkeypatch.setattr(mh_moe_module, "get_magi2_ep_group", lambda: tp_group)

    target = Magi2PreviewTransformer(_tiny_config())
    loaded = target.load_weights(checkpoint)

    assert loaded == set(target.state_dict())
    source_params = dict(source.state_dict())
    target_params = dict(target.state_dict())
    qkv_name = "block.layers.0.attention.linear_qkv.weight"
    qkv = target.block.layers[0].attention.linear_qkv
    torch.testing.assert_close(target_params[qkv_name], qkv.shard_checkpoint_weight(source_params[qkv_name]))
    sink_name = "block.layers.0.attention.sinks"
    torch.testing.assert_close(target_params[sink_name], source_params[sink_name][:, 1:2])
    moe_name = "block.layers.0.mlp.moe_mlp.W_down"
    moe = target.block.layers[0].mlp.moe_mlp
    torch.testing.assert_close(target_params[moe_name], moe.ep_slice(source_params[moe_name]))


def test_tp_local_moe_heads_reconstruct_single_rank_oracle() -> None:
    config = Magi2MultiHeadMoEConfig(
        hidden_size=8,
        num_heads=2,
        num_experts=3,
        top_k=2,
        expert_intermediate_size=5,
        params_dtype=torch.float32,
    )
    oracle = Magi2MultiHeadMoE(config, ep_group=_group(1, 0))
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        for parameter in oracle.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    inputs = torch.randn(7, 8, generator=generator)
    expected = oracle(inputs)

    local_outputs = []
    oracle_parameters = dict(oracle.named_parameters())
    for rank in range(2):
        local = Magi2MultiHeadMoE(config, ep_group=_group(2, rank))
        with torch.no_grad():
            for name, parameter in local.named_parameters():
                parameter.copy_(local.ep_slice(oracle_parameters[name]))
        local_input = inputs.view(7, 2, 4)[:, rank].contiguous()
        local_outputs.append(local(local_input))

    actual = torch.stack(local_outputs, dim=1).reshape_as(expected)
    torch.testing.assert_close(actual, expected)
