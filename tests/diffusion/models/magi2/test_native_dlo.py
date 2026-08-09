# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

import vllm_omni.diffusion.models.magi2.mh_moe as mh_moe_module
from vllm_omni.diffusion.models.magi2.configuration_magi2 import (
    Magi2MHCConfig,
    Magi2MoEConfig,
    Magi2PreviewConfig,
)
from vllm_omni.diffusion.models.magi2.mh_moe import (
    Magi2MultiHeadMoE,
    Magi2MultiHeadMoEConfig,
)
from vllm_omni.diffusion.models.magi2.modeling_magi2 import (
    Magi2PreviewTransformer,
)
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup
from vllm_omni.diffusion.models.magi2.pipeline_magi2 import (
    Magi2Pipeline,
    _validate_native_topology,
)
from vllm_omni.diffusion.offloader.block_discovery import get_blocks_from_dit
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
    DistributedLayerwiseOffloadBackend,
)
from vllm_omni.diffusion.offloader.offload_plan import supports_mmap_loading

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


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


def _topology_config(
    *,
    dp: int,
    tp: int,
    sp: int,
    distributed_offload: bool,
    dlo_allgather: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=dp,
            tensor_parallel_size=tp,
            sequence_parallel_size=sp,
            ulysses_degree=sp,
            ring_degree=1,
            allgather_degree=1,
            cfg_parallel_size=1,
            vae_patch_parallel_size=1,
            text_encoder_tp_size=1,
            enable_expert_parallel=False,
            use_hsdp=False,
        ),
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=distributed_offload,
        dlo_use_allgather=dlo_allgather,
        quantization_config=None,
        cache_backend="none",
        custom_pipeline_args={},
        additional_config={},
    )


@pytest.mark.parametrize(
    ("checkpoint_key", "model_key"),
    (
        (
            "block.layers.0.attention.to_qkv.weight",
            "transformer.block.layers.0.attention.to_qkv.weight",
        ),
        ("pre_adapter.rope.bands", "transformer.pre_adapter.rope.bands"),
        (
            "post_adapter.final_linear_video.weight",
            "transformer.post_adapter.final_linear_video.weight",
        ),
        ("text_encoder.layers.0.weight", None),
    ),
)
def test_mmap_checkpoint_keys_remap_to_the_registered_transformer_namespace(
    checkpoint_key: str,
    model_key: str | None,
) -> None:
    assert Magi2Pipeline._remap_ckpt_key(checkpoint_key) == model_key


def test_mmap_contract_covers_every_persistent_checkpoint_tensor() -> None:
    transformer = Magi2PreviewTransformer(_tiny_config())
    pipeline = Magi2Pipeline.__new__(Magi2Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = transformer

    parameter_names = {f"transformer.{name}" for name, _ in transformer.named_parameters()}
    remapped_names = {Magi2Pipeline._remap_ckpt_key(checkpoint_key) for checkpoint_key in transformer.state_dict()}

    assert supports_mmap_loading(pipeline)
    assert None not in remapped_names
    assert remapped_names <= parameter_names


def test_checkpoint_only_tensors_are_nontrainable_parameters() -> None:
    transformer = Magi2PreviewTransformer(_tiny_config())
    parameters = dict(transformer.named_parameters())
    buffers = dict(transformer.named_buffers())
    checkpoint_only_names = {
        "pre_adapter.rope.bands",
        "block.layers.0.mlp.moe_mlp.router.expert_bias",
        "block.layers.0.mlp.moe_mlp.router.expert_bias_ema",
    }

    assert checkpoint_only_names <= parameters.keys()
    assert checkpoint_only_names.isdisjoint(buffers)
    assert all(not parameters[name].requires_grad for name in checkpoint_only_names)


def test_mmap_block_discovery_preserves_the_released_block_hierarchy() -> None:
    transformer = Magi2PreviewTransformer(_tiny_config())

    attr_names, blocks = get_blocks_from_dit(transformer)

    assert attr_names == ["block"]
    assert blocks == list(transformer.block.layers)
    assert "layers" not in transformer._modules
    assert any(name.startswith("block.layers.0.") for name, _ in transformer.named_parameters())


def test_ep_sliced_parameters_expose_and_preserve_the_mmap_transform() -> None:
    moe = Magi2MultiHeadMoE(
        Magi2MultiHeadMoEConfig(
            hidden_size=6,
            num_heads=3,
            num_experts=2,
            top_k=1,
            expert_intermediate_size=4,
            params_dtype=torch.float32,
        ),
        ep_group=Magi2ParallelGroup(None, world_size=2, rank=1),
    )
    parameters = dict(moe.named_parameters())
    backend = object.__new__(DistributedLayerwiseOffloadBackend)
    backend._remember_mmap_param_attrs(moe)

    for name in sorted(moe._EP_SHARDED_PARAMETER_NAMES):
        parameter = parameters[name]
        transform = getattr(parameter, "mmap_weight_transform", None)
        assert callable(transform), name

        full_shape = (moe.num_heads * moe.num_experts, *parameter.shape[1:])
        checkpoint_tensor = torch.arange(
            math.prod(full_shape),
            dtype=parameter.dtype,
        ).reshape(full_shape)
        torch.testing.assert_close(transform(checkpoint_tensor), moe.ep_slice(checkpoint_tensor))

        replacement = nn.Parameter(
            torch.empty_like(parameter),
            requires_grad=parameter.requires_grad,
        )
        backend._attach_mmap_param_attrs(name, replacement)
        assert replacement.mmap_weight_transform is transform
        assert replacement.mmap_weight_transform_pending is True


def test_generic_mmap_loader_uses_the_real_block_path_and_keeps_ep_slicing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Magi2PreviewTransformer(_tiny_config())
    weights = {name: tensor.detach().cpu().contiguous() for name, tensor in source.state_dict().items()}
    preview_dir = tmp_path / "preview"
    preview_dir.mkdir()
    weight_file = preview_dir / "model.safetensors"
    save_file(weights, str(weight_file))
    (preview_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: weight_file.name for name in weights}})
    )

    ep_group = Magi2ParallelGroup(None, world_size=2, rank=1)
    monkeypatch.setattr(mh_moe_module, "get_magi2_ep_group", lambda: ep_group)
    with torch.device("meta"):
        target = Magi2PreviewTransformer(_tiny_config())
    pipeline = Magi2Pipeline.__new__(Magi2Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.checkpoint_root = str(tmp_path)
    pipeline.transformer = target

    moe_parameter_name = "block.layers.0.mlp.moe_mlp.gate"
    expected_local_shape = tuple(dict(target.named_parameters())[moe_parameter_name].shape)
    backend = object.__new__(DistributedLayerwiseOffloadBackend)
    backend.config = SimpleNamespace(model_path=str(tmp_path))
    backend._load_weights_via_mmap(
        pipeline,
        SimpleNamespace(dits=[target], dit_names=["transformer"]),
    )

    assert all(not parameter.is_meta for parameter in target.parameters())
    loaded = dict(target.named_parameters())[moe_parameter_name]
    assert tuple(loaded.shape) == tuple(weights[moe_parameter_name].shape)
    assert loaded.mmap_expected_shape == expected_local_shape
    assert loaded.mmap_weight_transform_pending is True
    assert tuple(loaded.mmap_weight_transform(loaded).shape) == expected_local_shape
    assert backend._mmap_model_to_ckpt[f"transformer.{moe_parameter_name}"][0] == moe_parameter_name


def test_mmap_loader_uses_pipeline_resolved_checkpoint_root(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Magi2PreviewTransformer(_tiny_config())
    weights = {name: tensor.detach().cpu().contiguous() for name, tensor in source.state_dict().items()}
    preview_dir = tmp_path / "preview"
    preview_dir.mkdir()
    weight_file = preview_dir / "model.safetensors"
    save_file(weights, str(weight_file))
    (preview_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: weight_file.name for name in weights}})
    )

    monkeypatch.setattr(
        mh_moe_module,
        "get_magi2_ep_group",
        lambda: Magi2ParallelGroup(None, world_size=1, rank=0),
    )
    with torch.device("meta"):
        target = Magi2PreviewTransformer(_tiny_config())
    pipeline = Magi2Pipeline.__new__(Magi2Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.checkpoint_root = str(tmp_path)
    pipeline.transformer = target

    backend = object.__new__(DistributedLayerwiseOffloadBackend)
    # This deliberately cannot be resolved or downloaded. The pipeline root
    # is authoritative because it already incorporates URL normalization and
    # the requested/pinned Hugging Face revision.
    backend.config = SimpleNamespace(model_path="invalid-owner/invalid-model")
    backend._load_weights_via_mmap(
        pipeline,
        SimpleNamespace(dits=[target], dit_names=["transformer"]),
    )

    assert all(not parameter.is_meta for parameter in target.parameters())


@pytest.mark.parametrize(
    ("dp", "tp", "sp", "distributed_offload", "dlo_allgather"),
    (
        pytest.param(1, 4, 1, False, True, id="tp4"),
        pytest.param(1, 2, 2, False, True, id="tp2-sp2"),
        pytest.param(1, 1, 4, False, True, id="sp4"),
        pytest.param(4, 1, 1, True, True, id="dlo-dp4-allgather"),
        pytest.param(2, 1, 2, True, True, id="dlo-dp2-sp2-allgather"),
        pytest.param(1, 1, 4, True, False, id="dlo-sp4-rank-local"),
    ),
)
def test_requested_four_worker_topologies_are_accepted(
    monkeypatch: pytest.MonkeyPatch,
    dp: int,
    tp: int,
    sp: int,
    distributed_offload: bool,
    dlo_allgather: bool,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    config = _topology_config(
        dp=dp,
        tp=tp,
        sp=sp,
        distributed_offload=distributed_offload,
        dlo_allgather=dlo_allgather,
    )

    _validate_native_topology(config)


@pytest.mark.parametrize(
    ("config", "message"),
    (
        pytest.param(
            _topology_config(dp=4, tp=1, sp=1, distributed_offload=False, dlo_allgather=True),
            "data parallelism currently requires distributed layerwise offload",
            id="dp-needs-dlo",
        ),
        pytest.param(
            _topology_config(dp=2, tp=2, sp=1, distributed_offload=True, dlo_allgather=True),
            "require tensor_parallel_size=1",
            id="dlo-dp-and-tp-cannot-mix",
        ),
        pytest.param(
            _topology_config(dp=1, tp=1, sp=4, distributed_offload=True, dlo_allgather=True),
            "dlo-no-use-allgather",
            id="sp-shards-cannot-allgather",
        ),
    ),
)
def test_invalid_dlo_topologies_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    config: SimpleNamespace,
    message: str,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(ValueError, match=message):
        _validate_native_topology(config)
