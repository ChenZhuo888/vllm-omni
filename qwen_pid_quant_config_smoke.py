#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image + PiD quantization-config routing smoke test.

This intentionally bypasses:
- full QwenImagePipeline construction
- tokenizer / text encoder / VAE
- PiD checkpoint loading
- Gemma
- any real forward pass

It only verifies the quantization-config path we are changing:

1. initialize a minimal single-rank vLLM environment;
2. build real vLLM-Omni QuantizationConfig / ComponentQuantizationConfig objects;
3. resolve the Qwen-Image "transformer" and "pid" component configs using the
   intended Qwen-Image pipeline contract;
4. construct a tiny real QwenImageTransformer2DModel and inspect one
   quantizable vLLM Linear;
5. construct a tiny real PiD FeedForward and inspect its ReplicatedLinear;
6. verify PiD ignored_layers still works by its internal layer prefix.

Run from the vllm-omni repository root:

    python tests/manual/qwen_pid_quant_config_smoke.py

No model checkpoint is required.
"""

from __future__ import annotations

import inspect
import os
import tempfile
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Iterator

import torch
import torch.distributed as dist

from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageTransformer2DModel,
)
from vllm_omni.diffusion.pid.pixeldit import FeedForward as PidFeedForward
from vllm_omni.quantization import (
    ComponentQuantizationConfig,
    build_quant_config,
)


def _maybe_vllm_config_context():
    """Provide the minimal vLLM config context required by model-parallel init."""
    try:
        from vllm.config import (
            DeviceConfig,
            VllmConfig,
            get_current_vllm_config_or_none,
            set_current_vllm_config,
        )
    except ImportError:
        return nullcontext()

    if get_current_vllm_config_or_none() is not None:
        return nullcontext()

    return set_current_vllm_config(
        VllmConfig(
            device_config=DeviceConfig(device="cpu"),
        )
    )


def _call_initialize_model_parallel() -> None:
    """Initialize TP=1 / PP=1 across vLLM API variants."""
    from vllm.distributed import initialize_model_parallel

    signature = inspect.signature(initialize_model_parallel)
    kwargs = {}

    if "tensor_model_parallel_size" in signature.parameters:
        kwargs["tensor_model_parallel_size"] = 1
    if "pipeline_model_parallel_size" in signature.parameters:
        kwargs["pipeline_model_parallel_size"] = 1

    initialize_model_parallel(**kwargs)


@contextmanager
def single_rank_vllm_environment() -> Iterator[None]:
    """Create the smallest vLLM runtime required by vLLM Linear layers."""
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        model_parallel_is_initialized,
    )

    created_dist = False
    created_model_parallel = False
    init_file_path: str | None = None

    with _maybe_vllm_config_context():
        try:
            if not dist.is_initialized():
                fd, init_file_path = tempfile.mkstemp(
                    prefix="vllm_qwen_pid_quant_config_"
                )
                os.close(fd)

                init_distributed_environment(
                    world_size=1,
                    rank=0,
                    distributed_init_method=f"file://{init_file_path}",
                    local_rank=0,
                    backend="gloo",
                )
                created_dist = True

            if not model_parallel_is_initialized():
                _call_initialize_model_parallel()
                created_model_parallel = True

            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()

            print("[0/5] Initializing single-rank vLLM environment")
            print(f"      torch.distributed initialized: {dist.is_initialized()}")
            print(
                "      backend:",
                dist.get_backend() if dist.is_initialized() else "N/A",
            )
            print(f"      TP rank/world-size: {tp_rank}/{tp_size}")

            if tp_rank != 0 or tp_size != 1:
                raise RuntimeError(
                    "This smoke test requires TP rank=0/world-size=1, "
                    f"got rank={tp_rank}, world-size={tp_size}"
                )

            yield

        finally:
            if created_model_parallel:
                destroy_model_parallel()
            if created_dist:
                destroy_distributed_environment()
            if init_file_path is not None:
                try:
                    os.unlink(init_file_path)
                except FileNotFoundError:
                    pass


def config_name(config) -> str:
    if config is None:
        return "None"
    return config.get_name()


def resolve_qwen_pid_configs(quant_config):
    """Expected Qwen-Image component-boundary routing contract.

    Normal QuantizationConfig:
        transformer <- config
        pid         <- None

    ComponentQuantizationConfig:
        transformer <- resolve("transformer")
        pid         <- resolve("pid")
    """
    if isinstance(quant_config, ComponentQuantizationConfig):
        return (
            quant_config.resolve("transformer"),
            quant_config.resolve("pid"),
        )

    return quant_config, None


def is_unquantized(layer) -> bool:
    return type(layer.quant_method).__name__ == "UnquantizedLinearMethod"


def print_layer(label: str, layer) -> None:
    print(f"      {label}")
    print(f"        prefix:       {layer.prefix}")
    print(f"        quant_config: {config_name(layer.quant_config)}")
    print(f"        quant_method: {type(layer.quant_method).__name__}")


def build_tiny_qwen_transformer(quant_config):
    """Build one real Qwen-Image transformer block, not the full 60-layer model."""
    parallel_config = SimpleNamespace(
        sequence_parallel_size=1,
        mask_sp_padding=False,
    )
    od_config = SimpleNamespace(
        parallel_config=parallel_config,
    )

    model = QwenImageTransformer2DModel(
        od_config=od_config,
        patch_size=2,
        in_channels=16,
        out_channels=16,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        guidance_embeds=False,
        axes_dims_rope=(4, 4, 8),
        quant_config=quant_config,
    )

    # This is an actually quantizable Linear. Entry/final projections are
    # intentionally quant_config=None in Qwen-Image and therefore are not
    # suitable for this check.
    layer = model.transformer_blocks[0].img_mlp.net[0].proj
    return model, layer


def build_tiny_pid_ffn(quant_config):
    """Build the real PiD FeedForward that contains migrated ReplicatedLinear."""
    module = PidFeedForward(
        dim=32,
        hidden_dim=64,
        quant_config=quant_config,
        prefix="patch_blocks.0.mlp_x",
    )
    return module


def assert_layer_state(
    *,
    case_name: str,
    component_name: str,
    layer,
    expect_quantized: bool,
) -> None:
    actual_quantized = not is_unquantized(layer)

    if actual_quantized != expect_quantized:
        raise RuntimeError(
            f"{case_name}: {component_name} quantization mismatch: "
            f"expected quantized={expect_quantized}, "
            f"got quant_method={type(layer.quant_method).__name__}, "
            f"prefix={layer.prefix!r}"
        )


def run_case(
    name: str,
    root_quant_config,
    *,
    expect_transformer_quantized: bool,
    expect_pid_quantized: bool,
) -> None:
    print(f"\n      case: {name}")
    print(f"        root config: {config_name(root_quant_config)}")

    transformer_quant_config, pid_quant_config = resolve_qwen_pid_configs(
        root_quant_config
    )

    print(
        "        resolved: "
        f"transformer={config_name(transformer_quant_config)}, "
        f"pid={config_name(pid_quant_config)}"
    )

    if isinstance(transformer_quant_config, ComponentQuantizationConfig):
        raise RuntimeError(
            f"{name}: transformer still received ComponentQuantizationConfig"
        )
    if isinstance(pid_quant_config, ComponentQuantizationConfig):
        raise RuntimeError(
            f"{name}: pid still received ComponentQuantizationConfig"
        )

    qwen_model, qwen_layer = build_tiny_qwen_transformer(
        transformer_quant_config
    )
    pid_ffn = build_tiny_pid_ffn(pid_quant_config)

    print_layer("Qwen-Image transformer Linear", qwen_layer)
    print_layer("PiD FeedForward w1", pid_ffn.w1)

    assert_layer_state(
        case_name=name,
        component_name="Qwen-Image transformer",
        layer=qwen_layer,
        expect_quantized=expect_transformer_quantized,
    )
    assert_layer_state(
        case_name=name,
        component_name="PiD",
        layer=pid_ffn.w1,
        expect_quantized=expect_pid_quantized,
    )

    del qwen_model
    del pid_ffn


def run_component_cases() -> None:
    print("[1/5] Building and routing quantization configs")

    cases = [
        (
            "none",
            None,
            False,
            False,
        ),
        (
            "plain-fp8",
            build_quant_config("fp8"),
            True,
            False,
        ),
        (
            "transformer-fp8-pid-none",
            build_quant_config(
                {
                    "transformer": {"method": "fp8"},
                    "pid": None,
                }
            ),
            True,
            False,
        ),
        (
            "transformer-none-pid-fp8",
            build_quant_config(
                {
                    "transformer": None,
                    "pid": {"method": "fp8"},
                }
            ),
            False,
            True,
        ),
        (
            "transformer-fp8-pid-fp8",
            build_quant_config(
                {
                    "transformer": {"method": "fp8"},
                    "pid": {"method": "fp8"},
                }
            ),
            True,
            True,
        ),
        (
            "default-fp8-explicit-pid-none",
            build_quant_config(
                {
                    "pid": None,
                    "default": {"method": "fp8"},
                }
            ),
            True,
            False,
        ),
        (
            "default-fp8-explicit-transformer-none",
            build_quant_config(
                {
                    "transformer": None,
                    "default": {"method": "fp8"},
                }
            ),
            False,
            True,
        ),
    ]

    print(f"      cases: {len(cases)}")

    print("[2/5] Constructing tiny real Qwen-Image/PiD modules per case")

    for (
        name,
        root_config,
        expect_transformer,
        expect_pid,
    ) in cases:
        run_case(
            name,
            root_config,
            expect_transformer_quantized=expect_transformer,
            expect_pid_quantized=expect_pid,
        )

    print("      component routing cases: PASS")


def verify_pid_ignored_layers() -> None:
    print("[3/5] Verifying PiD internal prefix-based ignored_layers")

    root_config = build_quant_config(
        {
            "transformer": None,
            "pid": {
                "method": "fp8",
                "ignored_layers": [
                    "patch_blocks.0.mlp_x.w1",
                ],
            },
        }
    )

    _, pid_quant_config = resolve_qwen_pid_configs(root_config)
    pid_ffn = build_tiny_pid_ffn(pid_quant_config)

    print_layer("PiD w1 (ignored)", pid_ffn.w1)
    print_layer("PiD w2 (not ignored)", pid_ffn.w2)
    print_layer("PiD w3 (not ignored)", pid_ffn.w3)

    if not is_unquantized(pid_ffn.w1):
        raise RuntimeError(
            "PiD w1 should be unquantized because its prefix is ignored"
        )
    if is_unquantized(pid_ffn.w2):
        raise RuntimeError(
            "PiD w2 should still be quantized"
        )
    if is_unquantized(pid_ffn.w3):
        raise RuntimeError(
            "PiD w3 should still be quantized"
        )

    print("      PiD prefix-based ignored_layers: PASS")


def verify_prefixes() -> None:
    print("[4/5] Verifying actual layer prefixes")

    fp8 = build_quant_config("fp8")

    qwen_model, qwen_layer = build_tiny_qwen_transformer(fp8)
    pid_ffn = build_tiny_pid_ffn(fp8)

    print(f"      Qwen prefix: {qwen_layer.prefix}")
    print(f"      PiD prefix:  {pid_ffn.w1.prefix}")

    if not qwen_layer.prefix.startswith("transformer_blocks."):
        raise RuntimeError(
            f"Unexpected Qwen-Image prefix: {qwen_layer.prefix!r}"
        )

    if not pid_ffn.w1.prefix.startswith("patch_blocks."):
        raise RuntimeError(
            f"Unexpected PiD prefix: {pid_ffn.w1.prefix!r}"
        )

    del qwen_model
    del pid_ffn

    print("      actual prefixes: PASS")


def run_test() -> None:
    run_component_cases()
    verify_pid_ignored_layers()
    verify_prefixes()

    print("[5/5] Final validation")
    print(
        "\nPASS: "
        "ComponentQuantizationConfig -> Qwen-Image component resolution / "
        "PiD component resolution -> concrete QuantizationConfig | None -> "
        "real vLLM Linear quant_method selection is connected."
    )


def main() -> None:
    torch.manual_seed(0)

    with single_rank_vllm_environment():
        run_test()


if __name__ == "__main__":
    main()
