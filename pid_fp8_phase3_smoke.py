#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Phase 3 smoke test for PiD online FP8 quantization.

Scope
-----
PiD only:
    - no Qwen-Image pipeline
    - no Gemma text encoder
    - no real prompt
    - no real Qwen latent

What this test proves
---------------------
1. build_quant_config("fp8") returns the expected online FP8 config.
2. Every migrated PiD FeedForward linear resolves to
   Fp8PerTensorOnlineLinearMethod.
3. Online FP8 weights start on meta device.
4. PiD production checkpoint loader goes through parameter.weight_loader.
5. BF16 checkpoint weights are really converted into FP8 weights.
6. weight_scale is generated.
7. A real FP8 linear kernel is selected.
8. A direct ReplicatedLinear FP8 forward succeeds.
9. The complete PiD 4-step SDE forward succeeds with finite output.

Recommended hardware
--------------------
Ada / Hopper or newer NVIDIA GPU, e.g. RTX 4090 / L40S / H100.

Usage
-----
python pid_fp8_phase3_smoke.py \
    --checkpoint /path/to/model_ema_bf16.pth

For debugging on hardware where vLLM selects Marlin fallback:

python pid_fp8_phase3_smoke.py \
    --checkpoint /path/to/model_ema_bf16.pth \
    --allow-marlin
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections import Counter
from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.kernels.linear.scaled_mm import (
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerTensorOnlineLinearMethod,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.pid.checkpoint import load_pid_checkpoint
from vllm_omni.diffusion.pid.config import (
    PID_SAMPLING_CONFIG,
    get_pid_net_config,
)
from vllm_omni.diffusion.pid.pid_net import PidNet
from vllm_omni.diffusion.pid.pixeldit import FeedForward
from vllm_omni.quantization import build_quant_config


# ---------------------------------------------------------------------------
# Small wrapper because load_pid_checkpoint() expects model.net
# ---------------------------------------------------------------------------


class PidNetOnlyWrapper(nn.Module):
    def __init__(self, net: PidNet):
        super().__init__()
        self.net = net


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PiD Phase-3 FP8 GPU smoke test",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PiD model_ema_bf16.pth",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Synthetic PiD output image size. 128 is enough for the smoke test.",
    )

    parser.add_argument(
        "--text-len",
        type=int,
        default=8,
        help="Synthetic caption embedding sequence length.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--allow-marlin",
        action="store_true",
        help=(
            "Allow Marlin FP8 fallback. By default the test requires "
            "a native non-Marlin FP8 kernel."
        ),
    )

    parser.add_argument(
        "--skip-sampler",
        action="store_true",
        help="Stop after direct FP8 ReplicatedLinear forward.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def init_single_rank_vllm() -> None:
    """
    vLLM parameters record TP rank/world-size during construction, so even
    ReplicatedLinear needs a valid single-rank vLLM model-parallel environment.
    """
    if torch.distributed.is_initialized():
        return

    init_file = tempfile.mkstemp(prefix="pid_fp8_dist_")[1]

    backend = current_platform.dist_backend

    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{init_file}",
        local_rank=0,
        backend=backend,
    )

    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )


def make_vllm_config() -> VllmConfig:
    """
    Online FP8 reads get_current_vllm_config().model_config.dtype while the
    linear method/kernel is being constructed.

    We do not need a full HF ModelConfig for this isolated PiD smoke test.
    """
    config = VllmConfig()

    config.model_config = SimpleNamespace(
        dtype=torch.bfloat16,
    )

    return config


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------


def collect_feedforward_fp8_linears(
        net: PidNet,
) -> list[tuple[str, ReplicatedLinear]]:
    """
    Verify every migrated FeedForward contains:
        w1 / w2 / w3
            -> ReplicatedLinear
            -> Fp8PerTensorOnlineLinearMethod
    """

    fp8_linears: list[tuple[str, ReplicatedLinear]] = []
    ff_count = 0

    for ff_name, module in net.named_modules():
        if not isinstance(module, FeedForward):
            continue

        ff_count += 1

        for attr in ("w1", "w3", "w2"):
            layer = getattr(module, attr)

            assert isinstance(layer, ReplicatedLinear), (
                f"{ff_name}.{attr}: expected ReplicatedLinear, "
                f"got {type(layer).__name__}"
            )

            method = layer.quant_method

            assert isinstance(method, Fp8PerTensorOnlineLinearMethod), (
                f"{ff_name}.{attr}: expected "
                "Fp8PerTensorOnlineLinearMethod, "
                f"got {type(method).__name__}"
            )

            fp8_linears.append(
                (f"{ff_name}.{attr}", layer)
            )

    assert ff_count > 0, "No FeedForward modules found."
    assert len(fp8_linears) == ff_count * 3

    print(f"      FeedForward modules: {ff_count}")
    print(f"      FP8 ReplicatedLinear: {len(fp8_linears)}")

    return fp8_linears


def verify_preload_state(
        fp8_linears: list[tuple[str, ReplicatedLinear]],
) -> None:
    """
    Online FP8 weights should be meta tensors before checkpoint loading.
    """

    non_meta = []

    for name, layer in fp8_linears:
        if layer.weight.device.type != "meta":
            non_meta.append(
                (
                    name,
                    str(layer.weight.device),
                    str(layer.weight.dtype),
                )
            )

    assert not non_meta, (
        "Online FP8 weights were expected to start on meta device, "
        f"but found non-meta layers: {non_meta[:5]}"
    )

    sample_name, sample_layer = fp8_linears[0]

    print(f"      sample layer: {sample_name}")
    print(f"      pre-load weight device: {sample_layer.weight.device}")
    print(f"      quant method: {type(sample_layer.quant_method).__name__}")


def verify_postload_state(
        fp8_linears: list[tuple[str, ReplicatedLinear]],
        device: torch.device,
        allow_marlin: bool,
) -> None:
    """
    This is the core Phase-3 check.

    After production checkpoint loading:
        meta BF16/FP32 logical weight
            ->
        actual CUDA FP8 qweight
            +
        weight_scale
            +
        selected FP8 kernel
    """

    expected_fp8_dtype = current_platform.fp8_dtype()

    failures: list[str] = []
    kernel_counter: Counter[str] = Counter()

    for name, layer in fp8_linears:
        method = layer.quant_method

        # 1. Must no longer be meta.
        if layer.weight.device.type == "meta":
            failures.append(
                f"{name}: weight is still meta after checkpoint loading"
            )
            continue

        # 2. Must live on the target CUDA device.
        if layer.weight.device != device:
            failures.append(
                f"{name}: weight device={layer.weight.device}, "
                f"expected={device}"
            )

        # 3. Must really be FP8.
        if layer.weight.dtype != expected_fp8_dtype:
            failures.append(
                f"{name}: weight dtype={layer.weight.dtype}, "
                f"expected={expected_fp8_dtype}"
            )

        # 4. Online quantization must generate weight_scale.
        if not hasattr(layer, "weight_scale"):
            failures.append(
                f"{name}: missing weight_scale"
            )
        else:
            weight_scale = layer.weight_scale

            if weight_scale.device.type == "meta":
                failures.append(
                    f"{name}: weight_scale is still meta"
                )
            elif not torch.isfinite(weight_scale.float()).all():
                failures.append(
                    f"{name}: weight_scale contains NaN/Inf"
                )

        # 5. There must be a real kernel object.
        fp8_linear = getattr(method, "fp8_linear", None)

        if fp8_linear is None:
            failures.append(
                f"{name}: quant_method.fp8_linear is None"
            )
            continue

        kernel_name = type(fp8_linear).__name__
        kernel_counter[kernel_name] += 1

        # For the main Phase-3 test we want actual W8A8 FP8 hardware path,
        # not the Marlin fallback used when native FP8 support is unavailable.
        if (
                not allow_marlin
                and isinstance(fp8_linear, MarlinFP8ScaledMMLinearKernel)
        ):
            failures.append(
                f"{name}: selected Marlin fallback kernel "
                f"({kernel_name}); use a native-FP8 GPU or pass "
                "--allow-marlin for fallback-only debugging."
            )

    if failures:
        preview = "\n".join(
            f"        - {x}" for x in failures[:20]
        )
        raise AssertionError(
            "FP8 post-load verification failed:\n"
            f"{preview}"
        )

    sample_name, sample_layer = fp8_linears[0]
    sample_method = sample_layer.quant_method

    print(f"      expected FP8 dtype: {expected_fp8_dtype}")
    print(f"      sample layer: {sample_name}")
    print(f"      post-load weight device: {sample_layer.weight.device}")
    print(f"      post-load weight dtype: {sample_layer.weight.dtype}")
    print(
        "      weight_scale: "
        f"shape={tuple(sample_layer.weight_scale.shape)}, "
        f"dtype={sample_layer.weight_scale.dtype}, "
        f"device={sample_layer.weight_scale.device}"
    )
    print(
        "      kernel: "
        f"{type(sample_method.fp8_linear).__name__}"
    )
    print(f"      kernel distribution: {dict(kernel_counter)}")


# ---------------------------------------------------------------------------
# Direct FP8 Linear forward
# ---------------------------------------------------------------------------


@torch.inference_mode()
def run_direct_fp8_linear_probe(
        fp8_linears: list[tuple[str, ReplicatedLinear]],
        device: torch.device,
) -> None:
    name, layer = fp8_linears[0]

    x = torch.randn(
        4,
        layer.input_size,
        device=device,
        dtype=torch.bfloat16,
    )

    y = layer(x)

    torch.cuda.synchronize(device)

    assert y.shape == (4, layer.output_size), (
        f"Unexpected output shape: {tuple(y.shape)}"
    )

    assert torch.isfinite(y.float()).all(), (
        f"{name}: direct FP8 output contains NaN/Inf"
    )

    print(f"      layer: {name}")
    print(f"      input:  {tuple(x.shape)} {x.dtype}")
    print(f"      output: {tuple(y.shape)} {y.dtype}")
    print(
        "      output stats: "
        f"min={y.float().min().item():.6f}, "
        f"max={y.float().max().item():.6f}, "
        f"mean={y.float().mean().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Exact PiD 4-step distilled SDE smoke loop
# ---------------------------------------------------------------------------


def net_output_to_x0(
        x_t: torch.Tensor,
        net_output: torch.Tensor,
        t: torch.Tensor,
) -> torch.Tensor:
    """
    PiD prediction_type='velocity':

        x0 = x_t - t * velocity

    Use double intermediate exactly like PidInferenceModel.
    """
    shape = [x_t.shape[0]] + [1] * (x_t.ndim - 1)
    t_shaped = t.double().view(*shape)

    return (
            x_t.double()
            - t_shaped * net_output.double()
    ).to(x_t.dtype)


@torch.inference_mode()
def run_four_step_sampler(
        net: PidNet,
        device: torch.device,
        image_size: int,
        text_len: int,
        batch_size: int,
        seed: int,
) -> torch.Tensor:
    net_config = get_pid_net_config("qwenimage")

    patch_size = net_config["patch_size"]
    sr_scale = net_config["sr_scale"]
    latent_down_factor = net_config["latent_spatial_down_factor"]
    lq_channels = net_config["lq_latent_channels"]
    text_dim = net_config["txt_embed_dim"]

    assert image_size % patch_size == 0, (
        f"image_size={image_size} must be divisible by "
        f"patch_size={patch_size}"
    )

    total_lq_downsample = sr_scale * latent_down_factor

    assert image_size % total_lq_downsample == 0, (
        f"image_size={image_size} must be divisible by "
        f"sr_scale * latent_down_factor={total_lq_downsample}"
    )

    lq_h = image_size // total_lq_downsample
    lq_w = image_size // total_lq_downsample

    generator = torch.Generator(
        device=device,
    ).manual_seed(seed)

    # PidInferenceModel uses a floating image/noise container and BF16
    # matmuls under autocast.
    noise = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    caption_embs = torch.randn(
        batch_size,
        text_len,
        text_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )

    lq_latent = torch.randn(
        batch_size,
        lq_channels,
        lq_h,
        lq_w,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )

    degrade_sigma = torch.zeros(
        batch_size,
        device=device,
        dtype=torch.bfloat16,
    )

    t_list = torch.tensor(
        PID_SAMPLING_CONFIG["student_t_list"],
        device=device,
        dtype=torch.float32,
    )

    expected_steps = PID_SAMPLING_CONFIG["student_sample_steps"]

    assert len(t_list) == expected_steps + 1, (
        f"Expected {expected_steps + 1} timestep boundaries, "
        f"got {len(t_list)}"
    )

    assert (
            PID_SAMPLING_CONFIG["prediction_type"]
            == "velocity"
    )

    assert (
            PID_SAMPLING_CONFIG["student_sample_type"]
            == "sde"
    )

    timescale = float(PID_SAMPLING_CONFIG["fm_timescale"])

    print(f"      noise:        {tuple(noise.shape)} {noise.dtype}")
    print(
        f"      caption_embs: {tuple(caption_embs.shape)} "
        f"{caption_embs.dtype}"
    )
    print(
        f"      lq_latent:    {tuple(lq_latent.shape)} "
        f"{lq_latent.dtype}"
    )
    print(f"      timesteps:    {t_list.tolist()}")

    x = noise

    # This mirrors PidInferenceModel._sample_loop().
    with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
    ):
        for step_index, (t_cur, t_next) in enumerate(
                zip(t_list[:-1], t_list[1:]),
                start=1,
        ):
            t_cur_batch = t_cur.expand(batch_size)
            t_scaled = t_cur_batch * timescale

            v_pred = net(
                x,
                t_scaled,
                caption_embs,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma,
            )

            if not torch.isfinite(v_pred.float()).all():
                raise AssertionError(
                    f"Step {step_index}: v_pred contains NaN/Inf"
                )

            if t_next.item() > 0:
                x0_pred = net_output_to_x0(
                    x,
                    v_pred,
                    t_cur_batch,
                )

                eps_infer = torch.randn(
                    x0_pred.shape,
                    device=x0_pred.device,
                    dtype=x0_pred.dtype,
                    generator=generator,
                )

                broadcast_shape = [
                                      batch_size
                                  ] + [1] * (x.ndim - 1)

                t_next_bcast = (
                    t_next.reshape(1)
                    .expand(broadcast_shape)
                )

                x = (
                        (1.0 - t_next_bcast) * x0_pred
                        + t_next_bcast * eps_infer
                )

            else:
                x = net_output_to_x0(
                    x,
                    v_pred,
                    t_cur_batch,
                )

            if not torch.isfinite(x.float()).all():
                raise AssertionError(
                    f"Step {step_index}: sample contains NaN/Inf"
                )

            torch.cuda.synchronize(device)

            x_float = x.float()

            print(
                f"      step {step_index}/{expected_steps}: "
                f"t={t_cur.item():.3f}"
                f" -> {t_next.item():.3f}, "
                f"v_dtype={v_pred.dtype}, "
                f"x_dtype={x.dtype}, "
                f"min={x_float.min().item():.5f}, "
                f"max={x_float.max().item():.5f}, "
                f"mean={x_float.mean().item():.5f}"
            )

    expected_shape = (
        batch_size,
        3,
        image_size,
        image_size,
    )

    assert x.shape == expected_shape, (
        f"Final shape={tuple(x.shape)}, "
        f"expected={expected_shape}"
    )

    assert torch.isfinite(x.float()).all(), (
        "Final PiD output contains NaN/Inf"
    )

    return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Phase 3 requires an NVIDIA CUDA GPU."
        )

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    device = torch.device(
        f"cuda:{args.device}"
    )

    torch.cuda.set_device(device)

    props = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)

    print("=" * 72)
    print("PiD Phase 3 - online FP8 GPU smoke test")
    print("=" * 72)

    print("[0/8] CUDA environment")
    print(f"      device: {device}")
    print(f"      GPU: {props.name}")
    print(
        "      compute capability: "
        f"{capability[0]}.{capability[1]}"
    )
    print(f"      torch: {torch.__version__}")
    print(f"      checkpoint: {args.checkpoint}")

    # ------------------------------------------------------------------
    # vLLM config must already exist when FP8 methods are constructed.
    # ------------------------------------------------------------------

    vllm_config = make_vllm_config()

    with set_current_vllm_config(vllm_config):
        print("[1/8] Initializing single-rank vLLM environment")

        init_single_rank_vllm()

        print(
            f"      torch.distributed initialized: "
            f"{torch.distributed.is_initialized()}"
        )

        if torch.distributed.is_initialized():
            print(
                f"      backend: "
                f"{torch.distributed.get_backend()}"
            )
            print(
                f"      rank/world-size: "
                f"{torch.distributed.get_rank()}/"
                f"{torch.distributed.get_world_size()}"
            )

        # --------------------------------------------------------------
        # Build formal FP8 config.
        # --------------------------------------------------------------

        print("[2/8] Building formal FP8 quantization config")

        quant_config = build_quant_config("fp8")

        assert quant_config is not None
        assert quant_config.get_name() == "fp8"

        # This must be online FP8: our checkpoint itself is BF16.
        assert not getattr(
            quant_config,
            "is_checkpoint_fp8_serialized",
            False,
        )

        print(
            f"      config type: "
            f"{type(quant_config).__name__}"
        )
        print(
            f"      quant name: "
            f"{quant_config.get_name()}"
        )
        print(
            "      checkpoint serialized FP8: "
            f"{getattr(quant_config, 'is_checkpoint_fp8_serialized', None)}"
        )

        # --------------------------------------------------------------
        # Match the formal diffusion model-loader construction contract:
        #
        #   set_default_torch_dtype(od_config.dtype)
        #   with target_device:
        #       construct model
        #
        # This is important for online FP8 because its logical weights are
        # created on meta device and remember where they should later be
        # materialized.
        # --------------------------------------------------------------

        print("[3/8] Building Qwen-Image PiD PidNet on CUDA")

        net_config = get_pid_net_config("qwenimage")

        torch.cuda.reset_peak_memory_stats(device)

        with set_default_torch_dtype(torch.bfloat16):
            with device:
                net = PidNet(
                    **net_config,
                    quant_config=quant_config,
                )

        wrapper = PidNetOnlyWrapper(net)
        net.eval()

        parameter_count = sum(
            p.numel()
            for p in net.parameters()
        )

        print(
            f"      parameters: "
            f"{parameter_count:,}"
        )

        fp8_linears = collect_feedforward_fp8_linears(net)

        # Current PiD architecture should naturally produce 84:
        # 28 FeedForward modules * 3 linear layers.
        #
        # We intentionally derive this rather than hard-code 84 so the
        # test remains meaningful if PiD architecture config changes.
        print(
            f"      expected from FeedForward graph: "
            f"{len(fp8_linears)}"
        )

        verify_preload_state(fp8_linears)

        # --------------------------------------------------------------
        # This is where online FP8 should actually happen.
        # --------------------------------------------------------------

        print("[4/8] Loading BF16 checkpoint through PiD production loader")

        with set_default_torch_dtype(torch.bfloat16):
            load_pid_checkpoint(
                wrapper,
                args.checkpoint,
            )

        net.eval()

        torch.cuda.synchronize(device)

        print(
            "      GPU memory allocated: "
            f"{torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GiB"
        )
        print(
            "      GPU peak allocated:   "
            f"{torch.cuda.max_memory_allocated(device) / 1024 ** 3:.2f} GiB"
        )

        # --------------------------------------------------------------
        # Confirm real FP8 storage + scales + kernel.
        # --------------------------------------------------------------

        print("[5/8] Verifying post-load FP8 state")

        verify_postload_state(
            fp8_linears,
            device=device,
            allow_marlin=args.allow_marlin,
        )

        # --------------------------------------------------------------
        # Isolate one ReplicatedLinear before testing whole PidNet.
        # --------------------------------------------------------------

        print("[6/8] Running direct FP8 ReplicatedLinear forward")

        run_direct_fp8_linear_probe(
            fp8_linears,
            device,
        )

        if args.skip_sampler:
            print("[7/8] 4-step sampler skipped by --skip-sampler")
        else:
            # ----------------------------------------------------------
            # Full PiD forward, exactly the part Phase 3 cares about.
            # ----------------------------------------------------------

            print("[7/8] Running complete 4-step PiD SDE forward")

            output = run_four_step_sampler(
                net=net,
                device=device,
                image_size=args.image_size,
                text_len=args.text_len,
                batch_size=args.batch_size,
                seed=args.seed,
            )

            output_float = output.float()

            print(
                "      final output: "
                f"shape={tuple(output.shape)}, "
                f"dtype={output.dtype}"
            )

            print(
                "      final stats: "
                f"min={output_float.min().item():.6f}, "
                f"max={output_float.max().item():.6f}, "
                f"mean={output_float.mean().item():.6f}"
            )

        print("[8/8] PASS")

        print()
        print("=" * 72)
        print("Phase 3 FP8 path verified:")
        print("  FP8 config")
        print("    -> FeedForward ReplicatedLinear")
        print("    -> Fp8PerTensorOnlineLinearMethod")
        print("    -> production checkpoint weight_loader")
        print("    -> FP8 weight + weight_scale")
        print("    -> real FP8 kernel")
        print("    -> ReplicatedLinear.forward")
        print("    -> complete PiD 4-step forward")
        print("=" * 72)


if __name__ == "__main__":
    main()
