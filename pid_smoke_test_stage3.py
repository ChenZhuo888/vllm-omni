#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PiD stage3 smoke test: ReplicatedLinear + production checkpoint loader.

This test intentionally bypasses:
- Qwen-Image
- Gemma
- PidInferenceModel

It verifies the PiD-only path:

1. create a minimal single-process vLLM distributed/model-parallel environment;
2. build the Qwen-Image PiD PidNet;
3. load the official PiD checkpoint through the production
   ``load_pid_checkpoint`` path;
4. verify a migrated ReplicatedLinear parameter owns a vLLM ``weight_loader``;
5. verify one ReplicatedLinear weight and the AdaLN checkpoint-key remapping;
6. feed random fake Gemma embeddings and random fake Qwen-Image latents;
7. run the official 4-step PiD SDE sampling loop;
8. verify every step and the final output are finite.

Run from the vllm-omni repository root so the local checkout is imported.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import os
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist

from vllm_omni.diffusion.pid.checkpoint import load_pid_checkpoint
from vllm_omni.diffusion.pid.config import (
    get_pid_net_config,
    get_pid_sampling_config,
)
from vllm_omni.diffusion.pid.pid_net import PidNet


class PidNetWrapper(torch.nn.Module):
    """Provide the ``.net`` attribute expected by load_pid_checkpoint."""

    def __init__(self, net: PidNet):
        super().__init__()
        self.net = net


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PiD stage3 ReplicatedLinear/checkpoint smoke test"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the PiD .pth checkpoint, e.g. model_ema_bf16.pth",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Execution device.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16"),
        default="auto",
        help="auto => float32 on CPU, bfloat16 on CUDA",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="PiD pixel-space output height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="PiD pixel-space output width",
    )
    parser.add_argument(
        "--text-length",
        type=int,
        default=8,
        help="Number of fake Gemma tokens. Small is enough for a smoke test.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_dtype(device: str, dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32 if device == "cpu" else torch.bfloat16


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.height <= 0 or args.width <= 0:
        raise ValueError("height/width must be positive")

    if args.height % 32 != 0 or args.width % 32 != 0:
        raise ValueError(
            "For the Qwen-Image PiD smoke test, height and width must be "
            "divisible by 32 (= sr_scale 4 * latent spatial down factor 8)."
        )

    if args.text_length <= 0 or args.text_length > 300:
        raise ValueError("text-length must be in [1, 300]")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but CUDA is not available"
        )


def _maybe_vllm_config_context():
    """Return a minimal vLLM config context when this vLLM version needs one.

    Newer vLLM versions require initialize_model_parallel() to run inside
    set_current_vllm_config(). Older versions do not. Keep this smoke test
    compatible with both.
    """
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
    """Initialize TP=1, PP=1 using the API supported by this vLLM checkout."""
    from vllm.distributed import initialize_model_parallel

    signature = inspect.signature(initialize_model_parallel)

    kwargs = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
    }
    if "prefill_context_model_parallel_size" in signature.parameters:
        kwargs["prefill_context_model_parallel_size"] = 1
    if "decode_context_model_parallel_size" in signature.parameters:
        kwargs["decode_context_model_parallel_size"] = 1
    if "backend" in signature.parameters:
        kwargs["backend"] = "gloo"

    initialize_model_parallel(**kwargs)


@contextmanager
def single_rank_vllm_environment() -> Iterator[None]:
    """Create the smallest vLLM runtime needed by BasevLLMParameter.

    ReplicatedLinear(disable_tp=True) itself does not shard through TP, but
    BasevLLMParameter still records TP rank/world-size during construction.
    Therefore this standalone test supplies a real rank-0/world-size-1
    distributed + model-parallel environment.
    """
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
                    prefix="vllm_pid_stage3_dist_"
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
            print(f"      backend: {dist.get_backend() if dist.is_initialized() else 'N/A'}")
            print(f"      TP rank/world-size: {tp_rank}/{tp_size}")

            if tp_rank != 0 or tp_size != 1:
                raise RuntimeError(
                    "Stage3 requires TP rank=0 and TP world-size=1, "
                    f"got rank={tp_rank}, world-size={tp_size}"
                )

            yield

        finally:
            # Only tear down resources this script created.
            if created_model_parallel:
                destroy_model_parallel()

            if created_dist:
                destroy_distributed_environment()

            if init_file_path is not None:
                try:
                    os.unlink(init_file_path)
                except FileNotFoundError:
                    pass


def assert_loaded_weight(
    net: PidNet,
    state_dict: dict[str, torch.Tensor],
    checkpoint_key: str,
    model_key: str,
) -> None:
    params = dict(net.named_parameters())

    if checkpoint_key not in state_dict:
        raise RuntimeError(
            f"Checkpoint key not found for verification: {checkpoint_key}"
        )

    if model_key not in params:
        raise RuntimeError(
            f"Model parameter not found for verification: {model_key}"
        )

    actual = params[model_key].detach().cpu()
    expected = state_dict[checkpoint_key].to(dtype=actual.dtype)

    try:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=0,
        )
    except AssertionError as exc:
        raise RuntimeError(
            "Loaded parameter does not exactly match checkpoint:\n"
            f"  checkpoint: {checkpoint_key}\n"
            f"  model:      {model_key}"
        ) from exc


def verify_checkpoint_loading(
    net: PidNet,
    checkpoint_path: Path,
) -> None:
    print(
        f"[2/5] Loading checkpoint through production loader: "
        f"{checkpoint_path}"
    )

    wrapper = PidNetWrapper(net)
    load_pid_checkpoint(wrapper, str(checkpoint_path))

    params = dict(net.named_parameters())

    replicated_weight_name = "patch_blocks.0.mlp_x.w1.weight"
    replicated_weight = params.get(replicated_weight_name)

    if replicated_weight is None:
        raise RuntimeError(
            f"Expected migrated parameter not found: {replicated_weight_name}"
        )

    has_weight_loader = hasattr(replicated_weight, "weight_loader")
    print(
        "      ReplicatedLinear parameter has weight_loader:",
        has_weight_loader,
    )

    if not has_weight_loader:
        raise RuntimeError(
            f"{replicated_weight_name} has no weight_loader; "
            "FeedForward may not be using ReplicatedLinear."
        )

    print(
        "      migrated parameter type:",
        type(replicated_weight).__name__,
    )

    # Reload the checkpoint only for exact-value verification.
    # The model itself was already loaded through load_pid_checkpoint above.
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # Verify one migrated ReplicatedLinear weight.
    assert_loaded_weight(
        net,
        state_dict,
        checkpoint_key="net.patch_blocks.0.mlp_x.w1.weight",
        model_key="patch_blocks.0.mlp_x.w1.weight",
    )
    print("      ReplicatedLinear weight exact match: PASS")

    # Verify official camel-case AdaLN checkpoint names were remapped to the
    # current snake-case model names by the production checkpoint loader.
    assert_loaded_weight(
        net,
        state_dict,
        checkpoint_key="net.patch_blocks.0.adaLN_modulation_img.0.weight",
        model_key="patch_blocks.0.ada_ln_modulation_img.0.weight",
    )
    print("      AdaLN image key remap exact match: PASS")

    assert_loaded_weight(
        net,
        state_dict,
        checkpoint_key="net.patch_blocks.0.adaLN_modulation_txt.0.weight",
        model_key="patch_blocks.0.ada_ln_modulation_txt.0.weight",
    )
    print("      AdaLN text key remap exact match: PASS")

    del state_dict
    del wrapper
    gc.collect()


def velocity_to_x0(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Convert velocity prediction to x0 using x0 = x_t - t * v."""
    shape = [x_t.shape[0]] + [1] * (x_t.ndim - 1)
    t_broadcast = t.double().view(*shape)

    return (
        x_t.double()
        - t_broadcast * velocity.double()
    ).to(x_t.dtype)


def run_four_step_sampler(
    net: PidNet,
    noise: torch.Tensor,
    caption_embs: torch.Tensor,
    lq_latent: torch.Tensor,
    degrade_sigma: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Run the same 4-step SDE structure used by PidInferenceModel."""

    sampling_config = get_pid_sampling_config()

    sample_type = sampling_config.get(
        "student_sample_type",
        "sde",
    )
    prediction_type = sampling_config.get(
        "prediction_type",
        "velocity",
    )

    if sample_type != "sde":
        raise ValueError(
            f"This smoke test expects student_sample_type='sde', "
            f"got {sample_type!r}"
        )

    if prediction_type != "velocity":
        raise ValueError(
            f"This smoke test currently expects prediction_type='velocity', "
            f"got {prediction_type!r}"
        )

    t_list = torch.tensor(
        sampling_config["student_t_list"],
        device=noise.device,
        dtype=torch.float32,
    )

    expected_steps = sampling_config["student_sample_steps"]
    actual_steps = len(t_list) - 1

    if actual_steps != expected_steps:
        raise RuntimeError(
            f"Sampling config mismatch: student_sample_steps={expected_steps}, "
            f"but student_t_list defines {actual_steps} steps"
        )

    timescale = float(sampling_config["fm_timescale"])

    generator = torch.Generator(device=noise.device)
    generator.manual_seed(seed)

    x = noise
    batch_size = x.shape[0]

    print(
        "      timestep schedule:",
        [float(v) for v in t_list.cpu()],
    )
    print(f"      fm_timescale: {timescale}")
    print(f"      sample type: {sample_type}")
    print(f"      prediction type: {prediction_type}")

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

        if not torch.isfinite(v_pred).all():
            raise RuntimeError(
                f"PidNet produced NaN/Inf at sampling step {step_index}"
            )

        x0_pred = velocity_to_x0(
            x_t=x,
            velocity=v_pred,
            t=t_cur_batch,
        )

        if not torch.isfinite(x0_pred).all():
            raise RuntimeError(
                f"x0 prediction produced NaN/Inf at sampling step {step_index}"
            )

        if t_next.item() > 0:
            eps_infer = torch.randn(
                x0_pred.shape,
                device=x0_pred.device,
                dtype=x0_pred.dtype,
                generator=generator,
            )

            shape = [batch_size] + [1] * (x.ndim - 1)
            t_next_broadcast = t_next.reshape(1).expand(shape)

            x = (
                (1.0 - t_next_broadcast) * x0_pred
                + t_next_broadcast * eps_infer
            )
        else:
            x = x0_pred

        is_finite = bool(torch.isfinite(x).all().item())

        print(
            f"      step {step_index}/{actual_steps}: "
            f"t={t_cur.item():.3f} -> {t_next.item():.3f}, "
            f"net_t={t_scaled[0].item():.1f}, "
            f"mean={x.float().mean().item():.6f}, "
            f"std={x.float().std().item():.6f}, "
            f"finite={is_finite}"
        )

        if not is_finite:
            raise RuntimeError(
                f"Sampler state contains NaN/Inf at step {step_index}"
            )

    return x


def run_test(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    dtype = resolve_dtype(
        args.device,
        args.dtype,
    )
    device = torch.device(args.device)

    print("[1/5] Building Qwen-Image PiD PidNet only")
    print(f"      device={device}, dtype={dtype}")

    net_config = get_pid_net_config("qwenimage")
    net = PidNet(**net_config)

    # Keep the same behavior as the previous smoke test:
    # construct on CPU, convert model storage dtype, then load checkpoint.
    net.to(dtype=dtype)
    net.eval()

    param_count = sum(
        parameter.numel()
        for parameter in net.parameters()
    )
    param_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in net.parameters()
    )

    print(f"      parameters={param_count:,}")
    print(
        f"      parameter storage ~= "
        f"{param_bytes / 1024**3:.2f} GiB"
    )

    verify_checkpoint_loading(
        net,
        args.checkpoint,
    )

    print(
        f"[3/5] Moving model to {device} "
        "and creating random conditions"
    )
    net.to(device)

    batch = 1
    latent_h = args.height // 32
    latent_w = args.width // 32

    noise = torch.randn(
        batch,
        3,
        args.height,
        args.width,
        device=device,
        dtype=dtype,
    )

    caption_embs = torch.randn(
        batch,
        args.text_length,
        2304,
        device=device,
        dtype=dtype,
    )

    lq_latent = torch.randn(
        batch,
        16,
        latent_h,
        latent_w,
        device=device,
        dtype=dtype,
    )

    degrade_sigma = torch.zeros(
        batch,
        device=device,
        dtype=dtype,
    )

    print("      noise:", tuple(noise.shape))
    print("      caption_embs:", tuple(caption_embs.shape))
    print("      lq_latent:", tuple(lq_latent.shape))
    print("      degrade_sigma:", tuple(degrade_sigma.shape))

    print("[4/5] Running PiD 4-step SDE sampler")

    with torch.inference_mode():
        output = run_four_step_sampler(
            net=net,
            noise=noise,
            caption_embs=caption_embs,
            lq_latent=lq_latent,
            degrade_sigma=degrade_sigma,
            seed=args.seed,
        )

    print("[5/5] Validating final output")

    expected_shape = (
        batch,
        3,
        args.height,
        args.width,
    )
    actual_shape = tuple(output.shape)
    is_finite = bool(
        torch.isfinite(output).all().item()
    )

    print(f"      output shape: {actual_shape}")
    print(f"      output dtype: {output.dtype}")
    print(f"      all finite:   {is_finite}")

    if actual_shape != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape: {actual_shape}, "
            f"expected {expected_shape}"
        )

    if not is_finite:
        raise RuntimeError(
            "Final output contains NaN or Inf"
        )

    print(
        "\nPASS: single-rank vLLM runtime -> "
        "PidNet -> ReplicatedLinear/ModelWeightParameter -> "
        "production weight_loader -> AdaLN remap -> "
        "4-step SDE sampler is connected."
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    with single_rank_vllm_environment():
        run_test(args)


if __name__ == "__main__":
    main()
