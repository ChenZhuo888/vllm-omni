#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PiD FP8 accuracy benchmark.

This benchmark intentionally bypasses Qwen-Image, Gemma, and PidInferenceModel.
It compares the PiD PidNet numerical outputs under identical inputs/noise for:

- BF16 baseline
- online FP8 per-tensor
- online FP8 per-block
- online FP8 per-channel
- offline/serialized FP8 per-tensor (dynamic activation)
- offline/serialized FP8 per-tensor (static activation)
- offline/serialized FP8 128x128 block (dynamic activation)

The core benchmark is independent of the quantization scheme:
1. Build PidNet with a case-specific quant_config.
2. Load weights through vllm_omni.diffusion.pid.checkpoint.load_pid_checkpoint.
3. For serialized/offline cases, explicitly finalize quantized layers after all
   FP8 weights/scales have been loaded.
4. Run a deterministic 4-step PiD SDE trajectory with pre-generated noise.
5. Compare:
   - first-step v_pred (pure same-input operator error);
   - state after every SDE step (cumulative trajectory error);
   - final raw x0;
   - final clamped x0 in [-1, 1].
6. Save metrics and quantization metadata to JSON.

Important offline-checkpoint contract
-------------------------------------
An offline case must point to an ACTUALLY quantized PiD checkpoint whose state
dict uses the same PiD ``net.*`` key namespace consumed by
``load_pid_checkpoint`` and contains the FP8 scale parameters required by the
selected scheme. Merely passing the original BF16 .pth while setting
``is_checkpoint_fp8_serialized=True`` is not a valid offline-quantization test.

Run from the vllm-omni repository root so the local checkout is imported.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import tempfile
from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterator

import torch
import torch.distributed as dist

from vllm_omni.diffusion.pid.checkpoint import load_pid_checkpoint
from vllm_omni.diffusion.pid.config import (
    get_pid_net_config,
    get_pid_sampling_config,
)
from vllm_omni.diffusion.pid.pid_net import PidNet


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantCase:
    name: str
    checkpoint: Path
    build_quant_config: Callable[[], object | None]
    description: str
    mode: str  # "baseline" | "online" | "offline"
    post_load_finalize: bool = False


@dataclass
class TestInputs:
    """All stochastic inputs, generated once on CPU and reused by every case."""

    initial_noise: torch.Tensor
    caption_embs: torch.Tensor
    lq_latent: torch.Tensor
    degrade_sigma: torch.Tensor
    sde_noises: list[torch.Tensor]


@dataclass
class RunOutput:
    """CPU float32 snapshots used for accuracy comparison."""

    first_v_pred: torch.Tensor
    step_states: list[torch.Tensor]
    final_x0: torch.Tensor
    final_clamped: torch.Tensor


@dataclass
class ErrorMetrics:
    mae: float
    rmse: float
    max_abs: float
    rel_l2: float
    cosine: float
    sqnr_db: float


class PidNetWrapper(torch.nn.Module):
    """Provide the ``.net`` attribute expected by load_pid_checkpoint."""

    def __init__(self, net: PidNet):
        super().__init__()
        self.net = net


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


ONLINE_CASE_NAMES = (
    "bf16",
    "online_fp8_per_tensor",
    "online_fp8_per_block",
    "online_fp8_per_channel",
)

OFFLINE_FP8_SCHEMES = (
    "fp8_tensor_dynamic",
    "fp8_tensor_static",
    "fp8_block128_dynamic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PiD online/offline FP8 numerical-accuracy benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Original BF16 PiD checkpoint. Used by the BF16 baseline and all "
            "online quantization cases."
        ),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["bf16", "online_fp8_per_tensor"],
        choices=ONLINE_CASE_NAMES,
        help="Built-in baseline/online cases to run.",
    )

    parser.add_argument(
        "--offline-fp8-case",
        action="append",
        nargs=3,
        metavar=("NAME", "SCHEME", "CHECKPOINT"),
        default=[],
        help=(
                "Add one serialized/offline FP8 case. SCHEME is one of: "
                + ", ".join(OFFLINE_FP8_SCHEMES)
                + ". May be repeated."
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Execution device. FP8 cases normally require CUDA/ROCm hardware.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="Reference activation/model dtype outside quantized weights.",
    )
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument(
        "--text-length",
        type=int,
        default=8,
        help="Fake Gemma token length. PiD accepts up to 300.",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--ignore-layer",
        action="append",
        default=[],
        help=(
            "Layer prefix to leave unquantized. May be repeated. "
            "Applied to built-in online/offline FP8 configs where supported."
        ),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("pid_quant_accuracy.json"),
        help="Where to save benchmark metadata and metrics.",
    )
    parser.add_argument(
        "--keep-case-outputs",
        action="store_true",
        help=(
            "Keep every case RunOutput in host memory until the benchmark ends. "
            "Normally outputs are compared immediately and released."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"BF16 checkpoint not found: {args.checkpoint}")

    if args.height <= 0 or args.width <= 0:
        raise ValueError("height/width must be positive")

    if args.height % 32 != 0 or args.width % 32 != 0:
        raise ValueError(
            "Qwen-Image PiD benchmark height/width must be divisible by 32 "
            "(sr_scale=4 * latent_spatial_down_factor=8)."
        )

    if not (1 <= args.text_length <= 300):
        raise ValueError("text-length must be in [1, 300]")

    if len(set(args.cases)) != len(args.cases):
        raise ValueError("--cases contains duplicate names")

    if "bf16" not in args.cases:
        raise ValueError(
            "--cases must include 'bf16' because every quantized case is "
            "measured against that baseline."
        )

    offline_names: set[str] = set()
    for name, scheme, checkpoint in args.offline_fp8_case:
        if name in ONLINE_CASE_NAMES:
            raise ValueError(
                f"Offline case name {name!r} collides with a built-in case name"
            )
        if name in offline_names:
            raise ValueError(f"Duplicate offline case name: {name!r}")
        offline_names.add(name)

        if scheme not in OFFLINE_FP8_SCHEMES:
            raise ValueError(
                f"Unsupported offline FP8 scheme {scheme!r}; expected one of "
                f"{OFFLINE_FP8_SCHEMES}"
            )

        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(
                f"Offline checkpoint for case {name!r} not found: {path}"
            )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"{args.device} was requested, but torch.cuda.is_available() is False"
        )

    if len(args.cases) + len(args.offline_fp8_case) > 1 and device.type == "cpu":
        raise RuntimeError(
            "Quantized benchmark cases are not intended for CPU execution. "
            "Use CUDA/ROCm hardware for FP8 accuracy testing."
        )


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(name)


# -----------------------------------------------------------------------------
# Minimal vLLM runtime
# -----------------------------------------------------------------------------


@contextmanager
def temporary_default_dtype(dtype: torch.dtype) -> Iterator[None]:
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def _make_vllm_config_context(device: torch.device, dtype: torch.dtype):
    """Create the minimal current-vLLM-config context needed by quant methods.

    Online/offline FP8 linear methods inspect
    ``get_current_vllm_config().model_config.dtype`` while they are created.
    A full HF ModelConfig is unnecessary for this standalone PiD benchmark, so
    attach a tiny dtype-bearing stub after VllmConfig validation.
    """
    from vllm.config import (
        DeviceConfig,
        VllmConfig,
        set_current_vllm_config,
    )

    cfg = VllmConfig(
        device_config=DeviceConfig(device=device.type),
    )
    cfg.model_config = SimpleNamespace(
        dtype=dtype,
        is_moe=False,
        hf_text_config=SimpleNamespace(
            model_type=None,
        ),
    )
    return set_current_vllm_config(cfg)


def _call_initialize_model_parallel(backend: str) -> None:
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
        kwargs["backend"] = backend

    initialize_model_parallel(**kwargs)


@contextmanager
def single_rank_vllm_environment(
        device: torch.device,
        dtype: torch.dtype,
) -> Iterator[None]:
    """Create rank-0/world-size-1 distributed + model-parallel state."""

    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        model_parallel_is_initialized,
    )

    created_dist = False
    created_model_parallel = False
    init_file_path: str | None = None

    dist_backend = "nccl" if device.type == "cuda" else "gloo"
    mp_backend = dist_backend

    with _make_vllm_config_context(device, dtype):
        try:
            if device.type == "cuda":
                torch.cuda.set_device(device)

            if not dist.is_initialized():
                fd, init_file_path = tempfile.mkstemp(
                    prefix="vllm_pid_quant_accuracy_"
                )
                os.close(fd)

                init_distributed_environment(
                    world_size=1,
                    rank=0,
                    distributed_init_method=f"file://{init_file_path}",
                    local_rank=device.index or 0,
                    backend=dist_backend,
                )
                created_dist = True

            if not model_parallel_is_initialized():
                _call_initialize_model_parallel(mp_backend)
                created_model_parallel = True

            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            rank = get_tensor_model_parallel_rank()
            world_size = get_tensor_model_parallel_world_size()
            if rank != 0 or world_size != 1:
                raise RuntimeError(
                    "Benchmark requires TP rank=0/world-size=1, got "
                    f"rank={rank}, world_size={world_size}"
                )

            print("[runtime] single-rank vLLM environment ready")
            print(f"          dist backend: {dist.get_backend()}")
            print(f"          TP rank/world-size: {rank}/{world_size}")
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


# -----------------------------------------------------------------------------
# Quantization case builders
# -----------------------------------------------------------------------------


def build_online_per_tensor_config(
        ignored_layers: list[str],
):
    """Exact online-FP8 family used by the existing PiD Phase-3 path."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    return Fp8Config(
        is_checkpoint_fp8_serialized=False,
        activation_scheme="dynamic",
        ignored_layers=ignored_layers,
    )


def build_online_shorthand_config(
        shorthand: str,
        ignored_layers: list[str],
):
    """Use current vLLM OnlineQuantizationConfig dispatch for block/channel."""
    from vllm.config.quantization import (
        QuantizationConfigArgs,
        resolve_quantization_config,
    )
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )

    resolved = resolve_quantization_config(shorthand, None)
    if resolved is None:
        raise RuntimeError(
            f"vLLM failed to resolve online quantization shorthand {shorthand!r}"
        )

    if ignored_layers:
        resolved = QuantizationConfigArgs(
            linear=resolved.linear,
            moe=resolved.moe,
            ignore=list(ignored_layers),
        )

    return OnlineQuantizationConfig(resolved)


def build_offline_fp8_config(
        scheme: str,
        ignored_layers: list[str],
):
    """Build vLLM's serialized/offline Fp8Config for a PiD FP8 checkpoint."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if scheme == "fp8_tensor_dynamic":
        return Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            ignored_layers=ignored_layers,
        )

    if scheme == "fp8_tensor_static":
        return Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="static",
            ignored_layers=ignored_layers,
        )

    if scheme == "fp8_block128_dynamic":
        return Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            ignored_layers=ignored_layers,
            weight_block_size=[128, 128],
        )

    raise ValueError(f"Unknown offline FP8 scheme: {scheme}")


def build_cases(args: argparse.Namespace) -> list[QuantCase]:
    bf16_checkpoint = args.checkpoint
    ignored = list(args.ignore_layer)

    builtin: dict[str, QuantCase] = {
        "bf16": QuantCase(
            name="bf16",
            checkpoint=bf16_checkpoint,
            build_quant_config=lambda: None,
            description="BF16 ReplicatedLinear baseline",
            mode="baseline",
        ),
        "online_fp8_per_tensor": QuantCase(
            name="online_fp8_per_tensor",
            checkpoint=bf16_checkpoint,
            build_quant_config=lambda: build_online_per_tensor_config(ignored),
            description=(
                "Online FP8: per-tensor weight + dynamic activation "
                "(Fp8PerTensorOnlineLinearMethod)"
            ),
            mode="online",
        ),
        "online_fp8_per_block": QuantCase(
            name="online_fp8_per_block",
            checkpoint=bf16_checkpoint,
            build_quant_config=lambda: build_online_shorthand_config(
                "fp8_per_block", ignored
            ),
            description=(
                "Online FP8: 128x128 block weight + dynamic block activation"
            ),
            mode="online",
        ),
        "online_fp8_per_channel": QuantCase(
            name="online_fp8_per_channel",
            checkpoint=bf16_checkpoint,
            build_quant_config=lambda: build_online_shorthand_config(
                "fp8_per_channel", ignored
            ),
            description=(
                "Online FP8: per-output-channel weight + dynamic per-token "
                "activation"
            ),
            mode="online",
        ),
    }

    cases = [builtin[name] for name in args.cases]

    for name, scheme, checkpoint in args.offline_fp8_case:
        checkpoint_path = Path(checkpoint)
        cases.append(
            QuantCase(
                name=name,
                checkpoint=checkpoint_path,
                build_quant_config=lambda s=scheme: build_offline_fp8_config(
                    s, ignored
                ),
                description=f"Offline serialized FP8: {scheme}",
                mode="offline",
                post_load_finalize=True,
            )
        )

    return cases


# -----------------------------------------------------------------------------
# Inputs and deterministic sampler
# -----------------------------------------------------------------------------


def build_test_inputs(
        height: int,
        width: int,
        text_length: int,
        seed: int,
) -> TestInputs:
    """Generate all random tensors ONCE on CPU in float32."""

    sampling_config = get_pid_sampling_config()
    t_list = list(sampling_config["student_t_list"])
    positive_next_steps = sum(float(t) > 0 for t in t_list[1:])

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    batch = 1
    latent_h = height // 32
    latent_w = width // 32

    initial_noise = torch.randn(
        batch,
        3,
        height,
        width,
        generator=generator,
        dtype=torch.float32,
    )
    caption_embs = torch.randn(
        batch,
        text_length,
        2304,
        generator=generator,
        dtype=torch.float32,
    )
    lq_latent = torch.randn(
        batch,
        16,
        latent_h,
        latent_w,
        generator=generator,
        dtype=torch.float32,
    )
    degrade_sigma = torch.zeros(
        batch,
        dtype=torch.float32,
    )

    sde_noises = [
        torch.randn(
            batch,
            3,
            height,
            width,
            generator=generator,
            dtype=torch.float32,
        )
        for _ in range(positive_next_steps)
    ]

    return TestInputs(
        initial_noise=initial_noise,
        caption_embs=caption_embs,
        lq_latent=lq_latent,
        degrade_sigma=degrade_sigma,
        sde_noises=sde_noises,
    )


def move_test_inputs(
        inputs: TestInputs,
        device: torch.device,
        dtype: torch.dtype,
) -> TestInputs:
    def move(x: torch.Tensor) -> torch.Tensor:
        return x.to(device=device, dtype=dtype, non_blocking=False)

    return TestInputs(
        initial_noise=move(inputs.initial_noise),
        caption_embs=move(inputs.caption_embs),
        lq_latent=move(inputs.lq_latent),
        degrade_sigma=move(inputs.degrade_sigma),
        sde_noises=[move(x) for x in inputs.sde_noises],
    )


def velocity_to_x0(
        x_t: torch.Tensor,
        velocity: torch.Tensor,
        t: torch.Tensor,
) -> torch.Tensor:
    """Match PiD's x0 = x_t - t * velocity with double intermediate."""

    shape = [x_t.shape[0]] + [1] * (x_t.ndim - 1)
    t_broadcast = t.double().view(*shape)

    return (
            x_t.double()
            - t_broadcast * velocity.double()
    ).to(x_t.dtype)


@torch.inference_mode()
def run_deterministic_sampler(
        net: PidNet,
        inputs: TestInputs,
) -> RunOutput:
    """Run PiD's 4-step SDE while injecting pre-generated SDE noise."""

    sampling_config = get_pid_sampling_config()

    sample_type = sampling_config.get("student_sample_type", "sde")
    prediction_type = sampling_config.get("prediction_type", "velocity")
    if sample_type != "sde":
        raise ValueError(
            f"Benchmark currently expects SDE sampling, got {sample_type!r}"
        )
    if prediction_type != "velocity":
        raise ValueError(
            "Benchmark currently expects velocity prediction, got "
            f"{prediction_type!r}"
        )

    t_list = torch.tensor(
        sampling_config["student_t_list"],
        device=inputs.initial_noise.device,
        dtype=torch.float32,
    )
    expected_steps = int(sampling_config["student_sample_steps"])
    actual_steps = len(t_list) - 1
    if expected_steps != actual_steps:
        raise RuntimeError(
            "Sampling config mismatch: "
            f"student_sample_steps={expected_steps}, t_list gives {actual_steps}"
        )

    timescale = float(sampling_config["fm_timescale"])

    x = inputs.initial_noise.clone()
    batch_size = x.shape[0]
    first_v_pred_cpu: torch.Tensor | None = None
    step_states_cpu: list[torch.Tensor] = []
    sde_noise_index = 0

    for step_index, (t_cur, t_next) in enumerate(
            zip(t_list[:-1], t_list[1:]),
            start=1,
    ):
        t_cur_batch = t_cur.expand(batch_size)
        t_scaled = t_cur_batch * timescale

        v_pred = net(
            x,
            t_scaled,
            inputs.caption_embs,
            lq_latent=inputs.lq_latent,
            degrade_sigma=inputs.degrade_sigma,
        )

        if not bool(torch.isfinite(v_pred).all().item()):
            raise RuntimeError(
                f"PidNet produced NaN/Inf at sampling step {step_index}"
            )

        if first_v_pred_cpu is None:
            first_v_pred_cpu = v_pred.detach().float().cpu()

        x0_pred = velocity_to_x0(
            x_t=x,
            velocity=v_pred,
            t=t_cur_batch,
        )
        if not bool(torch.isfinite(x0_pred).all().item()):
            raise RuntimeError(
                f"x0 prediction produced NaN/Inf at sampling step {step_index}"
            )

        if float(t_next.item()) > 0:
            if sde_noise_index >= len(inputs.sde_noises):
                raise RuntimeError("Not enough pre-generated SDE noises")

            eps_infer = inputs.sde_noises[sde_noise_index]
            sde_noise_index += 1

            shape = [batch_size] + [1] * (x.ndim - 1)
            t_next_broadcast = t_next.reshape(1).expand(shape)

            x = (
                    (1.0 - t_next_broadcast) * x0_pred
                    + t_next_broadcast * eps_infer
            )
        else:
            x = x0_pred

        if not bool(torch.isfinite(x).all().item()):
            raise RuntimeError(
                f"Sampler state contains NaN/Inf at step {step_index}"
            )

        step_states_cpu.append(x.detach().float().cpu())

    if first_v_pred_cpu is None:
        raise RuntimeError("Sampler produced no steps")

    if sde_noise_index != len(inputs.sde_noises):
        raise RuntimeError(
            "Unused pre-generated SDE noise: consumed "
            f"{sde_noise_index}, available {len(inputs.sde_noises)}"
        )

    final_x0 = step_states_cpu[-1]
    return RunOutput(
        first_v_pred=first_v_pred_cpu,
        step_states=step_states_cpu,
        final_x0=final_x0,
        final_clamped=final_x0.clamp(-1.0, 1.0),
    )


# -----------------------------------------------------------------------------
# Model loading / offline finalization / metadata
# -----------------------------------------------------------------------------


def build_model(
        quant_config: object | None,
        device: torch.device,
        dtype: torch.dtype,
) -> PidNet:
    """Construct the whole PidNet directly on the execution device.

    For online quantization this matters: online methods create meta weights and
    remember the current default device as the materialization target.
    """

    net_config = get_pid_net_config("qwenimage")

    with temporary_default_dtype(dtype):
        with torch.device(device):
            net = PidNet(
                **net_config,
                quant_config=quant_config,
            )

    net.eval()
    return net


def load_model_checkpoint(
        net: PidNet,
        checkpoint: Path,
) -> None:
    wrapper = PidNetWrapper(net)
    load_pid_checkpoint(wrapper, str(checkpoint))


def finalize_offline_quantized_layers(net: PidNet) -> int:
    """Finalize serialized/offline quantized LinearBase layers.

    Standard vLLM model loading performs this after all serialized FP8
    weights/scales are loaded. PiD currently has its own lightweight checkpoint
    loader, so the standalone benchmark performs the equivalent finalization
    explicitly for offline cases.
    """

    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )

    processed = 0

    for module in net.modules():
        if not isinstance(module, LinearBase):
            continue

        quant_method = getattr(module, "quant_method", None)
        if not isinstance(quant_method, QuantizeMethodBase):
            continue

        quant_method.process_weights_after_loading(module)

        # Some processing paths replace Parameter objects. Reconcile the
        # disable_tp/replicated parameter metadata just like vLLM's layerwise
        # loader does after processing.
        if hasattr(module, "update_param_tp_status"):
            module.update_param_tp_status()

        processed += 1

    return processed


def collect_quant_metadata(net: PidNet) -> dict:
    from vllm.model_executor.layers.linear import LinearBase

    method_counts: Counter[str] = Counter()
    weight_dtype_counts: Counter[str] = Counter()
    samples: list[dict] = []

    for name, module in net.named_modules():
        if not isinstance(module, LinearBase):
            continue

        quant_method = getattr(module, "quant_method", None)
        method_name = (
            type(quant_method).__name__
            if quant_method is not None
            else "None"
        )
        method_counts[method_name] += 1

        weight = getattr(module, "weight", None)
        weight_dtype = (
            str(weight.dtype)
            if isinstance(weight, torch.Tensor)
            else "N/A"
        )
        weight_dtype_counts[weight_dtype] += 1

        if len(samples) < 6:
            sample = {
                "layer": name,
                "module_type": type(module).__name__,
                "quant_method": method_name,
                "weight_dtype": weight_dtype,
                "weight_shape": (
                    list(weight.shape)
                    if isinstance(weight, torch.Tensor)
                    else None
                ),
            }

            for scale_name in (
                    "weight_scale",
                    "weight_scale_inv",
                    "input_scale",
            ):
                scale = getattr(module, scale_name, None)
                if isinstance(scale, torch.Tensor):
                    sample[scale_name] = {
                        "shape": list(scale.shape),
                        "dtype": str(scale.dtype),
                    }

            samples.append(sample)

    return {
        "quant_method_counts": dict(method_counts),
        "weight_dtype_counts": dict(weight_dtype_counts),
        "sample_layers": samples,
    }


# -----------------------------------------------------------------------------
# Metrics / reporting
# -----------------------------------------------------------------------------


def compute_metrics(
        reference: torch.Tensor,
        candidate: torch.Tensor,
) -> ErrorMetrics:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Shape mismatch: reference={reference.shape}, candidate={candidate.shape}"
        )

    ref = reference.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    out = candidate.detach().to(dtype=torch.float64, device="cpu").reshape(-1)

    if not bool(torch.isfinite(ref).all().item()):
        raise RuntimeError("Reference contains NaN/Inf")
    if not bool(torch.isfinite(out).all().item()):
        raise RuntimeError("Candidate contains NaN/Inf")

    diff = out - ref
    mae = diff.abs().mean().item()
    rmse = diff.square().mean().sqrt().item()
    max_abs = diff.abs().max().item()

    ref_norm = torch.linalg.vector_norm(ref).item()
    out_norm = torch.linalg.vector_norm(out).item()
    err_norm = torch.linalg.vector_norm(diff).item()

    tiny = 1e-30
    rel_l2 = err_norm / max(ref_norm, tiny)

    if ref_norm <= tiny or out_norm <= tiny:
        cosine = math.nan
    else:
        cosine = torch.dot(ref, out).item() / (ref_norm * out_norm)

    if err_norm <= tiny:
        sqnr_db = math.inf
    elif ref_norm <= tiny:
        sqnr_db = -math.inf
    else:
        sqnr_db = 20.0 * math.log10(ref_norm / err_norm)

    return ErrorMetrics(
        mae=mae,
        rmse=rmse,
        max_abs=max_abs,
        rel_l2=rel_l2,
        cosine=cosine,
        sqnr_db=sqnr_db,
    )


def compare_to_baseline(
        baseline: RunOutput,
        candidate: RunOutput,
) -> dict:
    if len(baseline.step_states) != len(candidate.step_states):
        raise RuntimeError("Baseline/candidate sampler step count differs")

    step_metrics = [
        compute_metrics(ref, cur)
        for ref, cur in zip(
            baseline.step_states,
            candidate.step_states,
        )
    ]

    return {
        "first_v_pred": asdict(
            compute_metrics(
                baseline.first_v_pred,
                candidate.first_v_pred,
            )
        ),
        "trajectory_steps": [
            asdict(metric) for metric in step_metrics
        ],
        "final_x0": asdict(
            compute_metrics(
                baseline.final_x0,
                candidate.final_x0,
            )
        ),
        "final_clamped": asdict(
            compute_metrics(
                baseline.final_clamped,
                candidate.final_clamped,
            )
        ),
    }


def format_float(value: float, scientific: bool = True) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if scientific:
        return f"{value:.4e}"
    return f"{value:.6f}"


def print_case_metrics(case_name: str, metrics: dict) -> None:
    rows: list[tuple[str, dict]] = [
        ("first_v_pred", metrics["first_v_pred"]),
    ]

    for idx, item in enumerate(metrics["trajectory_steps"], start=1):
        rows.append((f"state_after_step_{idx}", item))

    rows.append(("final_clamped", metrics["final_clamped"]))

    print()
    print(f"Accuracy vs BF16: {case_name}")
    print(
        f"{'point':<22}"
        f"{'rel_l2':>13}"
        f"{'cosine':>13}"
        f"{'SQNR(dB)':>13}"
        f"{'RMSE':>13}"
        f"{'max_abs':>13}"
    )
    print("-" * 87)

    for label, item in rows:
        print(
            f"{label:<22}"
            f"{format_float(item['rel_l2']):>13}"
            f"{format_float(item['cosine'], scientific=False):>13}"
            f"{format_float(item['sqnr_db'], scientific=False):>13}"
            f"{format_float(item['rmse']):>13}"
            f"{format_float(item['max_abs']):>13}"
        )


def json_safe(obj):
    """Convert +/-inf/nan floats into strings for portable strict JSON."""
    if isinstance(obj, dict):
        return {key: json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [json_safe(value) for value in obj]
    if isinstance(obj, tuple):
        return [json_safe(value) for value in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
    return obj


# -----------------------------------------------------------------------------
# Main benchmark
# -----------------------------------------------------------------------------


def run_case(
        case: QuantCase,
        cpu_inputs: TestInputs,
        device: torch.device,
        dtype: torch.dtype,
) -> tuple[RunOutput, dict]:
    print()
    print("=" * 88)
    print(f"CASE: {case.name}")
    print(f"mode: {case.mode}")
    print(f"description: {case.description}")
    print(f"checkpoint: {case.checkpoint}")
    print("=" * 88)

    quant_config = case.build_quant_config()

    net: PidNet | None = None
    try:
        net = build_model(
            quant_config=quant_config,
            device=device,
            dtype=dtype,
        )

        load_model_checkpoint(net, case.checkpoint)

        finalized_layers = 0
        if case.post_load_finalize:
            finalized_layers = finalize_offline_quantized_layers(net)
            print(
                "offline post-load finalized LinearBase layers:",
                finalized_layers,
            )

        metadata = collect_quant_metadata(net)
        metadata["offline_finalized_layers"] = finalized_layers

        print("quant methods:", metadata["quant_method_counts"])
        print("weight dtypes:", metadata["weight_dtype_counts"])

        case_inputs = move_test_inputs(
            cpu_inputs,
            device=device,
            dtype=dtype,
        )

        # Match PidInferenceModel production precision semantics:
        # keep numerically sensitive ops free to stay/promote to fp32, while
        # Linear/Conv matmuls execute under the requested low precision.
        autocast_ctx = (
            torch.autocast(
                device_type=device.type,
                dtype=dtype,
            )
            if dtype != torch.float32
            else nullcontext()
        )
        with autocast_ctx:
            output = run_deterministic_sampler(
                net,
                case_inputs,
            )

        del case_inputs

        return output, metadata

    finally:
        # Delete the model in THIS frame before empty_cache(). Deleting a copy
        # inside a helper would leave run_case()'s reference alive until return.
        if net is not None:
            del net
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    validate_args(args)

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    cases = build_cases(args)

    # Force BF16 baseline first even if the user listed it elsewhere.
    cases.sort(key=lambda case: 0 if case.name == "bf16" else 1)

    sampling_config = get_pid_sampling_config()

    print("=" * 88)
    print("PiD FP8 Accuracy Benchmark")
    print("=" * 88)
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(
            "compute capability:",
            ".".join(map(str, torch.cuda.get_device_capability(device))),
        )
    print(f"torch: {torch.__version__}")
    print(f"dtype: {dtype}")
    print(
        "input:",
        f"B=1 H={args.height} W={args.width} "
        f"text_len={args.text_length} seed={args.seed}",
    )
    print(
        "timestep schedule:",
        [float(v) for v in sampling_config["student_t_list"]],
    )
    print("cases:", [case.name for case in cases])

    # Generate random data before any model is built, so model construction and
    # quantization cannot perturb benchmark randomness.
    cpu_inputs = build_test_inputs(
        height=args.height,
        width=args.width,
        text_length=args.text_length,
        seed=args.seed,
    )

    results: dict = {
        "benchmark": "PiD FP8 accuracy",
        "baseline": "bf16",
        "device": str(device),
        "torch_version": torch.__version__,
        "dtype": str(dtype),
        "height": args.height,
        "width": args.width,
        "text_length": args.text_length,
        "seed": args.seed,
        "sampling_config": {
            "student_sample_type": sampling_config.get(
                "student_sample_type"
            ),
            "prediction_type": sampling_config.get("prediction_type"),
            "student_sample_steps": sampling_config.get(
                "student_sample_steps"
            ),
            "student_t_list": [
                float(v)
                for v in sampling_config["student_t_list"]
            ],
            "fm_timescale": float(sampling_config["fm_timescale"]),
        },
        "cases": {},
    }

    retained_outputs: dict[str, RunOutput] = {}
    baseline_output: RunOutput | None = None

    with single_rank_vllm_environment(device, dtype):
        for case in cases:
            output, metadata = run_case(
                case=case,
                cpu_inputs=cpu_inputs,
                device=device,
                dtype=dtype,
            )

            entry = {
                "mode": case.mode,
                "description": case.description,
                "checkpoint": str(case.checkpoint),
                "quant_metadata": metadata,
            }

            if case.name == "bf16":
                baseline_output = output
                entry["metrics_vs_bf16"] = None
                print("\nBF16 baseline captured.")
            else:
                if baseline_output is None:
                    raise RuntimeError(
                        "Internal error: quantized case ran before BF16 baseline"
                    )

                metrics = compare_to_baseline(
                    baseline_output,
                    output,
                )
                entry["metrics_vs_bf16"] = metrics
                print_case_metrics(case.name, metrics)

            results["cases"][case.name] = entry

            if args.keep_case_outputs:
                retained_outputs[case.name] = output
            elif case.name != "bf16":
                del output
                gc.collect()

    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(
            json_safe(results),
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    print()
    print("=" * 88)
    print("DONE")
    print(f"JSON result: {args.output_json}")
    print("=" * 88)


if __name__ == "__main__":
    main()
