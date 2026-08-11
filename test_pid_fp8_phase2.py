#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
PiD FP8 Phase 1 + Phase 2 combined smoke test.

This test verifies one continuous path:

1. Initialize a real single-rank vLLM runtime:
       VllmConfig
       -> torch.distributed
       -> TP=1 model-parallel group

2. Resolve the formal vLLM-Omni configuration:
       quantization_config="fp8"
       -> real Fp8Config

3. Build PiD through the real production constructor chain:
       PidDecodeMixin
       -> PidDecoder
       -> PidInferenceModel
       -> PidNet
       -> PixDiT_T2I
       -> MMDiTBlockT2I
       -> FeedForward
       -> ReplicatedLinear

4. Verify the real Fp8Config.get_quant_method() selects:
       Fp8PerTensorOnlineLinearMethod

5. Deliberately stop BEFORE FP8 create_weights/kernel initialization.
   ReplicatedLinear then uses UnquantizedLinearMethod only so that the
   existing BF16 checkpoint can still be loaded and executed on CPU.

6. Load the official PiD checkpoint through the production loader.

7. Verify:
       - ReplicatedLinear owns vLLM weight_loader
       - one migrated FeedForward weight exactly matches checkpoint
       - AdaLN checkpoint-key remapping still works
       - only FeedForward linears are quantization-controlled

8. Replace Gemma output with random embeddings and run the real
   PidInferenceModel 4-step sampling path.

This means:

Phase 1:
    ReplicatedLinear + production checkpoint loader + forward still work.

Phase 2:
    formal FP8 config reaches FeedForward and real vLLM FP8 dispatch.

NOT tested yet:
    FP8 create_weights
    FP8 weight conversion
    FP8 scale creation
    FP8 GEMM

Those belong to Phase 3.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import os
import tempfile
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PiD FP8 Phase1+Phase2 combined smoke test"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to official PiD model_ema_bf16.pth",
    )

    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )

    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help=(
            "PidInferenceModel precision. "
            "Use float32 for CPU smoke testing."
        ),
    )

    parser.add_argument(
        "--height",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--text-length",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    if args.height <= 0 or args.width <= 0:
        raise ValueError(
            "height/width must be positive"
        )

    if args.height % 32 != 0 or args.width % 32 != 0:
        raise ValueError(
            "For Qwen-Image PiD, height and width must be "
            "divisible by 32 (= sr_scale 4 * latent down factor 8)."
        )

    if args.text_length <= 0:
        raise ValueError(
            "text-length must be positive"
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested but CUDA is unavailable"
        )


# =============================================================================
# Real minimal vLLM runtime
# =============================================================================


def _maybe_vllm_config_context():
    """
    Newer vLLM requires initialize_model_parallel() to execute inside
    set_current_vllm_config().

    This is the same strategy as the previously working PiD smoke test.
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
    """
    Initialize TP=1 / PP=1 while remaining compatible with slightly
    different vLLM initialize_model_parallel signatures.
    """

    from vllm.distributed import initialize_model_parallel

    signature = inspect.signature(
        initialize_model_parallel
    )

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
    """
    Create the real minimum runtime required by BasevLLMParameter.

    Even ReplicatedLinear needs the TP group because ModelWeightParameter
    records TP rank/world-size at construction.
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
                    prefix="vllm_pid_fp8_phase2_dist_"
                )
                os.close(fd)

                init_distributed_environment(
                    world_size=1,
                    rank=0,
                    distributed_init_method=(
                        f"file://{init_file_path}"
                    ),
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

            print(
                "[0/8] Initializing real single-rank vLLM environment"
            )
            print(
                "      torch.distributed initialized:",
                dist.is_initialized(),
            )
            print(
                "      backend:",
                dist.get_backend()
                if dist.is_initialized()
                else "N/A",
            )
            print(
                f"      TP rank/world-size: {tp_rank}/{tp_size}"
            )

            if tp_rank != 0 or tp_size != 1:
                raise RuntimeError(
                    "Expected TP rank=0/world-size=1, "
                    f"got {tp_rank}/{tp_size}"
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


# =============================================================================
# Dummy Gemma
# =============================================================================


class DummyGemmaTextEncoder(nn.Module):
    """
    Do not load Gemma.

    PiD itself expects Qwen-Image PiD caption embeddings with width 2304.
    Random tensors are enough for this smoke test.
    """

    def __init__(
            self,
            *args,
            precision: str = "float32",
            text_length: int = 8,
            **kwargs,
    ):
        super().__init__()
        self.text_length = text_length

    def encode(
            self,
            captions: str | list[str],
    ) -> torch.Tensor:
        if isinstance(captions, str):
            batch_size = 1
        else:
            batch_size = len(captions)

        return torch.randn(
            batch_size,
            self.text_length,
            2304,
            dtype=torch.float32,
        )


# =============================================================================
# Dummy pipeline using the REAL PidDecodeMixin
# =============================================================================


def make_dummy_pipeline_class():
    from vllm_omni.diffusion.pid.mixin import PidDecodeMixin

    class DummyPidPipeline(nn.Module, PidDecodeMixin):
        PID_BACKBONE = "qwenimage"

        def __init__(self):
            nn.Module.__init__(self)

    return DummyPidPipeline


# =============================================================================
# Checkpoint verification
# =============================================================================


def assert_loaded_weight(
        net: nn.Module,
        state_dict: dict[str, torch.Tensor],
        checkpoint_key: str,
        model_key: str,
) -> None:
    params = dict(net.named_parameters())

    if checkpoint_key not in state_dict:
        raise RuntimeError(
            f"Checkpoint key missing: {checkpoint_key}"
        )

    if model_key not in params:
        raise RuntimeError(
            f"Model parameter missing: {model_key}"
        )

    actual = params[model_key].detach().cpu()

    expected = state_dict[
        checkpoint_key
    ].to(dtype=actual.dtype)

    try:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=0,
        )
    except AssertionError as exc:
        raise RuntimeError(
            "Loaded weight does not exactly match checkpoint:\n"
            f"  checkpoint: {checkpoint_key}\n"
            f"  model:      {model_key}"
        ) from exc


def verify_checkpoint_loading(
        net: nn.Module,
        checkpoint_path: Path,
) -> None:
    print(
        "[5/8] Verifying production checkpoint loading"
    )

    params = dict(net.named_parameters())

    replicated_weight_name = (
        "patch_blocks.0.mlp_x.w1.weight"
    )

    replicated_weight = params.get(
        replicated_weight_name
    )

    if replicated_weight is None:
        raise RuntimeError(
            "Migrated ReplicatedLinear parameter missing: "
            f"{replicated_weight_name}"
        )

    has_weight_loader = hasattr(
        replicated_weight,
        "weight_loader",
    )

    print(
        "      ReplicatedLinear parameter has weight_loader:",
        has_weight_loader,
    )

    print(
        "      migrated parameter type:",
        type(replicated_weight).__name__,
    )

    if not has_weight_loader:
        raise RuntimeError(
            f"{replicated_weight_name} has no weight_loader"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # Migrated FF weight.
    assert_loaded_weight(
        net,
        state_dict,
        checkpoint_key=(
            "net.patch_blocks.0.mlp_x.w1.weight"
        ),
        model_key=(
            "patch_blocks.0.mlp_x.w1.weight"
        ),
    )

    print(
        "      ReplicatedLinear weight exact match: PASS"
    )

    # Existing production key remapping.
    assert_loaded_weight(
        net,
        state_dict,
        checkpoint_key=(
            "net.patch_blocks.0."
            "adaLN_modulation_img.0.weight"
        ),
        model_key=(
            "patch_blocks.0."
            "ada_ln_modulation_img.0.weight"
        ),
    )

    print(
        "      AdaLN image key remap exact match: PASS"
    )

    assert_loaded_weight(
        net,
        state_dict,
        checkpoint_key=(
            "net.patch_blocks.0."
            "adaLN_modulation_txt.0.weight"
        ),
        model_key=(
            "patch_blocks.0."
            "ada_ln_modulation_txt.0.weight"
        ),
    )

    print(
        "      AdaLN text key remap exact match: PASS"
    )

    del state_dict
    gc.collect()


# =============================================================================
# Main test body
# =============================================================================


def run_test(
        args: argparse.Namespace,
) -> None:
    from vllm.model_executor.layers.linear import (
        LinearBase,
        ReplicatedLinear,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.layers.quantization.fp8 import (
        Fp8Config,
    )

    from vllm_omni.diffusion.data import (
        OmniDiffusionConfig,
    )

    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    # -------------------------------------------------------------------------
    # Record the REAL result selected by Fp8Config.get_quant_method().
    # -------------------------------------------------------------------------

    quant_method_events: list[dict] = []

    original_get_quant_method = (
        Fp8Config.get_quant_method
    )

    def traced_get_quant_method(
            self,
            layer,
            prefix,
    ):
        """
        Critical Phase-2 interception point.

        1. Call the REAL Fp8Config.get_quant_method().
        2. Record the real method selected by vLLM.
        3. Return UnquantizedLinearMethod so construction can continue
           using the existing BF16 checkpoint.

        Therefore we test real FP8 dispatch, while intentionally stopping
        before FP8 create_weights/kernel initialization.
        """

        real_method = original_get_quant_method(
            self,
            layer,
            prefix,
        )

        quant_method_events.append(
            {
                "layer_id": id(layer),
                "layer_type": type(layer).__name__,
                "prefix": prefix,
                "real_method": (
                    type(real_method).__name__
                    if real_method is not None
                    else None
                ),
            }
        )

        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()

        return real_method

    def forbid_fp8_kernel(*args, **kwargs):
        raise AssertionError(
            "Real FP8 kernel creation was reached. "
            "That belongs to Phase 3."
        )

    # Our real VllmConfig exists for distributed/model-parallel state.
    #
    # Its model_config is intentionally None in this lightweight runtime.
    # Fp8PerTensorOnlineLinearMethod.__init__ only needs model dtype, so
    # provide that one piece locally while keeping the actual vLLM runtime.
    fake_fp8_vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
        )
    )

    with ExitStack() as stack:

        # ---------------------------------------------------------------------
        # Real FP8 dispatch, but stop before actual FP8 weight/kernel creation.
        # ---------------------------------------------------------------------

        stack.enter_context(
            patch.object(
                Fp8Config,
                "get_quant_method",
                traced_get_quant_method,
            )
        )

        stack.enter_context(
            patch(
                "vllm.model_executor.layers.quantization."
                "online.fp8.get_current_vllm_config",
                return_value=fake_fp8_vllm_config,
            )
        )

        stack.enter_context(
            patch(
                "vllm.model_executor.layers.quantization."
                "online.fp8.cutlass_fp8_supported",
                return_value=False,
            )
        )

        stack.enter_context(
            patch(
                "vllm.model_executor.layers.quantization."
                "fp8.get_marlin_input_dtype",
                return_value=None,
            )
        )

        # Hard guard: these should never be called in this phase.
        stack.enter_context(
            patch(
                "vllm.model_executor.layers.quantization."
                "online.fp8.init_fp8_linear_kernel",
                side_effect=forbid_fp8_kernel,
            )
        )

        stack.enter_context(
            patch(
                "vllm.model_executor.layers.quantization."
                "fp8.init_fp8_linear_kernel",
                side_effect=forbid_fp8_kernel,
            )
        )

        # ---------------------------------------------------------------------
        # Remove ONLY Gemma.
        #
        # PiD config, PidDecoder, PidInferenceModel, PidNet and checkpoint
        # loader are all real production code.
        # ---------------------------------------------------------------------

        stack.enter_context(
            patch(
                "vllm_omni.diffusion.pid.pid_model."
                "GemmaTextEncoder",
                lambda *a, **kw: DummyGemmaTextEncoder(
                    *a,
                    text_length=args.text_length,
                    **kw,
                ),
            )
        )

        # Explicit device for standalone testing.
        stack.enter_context(
            patch(
                "vllm_omni.diffusion.pid.decoder."
                "get_local_device",
                return_value=device,
            )
        )

        # =====================================================================
        # Step 1: formal Omni FP8 config
        # =====================================================================

        print(
            "[1/8] Resolving formal Omni quantization_config"
        )

        od_config = OmniDiffusionConfig(
            model="pid-fp8-phase2-test",
            quantization_config="fp8",
        )

        quant_config = (
            od_config.quantization_config
        )

        if quant_config is None:
            raise RuntimeError(
                "quantization_config='fp8' resolved to None"
            )

        if not isinstance(
                quant_config,
                Fp8Config,
        ):
            raise RuntimeError(
                "Expected Fp8Config, got "
                f"{type(quant_config).__name__}"
            )

        if quant_config.get_name() != "fp8":
            raise RuntimeError(
                "Unexpected quant method: "
                f"{quant_config.get_name()}"
            )

        if (
                quant_config
                        .is_checkpoint_fp8_serialized
        ):
            raise RuntimeError(
                "This test expects online FP8 "
                "(BF16 source checkpoint)"
            )

        print(
            "      input: 'fp8'"
        )
        print(
            "      resolved:",
            type(quant_config).__name__,
        )
        print(
            "      activation scheme:",
            quant_config.activation_scheme,
        )
        print(
            "      serialized checkpoint:",
            quant_config.is_checkpoint_fp8_serialized,
        )

        # =====================================================================
        # Step 2:
        # build through real:
        #
        # PidDecodeMixin
        # -> PidDecoder
        # -> PidInferenceModel
        # -> PidNet
        #
        # AND production checkpoint loading.
        # =====================================================================

        print(
            "[2/8] Building full PiD construction chain"
        )

        od_config.pid_decode = {
            "enabled": True,
            "checkpoint_path": str(
                args.checkpoint
            ),
            "gemma_model": "/fake/gemma",
            "scale": 4,
            "num_steps": 4,
            "seed": args.seed,
            "degrade_sigma": 0.0,
            "precision": args.precision,
        }

        # Never torch.compile in this smoke test.
        od_config.enforce_eager = True

        DummyPidPipeline = (
            make_dummy_pipeline_class()
        )

        pipeline = DummyPidPipeline()

        # Real production PiD initialization.
        pipeline._init_pid_decoder(
            od_config
        )

        decoder = pipeline._pid_decoder

        if decoder is None:
            raise RuntimeError(
                "PidDecodeMixin did not create PidDecoder"
            )

        model = decoder._model

        if model is None:
            raise RuntimeError(
                "PidDecoder did not create PidInferenceModel"
            )

        net = model.net

        print(
            "      model:",
            type(model).__name__,
        )
        print(
            "      net:",
            type(net).__name__,
        )
        print(
            "      patch blocks:",
            len(net.patch_blocks),
        )

        parameter_count = sum(
            p.numel()
            for p in net.parameters()
        )

        parameter_bytes = sum(
            p.numel() * p.element_size()
            for p in net.parameters()
        )

        print(
            f"      parameters: {parameter_count:,}"
        )
        print(
            "      parameter storage ~= "
            f"{parameter_bytes / 1024 ** 3:.2f} GiB"
        )

        # =====================================================================
        # Step 3: verify config propagation
        # =====================================================================

        print(
            "[3/8] Verifying FP8 config propagation "
            "to FeedForward"
        )

        replicated_layers = {
            name: module
            for name, module
            in net.named_modules()
            if isinstance(
                module,
                ReplicatedLinear,
            )
        }

        expected_names = {
            (
                f"patch_blocks.{block_idx}."
                f"{stream}.{proj}"
            )
            for block_idx
            in range(len(net.patch_blocks))
            for stream
            in ("mlp_x", "mlp_y")
            for proj
            in ("w1", "w3", "w2")
        }

        actual_names = set(
            replicated_layers
        )

        if actual_names != expected_names:
            missing = sorted(
                expected_names - actual_names
            )
            unexpected = sorted(
                actual_names - expected_names
            )

            raise RuntimeError(
                "ReplicatedLinear scope mismatch.\n"
                f"Missing: {missing}\n"
                f"Unexpected: {unexpected}"
            )

        for name, layer in (
                replicated_layers.items()
        ):
            if (
                    layer.quant_config
                    is not quant_config
            ):
                raise RuntimeError(
                    f"{name}: did not receive "
                    "the original Omni Fp8Config object"
                )

        print(
            "      FeedForward "
            "ReplicatedLinear count:",
            len(replicated_layers),
        )

        print(
            "      all share exact "
            "Omni Fp8Config object: PASS"
        )

        # =====================================================================
        # Step 4: verify REAL FP8 dispatch + negative scope
        # =====================================================================

        print(
            "[4/8] Verifying real FP8 quant-method dispatch"
        )

        events_by_layer_id = {
            event["layer_id"]: event
            for event
            in quant_method_events
        }

        if (
                len(events_by_layer_id)
                != len(replicated_layers)
        ):
            raise RuntimeError(
                "get_quant_method event count mismatch: "
                f"{len(events_by_layer_id)} vs "
                f"{len(replicated_layers)}"
            )

        for name, layer in sorted(
                replicated_layers.items()
        ):
            event = events_by_layer_id.get(
                id(layer)
            )

            if event is None:
                raise RuntimeError(
                    f"{name}: "
                    "Fp8Config.get_quant_method "
                    "was never called"
                )

            real_method = event[
                "real_method"
            ]

            if (
                    real_method
                    != "Fp8PerTensorOnlineLinearMethod"
            ):
                raise RuntimeError(
                    f"{name}: expected "
                    "Fp8PerTensorOnlineLinearMethod, "
                    f"got {real_method}"
                )

            if not event[
                "prefix"
            ].endswith(name):
                raise RuntimeError(
                    f"{name}: invalid prefix "
                    f"{event['prefix']!r}"
                )

        print(
            "      all FeedForward linears:"
        )
        print(
            "          Fp8Config.get_quant_method"
            " -> Fp8PerTensorOnlineLinearMethod"
        )
        print(
            "      real FP8 dispatch: PASS"
        )

        # Attention must still be plain nn.Linear.
        for index, block in enumerate(
                net.patch_blocks
        ):
            for module_name in (
                    "qkv_x",
                    "qkv_y",
                    "proj_x",
                    "proj_y",
            ):
                module = getattr(
                    block.attn,
                    module_name,
                )

                if not isinstance(
                        module,
                        nn.Linear,
                ):
                    raise RuntimeError(
                        "Unexpected quantization scope: "
                        f"patch_blocks.{index}."
                        f"attn.{module_name} became "
                        f"{type(module).__name__}"
                    )

        # PiT MLP remains plain nn.Linear.
        for index, block in enumerate(
                net.pixel_blocks
        ):
            if not isinstance(
                    block.mlp.fc1,
                    nn.Linear,
            ):
                raise RuntimeError(
                    "PiT MLP fc1 unexpectedly "
                    "entered vLLM quantization"
                )

            if not isinstance(
                    block.mlp.fc2,
                    nn.Linear,
            ):
                raise RuntimeError(
                    "PiT MLP fc2 unexpectedly "
                    "entered vLLM quantization"
                )

        print(
            "      Attention/PiT remain nn.Linear: PASS"
        )

        # =====================================================================
        # Step 5: make sure Phase 1 checkpoint path still works
        # =====================================================================

        verify_checkpoint_loading(
            net,
            args.checkpoint,
        )

        # =====================================================================
        # Step 6: create random upstream conditions
        # =====================================================================

        print(
            "[6/8] Creating random upstream conditions"
        )

        batch_size = 1

        latent_h = args.height // 32
        latent_w = args.width // 32

        lq_latent = torch.randn(
            batch_size,
            16,
            latent_h,
            latent_w,
            device=device,
            dtype=torch.float32,
        )

        print(
            "      lq_latent:",
            tuple(lq_latent.shape),
        )

        print(
            "      fake Gemma embedding:",
            (
                batch_size,
                args.text_length,
                2304,
            ),
        )

        print(
            "      output size:",
            (
                args.height,
                args.width,
            ),
        )

        # =====================================================================
        # Step 7: run REAL PidInferenceModel 4-step path
        # =====================================================================

        print(
            "[7/8] Running real PiD 4-step sampler"
        )

        with torch.inference_mode():
            output = (
                model.generate_samples_from_batch(
                    lq_latent=lq_latent,
                    caption=["random-condition"],
                    output_size=(
                        args.height,
                        args.width,
                    ),
                    degrade_sigma=0.0,
                    num_steps=4,
                    seed=args.seed,
                )
            )

        # =====================================================================
        # Step 8: final validation
        # =====================================================================

        print(
            "[8/8] Validating final output"
        )

        expected_shape = (
            batch_size,
            3,
            args.height,
            args.width,
        )

        actual_shape = tuple(
            output.shape
        )

        all_finite = bool(
            torch.isfinite(
                output
            ).all().item()
        )

        print(
            "      output shape:",
            actual_shape,
        )

        print(
            "      output dtype:",
            output.dtype,
        )

        print(
            "      mean:",
            output.float().mean().item(),
        )

        print(
            "      std:",
            output.float().std().item(),
        )

        print(
            "      all finite:",
            all_finite,
        )

        if actual_shape != expected_shape:
            raise RuntimeError(
                "Unexpected output shape: "
                f"{actual_shape}, "
                f"expected {expected_shape}"
            )

        if not all_finite:
            raise RuntimeError(
                "Final output contains NaN/Inf"
            )

        print()
        print("=" * 80)
        print("PASS: PiD Phase 1 + Phase 2")
        print("=" * 80)
        print(
            """
Real VllmConfig / distributed / TP=1
        ↓
OmniDiffusionConfig(quantization_config="fp8")
        ↓
Fp8Config
        ↓
PidDecodeMixin
        ↓
PidDecoder
        ↓
PidInferenceModel
        ↓
PidNet
        ↓
FeedForward ReplicatedLinear
        ↓
real Fp8Config.get_quant_method()
        ↓
Fp8PerTensorOnlineLinearMethod
        ↓
[FP8 create_weights intentionally intercepted]
        ↓
BF16 checkpoint via production loader
        ↓
real PiD 4-step sampling
        ↓
finite output
"""
        )

        print(
            "Phase 3 remaining boundary:"
        )
        print(
            "      Fp8PerTensorOnlineLinearMethod.create_weights"
        )
        print(
            "      -> real FP8 weights/scales"
        )
        print(
            "      -> process_weights_after_loading"
        )
        print(
            "      -> real FP8 GEMM"
        )

        del output
        del lq_latent
        del net
        del model
        del decoder
        del pipeline

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Entry
# =============================================================================


def main() -> None:
    args = parse_args()
    validate_args(args)

    print("=" * 80)
    print("PiD FP8 Phase 1 + Phase 2 Combined Smoke Test")
    print("=" * 80)

    with single_rank_vllm_environment():
        run_test(args)


if __name__ == "__main__":
    main()
