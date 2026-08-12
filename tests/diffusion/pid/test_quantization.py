# SPDX-License-Identifier: Apache-2.0
"""PiD quantization integration tests for Qwen-Image.

IMPORTANT:
    Run this file from inside the repository's tests/ tree, e.g.
    tests/diffusion/pid/test_quantization.py

    It intentionally relies on the repository's own pytest fixtures:
      - default_vllm_config
      - init_fake_tp_group

This file intentionally does NOT test image quality or numerical closeness.

It validates the quantization control path in layers:

1. User-facing quantization_config parsing.
2. Qwen-Image component routing: transformer vs. PiD.
3. PiD propagation: pipeline -> PidDecoder -> PidInferenceModel -> PidNet.
4. FP8 layer selection, including ignored_layers.
5. VllmConfig / dtype wiring used by vLLM FP8 methods.
6. Real CUDA execution through the kernel selected by vLLM.

The CPU tests are lightweight and should diagnose routing/config failures without
loading Qwen-Image or the 1.4B PiD checkpoint. CUDA tests use a tiny real PiD
FeedForward block and synthetic weights.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
    set_current_vllm_config,
)
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerTensorOnlineLinearMethod,
)
from vllm.model_executor.utils import replace_parameter

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.pid.decoder import PidDecodeConfig, PidDecoder
from vllm_omni.diffusion.pid.pixeldit import FeedForward
from vllm_omni.quantization import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


# =============================================================================
# Helpers
# =============================================================================

def test_repo_vllm_test_environment(
    default_vllm_config,
    init_fake_tp_group,
):
    """Regression guard for the two vLLM runtime assumptions this suite needs."""
    from vllm.distributed.parallel_state import get_tp_group

    current = get_current_vllm_config()
    tp_group = get_tp_group()

    assert current is not None
    assert tp_group.world_size == 1
    assert tp_group.rank_in_group == 0



def _quant_name(config) -> str | None:
    return None if config is None else config.get_name()


def _online_fp8(*, ignored_layers: list[str] | None = None) -> Fp8Config:
    return Fp8Config(
        is_checkpoint_fp8_serialized=False,
        activation_scheme="dynamic",
        ignored_layers=ignored_layers,
    )


def _serialized_fp8(*, ignored_layers: list[str] | None = None) -> Fp8Config:
    return Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=ignored_layers,
    )


@contextmanager
def _torch_default_dtype(dtype: torch.dtype):
    """Keep torch's global default dtype deterministic for LinearMethod init."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


@contextmanager
def _vllm_config_scope(
    dtype: torch.dtype,
    quant_config=None,
):
    """Create the minimal *real* VllmConfig needed by vLLM linear kernels.

    FP8 LinearMethod constructors read model_config.dtype through
    get_current_vllm_config(), while kernel selection also reads kernel_config.
    Using VllmConfig here avoids the common test bug where a SimpleNamespace
    happens to provide dtype but misses some other config field.
    """
    vllm_config = VllmConfig()

    # Constructing a real ModelConfig would require a model/tokenizer config,
    # which is unrelated to these unit tests. The FP8 code path under test only
    # consumes model_config.dtype.
    object.__setattr__(
        vllm_config,
        "model_config",
        SimpleNamespace(dtype=dtype),
    )
    object.__setattr__(vllm_config, "quant_config", quant_config)

    with set_current_vllm_config(vllm_config):
        yield vllm_config


def _dummy_replicated_linear() -> ReplicatedLinear:
    """Return an uninitialized LinearBase instance for get_quant_method tests.

    Fp8Config.get_quant_method only needs isinstance(layer, LinearBase) for the
    selection step; no weights are touched here.
    """
    return object.__new__(ReplicatedLinear)


class _FakeTextEncoder:
    def to(self, _device):
        return self


class _FakeVAE:
    # Qwen pipeline only needs len(temperal_downsample) in __init__.
    temperal_downsample = [0, 1, 2]

    def to(self, _device):
        return self


def _capture_qwen_pipeline_routing(
    monkeypatch,
    quantization_config,
) -> dict[str, object]:
    """Run the real QwenImagePipeline.__init__ while mocking model IO.

    This deliberately exercises the production transformer/PiD split instead
    of reimplementing that split in the test.
    """
    import vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image as qwen_mod

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        qwen_mod,
        "get_local_device",
        MagicMock(return_value=torch.device("cpu")),
    )
    monkeypatch.setattr(
        qwen_mod,
        "prefetch_subfolders",
        MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        qwen_mod.FlowMatchEulerDiscreteScheduler,
        "from_pretrained",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(
        qwen_mod,
        "from_pretrained_with_prefetch",
        MagicMock(side_effect=[_FakeTextEncoder(), _FakeVAE()]),
    )
    monkeypatch.setattr(
        qwen_mod,
        "get_transformer_config_kwargs",
        MagicMock(return_value={}),
    )

    class _FakeTransformer(nn.Module):
        def __init__(self, *args, quant_config=None, **kwargs):
            super().__init__()
            captured["transformer"] = quant_config

    monkeypatch.setattr(
        qwen_mod,
        "QwenImageTransformer2DModel",
        _FakeTransformer,
    )
    monkeypatch.setattr(
        qwen_mod.Qwen2Tokenizer,
        "from_pretrained",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(
        qwen_mod.QwenImagePipeline,
        "setup_diffusion_pipeline_profiler",
        lambda *args, **kwargs: None,
    )

    def _fake_init_pid_decoder(
        self,
        od_config,
        quantization_config=None,
    ):
        captured["pid"] = quantization_config
        self._pid_config = None
        self._pid_decoder = None

    monkeypatch.setattr(
        qwen_mod.QwenImagePipeline,
        "_init_pid_decoder",
        _fake_init_pid_decoder,
    )

    od_config = SimpleNamespace(
        model="__unit_test_model__",
        parallel_config=SimpleNamespace(),
        tf_model_config=SimpleNamespace(),
        quantization_config=quantization_config,
        enable_diffusion_pipeline_profiler=False,
    )

    qwen_mod.QwenImagePipeline(od_config=od_config)
    return captured


def _all_ff_linears(ff: FeedForward) -> dict[str, ReplicatedLinear]:
    return {
        "w1": ff.w1,
        "w3": ff.w3,
        "w2": ff.w2,
    }


def _materialize_linear_for_cuda(linear: ReplicatedLinear) -> None:
    """Materialize synthetic weights in the format expected by its method."""
    method = linear.quant_method

    if isinstance(method, UnquantizedLinearMethod):
        with torch.no_grad():
            linear.weight.normal_(mean=0.0, std=0.02)
        return

    if isinstance(method, Fp8PerTensorOnlineLinearMethod):
        # Online FP8 intentionally creates a meta weight, then quantizes it
        # after checkpoint loading. Replace the meta parameter with a tiny
        # synthetic BF16 checkpoint weight and run the real post-load hook.
        loaded_weight = torch.randn(
            linear.output_size,
            linear.input_size,
            device="cuda",
            dtype=torch.bfloat16,
        ) * 0.02
        replace_parameter(linear, "weight", loaded_weight)
        method.process_weights_after_loading(linear)
        return

    if isinstance(method, Fp8LinearMethod):
        # Serialized FP8 expects FP8 weight + scale to already exist.
        loaded_weight = torch.randn(
            tuple(linear.weight.shape),
            device="cuda",
            dtype=torch.float32,
        ).clamp_(-2.0, 2.0)
        loaded_weight = loaded_weight.to(linear.weight.dtype)
        with torch.no_grad():
            linear.weight.copy_(loaded_weight)
            linear.weight_scale.fill_(1.0)
        method.process_weights_after_loading(linear)
        return

    raise AssertionError(f"Unexpected linear method: {type(method).__name__}")


# =============================================================================
# 1. Config parsing
# =============================================================================


@pytest.mark.cpu
class TestConfigParsing:
    def test_no_quantization(self):
        od_config = OmniDiffusionConfig(model="test", quantization_config=None)
        assert od_config.quantization_config is None

    def test_flat_fp8_config(self):
        od_config = OmniDiffusionConfig(
            model="test",
            quantization_config="fp8",
        )
        assert isinstance(od_config.quantization_config, Fp8Config)
        assert not od_config.quantization_config.is_checkpoint_fp8_serialized
        assert od_config.quantization_config.activation_scheme == "dynamic"

    def test_pid_only_component_config(self):
        od_config = OmniDiffusionConfig(
            model="test",
            quantization_config={
                "transformer": None,
                "pid": {"method": "fp8"},
            },
        )

        config = od_config.quantization_config
        assert isinstance(config, ComponentQuantizationConfig)
        assert config.resolve("transformer") is None
        assert _quant_name(config.resolve("pid")) == "fp8"

    def test_transformer_and_pid_can_use_different_fp8_modes(self):
        od_config = OmniDiffusionConfig(
            model="test",
            quantization_config={
                "transformer": {
                    "method": "fp8",
                    "is_checkpoint_fp8_serialized": False,
                },
                "pid": {
                    "method": "fp8",
                    "is_checkpoint_fp8_serialized": True,
                },
            },
        )

        config = od_config.quantization_config
        assert isinstance(config, ComponentQuantizationConfig)

        transformer = config.resolve("transformer")
        pid = config.resolve("pid")

        assert isinstance(transformer, Fp8Config)
        assert isinstance(pid, Fp8Config)
        assert not transformer.is_checkpoint_fp8_serialized
        assert pid.is_checkpoint_fp8_serialized

    def test_pid_can_receive_modelopt_fp8_config(self):
        od_config = OmniDiffusionConfig(
            model="test",
            quantization_config={
                "transformer": {"method": "fp8"},
                "pid": {
                    "quant_method": "modelopt",
                    "quant_algo": "FP8",
                    "ignore": [],
                    "producer": {"name": "modelopt"},
                },
            },
        )

        config = od_config.quantization_config
        assert isinstance(config, ComponentQuantizationConfig)
        assert _quant_name(config.resolve("transformer")) == "fp8"
        assert _quant_name(config.resolve("pid")) == "modelopt"

    def test_invalid_component_method_fails_early(self):
        with pytest.raises(ValueError, match="Unknown quantization method"):
            OmniDiffusionConfig(
                model="test",
                quantization_config={
                    "transformer": None,
                    "pid": {"method": "__not_a_quant_method__"},
                },
            )


# =============================================================================
# 2. Qwen-Image production routing: transformer vs PiD
# =============================================================================


@pytest.mark.cpu
class TestQwenImageComponentRouting:
    @pytest.mark.parametrize(
        ("spec", "expected_transformer", "expected_pid"),
        [
            (None, None, None),
            ("fp8", "fp8", None),
            (
                {
                    "transformer": {"method": "fp8"},
                    "pid": None,
                },
                "fp8",
                None,
            ),
            (
                {
                    "transformer": None,
                    "pid": {"method": "fp8"},
                },
                None,
                "fp8",
            ),
            (
                {
                    "transformer": {"method": "fp8"},
                    "pid": {
                        "quant_method": "modelopt",
                        "quant_algo": "FP8",
                        "ignore": [],
                        "producer": {"name": "modelopt"},
                    },
                },
                "fp8",
                "modelopt",
            ),
        ],
        ids=[
            "none",
            "flat-fp8-transformer-only",
            "component-transformer-only",
            "component-pid-only",
            "component-mixed-fp8-modelopt",
        ],
    )
    def test_real_pipeline_init_routes_components(
        self,
        monkeypatch,
        spec,
        expected_transformer,
        expected_pid,
    ):
        od_config = OmniDiffusionConfig(
            model="test",
            quantization_config=spec,
        )

        captured = _capture_qwen_pipeline_routing(
            monkeypatch,
            od_config.quantization_config,
        )

        assert _quant_name(captured["transformer"]) == expected_transformer
        assert _quant_name(captured["pid"]) == expected_pid


# =============================================================================
# 3. PiD propagation: pipeline -> decoder -> inference model -> PidNet
# =============================================================================


@pytest.mark.cpu
class TestPidQuantConfigPropagation:
    def test_mixin_forwards_quant_config_to_pid_decoder(
        self,
        monkeypatch,
    ):
        import vllm_omni.diffusion.pid.mixin as pid_mixin_mod
        from vllm_omni.diffusion.pid.mixin import PidDecodeMixin

        quant_config = _online_fp8()
        captured = {}

        class _FakePidDecoder(nn.Module):
            def __init__(
                self,
                *,
                config,
                backbone,
                enforce_eager,
                quant_config,
            ):
                super().__init__()
                captured["config"] = config
                captured["backbone"] = backbone
                captured["enforce_eager"] = enforce_eager
                captured["quant_config"] = quant_config
                captured["loaded"] = False

            def load_weights(self):
                captured["loaded"] = True

        monkeypatch.setattr(
            pid_mixin_mod,
            "PidDecoder",
            _FakePidDecoder,
        )

        class _DummyPipeline(nn.Module, PidDecodeMixin):
            PID_BACKBONE = "qwenimage"

            def __init__(self):
                nn.Module.__init__(self)

        pipeline = _DummyPipeline()
        od_config = SimpleNamespace(
            pid_decode=PidDecodeConfig(enabled=True),
            enforce_eager=True,
        )

        pipeline._init_pid_decoder(
            od_config,
            quantization_config=quant_config,
        )

        assert captured["quant_config"] is quant_config
        assert captured["backbone"] == "qwenimage"
        assert captured["enforce_eager"] is True
        assert captured["loaded"] is True
        assert "_pid_decoder" in pipeline._resident_modules

    def test_pid_decoder_forwards_quant_config_to_inference_model(
        self,
        monkeypatch,
    ):
        import vllm_omni.diffusion.pid.decoder as decoder_mod

        quant_config = _online_fp8()
        captured = {}

        class _FakeInferenceModel(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                captured.update(kwargs)

        monkeypatch.setattr(
            decoder_mod,
            "get_local_device",
            MagicMock(return_value=torch.device("cpu")),
        )
        mock_load = MagicMock(return_value=None)
        monkeypatch.setattr(
            decoder_mod,
            "load_pid_checkpoint",
            mock_load,
        )
        monkeypatch.setattr(
            decoder_mod,
            "PidInferenceModel",
            _FakeInferenceModel,
        )

        decoder = PidDecoder(
            config=PidDecodeConfig(
                enabled=True,
                checkpoint_path="fake-pid.pth",
                gemma_model="fake-gemma",
                precision="bfloat16",
            ),
            backbone="qwenimage",
            enforce_eager=True,
            quant_config=quant_config,
        )
        decoder.load_weights()

        assert captured["quant_config"] is quant_config
        assert captured["precision"] == "bfloat16"
        assert captured["enforce_eager"] is True
        mock_load.assert_called_once_with(decoder._model, "fake-pid.pth")

    @pytest.mark.parametrize(
        ("precision", "expected_dtype"),
        [
            ("float16", torch.float16),
            ("bfloat16", torch.bfloat16),
            ("float32", torch.float32),
        ],
    )
    def test_inference_model_forwards_quant_config_to_pid_net(
        self,
        monkeypatch,
        precision,
        expected_dtype,
    ):
        import vllm_omni.diffusion.pid.pid_model as pid_model_mod

        quant_config = _online_fp8()
        captured = {}

        class _FakePidNet(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                captured["net_kwargs"] = kwargs
                self._dummy = nn.Parameter(torch.zeros(1))

        class _FakeTextEncoder(nn.Module):
            def __init__(self, model_id, precision):
                super().__init__()
                captured["gemma_model_id"] = model_id
                captured["text_precision"] = precision

        monkeypatch.setattr(pid_model_mod, "PidNet", _FakePidNet)
        monkeypatch.setattr(
            pid_model_mod,
            "GemmaTextEncoder",
            _FakeTextEncoder,
        )

        model = pid_model_mod.PidInferenceModel(
            net_kwargs={"hidden_size": 32},
            gemma_model_id="fake-gemma",
            precision=precision,
            enforce_eager=True,
            quant_config=quant_config,
        )

        assert captured["net_kwargs"]["quant_config"] is quant_config
        assert captured["text_precision"] == precision
        assert model.precision == expected_dtype

        if expected_dtype == torch.float32:
            assert model.autocast_dtype is None
        else:
            assert model.autocast_dtype == expected_dtype


# =============================================================================
# 4. Layer selection and ignored_layers
# =============================================================================


@pytest.mark.cpu
class TestFp8LayerSelection:
    @pytest.mark.parametrize(
        "dtype",
        [torch.float16, torch.bfloat16],
        ids=["fp16", "bf16"],
    )
    def test_online_fp8_method_reads_input_dtype_from_vllm_config(
        self,
        dtype,
    ):
        quant_config = _online_fp8()
        layer = _dummy_replicated_linear()

        with _vllm_config_scope(dtype, quant_config) as vllm_config:
            assert get_current_vllm_config() is vllm_config

            method = quant_config.get_quant_method(
                layer,
                prefix="pid.test.ff.w1",
            )

        assert isinstance(method, Fp8PerTensorOnlineLinearMethod)
        assert method.input_dtype == dtype

    def test_fp8_supported_activation_dtypes_are_explicit(self):
        assert set(Fp8Config.get_supported_act_dtypes()) == {
            torch.float16,
            torch.bfloat16,
        }

    def test_exact_ignored_layer_stays_unquantized(self):
        prefix = "pid.test.ff.w1"
        quant_config = _online_fp8(ignored_layers=[prefix])

        method = quant_config.get_quant_method(
            _dummy_replicated_linear(),
            prefix=prefix,
        )

        assert isinstance(method, UnquantizedLinearMethod)

    def test_non_ignored_sibling_still_uses_fp8(self):
        quant_config = _online_fp8(
            ignored_layers=["pid.test.ff.w1"],
        )

        with _vllm_config_scope(torch.bfloat16, quant_config):
            method = quant_config.get_quant_method(
                _dummy_replicated_linear(),
                prefix="pid.test.ff.w3",
            )

        assert isinstance(method, Fp8PerTensorOnlineLinearMethod)

    def test_vllm_config_carries_same_quant_config_used_by_layer(self):
        quant_config = _online_fp8()

        with _vllm_config_scope(
            torch.bfloat16,
            quant_config,
        ) as vllm_config:
            method = quant_config.get_quant_method(
                _dummy_replicated_linear(),
                prefix="pid.test.ff.w2",
            )

            assert vllm_config.quant_config is quant_config
            assert vllm_config.model_config.dtype == torch.bfloat16
            assert method.input_dtype == vllm_config.model_config.dtype


# =============================================================================
# 5. CUDA: actual vLLM FP8 kernel execution
# =============================================================================


@hardware_test(res={"cuda": "L4"})
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA",
)
@pytest.mark.parametrize(
    ("ignored_layers", "expected_fp8"),
    [
        ([], {"w1", "w3", "w2"}),
        (["pid.test.ff.w1"], {"w3", "w2"}),
        (
            ["pid.test.ff.w1", "pid.test.ff.w2"],
            {"w3"},
        ),
    ],
    ids=[
        "all-fp8",
        "ignore-w1",
        "only-w3-fp8",
    ],
)
def test_online_fp8_real_kernel_dispatch_and_layer_control(
    default_vllm_config,
    init_fake_tp_group,
    ignored_layers,
    expected_fp8,
):
    """Run a real tiny PiD FeedForward through vLLM's selected CUDA kernel."""
    dtype = torch.bfloat16
    quant_config = _online_fp8(ignored_layers=ignored_layers)

    with _torch_default_dtype(dtype):
        with _vllm_config_scope(dtype, quant_config):
            # Make ReplicatedLinear parameters/biases land on CUDA.
            with torch.device("cuda"):
                ff = FeedForward(
                    dim=128,
                    hidden_dim=192,  # FeedForward converts this to 128.
                    quant_config=quant_config,
                    prefix="pid.test.ff",
                )

            linears = _all_ff_linears(ff)

            for linear in linears.values():
                _materialize_linear_for_cuda(linear)

            selected_kernels = {}

            for name, linear in linears.items():
                method = linear.quant_method
                if isinstance(method, Fp8PerTensorOnlineLinearMethod):
                    selected_kernels[name] = type(method.fp8_linear).__name__
                else:
                    assert isinstance(method, UnquantizedLinearMethod)

            assert set(selected_kernels) == expected_fp8

            x = torch.randn(
                2,
                4,
                128,
                device="cuda",
                dtype=dtype,
            )
            output = ff(x)

            assert output.shape == x.shape
            assert torch.isfinite(output).all()

            # Fp8PerTensorOnlineLinearMethod.apply delegates to the selected
            # fp8_linear kernel. No kernel call is mocked in this test.
            print(f"online FP8 kernels: {selected_kernels}")


@hardware_test(res={"cuda": "L4"})
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA",
)
def test_serialized_fp8_real_kernel_dispatch(
    default_vllm_config,
    init_fake_tp_group,
):
    """Exercise the pre-quantized/serialized FP8 LinearMethod path."""
    dtype = torch.bfloat16
    quant_config = _serialized_fp8()

    with _torch_default_dtype(dtype):
        with _vllm_config_scope(dtype, quant_config):
            with torch.device("cuda"):
                ff = FeedForward(
                    dim=128,
                    hidden_dim=192,
                    quant_config=quant_config,
                    prefix="pid.test.ff",
                )

            linears = _all_ff_linears(ff)

            selected_kernels = {}

            for name, linear in linears.items():
                assert isinstance(linear.quant_method, Fp8LinearMethod)
                _materialize_linear_for_cuda(linear)

                kernel = linear.quant_method.fp8_linear
                selected_kernels[name] = type(kernel).__name__

            x = torch.randn(
                2,
                4,
                128,
                device="cuda",
                dtype=dtype,
            )
            output = ff(x)

            assert output.shape == x.shape
            assert torch.isfinite(output).all()

            print(f"serialized FP8 kernels: {selected_kernels}")


# =============================================================================
# 6. One top-to-bottom contract test
# =============================================================================


@hardware_test(res={"cuda": "L4"})
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA",
)
def test_pid_component_config_reaches_real_cuda_kernel(
    default_vllm_config,
    init_fake_tp_group,
):
    """Config -> component resolve -> PiD layer -> quant method -> CUDA kernel.

    This is deliberately small: it proves the complete quantization contract
    without loading the Qwen-Image transformer, Gemma, or PiD checkpoint.
    """
    od_config = OmniDiffusionConfig(
        model="test",
        quantization_config={
            "transformer": None,
            "pid": {"method": "fp8"},
        },
    )
    component_config = od_config.quantization_config

    assert isinstance(component_config, ComponentQuantizationConfig)
    pid_quant_config = component_config.resolve("pid")
    assert isinstance(pid_quant_config, Fp8Config)

    dtype = torch.bfloat16

    with _torch_default_dtype(dtype):
        with _vllm_config_scope(dtype, pid_quant_config):
            with torch.device("cuda"):
                ff = FeedForward(
                    dim=128,
                    hidden_dim=192,
                    quant_config=pid_quant_config,
                    prefix="pid.integration.ff",
                )

            for linear in _all_ff_linears(ff).values():
                assert isinstance(
                    linear.quant_method,
                    Fp8PerTensorOnlineLinearMethod,
                )
                _materialize_linear_for_cuda(linear)

            kernel = ff.w1.quant_method.fp8_linear

            x = torch.randn(
                1,
                2,
                128,
                device="cuda",
                dtype=dtype,
            )
            output = ff(x)

            assert output.shape == x.shape
            assert torch.isfinite(output).all()

            print(
                "PiD integration kernel:",
                type(kernel).__name__,
            )
