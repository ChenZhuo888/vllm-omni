# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization-config routing tests for Qwen-Image + PiD.

These tests intentionally exercise the production QwenImagePipeline constructor
while mocking heavyweight model/tokenizer/checkpoint loading.

The contract under test is:

1. A normal QuantizationConfig keeps Qwen-Image's historical behavior:
   it is applied to the diffusion transformer, while PiD stays unquantized.

2. A ComponentQuantizationConfig is resolved at the Qwen-Image component
   boundary:
       "transformer" -> QwenImageTransformer2DModel
       "pid"         -> PidDecoder

3. The resolved PiD config is passed through PidDecodeMixin unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.pid.decoder import PidDecodeConfig
from vllm_omni.quantization import (
    ComponentQuantizationConfig,
    build_quant_config,
)


def _make_od_config(quantization_config):
    """Minimal OmniDiffusionConfig-shaped object for pipeline construction."""
    return SimpleNamespace(
        model="dummy-qwen-image",
        revision=None,
        parallel_config=SimpleNamespace(),
        tf_model_config=SimpleNamespace(),
        quantization_config=quantization_config,
        enable_diffusion_pipeline_profiler=False,
        enforce_eager=True,
        pid_decode=PidDecodeConfig(
            enabled=True,
            checkpoint_path="/unused/pid.pth",
            gemma_model="/unused/gemma",
        ),
    )


class _DummyTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(visual=object())

    def to(self, *args, **kwargs):
        return self


class _DummyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperal_downsample = [True, True, True]

    def to(self, *args, **kwargs):
        return self


class _CapturedTransformer(nn.Module):
    """Replacement for QwenImageTransformer2DModel that records quant_config."""

    def __init__(self, *, od_config, quant_config=None, **kwargs):
        super().__init__()
        self.received_quant_config = quant_config


def _build_pipeline(monkeypatch, quantization_config):
    """Construct the real QwenImagePipeline with heavyweight deps mocked."""
    import vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image as qwen_pipeline

    monkeypatch.setattr(
        qwen_pipeline,
        "get_local_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        qwen_pipeline,
        "prefetch_subfolders",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        qwen_pipeline.FlowMatchEulerDiscreteScheduler,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    def fake_from_pretrained_with_prefetch(fn, *args, subfolder=None, **kwargs):
        if subfolder == "text_encoder":
            return _DummyTextEncoder()
        if subfolder == "vae":
            return _DummyVAE()
        raise AssertionError(f"unexpected subfolder: {subfolder}")

    monkeypatch.setattr(
        qwen_pipeline,
        "from_pretrained_with_prefetch",
        fake_from_pretrained_with_prefetch,
    )
    monkeypatch.setattr(
        qwen_pipeline.Qwen2Tokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        qwen_pipeline,
        "get_transformer_config_kwargs",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        qwen_pipeline,
        "QwenImageTransformer2DModel",
        _CapturedTransformer,
    )
    monkeypatch.setattr(
        qwen_pipeline.QwenImagePipeline,
        "setup_diffusion_pipeline_profiler",
        lambda self, **kwargs: None,
    )

    # Do not load the real PiD checkpoint here. We only want to observe what
    # config QwenImagePipeline passes into the PiD component boundary.
    def capture_pid_init(self, od_config, quant_config=None):
        self._captured_pid_quant_config = quant_config

    monkeypatch.setattr(
        qwen_pipeline.QwenImagePipeline,
        "_init_pid_decoder",
        capture_pid_init,
    )

    pipeline = qwen_pipeline.QwenImagePipeline(
        od_config=_make_od_config(quantization_config)
    )
    return pipeline


def _assert_fp8(config):
    assert config is not None
    assert config.get_name() == "fp8"


@pytest.mark.parametrize(
    ("quantization_config", "transformer_expected", "pid_expected"),
    [
        pytest.param(
            None,
            None,
            None,
            id="no-quantization",
        ),
        pytest.param(
            build_quant_config("fp8"),
            "fp8",
            None,
            id="plain-fp8-transformer-only",
        ),
        pytest.param(
            build_quant_config(
                {
                    "transformer": {"method": "fp8"},
                    "pid": None,
                }
            ),
            "fp8",
            None,
            id="component-transformer-only",
        ),
        pytest.param(
            build_quant_config(
                {
                    "transformer": None,
                    "pid": {"method": "fp8"},
                }
            ),
            None,
            "fp8",
            id="component-pid-only",
        ),
        pytest.param(
            build_quant_config(
                {
                    "transformer": {"method": "fp8"},
                    "pid": {"method": "fp8"},
                }
            ),
            "fp8",
            "fp8",
            id="component-transformer-and-pid",
        ),
        pytest.param(
            build_quant_config(
                {
                    "pid": None,
                    "default": {"method": "fp8"},
                }
            ),
            "fp8",
            None,
            id="default-fp8-explicit-pid-none",
        ),
        pytest.param(
            build_quant_config(
                {
                    "transformer": None,
                    "default": {"method": "fp8"},
                }
            ),
            None,
            "fp8",
            id="default-fp8-explicit-transformer-none",
        ),
    ],
)
def test_qwen_image_routes_quantization_config_by_component(
    monkeypatch,
    quantization_config,
    transformer_expected,
    pid_expected,
):
    """QwenImagePipeline must hand each component its resolved concrete config."""
    pipeline = _build_pipeline(monkeypatch, quantization_config)

    transformer_config = pipeline.transformer.received_quant_config
    pid_config = pipeline._captured_pid_quant_config

    # The subcomponents should receive concrete QuantizationConfig | None,
    # never the ComponentQuantizationConfig router itself.
    assert not isinstance(transformer_config, ComponentQuantizationConfig)
    assert not isinstance(pid_config, ComponentQuantizationConfig)

    if transformer_expected is None:
        assert transformer_config is None
    else:
        _assert_fp8(transformer_config)

    if pid_expected is None:
        assert pid_config is None
    else:
        _assert_fp8(pid_config)


def test_plain_quant_config_preserves_original_transformer_object(monkeypatch):
    """A non-component config must keep Qwen-Image's original passthrough behavior."""
    fp8 = build_quant_config("fp8")

    pipeline = _build_pipeline(monkeypatch, fp8)

    assert pipeline.transformer.received_quant_config is fp8
    assert pipeline._captured_pid_quant_config is None


def test_component_resolution_preserves_exact_child_objects(monkeypatch):
    """Component resolution should pass the exact child configs, not rebuild them."""
    component = build_quant_config(
        {
            "transformer": {"method": "fp8"},
            "pid": {"method": "fp8"},
        }
    )
    assert isinstance(component, ComponentQuantizationConfig)

    expected_transformer = component.resolve("transformer")
    expected_pid = component.resolve("pid")

    pipeline = _build_pipeline(monkeypatch, component)

    assert pipeline.transformer.received_quant_config is expected_transformer
    assert pipeline._captured_pid_quant_config is expected_pid


def test_pid_mixin_passes_resolved_config_to_decoder(monkeypatch):
    """PidDecodeMixin must consume, not re-route, the already-resolved config."""
    import vllm_omni.diffusion.pid.mixin as pid_mixin

    fp8 = build_quant_config("fp8")

    class FakePidDecoder(nn.Module):
        def __init__(
            self,
            config,
            backbone,
            enforce_eager=False,
            quant_config=None,
        ):
            super().__init__()
            self.config = config
            self.backbone = backbone
            self.enforce_eager = enforce_eager
            self.quant_config = quant_config
            self.load_weights_called = False

        def load_weights(self):
            self.load_weights_called = True

    monkeypatch.setattr(pid_mixin, "PidDecoder", FakePidDecoder)

    class Host(nn.Module, pid_mixin.PidDecodeMixin):
        PID_BACKBONE = "qwenimage"

        def __init__(self):
            super().__init__()

    host = Host()
    od_config = SimpleNamespace(
        pid_decode=PidDecodeConfig(
            enabled=True,
            checkpoint_path="/unused/pid.pth",
            gemma_model="/unused/gemma",
        ),
        enforce_eager=True,
    )

    host._init_pid_decoder(
        od_config,
        quant_config=fp8,
    )

    assert host._pid_decoder is not None
    assert host._pid_decoder.quant_config is fp8
    assert host._pid_decoder.load_weights_called
    assert "_pid_decoder" in host._resident_modules


def test_pid_mixin_accepts_none_quant_config(monkeypatch):
    """Explicitly unquantized PiD should propagate None all the way to decoder."""
    import vllm_omni.diffusion.pid.mixin as pid_mixin

    class FakePidDecoder(nn.Module):
        def __init__(
            self,
            config,
            backbone,
            enforce_eager=False,
            quant_config=None,
        ):
            super().__init__()
            self.quant_config = quant_config

        def load_weights(self):
            pass

    monkeypatch.setattr(pid_mixin, "PidDecoder", FakePidDecoder)

    class Host(nn.Module, pid_mixin.PidDecodeMixin):
        PID_BACKBONE = "qwenimage"

        def __init__(self):
            super().__init__()

    host = Host()
    od_config = SimpleNamespace(
        pid_decode=PidDecodeConfig(
            enabled=True,
            checkpoint_path="/unused/pid.pth",
            gemma_model="/unused/gemma",
        ),
        enforce_eager=True,
    )

    host._init_pid_decoder(
        od_config,
        quant_config=None,
    )

    assert host._pid_decoder is not None
    assert host._pid_decoder.quant_config is None
