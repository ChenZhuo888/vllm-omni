# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.models.magi2.pipeline_magi2 import (
    MAGI2_AUDIO_SAMPLE_RATE,
    MAGI2_MODEL_REVISION,
    Magi2Pipeline,
    _magi2_post_process,
    _Magi2StagedComponent,
    _resolve_checkpoint_root,
    _single_image,
    _validate_native_topology,
)
from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model
from vllm_omni.errors import OmniClientError
from vllm_omni.model_extras.registry import get_extra_body_params

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class _FakeNativeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        video = np.zeros((2, 8, 12, 3), dtype=np.uint8)
        audio = np.zeros((32, 2), dtype=np.float32)
        return video, audio


def _pipeline() -> tuple[Magi2Pipeline, _FakeNativeRuntime]:
    pipe = Magi2Pipeline.__new__(Magi2Pipeline)
    nn.Module.__init__(pipe)
    pipe.deterministic = False
    runtime = _FakeNativeRuntime()

    def evaluate(_self, **kwargs):
        return runtime.evaluate(**kwargs)

    pipe._evaluate_preview = MethodType(evaluate, pipe)
    return pipe, runtime


def _request(prompt, **sampling_overrides):
    sampling = SimpleNamespace(
        width=None,
        height=None,
        fps=None,
        num_frames=1,
        num_inference_steps=None,
        num_outputs_per_prompt=1,
        seed=42,
        extra_args={},
    )
    for key, value in sampling_overrides.items():
        setattr(sampling, key, value)
    return SimpleNamespace(num_reqs=1, prompts=[prompt], sampling_params=sampling)


def _checkpoint_tree(root: Path) -> None:
    files = (
        "preview/model.safetensors.index.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "vae/Wan2.2_VAE.pth",
        "turbo_vae/TurboV3-Wan22-TinyShallow_7_7.json",
        "turbo_vae/checkpoint.ckpt",
        "stable-audio-open-1.0/model_config.json",
        "stable-audio-open-1.0/model.safetensors",
    )
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_official_id_detection_and_metadata():
    assert is_diffusion_model("sand-ai/MAGI-2-preview")
    assert is_diffusion_model("https://huggingface.co/sand-ai/MAGI-2-preview")
    assert resolve_model_class_name("sand-ai/MAGI-2-preview") == "Magi2Pipeline"
    assert DiffusionModelRegistry._try_load_model_cls("Magi2Pipeline") is Magi2Pipeline
    metadata = get_diffusion_model_metadata("Magi2Pipeline")
    assert metadata.supports_multimodal_inputs
    assert metadata.max_multimodal_image_inputs == 1
    assert {"seconds", "resolution"} <= get_extra_body_params("Magi2Pipeline")


def test_local_preview_signature_detection_without_refiner(tmp_path):
    _checkpoint_tree(tmp_path)
    assert is_diffusion_model(str(tmp_path))
    assert resolve_model_class_name(str(tmp_path)) == "Magi2Pipeline"
    config = OmniDiffusionConfig(model=str(tmp_path))
    config.enrich_config()
    assert config.model_class_name == "Magi2Pipeline"
    assert config.supports_multimodal_inputs


def test_huggingface_url_resolves_pinned_snapshot(tmp_path, monkeypatch):
    _checkpoint_tree(tmp_path)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *, repo_id, revision: (
            str(tmp_path)
            if (repo_id, revision) == ("sand-ai/MAGI-2-preview", MAGI2_MODEL_REVISION)
            else pytest.fail("unexpected snapshot request")
        ),
    )
    assert _resolve_checkpoint_root(
        "https://huggingface.co/sand-ai/MAGI-2-preview",
        None,
    ) == str(tmp_path.resolve())


def test_forward_uses_native_540p_preview_defaults(monkeypatch):
    pipe, runtime = _pipeline()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = pipe(_request("A fox walks through snow"))

    call = runtime.calls[0]
    assert call["prompt"] == "A fox walks through snow"
    assert call["image"] is None
    assert (call["width"], call["height"]) == (896, 512)
    assert call["num_inference_steps"] == 100
    assert result.output["metadata"]["video"]["fps"] == 12.5
    assert result.output["metadata"]["audio"]["sample_rate"] == MAGI2_AUDIO_SAMPLE_RATE
    assert result.stage_durations["magi2_preview_e2e"] >= 0


def test_forward_maps_272p_i2v_and_output_resize(monkeypatch):
    pipe, runtime = _pipeline()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    image = Image.new("RGB", (16, 9), "white")
    prompt = {
        "prompt": "The first frame begins moving",
        "multi_modal_data": {"image": image},
    }
    result = pipe(
        _request(
            prompt,
            num_inference_steps=1,
            extra_args={
                "resolution": "272p",
                "output_width": 6,
                "output_height": 4,
            },
        )
    )

    call = runtime.calls[0]
    assert call["image"] is image
    assert (call["width"], call["height"]) == (448, 256)
    assert call["num_inference_steps"] == 1
    assert result.output["payload"]["video"].shape == (2, 4, 6, 3)


def test_forward_uses_generator_seed(monkeypatch):
    pipe, _ = _pipeline()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    seeded: list[int] = []
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2._seed_request",
        lambda seed, deterministic: seeded.append(seed),
    )
    pipe(
        _request(
            "A fox walks through snow",
            seed=None,
            generator=torch.Generator().manual_seed(7),
        )
    )
    assert seeded == [7]


@pytest.mark.parametrize(
    ("prompt", "extra_args", "message"),
    [
        ("", {}, "non-empty prompt"),
        ("prompt", {"seconds": 5}, "10-second clips"),
        ("prompt", {"duration": 5}, "10-second clips"),
        ("prompt", {"resolution": "720p"}, "Unsupported native"),
        ("prompt", {"resolution": "1080p"}, "Unsupported native"),
        ("prompt", {"resolution": "540p", "use_refiner": True}, "refiner"),
    ],
)
def test_forward_rejects_invalid_preview_requests(
    monkeypatch,
    prompt,
    extra_args,
    message,
):
    pipe, runtime = _pipeline()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(OmniClientError, match=message):
        pipe(_request(prompt, extra_args=extra_args))
    assert not runtime.calls


def test_forward_rejects_multiple_images(monkeypatch):
    pipe, _ = _pipeline()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    prompt = {
        "prompt": "animate",
        "multi_modal_data": {"image": [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]},
    }
    with pytest.raises(OmniClientError, match="at most one input image"):
        pipe(_request(prompt))


def test_pathlike_image_is_normalized():
    assert _single_image(Path("first-frame.png")) == "first-frame.png"


def test_forward_rejects_multiple_outputs(monkeypatch):
    pipe, _ = _pipeline()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(OmniClientError, match="exactly one output"):
        pipe(_request("animate", num_outputs_per_prompt=2))


def test_post_process_keeps_dynamic_metadata():
    payload = {
        "video": "v",
        "audio": "a",
        "audio_sample_rate": 44100,
        "fps": 12.5,
    }
    assert _magi2_post_process(payload) == payload


def _topology_config(**overrides):
    parallel = SimpleNamespace(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        tensor_parallel_size=1,
        sequence_parallel_size=1,
        ulysses_degree=1,
        ring_degree=1,
        allgather_degree=1,
        cfg_parallel_size=1,
        vae_patch_parallel_size=1,
        text_encoder_tp_size=1,
        enable_expert_parallel=False,
        use_hsdp=False,
    )
    config = SimpleNamespace(
        parallel_config=parallel,
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
        dlo_use_allgather=True,
        quantization_config=None,
        cache_backend="none",
        custom_pipeline_args={},
        additional_config={"magi2_allow_unsupported_topology": True},
    )
    for key, value in overrides.items():
        if hasattr(parallel, key):
            setattr(parallel, key, value)
        else:
            setattr(config, key, value)
    return config


def test_native_topology_rejects_tensor_parallelism():
    config = _topology_config(tensor_parallel_size=2)
    with pytest.raises(ValueError, match="tensor_parallel_size=2"):
        _validate_native_topology(config)


def test_native_topology_requires_dlo_rank_local_mode():
    config = _topology_config(enable_distributed_layerwise_offload=True)
    with pytest.raises(ValueError, match="dlo-no-use-allgather"):
        _validate_native_topology(config)
    config.dlo_use_allgather = False
    _validate_native_topology(config)


def test_offload_plan_stages_every_auxiliary_component():
    assert Magi2Pipeline._offload_plan.on_demand_component_paths == frozenset(
        {
            "text_encoder",
            "image_vae",
            "video_decoder",
            "audio_decoder",
        }
    )


def test_staged_component_keeps_one_cpu_master(monkeypatch):
    inner = nn.Linear(2, 2)
    stager = Mock()
    stager_factory = Mock(return_value=stager)
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2.PinnedModuleStager",
        stager_factory,
    )

    component = _Magi2StagedComponent(
        inner,
        torch.device("cuda:0"),
        pin_memory=False,
    )
    component.load_to_device()
    component.offload_to_cpu()

    assert component.module is inner
    stager_factory.assert_called_once_with(
        inner,
        torch.device("cuda:0"),
        pin_memory=False,
    )
    stager.load.assert_called_once_with()
    stager.offload.assert_called_once_with()


def test_component_stage_releases_on_failure(monkeypatch):
    stager = Mock()
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2.PinnedModuleStager",
        Mock(return_value=stager),
    )
    component = _Magi2StagedComponent(
        nn.Identity(),
        torch.device("cuda:0"),
        pin_memory=False,
    )
    pipeline = object.__new__(Magi2Pipeline)
    nn.Module.__init__(pipeline)
    pipeline._offload_aux_after_use = True

    with pytest.raises(RuntimeError, match="encode failed"):
        with pipeline._component_on_device(component) as resident:
            assert resident is component.module
            stager.load.assert_called_once_with()
            stager.offload.assert_not_called()
            raise RuntimeError("encode failed")

    stager.offload.assert_called_once_with()


def test_prompt_pair_uses_one_text_encoder_residency_window(monkeypatch):
    stager = Mock()
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2.PinnedModuleStager",
        Mock(return_value=stager),
    )
    text_encoder = nn.Module()
    text_encoder.encode = Mock(side_effect=(torch.tensor([[1.0]]), torch.tensor([[2.0]])))
    component = _Magi2StagedComponent(
        text_encoder,
        torch.device("cuda:0"),
        pin_memory=False,
    )
    pipeline = object.__new__(Magi2Pipeline)
    nn.Module.__init__(pipeline)
    pipeline._is_output_rank = True
    pipeline._offload_aux_after_use = True
    pipeline._parallel_group = SimpleNamespace(world_size=1)
    pipeline.device_str = "cpu"
    pipeline.dtype = torch.float32
    pipeline.text_encoder = component

    positive, negative = pipeline._encode_prompts(("positive", "negative"))

    assert torch.equal(positive, torch.tensor([[1.0]]))
    assert torch.equal(negative, torch.tensor([[2.0]]))
    assert text_encoder.encode.call_args_list == [
        (("positive",),),
        (("negative",),),
    ]
    stager.load.assert_called_once_with()
    stager.offload.assert_called_once_with()


def test_decode_audio_preserves_batch_for_released_preview_shape(monkeypatch):
    class ShapeCheckingDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_shape: tuple[int, ...] | None = None

        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            self.seen_shape = tuple(latent.shape)
            if self.seen_shape != (1, 64, 250):
                raise AssertionError(f"unexpected decoder input shape {self.seen_shape}")
            return torch.zeros(1, 2, 512_000)

    decoder = ShapeCheckingDecoder()
    stager = Mock()
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2.PinnedModuleStager",
        Mock(return_value=stager),
    )
    component = _Magi2StagedComponent(
        decoder,
        torch.device("cpu"),
        pin_memory=False,
    )
    pipeline = object.__new__(Magi2Pipeline)
    nn.Module.__init__(pipeline)
    pipeline._is_output_rank = True
    pipeline._offload_aux_after_use = True
    pipeline.device_str = "cpu"
    pipeline.audio_decoder = component

    def fake_resample(audio: np.ndarray, target_length: int) -> np.ndarray:
        assert audio.shape == (512_000, 2)
        assert target_length == 441_000
        return np.zeros((target_length, 2), dtype=audio.dtype)

    monkeypatch.setattr("scipy.signal.resample", fake_resample)
    audio = pipeline._decode_audio(torch.zeros(1, 250, 64))

    assert decoder.seen_shape == (1, 64, 250)
    assert audio is not None
    assert audio.shape == (441_000, 2)
    stager.load.assert_called_once_with()
    stager.offload.assert_called_once_with()


class _TinyMagiTransformer(nn.Module):
    _layerwise_offload_blocks_attrs = ["block"]

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])


class _NoEagerMoveComponent(_Magi2StagedComponent):
    def to(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("staged MAGI-2 component was moved eagerly")


def _offload_pipeline(monkeypatch):
    stagers = [Mock() for _ in range(4)]
    stager_factory = Mock(side_effect=stagers)
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2.PinnedModuleStager",
        stager_factory,
    )
    pipeline = object.__new__(Magi2Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = _TinyMagiTransformer()
    components = [
        _NoEagerMoveComponent(
            nn.Linear(2, 2),
            torch.device("cuda:0"),
            pin_memory=False,
        )
        for _ in range(4)
    ]
    (
        pipeline.text_encoder,
        pipeline.image_vae,
        pipeline.video_decoder,
        pipeline.audio_decoder,
    ) = components
    return pipeline, stagers


class _FakeLayerwiseHook:
    def __init__(self) -> None:
        self._prev_hook = None
        self.current_slot = 0
        self.prefetch_layer = Mock()
        self.get_weights = Mock()


def test_ordinary_layerwise_enable_preserves_staged_aux_hierarchy(
    monkeypatch,
):
    from vllm_omni.diffusion.offloader.base import (
        OffloadConfig,
        OffloadStrategy,
    )
    from vllm_omni.diffusion.offloader.layerwise_backend import (
        LayerWiseOffloadBackend,
    )

    pipeline, stagers = _offload_pipeline(monkeypatch)
    hooks: list[_FakeLayerwiseHook] = []

    def fake_apply(*args, **kwargs):
        del args, kwargs
        hook = _FakeLayerwiseHook()
        hooks.append(hook)
        return hook

    monkeypatch.setattr(
        "vllm_omni.diffusion.offloader.layerwise_backend.apply_block_hook",
        fake_apply,
    )
    backend = LayerWiseOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.LAYER_WISE,
            pin_cpu_memory=False,
        ),
        torch.device("cpu"),
    )

    backend.enable(pipeline)

    assert backend.enabled
    assert len(hooks) == len(pipeline.transformer.block)
    assert all(stager.offload.call_count == 1 for stager in stagers)
    assert all(stager.load.call_count == 0 for stager in stagers)


def test_dlo_no_allgather_enable_preserves_staged_aux_and_streams_blocks(
    monkeypatch,
):
    from vllm_omni.diffusion.offloader.base import (
        OffloadConfig,
        OffloadStrategy,
    )
    from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
        DistributedLayerwiseOffloadBackend,
    )

    pipeline, stagers = _offload_pipeline(monkeypatch)
    hooks: list[_FakeLayerwiseHook] = []

    def fake_apply(*args, **kwargs):
        del args, kwargs
        hook = _FakeLayerwiseHook()
        hooks.append(hook)
        return hook

    monkeypatch.setattr(
        "vllm_omni.diffusion.offloader.distributed_layerwise_backend.apply_distributed_block_hook",
        fake_apply,
    )
    backend = DistributedLayerwiseOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
            pin_cpu_memory=False,
            dlo_use_allgather=False,
        ),
        torch.device("cpu"),
    )
    backend._allocate_shared_buffers = Mock(return_value=[{}, {}])
    backend._cleanup_after_loading = Mock()
    backend._release_mmap_handles = Mock()

    backend.enable(pipeline)

    assert backend.enabled
    assert backend.dp_size == 1
    assert len(hooks) == len(pipeline.transformer.block)
    assert all(stager.offload.call_count == 1 for stager in stagers)
    assert all(stager.load.call_count == 0 for stager in stagers)
    hooks[-1].prefetch_layer.assert_called_once_with(
        slot=hooks[0].current_slot,
        non_blocking=False,
    )
