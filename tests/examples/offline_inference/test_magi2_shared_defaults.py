# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

REPO_ROOT = Path(__file__).parents[3]


def _load_example(name: str, relative_path: str) -> ModuleType:
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def text_to_video() -> ModuleType:
    return _load_example(
        "magi2_text_to_video_example",
        "examples/offline_inference/text_to_video/text_to_video.py",
    )


@pytest.fixture(scope="module")
def image_to_video() -> ModuleType:
    return _load_example(
        "magi2_image_to_video_example",
        "examples/offline_inference/image_to_video/image_to_video.py",
    )


@pytest.mark.parametrize(
    ("extra_body", "expected_size"),
    [
        (None, (896, 512)),
        ({"resolution": "540p"}, (896, 512)),
        ({"resolution": "272p"}, (448, 256)),
    ],
)
def test_shared_text_to_video_uses_native_preview_defaults(
    text_to_video: ModuleType,
    extra_body: dict[str, object] | None,
    expected_size: tuple[int, int],
) -> None:
    preset = text_to_video._detect_preset(
        "sand-ai/MAGI-2-preview",
        "Magi2Pipeline",
        extra_body,
    )
    assert (preset["width"], preset["height"]) == expected_size
    assert preset["num_frames"] == 125
    assert preset["num_inference_steps"] == 100
    assert preset["fps"] == 12.5


@pytest.mark.parametrize(
    ("extra_body", "expected_size"),
    [
        (None, (896, 512)),
        ({"resolution": "540p"}, (896, 512)),
        ({"resolution": "272p"}, (448, 256)),
    ],
)
def test_shared_image_to_video_uses_native_preview_defaults(
    image_to_video: ModuleType,
    extra_body: dict[str, object] | None,
    expected_size: tuple[int, int],
) -> None:
    assert image_to_video._magi2_preview_dimensions(extra_body) == expected_size


@pytest.mark.parametrize("resolution", ["720p", "preview", "unknown"])
def test_shared_examples_reject_non_preview_resolution(
    text_to_video: ModuleType,
    image_to_video: ModuleType,
    resolution: str,
) -> None:
    extra_body = {"resolution": resolution}
    with pytest.raises(ValueError, match="272p.*540p"):
        text_to_video._magi2_preview_preset(extra_body)
    with pytest.raises(ValueError, match="272p.*540p"):
        image_to_video._magi2_preview_dimensions(extra_body)


@pytest.mark.parametrize("example_fixture", ["text_to_video", "image_to_video"])
def test_shared_video_dlo_cli_defaults(
    example_fixture: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = request.getfixturevalue(example_fixture)
    monkeypatch.setattr(sys, "argv", [str(example.__file__)])

    args = example.parse_args()

    assert args.enable_distributed_layerwise_offload is False
    assert args.dlo_use_allgather is True
    assert args.dlo_resident_layers == 0
    assert example._distributed_layerwise_offload_kwargs(args) == {
        "enable_distributed_layerwise_offload": False,
        "dlo_use_allgather": True,
        "dlo_resident_layers": 0,
    }


@pytest.mark.parametrize("example_fixture", ["text_to_video", "image_to_video"])
def test_shared_video_dlo_cli_forwards_rank_local_options(
    example_fixture: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = request.getfixturevalue(example_fixture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(example.__file__),
            "--enable-distributed-layerwise-offload",
            "--dlo-no-use-allgather",
            "--dlo-resident-layers",
            "7",
        ],
    )

    args = example.parse_args()

    assert example._distributed_layerwise_offload_kwargs(args) == {
        "enable_distributed_layerwise_offload": True,
        "dlo_use_allgather": False,
        "dlo_resident_layers": 7,
    }
