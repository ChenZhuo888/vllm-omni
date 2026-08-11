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


def test_shared_text_to_video_forwards_hsdp_cli(text_to_video: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(text_to_video.__file__),
            "--ulysses-degree",
            "3",
            "--use-hsdp",
            "--hsdp-shard-size",
            "3",
        ],
    )

    config = text_to_video._parallel_config_from_args(text_to_video.parse_args())

    assert config.ulysses_degree == 3
    assert config.use_hsdp is True
    assert config.hsdp_shard_size == 3
