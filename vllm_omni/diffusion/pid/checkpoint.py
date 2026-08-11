# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simple checkpoint loader for PiD model weights."""

from __future__ import annotations

import logging
from collections import OrderedDict

import torch

from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = logging.getLogger(__name__)


def load_pid_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> None:
    """Load PiD checkpoint, stripping 'net.' prefix from state dict keys.

    PiD checkpoints saved via ``PidDistillModel.state_dict()`` have all keys
    prefixed with ``"net."``. We strip this to match ``PidInferenceModel.net``
    (a ``PidNet`` instance).

    Missing LQ-projection keys are expected when loading a checkpoint that
    was fine-tuned from a base T2I model (LQ modules are zero-init anyway).
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    net_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("net.") and not k.startswith("net_ema."):
            name = k[len("net."):]
            name = _remap_pid_checkpoint_key(name)
            net_sd[name] = v

    params_dict = dict(model.net.named_parameters())

    loaded_params = set()
    unexpected = []

    for name, loaded_weight in net_sd.items():
        param = params_dict.get(name)

        if param is None:
            unexpected.append(name)
            continue

        weight_loader = getattr(
            param,
            "weight_loader",
            default_weight_loader,
        )
        weight_loader(param, loaded_weight)

        loaded_params.add(name)

    missing = [
        name
        for name in params_dict
        if name not in loaded_params
    ]

    lq_missing = [k for k in missing if "lq_proj" in k or "pit_lq" in k]
    other_missing = [k for k in missing if "lq_proj" not in k and "pit_lq" not in k]

    if lq_missing:
        logger.info(
            "Expected missing LQ keys (%d keys) -- LQ modules are zero-init.",
            len(lq_missing),
        )
    if other_missing:
        logger.warning("Missing keys: %s", other_missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)

def _remap_pid_checkpoint_key(key: str) -> str:
    key = key.replace(
        ".adaLN_modulation_img.",
        ".ada_ln_modulation_img.",
    )
    key = key.replace(
        ".adaLN_modulation_txt.",
        ".ada_ln_modulation_txt.",
    )
    return key
