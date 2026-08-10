# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simple checkpoint loader for PiD model weights."""

from __future__ import annotations

import logging
from collections import OrderedDict

import torch

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
    state_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )

    net_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("net.") and not k.startswith("net_ema."):
            net_sd[k[len("net."):]] = v

    missing, unexpected = model.net.load_state_dict(net_sd, strict=False)

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
