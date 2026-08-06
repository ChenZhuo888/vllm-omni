# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.magi2.attention import (
    correct_out_lse_with_sink,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def test_sink_correction_matches_explicit_zero_value_tokens():
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn(2, 5, 3, generator=generator)
    values = torch.randn(2, 5, 3, 4, generator=generator)
    sinks = torch.randn(2, 3, generator=generator)

    real_weights = logits.softmax(dim=1)
    plain_out = (real_weights.unsqueeze(-1) * values).sum(dim=1)
    real_lse = torch.logsumexp(logits, dim=1)
    corrected, corrected_lse = correct_out_lse_with_sink(
        plain_out,
        real_lse.transpose(0, 1),
        sinks,
    )

    sink_logits = sinks.unsqueeze(0).expand(2, -1, -1)
    explicit_lse = torch.logsumexp(
        torch.cat((logits, sink_logits), dim=1),
        dim=1,
    )
    expected = plain_out * torch.exp(real_lse - explicit_lse).unsqueeze(-1)
    torch.testing.assert_close(corrected, expected)
    torch.testing.assert_close(corrected_lse.transpose(0, 1), explicit_lse)


def test_sink_correction_without_sinks_is_identity():
    out = torch.randn(4, 2, 8)
    lse = torch.randn(2, 4)
    corrected, corrected_lse = correct_out_lse_with_sink(out, lse, None)
    assert corrected is out
    assert corrected_lse is lse


def test_sink_correction_validates_head_count():
    with pytest.raises(ValueError, match="head counts differ"):
        correct_out_lse_with_sink(
            torch.randn(4, 2, 8),
            torch.randn(2, 4),
            torch.randn(1, 3),
        )
