# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request fields supported by the MAGI-2 Preview pipeline."""

from __future__ import annotations

MAGI2_EXTRA_BODY_PARAMS = frozenset(
    {
        "seconds",
        "resolution",
        "image_path",
        "output_width",
        "output_height",
        "deterministic",
    }
)
MAGI2_EXTRA_OUTPUT_PARAMS = frozenset()
