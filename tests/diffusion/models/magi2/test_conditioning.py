# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json

from vllm_omni.diffusion.models.magi2.conditioning import (
    json_to_compact_markdown,
    normalize_prompt,
)


def test_normalize_prompt_leaves_plain_text_unchanged():
    prompt = "  A fox walks through fresh snow.  "
    assert normalize_prompt(prompt) == prompt


def test_json_prompt_matches_magi_compact_format():
    prompt = {
        "global_layer": {
            "context": "  winter   forest ",
            "description": "A fox listens.",
            "aesthetics": {
                "style": "cinematic",
                "mood_atmosphere": "quiet",
                "color_scheme": "blue and amber",
            },
            "audio_baseline": {
                "dialogue": {"language": "English", "speaker_tags": ["narrator"]},
                "ambience": "wind",
            },
            "alive_subjects_static": [
                {
                    "subject_id": "fox",
                    "description": "red fox",
                    "position": "center",
                    "visual_attributes": {"clothing": None, "appearance_details": "snowy fur"},
                }
            ],
        },
        "dynamic_layer": {
            "timeline_segments": [
                {
                    "segment_basic_info": {
                        "timestamp_range": "0-5s",
                        "segment_description": "The fox turns.",
                    },
                    "audio": {"dialogue_lines": [{"speaker": "narrator", "text": "Listen.", "timestamp": "2s"}]},
                    "alive_subjects": [
                        {"subject_id": "fox", "action": {"primary_action": "turns", "interaction": "listens"}}
                    ],
                }
            ]
        },
        "reference_layer": ["reference: <Figure 1>"],
    }
    expected = "\n".join(
        [
            "context: winter forest",
            "A fox listens.",
            "aesthetics: style=cinematic; mood=quiet; color=blue and amber",
            "audio: language=English; speakers=narrator; ambience=wind",
            "subjects:",
            " - fox | red fox (center) :: appearance_details=snowy fur",
            "timeline:",
            " - 0-5s The fox turns.",
            "   - dialogue narrator: Listen. (2s)",
            "   - action fox: turns; listens",
            "reference: <Figure 1>",
        ]
    )
    encoded = json.dumps(prompt)
    assert json_to_compact_markdown(encoded) == expected
    assert normalize_prompt(encoded) == expected


def test_non_magi_json_is_preserved():
    raw = '{"prompt": "hello"}'
    assert json_to_compact_markdown(raw) == raw
    assert normalize_prompt(raw) == raw
