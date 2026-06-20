#!/usr/bin/env python3
"""
claim_reviewer.py — Anthropic (Claude) backend for the evidence-review agent.

Thin wrapper over the provider-neutral logic in core.py. Kept so the original
Claude path still works; select it with PROVIDER=anthropic. The Gemini backend
lives in gemini_reviewer.py and is the default.
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import anthropic

import core

MODEL = os.environ.get("REVIEW_MODEL", "claude-opus-4-8")
MAX_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "1024"))


def make_client() -> anthropic.Anthropic:
    core.load_env()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: ANTHROPIC_API_KEY not set (env or .venv/.env).")
    return anthropic.Anthropic(api_key=api_key)


def review_claim(claim_row: dict, history_by_user: dict, requirements: list[dict],
                 client: anthropic.Anthropic, dataset_dir: Path) -> dict:
    user_id = claim_row["user_id"]
    claim_object = claim_row["claim_object"]

    image_blocks, labels, missing = [], [], []
    for raw_path in core.parse_image_paths(claim_row["image_paths"]):
        label = core.image_id_from_path(raw_path)
        labels.append(label)
        b64, mime = core.encode_image_b64(dataset_dir / raw_path)
        if b64:
            image_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            })
        else:
            missing.append(label)

    history = history_by_user.get(user_id)
    prompt_text = core.build_prompt_text(
        claim_row,
        core.user_history_block(history),
        core.evidence_block(requirements, claim_object),
        labels,
    )
    content = [{"type": "text", "text": prompt_text}, *image_blocks]

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=core.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        start, end = text.find("{"), text.rfind("}")
        result = json.loads(text[start:end + 1] if start != -1 else text)
        usage = {"input_tokens": resp.usage.input_tokens,
                 "output_tokens": resp.usage.output_tokens}
    except Exception as e:  # noqa: BLE001
        result = core.error_result(str(e))
        usage = {"input_tokens": 0, "output_tokens": 0}

    normalised = core.normalise_result(result, claim_object, history, missing)
    return core.build_output_row(claim_row, normalised, usage)
