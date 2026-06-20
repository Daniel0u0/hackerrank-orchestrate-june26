#!/usr/bin/env python3
"""
gemini_reviewer.py — Gemini (Google) backend for the evidence-review agent.

Uses the `google-genai` SDK. One multimodal call per claim: the system
instruction carries the analyst policy, and the user turn carries the prompt
text plus every submitted image as inline image parts. Gemini's native JSON
mode (`response_mime_type="application/json"`) is used so the reply parses
cleanly; the result is then clamped to the allowed value lists in core.py.

API key resolution order:
  1. GEMINI_API_KEY / GOOGLE_API_KEY already in the environment
  2. .venv/.env  (loaded via core.load_env)
"""

from __future__ import annotations

import os
import re
import json
import time
import random
from pathlib import Path

from google import genai
from google.genai import types

import core

MODEL = os.environ.get("REVIEW_MODEL", "gemini-2.5-flash")
MAX_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "1024"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "6"))


def _generate_with_retry(client, contents, cfg):
    """Call Gemini, retrying on 429 / 503 with exponential backoff.

    Honours the server-suggested `retryDelay` when present, otherwise uses
    exponential backoff with jitter. Re-raises non-retryable errors.
    """
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(
                model=MODEL, contents=contents, config=cfg,
            )
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            retryable = ("429" in msg or "RESOURCE_EXHAUSTED" in msg
                         or "503" in msg or "UNAVAILABLE" in msg
                         or "overloaded" in msg.lower())
            if not retryable or attempt == MAX_RETRIES - 1:
                raise
            m = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)", msg)
            wait = float(m.group(1)) if m else delay
            wait += random.uniform(0, 1)  # jitter to avoid thundering herd
            print(f"    rate-limited; retry {attempt+1}/{MAX_RETRIES} in {wait:.1f}s",
                  flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 60)


def make_client() -> genai.Client:
    core.load_env()  # populate from .venv/.env if not already set
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "ERROR: GEMINI_API_KEY not set. Put it in .venv/.env as "
            "GEMINI_API_KEY=... or export it in the environment."
        )
    return genai.Client(api_key=api_key)


def _config() -> types.GenerateContentConfig:
    # thinking_budget=0 disables the model's internal reasoning tokens. For this
    # constrained extraction task they add latency/cost and, worse, can consume
    # the entire max_output_tokens budget and truncate the JSON answer.
    return types.GenerateContentConfig(
        system_instruction=core.SYSTEM_PROMPT,
        temperature=0.0,                        # deterministic where possible
        max_output_tokens=MAX_TOKENS,
        response_mime_type="application/json",  # native structured output
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )


def review_claim(claim_row: dict, history_by_user: dict, requirements: list[dict],
                 client: genai.Client, dataset_dir: Path) -> dict:
    """Run one claim through Gemini and return a full output-row dict."""
    user_id = claim_row["user_id"]
    claim_object = claim_row["claim_object"]

    image_parts, labels, missing = [], [], []
    for raw_path in core.parse_image_paths(claim_row["image_paths"]):
        label = core.image_id_from_path(raw_path)
        labels.append(label)
        data, mime = core.read_image_bytes(dataset_dir / raw_path)
        if data:
            image_parts.append(types.Part.from_bytes(data=data, mime_type=mime))
        else:
            missing.append(label)

    history = history_by_user.get(user_id)
    prompt_text = core.build_prompt_text(
        claim_row,
        core.user_history_block(history),
        core.evidence_block(requirements, claim_object),
        labels,
    )
    contents = [types.Part.from_text(text=prompt_text), *image_parts]

    try:
        resp = _generate_with_retry(client, contents, _config())
        text = (resp.text or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        start, end = text.find("{"), text.rfind("}")
        result = json.loads(text[start:end + 1] if start != -1 else text)
        um = resp.usage_metadata
        usage = {
            "input_tokens": getattr(um, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
        }
    except Exception as e:  # noqa: BLE001 — any failure -> safe fallback row
        result = core.error_result(str(e))
        usage = {"input_tokens": 0, "output_tokens": 0}

    normalised = core.normalise_result(result, claim_object, history, missing)
    return core.build_output_row(claim_row, normalised, usage)
