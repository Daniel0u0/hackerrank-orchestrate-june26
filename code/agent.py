#!/usr/bin/env python3
"""
agent.py — provider selector for the evidence-review agent.

Picks the backend from the PROVIDER env var (default: gemini) and exposes a
uniform `make_client()` / `review_claim()` pair plus the active MODEL name.
Both entry points (main.py, evaluation/main.py) import from here so swapping
providers is a one-line env change.
"""

from __future__ import annotations

import os

import core

core.load_env()  # ensure .venv/.env is loaded before reading PROVIDER/keys

PROVIDER = os.environ.get("PROVIDER", "gemini").strip().lower()

if PROVIDER in ("gemini", "google"):
    import gemini_reviewer as backend
elif PROVIDER in ("anthropic", "claude"):
    import claim_reviewer as backend
else:
    raise SystemExit(f"Unknown PROVIDER={PROVIDER!r}; use 'gemini' or 'anthropic'.")

MODEL = backend.MODEL
make_client = backend.make_client
review_claim = backend.review_claim
