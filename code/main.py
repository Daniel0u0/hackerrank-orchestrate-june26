#!/usr/bin/env python3
"""
main.py — Multi-Modal Evidence Review System (test entry point).

Reads:  dataset/claims.csv, dataset/user_history.csv,
        dataset/evidence_requirements.csv, dataset/images/test/...
Writes: output.csv  (repo root)

Usage (from repo root):
    # put GEMINI_API_KEY=... in .venv/.env (default provider: gemini)
    python code/main.py

Optional environment variables:
    PROVIDER       gemini (default) or anthropic
    REVIEW_MODEL   override model id (default: gemini-2.5-flash)
    LIMIT          process only the first N claims (smoke test)
    REQUEST_SLEEP  seconds to sleep between calls (default: 0.5)
"""

import os
import sys
import time
from pathlib import Path

# Make the sibling shared modules importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent))
import core            # noqa: E402
import agent as cr     # noqa: E402  (provider-selected backend + core re-exports)

REPO_ROOT   = Path(__file__).parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
OUTPUT_FILE = REPO_ROOT / "output.csv"


def main() -> None:
    client = cr.make_client()

    claims       = core.load_csv_as_dicts(DATASET_DIR / "claims.csv")
    history_rows = core.load_csv_as_dicts(DATASET_DIR / "user_history.csv")
    requirements = core.load_csv_as_dicts(DATASET_DIR / "evidence_requirements.csv")
    history_by_user = {r["user_id"]: r for r in history_rows}

    limit = int(os.environ.get("LIMIT", "0"))
    if limit:
        claims = claims[:limit]
    sleep_s = float(os.environ.get("REQUEST_SLEEP", "0.5"))

    rows, tot_in, tot_out, n_images = [], 0, 0, 0
    total = len(claims)
    t0 = time.time()
    for i, claim in enumerate(claims, 1):
        case = claim["image_paths"].split("/")[2] if "/" in claim["image_paths"] else "?"
        print(f"[{i}/{total}] {claim['user_id']} / {case} ...", flush=True)
        row = cr.review_claim(claim, history_by_user, requirements, client, DATASET_DIR)
        n_images += len(core.parse_image_paths(claim["image_paths"]))
        tot_in  += row["_usage"]["input_tokens"]
        tot_out += row["_usage"]["output_tokens"]
        rows.append(row)
        if i < total:
            time.sleep(sleep_s)
    elapsed = time.time() - t0

    core.write_output_csv(rows, OUTPUT_FILE)

    # ── Operational summary (also see evaluation/evaluation_report.md) ─────────
    print("\n── Run summary ────────────────────────────────────────────────")
    print(f"  Claims processed : {total}")
    print(f"  Images processed : {n_images}")
    print(f"  Model calls      : {total} (1 per claim, all images batched)")
    print(f"  Model            : {cr.MODEL}")
    print(f"  Input tokens     : {tot_in:,}")
    print(f"  Output tokens    : {tot_out:,}")
    print(f"  Wall-clock time  : {elapsed:.0f}s ({elapsed / max(total,1):.1f}s/claim)")
    print(f"  Output written   : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
