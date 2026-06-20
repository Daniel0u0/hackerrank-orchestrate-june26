# Multi-Modal Evidence Review — Solution

Verifies damage claims (car / laptop / package) by reasoning over submitted
images, the claim conversation, user risk history, and minimum-evidence
requirements. **Images are the primary source of truth**; user history only
adds risk context and never flips a decision the images clearly establish.

## Approach

For each claim the system makes **one** Claude vision call that receives:

- the **conversation** (the model extracts the final concrete claim — object,
  part, issue — ignoring earlier hedging),
- the **user history** row (risk context only),
- the **relevant minimum-evidence requirements** for that object, and
- **all submitted images**, batched into the single request.

The model returns strict JSON, which is then **clamped to the allowed value
lists** in `problem_statement.md` and merged with deterministic rules
(missing-image → `damage_not_visible`; inherit `user_history_risk` /
`manual_review_required` from the history file). Prompt-injection attempts in
the conversation or inside images are ignored and flagged
`text_instruction_present`.

All provider-neutral logic lives in [`core.py`](core.py). The model call is
isolated in pluggable backends so the test and evaluation entry points never
drift apart and providers can be swapped with one env var.

## Files

| Path | Purpose |
|---|---|
| `code/core.py` | Provider-neutral: prompts, schema, allowed-value clamping, CSV/image helpers, `.env` loader. |
| `code/gemini_reviewer.py` | **Gemini** backend (`google-genai`). Default provider. |
| `code/claim_reviewer.py` | **Anthropic** backend (`anthropic`). Used when `PROVIDER=anthropic`. |
| `code/agent.py` | Selects the backend from `PROVIDER` and re-exports `make_client` / `review_claim`. |
| `code/main.py` | Runs on `dataset/claims.csv` → writes `output.csv` at repo root. |
| `code/evaluation/main.py` | Runs on `dataset/sample_claims.csv`, scores vs. labels, writes `evaluation/evaluation_report.md`. |

## Setup

The API key is read from `.venv/.env` (or the environment). Create it with:

```
# .venv/.env
GEMINI_API_KEY=your_key_here
```

Get a Gemini key at <https://aistudio.google.com/apikey>. `GOOGLE_API_KEY` is
also accepted. Keys are read from env / `.env` only — never hardcoded.

## Running

```bash
pip install -r requirements.txt
python code/evaluation/main.py               # evaluate on labeled sample data
python code/main.py                          # produce output.csv for the test set
```

### Environment variables

| Var | Default | Effect |
|---|---|---|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | **required for Gemini**; read from `.venv/.env` or env. |
| `ANTHROPIC_API_KEY` | — | required only when `PROVIDER=anthropic`. |
| `PROVIDER` | `gemini` | `gemini` or `anthropic`. |
| `REVIEW_MODEL` | `gemini-2.5-flash` | model id to use. |
| `LIMIT` | `0` (all) | process only the first N claims (smoke test). |
| `REQUEST_SLEEP` | `0.5` | seconds between calls (RPM throttle). |
| `PRICE_IN_PER_M` / `PRICE_OUT_PER_M` | `0.30` / `2.50` | pricing assumptions for the cost report. |

> Note: the Gemini backend sets `thinking_budget=0`. Gemini 2.5 models otherwise
> spend output tokens on internal reasoning, which adds cost/latency and can
> truncate the JSON answer for this constrained extraction task.

## Output

`output.csv` has exactly the 14 columns from `problem_statement.md`, in order.
Every field is constrained to the allowed value lists; failures degrade to a
safe `not_enough_information` + `manual_review_required` row rather than
crashing the batch.

See `evaluation/evaluation_report.md` (generated) for accuracy per column, a
`claim_status` confusion matrix, and the cost / latency / rate-limit analysis.
