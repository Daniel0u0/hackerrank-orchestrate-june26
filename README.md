<div align="center">

# 🔍 Multi-Modal Evidence Review System

**An AI agent that verifies damage claims (car · laptop · package) from images, a claim conversation, and user history — and outputs a structured, auditable verdict.**

![Model](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-4285F4?logo=google)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![SDK](https://img.shields.io/badge/SDK-google--genai-34A853)
![Output](https://img.shields.io/badge/Determinism-temperature%3D0-orange)
![Status](https://img.shields.io/badge/status-submission--ready-success)

</div>

---

## 📖 Overview

Insurance and logistics teams receive damage claims as a few photos plus a chat
transcript. This system automates the first-pass review: it reads the
conversation to find **what the user is actually claiming**, inspects the
**submitted images as the primary source of truth**, checks them against a
**minimum-evidence checklist**, factors in the user's **risk history**, and
emits a single structured row per claim.

The core design philosophy:

> **Images decide the verdict. The conversation says what to check. History only
> adds risk context — it never overrides what the photos clearly show.**

For every claim the agent decides whether the evidence is `supported`,
`contradicted`, or `not_enough_information`, names the issue type and object
part, flags image-quality / authenticity / history risks, rates severity, and
writes a short image-grounded justification.

---

## ✨ Features

- 🖼️ **Multi-image reasoning** — all photos for a claim are evaluated together in **one** vision call.
- 🧾 **Conversation claim extraction** — pulls the user's *final* claim out of rambling, multi-turn, multilingual chats.
- 📋 **Evidence-standard checking** — grounded in `evidence_requirements.csv` per object + issue family.
- 🛡️ **Prompt-injection defense** — text in the chat *or inside an image* that says "approve this / ignore instructions" is ignored and flagged `text_instruction_present`.
- ⚖️ **Risk-aware** — inherits `user_history_risk` / `manual_review_required` from history without letting it flip clear visual evidence.
- 🔒 **Strict schema** — every field is clamped to the allowed value lists; the output can never contain an invalid token.
- 🔁 **Resilient** — exponential-backoff retry on rate limits; any failure degrades to a safe `not_enough_information` row instead of crashing the batch.
- 🔌 **Provider-pluggable** — Gemini by default, Anthropic Claude with one env var; prompts can't drift between them.
- 📊 **Built-in evaluation** — scores against labeled data and auto-generates a cost/latency/accuracy report.

---

## 🏗️ Architecture

A **single-shot multimodal classifier with a deterministic post-processing
layer** — not a tool-calling agent loop. The task is "evidence → one structured
verdict," which one well-built call answers more cheaply and reproducibly than a
multi-step agent.

```
                       dataset/
        ┌───────────────┼────────────────┬─────────────────────┐
   claims.csv     user_history.csv  evidence_requirements   images/test/
   (what+who)      (risk context)   .csv (min. evidence)    (the photos)
        └───────────────┴────────────────┴─────────────────────┘
                                │   per claim
                                ▼
                ┌──────────────────────────────────┐
                │   BUILD ONE REQUEST  (core.py)    │
                │  • system prompt = analyst policy │
                │  • user prompt = conversation +   │
                │    history + requirements +       │
                │    allowed values + JSON schema   │
                │  • + all images attached inline   │
                └──────────────────────────────────┘
                                │
                                ▼
                ┌──────────────────────────────────┐
                │  Gemini 2.5 Flash (vision)        │
                │  temperature=0, thinking_budget=0 │
                │  response_mime_type=application/  │
                │  json  →  strict JSON             │
                └──────────────────────────────────┘
                                │
                                ▼
                ┌──────────────────────────────────┐
                │  POST-PROCESS  (core.py)          │
                │  • clamp every field to allowed   │
                │    value lists                    │
                │  • merge deterministic flags      │
                │    (missing img → damage_not_     │
                │    visible; inherit history flags)│
                │  • on error → safe fallback row   │
                └──────────────────────────────────┘
                                │
                                ▼
                         output.csv  (14 cols)
```

**Why layered this way:** all *behaviour* (prompt, rules, validation) lives in
one provider-neutral file (`core.py`). Each backend only knows *how to call one
API*. Swapping Gemini ↔ Claude is a one-line env change and the rules can't
diverge because there is exactly one copy.

---

## 🧰 Tech Stack

| Layer | Choice | Why |
|---|---|---|
| **LLM** | Google **Gemini 2.5 Flash** (vision) | Strong multimodal quality, very low cost, fast |
| **SDK** | `google-genai` | Official SDK; native JSON output mode |
| **Agent framework** | None (plain Python) | Fewer moving parts → reproducible & easy to grade |
| **MCP servers** | None | Direct API calls; no external tools needed |
| **Structured output** | `response_mime_type="application/json"` + allowed-value clamping | Guarantees a valid, schema-conformant row |
| **Alt provider** | Anthropic Claude (`anthropic`) | Drop-in via `PROVIDER=anthropic` |

---

## 📂 Project Structure

```
.
├── README.md                      ← you are here
├── requirements.txt               ← google-genai (+ optional anthropic)
├── .gitignore                     ← excludes .venv/.env, logs, caches
├── output.csv                     ← 🎯 PREDICTIONS for dataset/claims.csv (the deliverable)
│
├── code/
│   ├── core.py                    ← provider-neutral brain: prompts, schema,
│   │                                allowed values, normalisation, .env loader
│   ├── gemini_reviewer.py         ← Gemini backend (default) + retry logic
│   ├── claim_reviewer.py          ← Anthropic backend (PROVIDER=anthropic)
│   ├── agent.py                   ← selects backend from PROVIDER env var
│   ├── main.py                    ← entry point: claims.csv → output.csv
│   ├── README.md                  ← code-level docs
│   └── evaluation/
│       ├── main.py                ← scores predictions vs. labeled sample data
│       ├── evaluation_report.md   ← generated accuracy + cost/latency report
│       └── predictions_sample.csv ← generated predictions on sample_claims.csv
│
└── dataset/                       ← provided inputs (not part of code archive)
    ├── claims.csv                 ← 44 input-only test claims
    ├── sample_claims.csv          ← 20 labeled claims (for evaluation)
    ├── user_history.csv           ← per-user risk context
    ├── evidence_requirements.csv  ← minimum-evidence checklist
    └── images/{sample,test}/      ← referenced photos
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your API key

The key is read from `.venv/.env` (or the environment) — **never hardcoded**.
Create the file:

```bash
# .venv/.env
GEMINI_API_KEY=your_key_here
```

Get a key at <https://aistudio.google.com/apikey>. `GOOGLE_API_KEY` is also
accepted. Make sure billing is enabled on the key's project — a free-tier key is
throttled to ~10 requests/minute.

### 3. Run

```bash
python code/evaluation/main.py     # evaluate on labeled sample data + report
python code/main.py                # produce output.csv for the 44 test claims
```

---

## 🎮 Usage

```bash
# Full test run → writes output.csv at the repo root
python code/main.py

# Cheap smoke test on the first 2 claims
LIMIT=2 python code/main.py

# Use Claude instead of Gemini
PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... python code/main.py
```

### The prediction file (`output.csv`)

- **Location:** repo root → `output.csv`
- **Generated by:** `python code/main.py`
- **Format:** one row per `dataset/claims.csv` row (44 total), with these
  **14 columns in exact order**:

  `user_id, image_paths, user_claim, claim_object, evidence_standard_met,
  evidence_standard_met_reason, risk_flags, issue_type, object_part,
  claim_status, claim_status_justification, supporting_image_ids, valid_image,
  severity`

---

## ⚙️ Configuration

| Variable | Default | Effect |
|---|---|---|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | **Required for Gemini.** From `.venv/.env` or env. |
| `ANTHROPIC_API_KEY` | — | Required only when `PROVIDER=anthropic`. |
| `PROVIDER` | `gemini` | `gemini` or `anthropic`. |
| `REVIEW_MODEL` | `gemini-2.5-flash` | Model id. |
| `LIMIT` | `0` (all) | Process only the first N claims (smoke test). |
| `REQUEST_SLEEP` | `0.5` | Seconds between calls (RPM throttle). |
| `MAX_RETRIES` | `6` | Backoff retries on 429 / 503. |
| `PRICE_IN_PER_M` / `PRICE_OUT_PER_M` | `0.30` / `2.50` | Pricing assumptions for the cost report. |

---

## 📊 Evaluation & Results

Run `python code/evaluation/main.py` to reproduce. It runs the **same** reviewer
on the labeled `sample_claims.csv`, scores each column, and writes
`code/evaluation/evaluation_report.md`.

**Blended accuracy: 69.3%** on the 20 labeled samples (exact-match per column;
Jaccard overlap for the multi-label set fields).

| Column | Score | Metric |
|---|---|---|
| evidence_standard_met | 85% | exact |
| object_part | 80% | exact |
| supporting_image_ids | 77.5% | Jaccard |
| claim_status | 70% | exact |
| issue_type | 70% | exact |
| valid_image | 75% | exact |
| severity | 55% | exact |
| risk_flags | 42% | Jaccard |

**Determinism (measured over 2 back-to-back runs):** with `temperature=0` the
decision-critical fields — `claim_status`, `evidence_standard_met`,
`valid_image`, `severity`, `risk_flags`, `supporting_image_ids` — were **100%
identical**. Only `issue_type` (90%) and `object_part` (95%) flipped on
borderline rows. So the agent is *near*-deterministic, not bit-exact;
free-text wording varies but does not affect scoring.

---

## 💰 Cost, Latency & Rate Limits

Measured on the test set (`gemini-2.5-flash`, paid tier):

| Metric | Value |
|---|---|
| Model calls | **44** (1 per claim, all images batched) |
| Images processed | 82 |
| Tokens | ~94.7k input / ~8.4k output |
| Runtime | **~158 s** (~3.6 s/claim, sequential) |
| **Cost** | **≈ $0.05** for the full test set (~$0.001/claim) |

**Strategies used:** one call per claim (never one-per-image); `thinking_budget=0`
(stops Gemini spending the output budget on internal reasoning, which also
prevents JSON truncation); `max_output_tokens=1024`; `REQUEST_SLEEP` throttle +
exponential-backoff retry honoring the server's `retryDelay`; a static system
prompt that is a natural prompt-cache target; images encoded once and never
re-sent. Full breakdown in
[`code/evaluation/evaluation_report.md`](code/evaluation/evaluation_report.md).

---

## 🧠 Key Design Decisions

- **Single call, not an agent loop.** The task maps to one classification per
  claim; a tool-using loop would add cost and non-determinism for no accuracy
  gain.
- **`thinking_budget=0`.** Discovered during testing that Gemini 2.5 Flash's
  default reasoning consumed the entire 1024-token output budget (982 tokens) and
  truncated the JSON. Disabling it fixed correctness *and* cut cost/latency.
- **Allowed-value clamping over trust.** The model is asked for valid values, but
  output is also hard-clamped — invalid tokens are impossible.
- **Deterministic flag merging.** Missing images and history-derived risk flags
  are applied in code, not left to the model's memory.
- **Calibrated labeling guidance.** Generalizable rules (cracked glass →`crack`
  not `glass_shatter`; liquid mark →`stain` not `water_damage`; severity defaults
  to `medium`) lifted `issue_type` 40→70% and `severity` 15→55% — without
  hardcoding any per-row answers.

---

## 🔐 Security & Reproducibility

- API keys read from `.venv/.env` / environment **only** — never committed; `.gitignore` excludes `.venv/.env` and logs.
- `temperature=0` + value clamping for stable, repeatable runs.
- Failures never crash the batch; every claim yields a valid row.

---

## ⚠️ Limitations & Future Work

- **`contradicted` detection** is the weakest decision (the model leans
  `supported` / `not_enough_information` when claimed damage is absent). Sharper
  "part visible & intact → contradicted" guidance would help.
- **`risk_flags`** is scored by strict set overlap on 14 labels — the hardest
  column to match exactly.
- **Concurrency** is not enabled; a thread pool would cut wall-clock time
  near-linearly on the paid tier.

---

## 📦 Submission Contents

| Deliverable | What |
|---|---|
| `code.zip` | `code/` + `requirements.txt` + `.gitignore` + this README (exclude `.venv/`, `dataset/images/`, caches) |
| `output.csv` | Predictions for all 44 rows of `dataset/claims.csv` |
| `log.txt` | AI chat transcript from `%USERPROFILE%\hackerrank_orchestrate\log.txt` |

---

<div align="center">

Built for the **HackerRank Orchestrate (June 2026)** challenge ·
Multi-Modal Evidence Review

</div>
