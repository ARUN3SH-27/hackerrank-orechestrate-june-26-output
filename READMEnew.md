# Multi-Modal Insurance Claim Verifier

A vision-language model pipeline that verifies damage claims (car / laptop / package) by inspecting submitted photos, cross-checking against per-object evidence requirements and user history, and producing a structured decision per claim.

Built for the **HackerRank Orchestrate — June 2026** hackathon (24-hour challenge).

---

## What it does

For each claim, the system:

- Inspects one or more submitted images using a vision-language model
- Extracts the actual damage assertion from the customer conversation
- Cross-checks image findings against minimum evidence requirements per object type
- Incorporates user claim history as risk context (without overriding clear visual evidence)
- Outputs a fully structured, schema-validated decision row

Output fields per claim: `evidence_standard_met`, `issue_type`, `object_part`, `claim_status` (`supported` / `contradicted` / `not_enough_information`), `severity`, `risk_flags`, `supporting_image_ids`, and short justifications grounded in the images.

---

## Architecture

### Two competing strategies

The core design choice was decoupling *observation* from *judgment* — rather than asking the model to inspect and decide in one shot, the primary strategy separates the two steps:

**`per_image` (used for final output)**
One vision call per submitted photo, asking only *"what is literally visible here"* — object match, part, issue type, image quality flags. A separate text-only call then aggregates all per-image notes together with the conversation and user history to produce the final verdict. More calls, but each judgment is anchored to exactly one photo, which matters most on multi-image claims where a single combined call can conflate evidence across images.

**`single_call` (baseline)**
One vision call per claim decides everything directly. Cheaper and faster, but the underlying model only accepts one image per request — so multi-image claims are evaluated from the first photo only. Real structural limitation.

Both strategies were built, run, and scored against the labeled sample set before choosing.

### Pipeline per claim

```
claims.csv
    │
    ├─ evidence_requirements.csv  ──► resolve per-object checklist
    ├─ user_history.csv           ──► load risk context
    │
    ▼
image_utils.py     resize ≤1024px, recompress ≤170KB JPEG
                   (handles AVIF/WebP mislabeled as .jpg transparently)
    │
    ▼
[per_image strategy]
    vision call × N images   ──► per-photo observations (JSON)
    text call × 1            ──► aggregate + final decision (JSON)
    │
    ▼
schema.py          normalize all fields against allowed-value enums
                   fall back to unknown / not_enough_information on malformed output
    │
    ▼
output.csv
```

### Reliability and cost controls

| Concern | Approach |
|---|---|
| Rate limits | Bounded `ThreadPoolExecutor` (4 workers, configurable) to stay under NIM's free-tier RPM cap |
| Transient failures | `tenacity` exponential backoff on 429/5xx |
| Repeated runs | On-disk response cache keyed by SHA-256 of request payload — reruns cost zero additional API calls |
| Bad model output | Per-field normalization in `schema.py`; malformed JSON falls back to safe defaults rather than crashing the row |
| Batch resilience | Per-row try/except in `main.py` — one failed claim degrades to `not_enough_information / manual_review_required`, never aborts the run |

---

## Results

Evaluated against 20 labeled rows in `dataset/sample_claims.csv`:

| Field | per_image | single_call |
|---|---|---|
| `evidence_standard_met` | 0.80 | 0.80 |
| `issue_type` | 0.55 | 0.40 |
| `object_part` | 0.80 | 0.75 |
| `claim_status` | 0.60 | 0.60 |
| `valid_image` | 0.90 | 0.80 |
| `severity` | 0.45 | 0.55 |
| `risk_flags` (Jaccard) | 0.41 | 0.61 |
| `supporting_image_ids` (Jaccard) | 0.80 | 0.75 |
| **Overall** | **0.664** | **0.657** |

| Operational | per_image | single_call |
|---|---|---|
| Model calls (20 claims, 29 images) | 49 | 20 |
| Prompt tokens | 163,247 | 100,659 |
| Completion tokens | 4,998 | 2,114 |

`per_image` wins on `issue_type` (+15pp) and `supporting_image_ids` (+5pp) — the fields most directly driven by per-photo grounding. `single_call` wins on `risk_flags` and `severity`, likely because the aggregated single prompt picks up conversation-level signals more directly. The trade-off was accepted in favour of the fields most central to claim integrity.

Full breakdown: [`code/evaluation/evaluation_report.md`](code/evaluation/evaluation_report.md)

---

## Setup

**Requirements:** Python 3.10+, pip

```bash
git clone <this-repo>
cd code
pip install -r requirements.txt
```

Create a `.env` file in `code/` with your NVIDIA NIM API key:

```
NVIDIA_API_KEY=your_key_here
NIM_VISION_MODEL=meta/llama-3.2-11b-vision-instruct
NIM_TEXT_MODEL=meta/llama-3.1-8b-instruct
```

Get a free NVIDIA NIM key at [build.nvidia.com](https://build.nvidia.com).

---

## Run

**Full test set → `output.csv`:**

```bash
python main.py --input claims.csv --output ../output.csv --strategy per_image
```

**Smoke test (first 2 rows):**

```bash
python main.py --input sample_claims.csv --limit 2 --strategy per_image
```

**Run both strategies and generate evaluation report:**

```bash
python evaluation/main.py --strategies per_image,single_call
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--strategy` | `per_image` | `per_image` or `single_call` |
| `--workers` | `4` | Thread pool size |
| `--limit` | none | Cap rows for testing |

---

## Repo structure

```
.
├── problem_statement.md
├── AGENTS.md
├── dataset/
│   ├── claims.csv
│   ├── sample_claims.csv
│   ├── user_history.csv
│   ├── evidence_requirements.csv
│   └── images/
│       ├── sample/
│       └── test/
└── code/
    ├── main.py
    ├── requirements.txt
    ├── lib/
    │   ├── data_loader.py       # CSV + image-path loading
    │   ├── image_utils.py       # resize / compress / base64-encode images
    │   ├── nim_client.py        # NIM API client, caching, retries, call stats
    │   ├── pipeline.py          # per_image and single_call strategy implementations
    │   ├── prompts.py           # prompt templates (JSON-only output enforced)
    │   └── schema.py            # allowed-value enums + output normalization
    └── evaluation/
        ├── main.py              # scoring harness + report generator
        ├── evaluation_report.md # generated — strategy comparison + operational analysis
        └── metrics.json         # generated — raw per-field scores
```

---

## Models used

| Role | Model | Provider |
|---|---|---|
| Image inspection | `meta/llama-3.2-11b-vision-instruct` | NVIDIA NIM |
| Text aggregation | `meta/llama-3.1-8b-instruct` | NVIDIA NIM |

Both accessed via NVIDIA NIM's OpenAI-compatible chat endpoint. Free-tier credits used throughout development and evaluation.

---

## Known limitations

- Vision model (11B) is not adversarially robust — submitted images could contain misleading overlays or be non-original; the pipeline flags these via `risk_flags` but does not block on them
- `CallStats` counters use unsynchronized increments across threads — token/call counts in the summary are approximate under real concurrency, though output rows are always correct
- Severity accuracy (0.45) is the weakest field — the model struggles to calibrate damage magnitude from photos alone without reference scale
- This is a decision-support prototype, not a production claims system — any real deployment would require mandatory human review on non-trivial decisions and a much larger validated accuracy baseline
