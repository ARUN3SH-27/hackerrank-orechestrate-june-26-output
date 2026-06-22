# Multi-Modal Evidence Review — Solution

Verifies damage claims (car / laptop / package) by inspecting submitted images with a
vision-language model, cross-checking against per-object minimum evidence requirements
and user claim history, and producing a structured decision per claim.

## Setup

```bash
cd code
pip install -r requirements.txt
cp .env.example .env   # then fill in NVIDIA_API_KEY
```

Secrets are read only from environment variables (loaded via `.env` with `python-dotenv`);
nothing is hardcoded. Uses [NVIDIA NIM](https://build.nvidia.com) hosted inference:

- `NIM_VISION_MODEL` (default `meta/llama-3.2-11b-vision-instruct`) — image inspection.
- `NIM_TEXT_MODEL` (default `meta/llama-3.1-8b-instruct`) — text-only aggregation/synthesis.

## Run

```bash
# Full test set -> ../output.csv
python main.py --input claims.csv --output ../output.csv --strategy per_image

# Quick smoke test on a couple of rows
python main.py --input sample_claims.csv --limit 2 --strategy per_image
```

Flags: `--strategy {per_image,single_call}`, `--workers N` (bounded thread pool, default 4),
`--limit N` (cap rows for testing).

## Evaluation

```bash
python evaluation/main.py --strategies per_image,single_call
```

Runs both strategies against the labeled `dataset/sample_claims.csv`, scores per-field
accuracy (exact match for scalar fields, Jaccard overlap for the multi-value
`risk_flags`/`supporting_image_ids` fields), and writes:

- `evaluation/evaluation_report.md` — strategy comparison + operational analysis.
- `evaluation/metrics.json` — raw scores.
- `evaluation/sample_predictions_<strategy>.csv` — per-strategy predictions on the sample set.

## Design

**Two strategies**, selectable via `--strategy`:

- **`per_image`** (winner, used for `output.csv`): one vision call per submitted image,
  asking only for what is literally visible (object/part match, issue type, image-quality
  flags) -- grounded and auditable per photo. A second, text-only call synthesizes the final
  decision from all per-image notes + the conversation + user history. More calls, but each
  judgment is anchored to exactly one photo, which matters most on multi-image claims where a
  single combined call can conflate evidence across images.
- **`single_call`**: one vision call decides everything directly. Cheaper and faster, but the
  underlying model only accepts one image per request, so multi-image claims are evaluated
  from the first image only -- a real coverage limitation reflected in its lower score on
  multi-image rows in `dataset/sample_claims.csv` (see `evaluation_report.md`).

**Pipeline per claim:**
1. Resolve `claim_object` evidence checklist from `dataset/evidence_requirements.csv`.
2. Load and inspect each image (resized to ≤1024px, recompressed to ≤~170KB JPEG to satisfy
   the NIM inline-image size limit and keep token usage down). Several dataset images are
   AVIF/WebP/PNG mislabeled with a `.jpg` extension; `pillow-avif-plugin` plus Pillow's format
   sniffing handle all of them transparently.
3. Aggregate per-image findings with the conversation and `dataset/user_history.csv` risk
   context into the final structured decision.
4. Normalize/validate every field against the allowed-value lists in `lib/schema.py`,
   falling back to `unknown`/`none`/`not_enough_information` on any malformed model output
   rather than failing the row.

**Reliability and cost controls** (see `lib/nim_client.py`, `lib/image_utils.py`):
- Bounded `ThreadPoolExecutor` concurrency (not unbounded) to respect NIM TPM/RPM limits.
- Exponential-backoff retries (`tenacity`) on 429/5xx.
- On-disk response cache keyed by exact request hash (`code/.cache/`) — re-running
  evaluation or resuming an interrupted full run costs zero extra calls for unchanged inputs.
- Per-row error isolation in `main.py`: a failed claim degrades to a safe
  `not_enough_information` / `manual_review_required` row instead of aborting the run.

## Files

```
code/
├── main.py                  # CLI entry point -> output.csv
├── lib/
│   ├── data_loader.py        # CSV + image-path loading
│   ├── image_utils.py        # resize/compress/encode images for inline transport
│   ├── nim_client.py          # NIM chat client, caching, retries, call/token stats
│   ├── prompts.py             # prompt templates for both strategies
│   ├── pipeline.py            # per_image / single_call strategy implementations
│   └── schema.py               # allowed-value lists + output normalization
└── evaluation/
    ├── main.py                # scoring harness + report generator
    ├── evaluation_report.md    # strategy comparison + operational analysis (generated)
    └── metrics.json             # raw scores (generated)
```
