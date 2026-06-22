"""Entry point: run the claim-review pipeline over a claims CSV and write output.csv.

Usage:
    python main.py [--input dataset/claims.csv] [--output ../output.csv]
                    [--strategy per_image|single_call] [--limit N] [--workers N]
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from lib import schema
from lib.data_loader import load_claims, load_user_history, load_evidence_requirements
from lib.pipeline import run_claim
from lib.nim_client import CallStats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="claims.csv",
                         help="filename under dataset/, e.g. claims.csv or sample_claims.csv")
    parser.add_argument("--output", default=os.path.join(REPO_ROOT, "output.csv"))
    parser.add_argument("--strategy", default="per_image", choices=["per_image", "single_call"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    rows = load_claims(args.input)
    if args.limit:
        rows = rows[: args.limit]
    history_map = load_user_history()
    requirements = load_evidence_requirements()

    results = [None] * len(rows)
    start = time.time()

    def work(i, row):
        try:
            return i, run_claim(row, history_map, requirements, strategy=args.strategy)
        except Exception as e:  # keep going; emit a safe fallback row
            fallback = {
                "user_id": row["user_id"], "image_paths": row["image_paths"],
                "user_claim": row["user_claim"], "claim_object": row["claim_object"],
                "evidence_standard_met": "false",
                "evidence_standard_met_reason": f"pipeline error: {e}",
                "risk_flags": "manual_review_required", "issue_type": "unknown",
                "object_part": "unknown", "claim_status": "not_enough_information",
                "claim_status_justification": f"Automated review failed: {e}",
                "supporting_image_ids": "none", "valid_image": "false", "severity": "unknown",
            }
            return i, {col: fallback[col] for col in schema.OUTPUT_COLUMNS}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, i, row) for i, row in enumerate(rows)]
        done = 0
        for fut in as_completed(futures):
            i, result = fut.result()
            results[i] = result
            done += 1
            print(f"[{done}/{len(rows)}] processed {result['user_id']}", file=sys.stderr)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=schema.OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start
    print(f"\nWrote {len(results)} rows to {args.output}")
    print(f"Strategy: {args.strategy} | elapsed: {elapsed:.1f}s | "
          f"model calls: {CallStats.calls} | cache hits: {CallStats.cache_hits} | "
          f"prompt_tokens: {CallStats.prompt_tokens} | completion_tokens: {CallStats.completion_tokens}")


if __name__ == "__main__":
    main()
