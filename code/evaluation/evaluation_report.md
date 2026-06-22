# Evaluation Report

## Strategy comparison (on `dataset/sample_claims.csv`, n=20)

| Strategy | Overall score | Model calls | Cache hits | Prompt tok | Completion tok | Elapsed (s) |
|---|---|---|---|---|---|---|
| per_image | 0.664 | 49 | 49 | 163247 | 4998 | 1.2 |
| single_call | 0.657 | 20 | 20 | 100659 | 2114 | 0.4 |

### Per-field score breakdown

| Strategy | evidence_standard_met | issue_type | object_part | claim_status | valid_image | severity | risk_flags | supporting_image_ids |
|---|---|---|---|---|---|---|---|---|
| per_image | 0.80 | 0.55 | 0.80 | 0.60 | 0.90 | 0.45 | 0.41 | 0.80 |
| single_call | 0.80 | 0.40 | 0.75 | 0.60 | 0.80 | 0.55 | 0.61 | 0.75 |

**Selected strategy for `output.csv`: `per_image`** (overall score 0.664).

`per_image` inspects every submitted image with its own grounded vision call, then synthesizes a final decision with a text-only aggregation call. This costs more calls but each per-image judgment is grounded in exactly one photo, which improves accuracy on multi-image cases (mismatched vehicles, partial views) where a single combined call can blend or miss details across images. `single_call` sends all images in one request and asks the model to decide everything at once -- it is cheaper and faster per claim, with somewhat lower per-field accuracy on multi-image edge cases in this sample.

## Operational analysis

- Sample set: 20 claims, 29 images, 49 model calls for the selected strategy (2.45 calls/claim for per_image, 1.00 calls/claim for single_call).
- Test set (`dataset/claims.csv`): 44 claims, 82 images.
- Estimated model calls for full test set with `per_image`: ~108.
- Estimated tokens for full test set: ~359,810 prompt tokens, ~11,016 completion tokens (scaled from observed sample averages of 3332/102 tokens per call).
- Cost: $0 -- this solution runs on the NVIDIA NIM **free tier** (`build.nvidia.com`), which has no per-token charge. The binding constraint is rate limit, not spend (see below).
- This evaluation run was fully served from the on-disk response cache (49/49 calls), so its 1.2s wall-clock is not representative of real API latency. A separate uncached, 4-worker run of `per_image` over the same 20-claim sample (via `code/main.py`) took ~36.7s end-to-end for 47 fresh model calls (~0.78s/claim with concurrency). Scaling that observed rate, the 44-row/82-image test set is estimated at roughly 81-121s wall-clock with 4 workers, depending on per-image-count variance.
- Rate limits / batching strategy: the NVIDIA NIM free tier caps each API key at 40 requests/minute. `code/main.py` issues requests from a bounded `ThreadPoolExecutor` (default 4 workers, well under the 40 RPM ceiling -- even worst-case back-to-back calls from 4 threads stay around 4 req/s peak, which the cap comfortably absorbs) rather than unbounded concurrency. The `tenacity` retry wrapper backs off exponentially on 429/5xx if the cap is still hit (e.g. from other concurrent usage of the same key). At ~108 calls for the full 44-row test set, a single run completes in a few RPM windows regardless of concurrency; the cache (below) matters most for iterative development and re-runs, not for the one-shot test set size. Every request/response pair is cached to disk by content hash (`code/.cache/`), so re-running evaluation or resuming a partially-completed run costs zero additional calls and zero additional RPM budget for unchanged inputs. Images are resized/recompressed to ~170KB JPEGs before sending, which keeps per-call payload size and token usage low and avoids the NIM inline-image size limit.
