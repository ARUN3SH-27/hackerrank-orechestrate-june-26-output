"""CSV and image loading helpers."""
import csv
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_claims(filename):
    return read_csv(os.path.join(DATASET_DIR, filename))


def load_user_history():
    rows = read_csv(os.path.join(DATASET_DIR, "user_history.csv"))
    return {r["user_id"]: r for r in rows}


def load_evidence_requirements():
    return read_csv(os.path.join(DATASET_DIR, "evidence_requirements.csv"))


def evidence_text_for(claim_object, requirements):
    lines = []
    for r in requirements:
        if r["claim_object"] in (claim_object, "all"):
            lines.append(f"- [{r['requirement_id']}] ({r['applies_to']}): {r['minimum_image_evidence']}")
    return "\n".join(lines)


def image_paths_to_abs(image_paths_field):
    """'images/test/case_001/img_1.jpg;...' -> [(image_id, abs_path), ...]"""
    out = []
    for rel in image_paths_field.split(";"):
        rel = rel.strip()
        if not rel:
            continue
        image_id = os.path.splitext(os.path.basename(rel))[0]
        abs_path = os.path.join(DATASET_DIR, rel)
        out.append((image_id, abs_path))
    return out


def user_history_text(user_id, history_map):
    h = history_map.get(user_id)
    if not h:
        return "No history available for this user."
    return (
        f"past_claim_count={h['past_claim_count']}, accepted={h['accept_claim']}, "
        f"manual_review={h['manual_review_claim']}, rejected={h['rejected_claim']}, "
        f"last_90_days={h['last_90_days_claim_count']}, history_flags={h['history_flags']}, "
        f"summary: {h['history_summary']}"
    )


def user_history_flags(user_id, history_map):
    h = history_map.get(user_id)
    if not h:
        return []
    raw = h.get("history_flags", "") or ""
    return [f.strip() for f in raw.split(";") if f.strip() and f.strip().lower() != "none"]
