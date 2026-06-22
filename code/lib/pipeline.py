"""Two claim-review strategies:

- per_image: one vision call per image (grounded, cheap per-call) + one text aggregation call.
- single_call: one vision call with all images attached, model decides everything directly.
"""
import json
import re

from . import nim_client, prompts, schema
from .data_loader import (
    evidence_text_for, image_paths_to_abs, user_history_text, user_history_flags,
)
from .image_utils import load_and_encode


def _extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _normalize_final(parsed, claim_object, valid_image_ids):
    risk_flags = schema.norm_risk_flags(parsed.get("risk_flags"))
    def b(v, default):
        return "true" if schema.norm_bool(v, default) else "false"

    return {
        "evidence_standard_met": b(parsed.get("evidence_standard_met"), False),
        "evidence_standard_met_reason": str(parsed.get("evidence_standard_met_reason", "")).strip()
            or "Insufficient model output; defaulted.",
        "risk_flags": ";".join(risk_flags),
        "issue_type": schema.norm_enum(parsed.get("issue_type"), schema.ISSUE_TYPE, "unknown"),
        "object_part": schema.norm_object_part(parsed.get("object_part"), claim_object),
        "claim_status": schema.norm_enum(parsed.get("claim_status"), schema.CLAIM_STATUS,
                                          "not_enough_information"),
        "claim_status_justification": str(parsed.get("claim_status_justification", "")).strip()
            or "Model did not return a justification.",
        "supporting_image_ids": ";".join(
            schema.norm_image_ids(parsed.get("supporting_image_ids"), valid_image_ids)
        ),
        "valid_image": b(parsed.get("valid_image"), True),
        "severity": schema.norm_enum(parsed.get("severity"), schema.SEVERITY, "unknown"),
    }


def _apply_history_risk(result, history_flags):
    flags = schema.norm_risk_flags(result["risk_flags"])
    flags = [f for f in flags if f != "none"]
    if "user_history_risk" in history_flags and "user_history_risk" not in flags:
        flags.append("user_history_risk")
    if "manual_review_required" in history_flags and "manual_review_required" not in flags:
        flags.append("manual_review_required")
    result["risk_flags"] = ";".join(flags) if flags else "none"
    return result


def run_per_image(row, history_map, requirements):
    claim_object = row["claim_object"].strip().lower()
    user_claim = row["user_claim"]
    images = image_paths_to_abs(row["image_paths"])
    image_ids = [iid for iid, _ in images]
    evidence_text = evidence_text_for(claim_object, requirements)
    h_text = user_history_text(row["user_id"], history_map)
    h_flags = user_history_flags(row["user_id"], history_map)

    per_image_notes = []
    for image_id, abs_path in images:
        data_uri, _, _ = load_and_encode(abs_path)
        prompt = prompts.per_image_prompt(claim_object, user_claim, evidence_text, image_id)
        raw = nim_client.vision_chat(prompt, [data_uri], system=prompts.PER_IMAGE_SYSTEM)
        parsed = _extract_json(raw)
        per_image_notes.append(json.dumps(parsed) if parsed else raw.strip()[:400])

    agg_prompt = prompts.aggregate_prompt(
        claim_object, user_claim, evidence_text, h_text, h_flags, per_image_notes, image_ids
    )
    raw = nim_client.text_chat(agg_prompt, system=prompts.AGGREGATE_SYSTEM)
    parsed = _extract_json(raw)
    result = _normalize_final(parsed, claim_object, image_ids)
    return _apply_history_risk(result, h_flags)


def run_single_call(row, history_map, requirements):
    claim_object = row["claim_object"].strip().lower()
    user_claim = row["user_claim"]
    images = image_paths_to_abs(row["image_paths"])
    image_ids = [iid for iid, _ in images]
    evidence_text = evidence_text_for(claim_object, requirements)
    h_text = user_history_text(row["user_id"], history_map)
    h_flags = user_history_flags(row["user_id"], history_map)

    # The vision model used here only accepts a single image per request, so single_call
    # is restricted to the first submitted image -- a real cost/coverage trade-off vs.
    # per_image, which is reflected in the evaluation comparison.
    data_uri, _, _ = load_and_encode(images[0][1])
    prompt = prompts.single_call_prompt(claim_object, user_claim, evidence_text, h_text,
                                         h_flags, image_ids)
    raw = nim_client.vision_chat(prompt, [data_uri], system=prompts.SINGLE_CALL_SYSTEM,
                                  max_tokens=700)
    parsed = _extract_json(raw)
    result = _normalize_final(parsed, claim_object, image_ids)
    return _apply_history_risk(result, h_flags)


STRATEGIES = {
    "per_image": run_per_image,
    "single_call": run_single_call,
}


def run_claim(row, history_map, requirements, strategy="per_image"):
    fn = STRATEGIES[strategy]
    result = fn(row, history_map, requirements)
    out = {
        "user_id": row["user_id"],
        "image_paths": row["image_paths"],
        "user_claim": row["user_claim"],
        "claim_object": row["claim_object"],
    }
    out.update(result)
    return {col: out[col] for col in schema.OUTPUT_COLUMNS}
