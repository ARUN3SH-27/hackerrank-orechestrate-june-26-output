"""Allowed-value lists and normalization for the evidence-review output schema."""

CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ISSUE_TYPE = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
}

OBJECT_PART = {
    "car": {"front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
            "headlight", "taillight", "fender", "quarter_panel", "body", "unknown"},
    "laptop": {"screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port",
               "base", "body", "unknown"},
    "package": {"box", "package_corner", "package_side", "seal", "label",
                "contents", "item", "unknown"},
}

RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required",
}

SEVERITY = {"none", "low", "medium", "high", "unknown"}

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]


def norm_bool(value, default=False):
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "yes", "1"):
        return True
    if s in ("false", "no", "0"):
        return False
    return default


def norm_enum(value, allowed, default):
    s = str(value).strip().lower().replace(" ", "_") if value is not None else ""
    return s if s in allowed else default


def norm_object_part(value, claim_object):
    allowed = OBJECT_PART.get(claim_object, {"unknown"})
    return norm_enum(value, allowed, "unknown")


def norm_risk_flags(values):
    if values is None:
        return ["none"]
    if isinstance(values, str):
        values = [v.strip() for v in values.split(";")]
    out = []
    for v in values:
        s = str(v).strip().lower().replace(" ", "_")
        if s and s in RISK_FLAGS and s != "none" and s not in out:
            out.append(s)
    return out or ["none"]


def norm_image_ids(values, valid_ids):
    if values is None:
        return ["none"]
    if isinstance(values, str):
        values = [v.strip() for v in values.split(";")]
    out = [v for v in values if v in valid_ids and v not in ("none", "")]
    seen = []
    for v in out:
        if v not in seen:
            seen.append(v)
    return seen or ["none"]
