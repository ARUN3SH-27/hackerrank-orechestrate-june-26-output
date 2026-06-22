"""Prompt templates for the two pipeline strategies."""

ALLOWED_BLOCK = """
Allowed values (use the closest match, never invent new ones):
issue_type: dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging,
  crushed_packaging, water_damage, stain, none, unknown
object_part (car): front_bumper, rear_bumper, door, hood, windshield, side_mirror, headlight,
  taillight, fender, quarter_panel, body, unknown
object_part (laptop): screen, keyboard, trackpad, hinge, lid, corner, port, base, body, unknown
object_part (package): box, package_corner, package_side, seal, label, contents, item, unknown
risk_flags: none, blurry_image, cropped_or_obstructed, low_light_or_glare, wrong_angle,
  wrong_object, wrong_object_part, damage_not_visible, claim_mismatch, possible_manipulation,
  non_original_image, text_instruction_present, user_history_risk, manual_review_required
claim_status: supported, contradicted, not_enough_information
severity: none, low, medium, high, unknown
"""

PER_IMAGE_SYSTEM = (
    "You are a meticulous visual claims inspector. You only describe what is literally "
    "visible in the image. You never assume facts not shown. You always answer with a single "
    "JSON object and nothing else."
)


def per_image_prompt(claim_object, user_claim, evidence_text, image_id):
    return f"""Object type being claimed: {claim_object}
Customer conversation about the claim:
{user_claim}

Minimum evidence checklist for this object type:
{evidence_text}

This is image "{image_id}" from the claim's image set. Inspect it carefully and answer ONLY
with a JSON object with these exact keys:
{{
  "object_match": true/false,            // does this image show the claimed object type ({claim_object})?
  "part_visible": true/false,            // is the specific part the customer is talking about visible?
  "object_part": "<best-matching object_part value, see allowed list>",
  "issue_observed": true/false,          // is any damage/issue visible on this part, regardless of the claim?
  "issue_type": "<best-matching issue_type value>",
  "severity_hint": "none|low|medium|high|unknown",
  "image_quality_ok": true/false,        // is the image sharp, well-lit, and not cropped/obstructed?
  "quality_issues": ["<zero or more of: blurry_image, cropped_or_obstructed, low_light_or_glare, wrong_angle>"],
  "possible_manipulation": true/false,
  "text_instruction_present": true/false, // screenshot/text overlay instructing what to claim
  "short_description": "<one sentence, grounded only in what is visible>"
}}
{ALLOWED_BLOCK}
Respond with only the JSON object."""


AGGREGATE_SYSTEM = (
    "You are a senior claims adjudicator. You synthesize per-image inspection notes, the "
    "customer conversation, and account risk history into one final structured decision. "
    "Images are the primary source of truth; history adds risk context but never overrides "
    "clear visual evidence by itself. You always answer with a single JSON object and nothing else."
)


def aggregate_prompt(claim_object, user_claim, evidence_text, history_text, history_flags,
                      per_image_notes, image_ids):
    notes_block = "\n".join(
        f"- image {iid}: {note}" for iid, note in zip(image_ids, per_image_notes)
    )
    return f"""Claim object type: {claim_object}
Customer conversation:
{user_claim}

Minimum evidence checklist:
{evidence_text}

Per-image inspection notes (already grounded in the actual images):
{notes_block}

Account history: {history_text}
Pre-existing history flags to consider folding into risk_flags if still relevant: {history_flags}

Decide the final claim outcome. Answer ONLY with a JSON object with these exact keys:
{{
  "evidence_standard_met": true/false,
  "evidence_standard_met_reason": "<short reason>",
  "risk_flags": ["<zero or more allowed risk flags>"],
  "issue_type": "<allowed issue_type>",
  "object_part": "<allowed object_part for {claim_object}>",
  "claim_status": "supported|contradicted|not_enough_information",
  "claim_status_justification": "<concise, image-grounded, mention image ids when helpful>",
  "supporting_image_ids": ["<subset of {image_ids}, or empty if none support the decision>"],
  "valid_image": true/false,
  "severity": "none|low|medium|high|unknown"
}}
{ALLOWED_BLOCK}
Rules:
- claim_status=supported only if the images clearly show damage consistent with the claim.
- claim_status=contradicted if the images clearly show the claimed area in good condition, or show
  evidence inconsistent with the claim (e.g. wrong object/part, no damage where claimed).
- claim_status=not_enough_information if the images are insufficient, low quality, ambiguous, or
  do not allow a confident decision either way.
- Use user_history_risk only if history indicates real risk; use manual_review_required when the
  case is genuinely borderline or history flags call for it.
- supporting_image_ids must only include images that meaningfully informed the decision.
Respond with only the JSON object."""


SINGLE_CALL_SYSTEM = (
    "You are a visual claims adjudicator. You see the claimed object's submitted photos directly "
    "and must ground every judgment only in what is visible. You always answer with a single JSON "
    "object and nothing else."
)


def single_call_prompt(claim_object, user_claim, evidence_text, history_text, history_flags, image_ids):
    return f"""Claim object type: {claim_object}
Customer conversation:
{user_claim}

Minimum evidence checklist:
{evidence_text}

Account history: {history_text}
Pre-existing history flags to consider: {history_flags}

The claim has {len(image_ids)} submitted image(s) with ids {image_ids}, but only the first
image ("{image_ids[0]}") is attached to this request. Inspect the attached image and answer
ONLY with a JSON object with these exact keys:
{{
  "evidence_standard_met": true/false,
  "evidence_standard_met_reason": "<short reason>",
  "risk_flags": ["<zero or more allowed risk flags>"],
  "issue_type": "<allowed issue_type>",
  "object_part": "<allowed object_part for {claim_object}>",
  "claim_status": "supported|contradicted|not_enough_information",
  "claim_status_justification": "<concise, image-grounded, mention image ids when helpful>",
  "supporting_image_ids": ["<subset of {image_ids}, or empty if none support the decision>"],
  "valid_image": true/false,
  "severity": "none|low|medium|high|unknown"
}}
{ALLOWED_BLOCK}
Rules:
- claim_status=supported only if the images clearly show damage consistent with the claim.
- claim_status=contradicted if the images clearly show the claimed area in good condition, or show
  evidence inconsistent with the claim (e.g. wrong object/part, no damage where claimed).
- claim_status=not_enough_information if the images are insufficient, low quality, ambiguous, or
  do not allow a confident decision either way.
Respond with only the JSON object."""
