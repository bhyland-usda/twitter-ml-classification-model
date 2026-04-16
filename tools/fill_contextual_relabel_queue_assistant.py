#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

LABEL_ACTIONABLE = "actionable"
LABEL_NON_ACTIONABLE = "non_actionable"

REQUIRED_COLUMNS = (
    "rank",
    "review_bucket",
    "review_priority_score",
    "comment_id",
    "predicted_label",
    "final_label",
    "confidence",
    "final_confidence",
    "guardrail_decision_source",
    "guardrail_rule_ids",
    "guardrail_categories",
    "review_hint",
    "suggested_review_action",
    "parent_text",
    "text",
    "reviewer_label",
    "reviewer_notes",
)

REVIEW_BUCKET_SURVIVED = "survived_actionable"
REVIEW_BUCKET_DOWNGRADED = "downgraded_actionable"

ACTIONABLE_OVERRIDE_IDS: Set[str] = {
    "2039148788296585625",
    "2037908787730714759",
    "2037744491596304580",
    "2037904189951504644",
    "2037585171130556561",
    "2037719965369819588",
    "2037707580504436872",
    "2037707622564913264",
    "2038191815094940118",
    "2037912327459115385",
    "2037906312298918026",
    "2037610364318228923",
    "2037593029247168754",
    "2037992934088540666",
    "2037638130434494844",
    "2037848593827041372",
    "2037963660237136200",
    "2038046519622873588",
    "2037684090112037244",
    "2038212769732808783",
    "2037931363211325777",
    "2037693289982288217",
    "2038045758625063046",
    "2037998105048555606",
    "2039346804718747842",
    "2038279000477336059",
    "2037700178464092340",
    "2037622638671999283",
    "2037932635624755560",
    "2037306786110144995",
    "2037611625230877144",
    "2037933939264831990",
    "2037908265720258715",
    "2039345572654506267",
    "2037989643157713265",
    "2037924641608475117",
    "2038262449560866911",
    "2038256018799362370",
    "2038235856197300531",
    "2037933340754383307",
    "2037664314295722110",
    "2038101365763817499",
    "2038034196598292543",
    "2039444457477562880",
    "2037919989131358355",
    "2039450631056842960",
    "2037906778592100375",
    "2039448369647517994",
    "2037937545716044070",
    "2038005022034731031",
    "2037979278566908239",
    "2038046474185949416",
    "2037902940632207640",
    "2037591054593974713",
    "2037702719654252558",
    "2037689322019619276",
    "2037957003197222916",
    "2037696050064965940",
    "2037693558262480921",
    "2039438393889955936",
    "2037594038442836028",
}

NON_ACTIONABLE_OVERRIDE_IDS: Set[str] = {
    "2037644965371850808",
}

SEVERE_ABUSE_RE = re.compile(
    r"\b("
    r"piece of shit|bullshit|fuck(?:ing|er|ers)?|ass clown|bitch|loser|fraud|"
    r"traitor|idiot|stupid|cringe|disgusting grifter|jokers?|cowards?|"
    r"licking .* starfish|suck on that|orange god|hell no"
    r")\b",
    re.IGNORECASE,
)

MODERATE_TOXIC_RE = re.compile(
    r"\b("
    r"you fools|you people|liar|lying|tone[- ]deaf|gaslighting|fraud|"
    r"you are not serious people|you are a joke|you are one of the losers|"
    r"bankrupting farmers|poisoning us|destroying our soil|you're a fraud"
    r")\b",
    re.IGNORECASE,
)

SPAM_RE = re.compile(
    r"\b("
    r"telegram|follow me|respond with|x inbox|financial advice|gifts|"
    r"investment community|come say hi|thank you for being a sincere legitimate person"
    r")\b",
    re.IGNORECASE,
)

PRAISE_OR_CHATTER_RE = re.compile(
    r"\b("
    r"great!?|fantastic!?|thanks!?|thank you!?|favorite food|"
    r"yes yes yes|exactly|i hope that's the case|farmers are the backbone|"
    r"the future of agriculture looks bright|your address .* excellent|"
    r"onlyfarms|soft porn for destitute farmers"
    r")\b",
    re.IGNORECASE,
)

OFF_TOPIC_GEOPOLITICS_RE = re.compile(
    r"\b("
    r"iran|israel|tehran|war crimes|gaza|church of the holy sepulchre|"
    r"buddgoptreason|pharmaceutical facility|cardinal|mass"
    r")\b",
    re.IGNORECASE,
)

POLICY_OBJECT_RE = re.compile(
    r"\b("
    r"mcool|mandatory country of origin|country of origin|origin labels?|"
    r"label(?:ing)?|glyphosate|chemicals?|lab(?: |-)?meat|fertilizer|"
    r"diesel|fuel costs?|farm(?:er|ers|ing)?|ranch(?:er|ers)?|wild horses?|"
    r"school lunches?|food stamp fraud|snap|forest service|epa|"
    r"prime act|coalition|grants?|relief|beef|argentin(?:a|ian)|"
    r"foreign beef|foreign ownership|ccp|farmland|processing plants?|"
    r"slaughterhouses?|food prices?|egg prices?|eggs affordable|"
    r"verification process|tariffs?|import(?:ing|s)?|subsid(?:y|ies)|"
    r"fuel standard|renewable fuels?|food freedom|amish farmer|"
    r"local butchers?|local consumers?|school lunches?|poison|"
    r"human remains|humanure|geoengineering|school lunches|"
    r"food supply|made in america|product of usa|farmer aid|small farmers?|"
    r"family farms?|rural america|meatpacking"
    r")\b",
    re.IGNORECASE,
)

ACTION_REQUEST_RE = re.compile(
    r"\b("
    r"please\b|stop\b|ban\b|get rid of\b|do something about\b|"
    r"make .* mandatory\b|make sure\b|protect\b|save\b|"
    r"assign someone\b|ask someone to contact me\b|contact me\b|"
    r"provide (?:relief|update|updates)\b|give (?:aid|grants?|guarantees?)\b|"
    r"focus on\b|shut down\b|allow\b|don't allow\b|"
    r"pass mcool\b|pass prime act\b|make eggs affordable\b|"
    r"make america healthy again\b"
    r")",
    re.IGNORECASE,
)

QUESTION_REQUEST_RE = re.compile(
    r"\b("
    r"when will|how do you plan|what are you doing|why aren't you|"
    r"why dont you|why don't you|will you|can you|is it mandatory|"
    r"where is the|where are the|what changed|how about|"
    r"are we going to|do you plan on|do you plan to"
    r")\b",
    re.IGNORECASE,
)

DIRECT_CONTACT_RE = re.compile(
    r"\b("
    r"contact me|ask someone to contact me|reach my management|"
    r"assign someone|brief your office"
    r")\b",
    re.IGNORECASE,
)

SHORT_AMBIGUOUS_RE = re.compile(
    r"^(great!?|thanks!?|thank you!?|exactly!?|yes!?|yes yes yes!?|"
    r"i hope that's the case\.?|literally my favorite food\.?|"
    r"your address to our farmers was excellent\.? thanks\.?)$",
    re.IGNORECASE,
)

MANDATORY_LABEL_RE = re.compile(
    r"\b("
    r"mcool|mandatory country of origin|mandatory .* label|"
    r"country of origin labels? mandatory|mandatory labeling"
    r")\b",
    re.IGNORECASE,
)

WILD_HORSE_RE = re.compile(
    r"\bwild horses?\b|\bprotect .* horses?\b|\bsave .* horses?\b", re.IGNORECASE
)
COST_RELIEF_RE = re.compile(
    r"\b("
    r"fertilizer|diesel|fuel costs?|egg prices?|eggs affordable|"
    r"relief|grants?|aid|costs? .* rising|rising costs?"
    r")\b",
    re.IGNORECASE,
)

VOLUNTARY_LABEL_RE = re.compile(
    r"\b(voluntary|already on the books|biden era program|didn't make it mandatory)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill a contextual relabel queue with assistant-proposed "
            "reviewer_label and reviewer_notes values."
        )
    )
    parser.add_argument(
        "--input", required=True, help="Input contextual relabel queue CSV."
    )
    parser.add_argument(
        "--output", required=True, help="Output CSV with assistant-filled labels."
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite rows that already contain reviewer_label or reviewer_notes.",
    )
    return parser.parse_args()


def clean(value: str | None) -> str:
    if value is None:
        return ""
    return value.replace("\ufeff", "").replace("\x00", "").strip()


def load_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"[ERROR] CSV has no header row: {path}")
        fieldnames = [clean(name) for name in reader.fieldnames]
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise SystemExit(
                f"[ERROR] input CSV missing required columns {missing}: {path}"
            )

        rows: List[Dict[str, str]] = []
        for row in reader:
            normalized = {clean(k): clean(v) for k, v in row.items()}
            rows.append(normalized)

    if not rows:
        raise SystemExit(f"[ERROR] input CSV has zero rows: {path}")

    return rows, fieldnames


def write_rows(
    path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def has_policy_object(text: str) -> bool:
    return bool(POLICY_OBJECT_RE.search(text))


def has_action_request(text: str) -> bool:
    return bool(ACTION_REQUEST_RE.search(text) or QUESTION_REQUEST_RE.search(text))


def classify_row(row: Dict[str, str]) -> Tuple[str, str]:
    bucket = clean(row.get("review_bucket"))
    text = normalize_space(row.get("text", ""))
    parent_text = normalize_space(row.get("parent_text", ""))
    combined = f"{parent_text} {text}".strip()
    lowered = combined.lower()

    reasons: List[str] = []
    action_score = 0
    non_action_score = 0

    if not text:
        return LABEL_NON_ACTIONABLE, "assistant: empty text"

    comment_id = clean(row.get("comment_id"))

    if comment_id in ACTIONABLE_OVERRIDE_IDS:
        return (
            LABEL_ACTIONABLE,
            "assistant: manual override actionable; specific contextual request or policy ask",
        )

    if comment_id in NON_ACTIONABLE_OVERRIDE_IDS:
        return (
            LABEL_NON_ACTIONABLE,
            "assistant: manual override non-actionable; sarcasm / low-signal complaint",
        )

    if SPAM_RE.search(combined):
        return LABEL_NON_ACTIONABLE, "assistant: spam / outreach solicitation"
    if SEVERE_ABUSE_RE.search(text):
        return (
            LABEL_NON_ACTIONABLE,
            "assistant: derogatory / abusive language overrides actionability",
        )
    if OFF_TOPIC_GEOPOLITICS_RE.search(combined) and not has_policy_object(combined):
        return (
            LABEL_NON_ACTIONABLE,
            "assistant: off-topic geopolitics without concrete USDA-relevant ask",
        )
    if SHORT_AMBIGUOUS_RE.match(text):
        return (
            LABEL_NON_ACTIONABLE,
            "assistant: short affirmation / chatter without actionable content",
        )

    if MODERATE_TOXIC_RE.search(text):
        non_action_score += 2
        reasons.append("toxic_tone")

    if (
        PRAISE_OR_CHATTER_RE.search(text)
        and len(text) < 90
        and not has_action_request(text)
    ):
        non_action_score += 3
        reasons.append("generic_chatter_or_praise")

    if DIRECT_CONTACT_RE.search(text):
        action_score += 4
        reasons.append("direct_contact_request")

    if WILD_HORSE_RE.search(text):
        action_score += 4
        reasons.append("protective_request_wild_horses")

    if MANDATORY_LABEL_RE.search(text):
        action_score += 4
        reasons.append("mandatory_labeling_request")

    if COST_RELIEF_RE.search(text) and has_action_request(text):
        action_score += 3
        reasons.append("cost_relief_request")

    if has_policy_object(text):
        action_score += 2
        reasons.append("policy_object_present")

    if has_action_request(text):
        action_score += 2
        reasons.append("request_structure_present")

    if VOLUNTARY_LABEL_RE.search(text):
        action_score += 1
        reasons.append("criticizes_voluntary_policy")

    if "please" in lowered:
        action_score += 1
        reasons.append("explicit_please")

    if text.endswith("?") and has_policy_object(text):
        action_score += 1
        reasons.append("policy_question")

    if len(re.sub(r"[^A-Za-z0-9]+", "", text)) < 12:
        non_action_score += 3
        reasons.append("very_low_information")

    if bucket == REVIEW_BUCKET_SURVIVED:
        if non_action_score >= 4 and action_score < 4:
            return (
                LABEL_NON_ACTIONABLE,
                "assistant: survived queue row downgraded due to strong non-actionable cues: "
                + ", ".join(reasons),
            )
        return (
            LABEL_ACTIONABLE,
            "assistant: survived actionable retained; cues="
            + (
                ", ".join(reasons)
                if reasons
                else "survived_without_contradictory_signals"
            ),
        )

    if bucket == REVIEW_BUCKET_DOWNGRADED:
        if action_score >= 5 and non_action_score <= 2:
            return (
                LABEL_ACTIONABLE,
                "assistant: contextual policy / request signal strong enough to relabel actionable; cues="
                + ", ".join(reasons),
            )
        if action_score >= 4 and "direct_contact_request" in reasons:
            return (
                LABEL_ACTIONABLE,
                "assistant: explicit request for agency contact / follow-up; cues="
                + ", ".join(reasons),
            )
        if action_score >= 4 and "mandatory_labeling_request" in reasons:
            return (
                LABEL_ACTIONABLE,
                "assistant: concrete labeling / policy ask; cues=" + ", ".join(reasons),
            )
        if action_score >= 4 and "protective_request_wild_horses" in reasons:
            return (
                LABEL_ACTIONABLE,
                "assistant: concrete protection request tied to contextual policy topic; cues="
                + ", ".join(reasons),
            )
        if action_score >= 4 and "cost_relief_request" in reasons:
            return (
                LABEL_ACTIONABLE,
                "assistant: concrete relief / aid request; cues=" + ", ".join(reasons),
            )
        return (
            LABEL_NON_ACTIONABLE,
            "assistant: downgraded row remains non-actionable; insufficient concrete action despite context; cues="
            + (", ".join(reasons) if reasons else "no_strong_action_signal"),
        )

    # Fallback for unexpected queue buckets.
    if action_score > non_action_score and action_score >= 5:
        return (
            LABEL_ACTIONABLE,
            "assistant: fallback actionable due to strong policy / request cues="
            + ", ".join(reasons),
        )

    return (
        LABEL_NON_ACTIONABLE,
        "assistant: fallback non-actionable; no sufficiently concrete contextual ask; cues="
        + (", ".join(reasons) if reasons else "none"),
    )


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"[ERROR] input file not found: {input_path}")

    rows, fieldnames = load_rows(input_path)

    filled_rows: List[Dict[str, str]] = []
    proposed_counts = {LABEL_ACTIONABLE: 0, LABEL_NON_ACTIONABLE: 0}
    preserved_existing = 0

    for row in rows:
        current_label = clean(row.get("reviewer_label"))
        current_notes = clean(row.get("reviewer_notes"))

        if not args.overwrite_existing and (current_label or current_notes):
            preserved_existing += 1
            filled_rows.append(dict(row))
            continue

        proposed_label, proposed_notes = classify_row(row)
        updated = dict(row)
        updated["reviewer_label"] = proposed_label
        updated["reviewer_notes"] = proposed_notes
        filled_rows.append(updated)
        proposed_counts[proposed_label] += 1

    write_rows(output_path, filled_rows, fieldnames)

    print(f"[INFO] input={input_path}")
    print(f"[INFO] output={output_path}")
    print(f"[INFO] rows={len(rows)}")
    print(f"[INFO] preserved_existing={preserved_existing}")
    print(f"[INFO] proposed_actionable={proposed_counts[LABEL_ACTIONABLE]}")
    print(f"[INFO] proposed_non_actionable={proposed_counts[LABEL_NON_ACTIONABLE]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
