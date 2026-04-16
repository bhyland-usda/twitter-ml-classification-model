#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

LABEL_ACTIONABLE = "actionable"
LABEL_NON_ACTIONABLE = "non_actionable"
ALLOWED_LABELS = {LABEL_ACTIONABLE, LABEL_NON_ACTIONABLE}

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

TOKEN_RE = re.compile(r"[a-z0-9']+")
MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")
WHITESPACE_RE = re.compile(r"\s+")

REQUEST_PATTERNS: Tuple[Tuple[str, float, str], ...] = (
    (r"\bmandatory\b", 2.0, "mandatory_request"),
    (r"\bmcool\b", 3.0, "mcool_request"),
    (r"\bcountry of origin\b", 2.5, "country_of_origin"),
    (r"\blabel(?:ing)?\b", 1.2, "labeling_policy"),
    (r"\bcontact me\b", 3.0, "direct_contact_request"),
    (r"\bask someone to contact me\b", 3.5, "office_contact_request"),
    (r"\bplease ask someone\b", 2.5, "office_contact_request"),
    (r"\bassign someone\b", 2.5, "delegation_request"),
    (r"\bcapital loans?\b", 2.5, "capital_support"),
    (r"\bgrants?\b", 2.2, "grant_request"),
    (r"\brelief\b", 1.8, "relief_request"),
    (r"\bfertilizer\b", 1.8, "fertilizer_issue"),
    (r"\bdiesel\b", 1.5, "diesel_issue"),
    (r"\bfuel costs?\b", 1.5, "fuel_cost_issue"),
    (r"\bimport(?:ing|s)? beef\b", 2.0, "import_beef_policy"),
    (r"\bforeign (?:products?|beef|ownership)\b", 2.0, "foreign_policy_request"),
    (r"\blab(?: |-)?created meat\b", 1.5, "lab_meat_issue"),
    (r"\blab(?: |-)?grown meat\b", 1.5, "lab_meat_issue"),
    (r"\borganic beef\b", 1.5, "organic_beef_request"),
    (r"\bnon[- ]?gmo seeds?\b", 1.5, "seed_request"),
    (r"\bglyphosate\b", 1.8, "glyphosate_policy"),
    (r"\bepa guidelines?\b", 2.0, "epa_guidelines_request"),
    (r"\breview the tariffs?\b", 2.2, "tariff_review_request"),
    (r"\btariffs?\b", 1.4, "tariff_issue"),
    (r"\bwild horses?\b", 2.4, "wild_horse_request"),
    (r"\bschool lunches?\b", 2.2, "school_lunch_request"),
    (r"\bfood freedom\b", 2.0, "food_freedom_request"),
    (r"\bmake eggs affordable again\b", 1.2, "egg_price_request"),
    (r"\bmake sure\b", 1.8, "explicit_make_sure"),
    (r"\bprotect\b", 1.5, "protect_request"),
    (r"\bsave\b", 1.5, "save_request"),
    (r"\bstop\b", 1.2, "stop_request"),
    (r"\bban\b", 1.5, "ban_request"),
    (r"\bget rid of\b", 1.4, "remove_request"),
    (r"\bdon't allow\b", 1.8, "disallow_request"),
    (r"\bwhy aren't\b", 1.5, "why_arent_request"),
    (r"\bwhen will\b", 1.5, "when_will_request"),
    (r"\bwhat are you doing\b", 1.8, "what_are_you_doing_request"),
    (r"\bhow do you plan\b", 1.8, "how_do_you_plan_request"),
    (r"\bwhere is the\b", 1.5, "where_is_request"),
    (r"\bwhy do(?:n't| not) you\b", 1.6, "why_dont_you_request"),
    (r"\bdo something\b", 1.2, "do_something_request"),
)

NON_ACTIONABLE_PATTERNS: Tuple[Tuple[str, float, str], ...] = (
    (r"\bdr oz\b", 4.0, "wrong_agency_droz"),
    (r"\bhasa\b", 4.0, "wrong_agency_hasa"),
    (r"\btelegram\b", 6.0, "spam_telegram"),
    (r"\bfollow me\b", 6.0, "spam_follow_me"),
    (r"\bx inbox\b", 4.0, "spam_inbox"),
    (r"\bfinancial advices?\b", 5.0, "spam_financial"),
    (r"\binvestment\b", 4.0, "spam_investment"),
    (r"\bhammer family\b", 5.0, "conspiracy_hammer"),
    (r"\breceivers? in our heads?\b", 6.0, "conspiracy_receivers"),
    (r"\bmanchurian candidates?\b", 6.0, "conspiracy_manchurian"),
    (r"\bspiritual darkness\b", 5.0, "conspiracy_spiritual"),
    (r"\bgeoengineering\b", 3.5, "conspiracy_geoengineering"),
    (r"\bchem trails?\b", 3.5, "conspiracy_chemtrails"),
    (r"\bdarpa cornucopia\b", 4.5, "conspiracy_darpa"),
    (r"\biran\b", 4.0, "off_topic_iran"),
    (r"\bisrael\b", 4.0, "off_topic_israel"),
    (r"\bchurch of the holy sepulchre\b", 5.0, "off_topic_religious"),
    (r"\bmemecoin\b", 5.0, "off_topic_memecoin"),
    (r"\bfavorite food\b", 4.0, "generic_chatter"),
    (r"\bonlyfarms\b", 5.0, "mockery_onlyfarms"),
    (r"\bsoft porn\b", 5.0, "mockery_onlyfarms"),
    (r"\bgreat!?$\b", 4.0, "generic_praise"),
    (r"\bthanks!?$\b", 4.0, "generic_praise"),
    (r"\bthank you!?$\b", 4.0, "generic_praise"),
    (r"\byes yes yes\b", 4.0, "generic_affirmation"),
    (r"\bexactly\b", 3.0, "generic_affirmation"),
    (r"\bdo you understand\?\b", 4.0, "low_information_question"),
    (r"\bwhat changed\?\b", 3.0, "medical_question"),
    (r"\bfood allergy\b", 4.0, "medical_question"),
)

SEVERE_ABUSE_RE = re.compile(
    r"\b("
    r"piece of shit|bullshit|fuck(?:ing|er|ers)?|ass clown|bitch|loser|"
    r"traitor|idiot|stupid|cringe|disgusting grifter|cowards?|"
    r"licking .* starfish|orange god|lying pos|shut the fuck up|stfu"
    r")\b",
    re.IGNORECASE,
)

MODERATE_ABUSE_RE = re.compile(
    r"\b("
    r"fraud|tone[- ]deaf|gaslighting|joke|jokers?|propaganda machine|"
    r"you people|poisoning us|destroying our soil|self promotion"
    r")\b",
    re.IGNORECASE,
)

QUESTION_ONLY_RE = re.compile(r"^\?+$")
USER_REVIEW_PREFIX = "user_review_"
ASSISTANT_GUIDED_PREFIX = "assistant_user_guided:"
UNCERTAIN_NOTE_TAG = "[uncertain]"


@dataclass(frozen=True)
class QueueExample:
    row: Dict[str, str]
    combined_text: str
    tokens: frozenset[str]
    label: str
    notes: str

    def comment_id(self) -> str:
        return clean_cell(self.row.get("comment_id"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use human-reviewed contextual queue rows to autofill the remaining queue rows "
            "with user-guided labels and notes."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input contextual queue CSV containing a mix of human-reviewed and unreviewed rows.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path for the fully filled contextual queue.",
    )
    parser.add_argument(
        "--uncertain-output",
        default="",
        help=(
            "Optional output CSV path for rows classified with low margin / lower confidence "
            "so they can be reviewed manually."
        ),
    )
    parser.add_argument(
        "--overwrite-non-human-rows",
        action="store_true",
        help=(
            "If set, recompute all non-human rows even when they already contain assistant notes. "
            "Human-reviewed rows are always preserved."
        ),
    )
    parser.add_argument(
        "--uncertain-margin",
        type=float,
        default=2.0,
        help="Absolute score margin below which a row is marked uncertain (default: 2.0).",
    )
    return parser.parse_args()


def clean_cell(value: str | None) -> str:
    if value is None:
        return ""
    return value.replace("\ufeff", "").replace("\x00", "").strip()


def normalize_space(text: str) -> str:
    return WHITESPACE_RE.sub(" ", clean_cell(text)).strip()


def normalize_for_tokens(text: str) -> str:
    text = normalize_space(text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    return text.lower()


def tokenize(text: str) -> List[str]:
    normalized = normalize_for_tokens(text)
    return [token for token in TOKEN_RE.findall(normalized) if len(token) > 2]


def token_set(text: str) -> frozenset[str]:
    return frozenset(tokenize(text))


def parse_float(value: str | None, default: float = 0.0) -> float:
    try:
        parsed = float(clean_cell(value))
    except Exception:
        return default
    if not math.isfinite(parsed):
        return default
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def read_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"[ERROR] CSV has no header row: {path}")
        fieldnames = [clean_cell(name) for name in reader.fieldnames]
        rows: List[Dict[str, str]] = []
        for raw in reader:
            rows.append({clean_cell(k): clean_cell(v) for k, v in raw.items()})
    if not rows:
        raise SystemExit(f"[ERROR] input CSV has zero rows: {path}")
    missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
    if missing:
        raise SystemExit(
            f"[ERROR] input CSV missing required columns {missing}: {path}"
        )
    return rows, fieldnames


def build_combined_text(row: Dict[str, str]) -> str:
    parent_text = normalize_space(row.get("parent_text", ""))
    text = normalize_space(row.get("text", ""))
    return f"{parent_text} {text}".strip()


def is_human_reviewed(row: Dict[str, str]) -> bool:
    return clean_cell(row.get("reviewer_notes")).startswith(USER_REVIEW_PREFIX)


def should_autofill(row: Dict[str, str], overwrite_non_human_rows: bool) -> bool:
    if is_human_reviewed(row):
        return False
    if overwrite_non_human_rows:
        return True
    return not clean_cell(row.get("reviewer_notes")).startswith(ASSISTANT_GUIDED_PREFIX)


def build_seed_examples(rows: Sequence[Dict[str, str]]) -> List[QueueExample]:
    examples: List[QueueExample] = []
    for row in rows:
        if not is_human_reviewed(row):
            continue
        label = clean_cell(row.get("reviewer_label")).lower()
        if label not in ALLOWED_LABELS:
            continue
        combined = build_combined_text(row)
        examples.append(
            QueueExample(
                row=row,
                combined_text=combined,
                tokens=token_set(combined),
                label=label,
                notes=clean_cell(row.get("reviewer_notes")),
            )
        )
    if not examples:
        raise SystemExit(
            "[ERROR] no human-reviewed rows found; cannot build user-guided autofill"
        )
    return examples


def build_token_weights(
    examples: Sequence[QueueExample],
) -> Dict[str, Dict[str, float]]:
    doc_counts: Dict[str, Counter[str]] = {
        LABEL_ACTIONABLE: Counter(),
        LABEL_NON_ACTIONABLE: Counter(),
    }
    label_counts = Counter(example.label for example in examples)

    for example in examples:
        for token in example.tokens:
            doc_counts[example.label][token] += 1

    weights: Dict[str, Dict[str, float]] = {
        LABEL_ACTIONABLE: defaultdict(float),
        LABEL_NON_ACTIONABLE: defaultdict(float),
    }

    for token in set(doc_counts[LABEL_ACTIONABLE]) | set(
        doc_counts[LABEL_NON_ACTIONABLE]
    ):
        act_count = doc_counts[LABEL_ACTIONABLE].get(token, 0)
        non_count = doc_counts[LABEL_NON_ACTIONABLE].get(token, 0)
        act_rate = (act_count + 1.0) / (label_counts[LABEL_ACTIONABLE] + 2.0)
        non_rate = (non_count + 1.0) / (label_counts[LABEL_NON_ACTIONABLE] + 2.0)
        delta = math.log(act_rate / non_rate)

        if delta > 0:
            weights[LABEL_ACTIONABLE][token] = delta
        elif delta < 0:
            weights[LABEL_NON_ACTIONABLE][token] = -delta

    return weights


def jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    if union == 0:
        return 0.0
    return intersection / union


def nearest_similarity_scores(
    candidate_tokens: frozenset[str],
    examples: Sequence[QueueExample],
) -> Dict[str, float]:
    best = {LABEL_ACTIONABLE: 0.0, LABEL_NON_ACTIONABLE: 0.0}
    for example in examples:
        similarity = jaccard_similarity(candidate_tokens, example.tokens)
        if similarity > best[example.label]:
            best[example.label] = similarity
    return best


def apply_weighted_token_scores(
    candidate_tokens: frozenset[str],
    token_weights: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    scores = {LABEL_ACTIONABLE: 0.0, LABEL_NON_ACTIONABLE: 0.0}
    for token in candidate_tokens:
        scores[LABEL_ACTIONABLE] += token_weights[LABEL_ACTIONABLE].get(token, 0.0)
        scores[LABEL_NON_ACTIONABLE] += token_weights[LABEL_NON_ACTIONABLE].get(
            token, 0.0
        )
    return scores


def apply_rule_scores(
    text: str, row: Dict[str, str]
) -> Tuple[Dict[str, float], List[str]]:
    scores = {LABEL_ACTIONABLE: 0.0, LABEL_NON_ACTIONABLE: 0.0}
    reasons: List[str] = []

    normalized_text = normalize_space(text)
    lowered = normalized_text.lower()

    if QUESTION_ONLY_RE.match(normalized_text):
        scores[LABEL_NON_ACTIONABLE] += 5.0
        reasons.append("question_marks_only")

    for pattern, weight, reason in REQUEST_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            scores[LABEL_ACTIONABLE] += weight
            reasons.append(reason)

    for pattern, weight, reason in NON_ACTIONABLE_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            scores[LABEL_NON_ACTIONABLE] += weight
            reasons.append(reason)

    severe_abuse = bool(SEVERE_ABUSE_RE.search(normalized_text))
    moderate_abuse = bool(MODERATE_ABUSE_RE.search(normalized_text))
    has_explicit_request = any(
        re.search(pattern, lowered, re.IGNORECASE) for pattern, _, _ in REQUEST_PATTERNS
    )

    if severe_abuse:
        scores[LABEL_NON_ACTIONABLE] += 4.0 if has_explicit_request else 7.0
        reasons.append("severe_abuse")
    elif moderate_abuse:
        scores[LABEL_NON_ACTIONABLE] += 2.0
        reasons.append("moderate_abuse")

    review_bucket = clean_cell(row.get("review_bucket"))
    if review_bucket == "survived_actionable":
        scores[LABEL_ACTIONABLE] += 2.0
        reasons.append("survived_actionable_bucket")
    elif review_bucket == "downgraded_actionable":
        scores[LABEL_NON_ACTIONABLE] += 0.5
        reasons.append("downgraded_actionable_bucket")

    decision_sources = clean_cell(row.get("guardrail_decision_source")).lower()
    if "missing_concrete_action" in decision_sources:
        scores[LABEL_NON_ACTIONABLE] += 1.5
        reasons.append("guardrail_missing_concrete_action")
    if "question_without_concrete_action" in decision_sources:
        scores[LABEL_NON_ACTIONABLE] += 1.0
        reasons.append("guardrail_question_without_concrete_action")
    if "missing_action_cue" in decision_sources:
        scores[LABEL_NON_ACTIONABLE] += 0.75
        reasons.append("guardrail_missing_action_cue")
    if "safety_veto" in decision_sources or "scope_veto" in decision_sources:
        scores[LABEL_NON_ACTIONABLE] += 3.0
        reasons.append("guardrail_veto")

    categories = clean_cell(row.get("guardrail_categories")).lower()
    if any(
        category in categories
        for category in (
            "abuse",
            "violence",
            "xenophobia",
            "conspiracy",
            "scam_spam",
            "health_misinformation",
            "out_of_scope",
            "off_topic_geopolitics",
        )
    ):
        scores[LABEL_NON_ACTIONABLE] += 3.0
        reasons.append("guardrail_category_veto")

    if (
        lowered.startswith("@")
        and "@secrollins" not in lowered
        and "@usda" not in lowered
    ):
        scores[LABEL_NON_ACTIONABLE] += 1.5
        reasons.append("addressed_to_other_user")

    if len(TOKEN_RE.findall(lowered)) < 6:
        scores[LABEL_NON_ACTIONABLE] += 2.0
        reasons.append("low_information_length")

    return scores, reasons


def classify_candidate(
    row: Dict[str, str],
    seed_examples: Sequence[QueueExample],
    token_weights: Dict[str, Dict[str, float]],
    uncertain_margin: float,
) -> Tuple[str, str, float, bool]:
    combined_text = build_combined_text(row)
    candidate_tokens = token_set(combined_text)

    base_scores = {
        LABEL_ACTIONABLE: 0.0,
        LABEL_NON_ACTIONABLE: 0.0,
    }

    weighted = apply_weighted_token_scores(candidate_tokens, token_weights)
    base_scores[LABEL_ACTIONABLE] += weighted[LABEL_ACTIONABLE]
    base_scores[LABEL_NON_ACTIONABLE] += weighted[LABEL_NON_ACTIONABLE]

    similarity = nearest_similarity_scores(candidate_tokens, seed_examples)
    base_scores[LABEL_ACTIONABLE] += similarity[LABEL_ACTIONABLE] * 12.0
    base_scores[LABEL_NON_ACTIONABLE] += similarity[LABEL_NON_ACTIONABLE] * 12.0

    rule_scores, reasons = apply_rule_scores(combined_text, row)
    base_scores[LABEL_ACTIONABLE] += rule_scores[LABEL_ACTIONABLE]
    base_scores[LABEL_NON_ACTIONABLE] += rule_scores[LABEL_NON_ACTIONABLE]

    actionable_score = base_scores[LABEL_ACTIONABLE]
    non_actionable_score = base_scores[LABEL_NON_ACTIONABLE]
    margin = actionable_score - non_actionable_score

    label = LABEL_ACTIONABLE if margin >= 0.0 else LABEL_NON_ACTIONABLE
    uncertain = abs(margin) < uncertain_margin

    note = (
        f"{ASSISTANT_GUIDED_PREFIX} "
        f"label={label}; "
        f"actionable_score={actionable_score:.3f}; "
        f"non_actionable_score={non_actionable_score:.3f}; "
        f"similarity_actionable={similarity[LABEL_ACTIONABLE]:.3f}; "
        f"similarity_non_actionable={similarity[LABEL_NON_ACTIONABLE]:.3f}; "
        f"rules={','.join(reasons) if reasons else 'none'}"
    )

    if uncertain:
        note += f" {UNCERTAIN_NOTE_TAG}"

    return label, note, abs(margin), uncertain


def write_rows(
    path: Path,
    rows: Sequence[Dict[str, str]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    uncertain_output_path = (
        Path(args.uncertain_output) if args.uncertain_output else None
    )

    if not input_path.exists():
        raise SystemExit(f"[ERROR] input file not found: {input_path}")
    if args.uncertain_margin <= 0.0:
        raise SystemExit("[ERROR] --uncertain-margin must be > 0")

    rows, fieldnames = read_rows(input_path)
    seed_examples = build_seed_examples(rows)
    token_weights = build_token_weights(seed_examples)

    filled_rows: List[Dict[str, str]] = []
    uncertain_rows: List[Dict[str, str]] = []

    stats = Counter()
    stats["rows_total"] = len(rows)
    stats["seed_human_rows"] = len(seed_examples)

    for row in rows:
        updated = dict(row)

        if is_human_reviewed(row):
            stats["rows_preserved_human_review"] += 1
            filled_rows.append(updated)
            continue

        if not should_autofill(row, args.overwrite_non_human_rows):
            stats["rows_preserved_existing_non_human"] += 1
            filled_rows.append(updated)
            continue

        label, notes, margin_abs, uncertain = classify_candidate(
            row=updated,
            seed_examples=seed_examples,
            token_weights=token_weights,
            uncertain_margin=args.uncertain_margin,
        )

        updated["reviewer_label"] = label
        updated["reviewer_notes"] = notes
        filled_rows.append(updated)

        stats["rows_autofilled"] += 1
        stats[f"autofilled_{label}"] += 1

        if uncertain:
            uncertain_rows.append(updated)
            stats["rows_uncertain"] += 1

        if margin_abs >= 6.0:
            stats["rows_high_margin"] += 1
        elif margin_abs >= 3.0:
            stats["rows_medium_margin"] += 1
        else:
            stats["rows_low_margin"] += 1

    write_rows(output_path, filled_rows, fieldnames)

    if uncertain_output_path:
        write_rows(uncertain_output_path, uncertain_rows, fieldnames)

    label_counts = Counter(
        clean_cell(row.get("reviewer_label")).lower() for row in filled_rows
    )
    uncertain_label_counts = Counter(
        clean_cell(row.get("reviewer_label")).lower() for row in uncertain_rows
    )

    print(f"[INFO] input={input_path}")
    print(f"[INFO] output={output_path}")
    if uncertain_output_path:
        print(f"[INFO] uncertain_output={uncertain_output_path}")

    for key in (
        "rows_total",
        "seed_human_rows",
        "rows_preserved_human_review",
        "rows_preserved_existing_non_human",
        "rows_autofilled",
        "autofilled_actionable",
        "autofilled_non_actionable",
        "rows_uncertain",
        "rows_high_margin",
        "rows_medium_margin",
        "rows_low_margin",
    ):
        print(f"[INFO] {key}={stats.get(key, 0)}")

    print(
        "[INFO] final_label_distribution="
        f"actionable:{label_counts.get(LABEL_ACTIONABLE, 0)} "
        f"non_actionable:{label_counts.get(LABEL_NON_ACTIONABLE, 0)}"
    )
    if uncertain_output_path:
        print(
            "[INFO] uncertain_label_distribution="
            f"actionable:{uncertain_label_counts.get(LABEL_ACTIONABLE, 0)} "
            f"non_actionable:{uncertain_label_counts.get(LABEL_NON_ACTIONABLE, 0)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
