#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple


REQUIRED_COLUMNS = {
    "comment_id",
    "text",
    "predicted_label",
    "confidence",
    "guardrail_triggered",
    "guardrail_decision_source",
    "guardrail_rule_ids",
    "guardrail_categories",
    "final_label",
    "final_confidence",
}


def parse_sources(raw: str) -> Set[str]:
    return {x.strip() for x in (raw or "").split("|") if x.strip()}


def parse_float(raw: str, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return default
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def find_latest_non_actionable_log(queues_dir: Path) -> Path:
    candidates = sorted(queues_dir.glob("non_actionable_log_*.csv"))
    if not candidates:
        raise SystemExit(
            f"[ERROR] No non_actionable_log_*.csv files found in {queues_dir}"
        )
    return candidates[-1]


def ensure_columns(rows: List[Dict[str, str]], path: Path) -> None:
    if not rows:
        raise SystemExit(f"[ERROR] CSV has zero rows: {path}")
    columns = set(rows[0].keys())
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise SystemExit(
            f"[ERROR] CSV missing required columns {sorted(missing)} in {path}"
        )


def build_review_hint(
    sources: Set[str],
    categories: Set[str],
    confidence: float,
) -> str:
    if "question_without_concrete_action" in sources and confidence >= 0.95:
        return "High-confidence question-style downgrade: check if operational ask exists."
    if "missing_concrete_action" in sources and confidence >= 0.95:
        return "High-confidence structural downgrade: verify if concrete action target exists."
    if "missing_action_cue" in sources:
        return "No explicit action cue: verify whether implied request should still be actionable."
    if "rule_match" in sources:
        return "Rule-match downgrade: verify policy/category assignment."
    if "health_misinformation" in categories:
        return "Health-misinformation category: verify safety veto decision."
    return "General borderline candidate: manual review recommended."


def score_candidate(
    confidence: float,
    sources: Set[str],
    categories: Set[str],
) -> float:
    score = confidence * 100.0

    if "question_without_concrete_action" in sources:
        score += 6.0
    if "missing_concrete_action" in sources:
        score += 5.0
    if "missing_action_cue" in sources:
        score += 3.0

    # De-prioritize explicit policy/safety veto rows for "borderline" review.
    if "safety_veto" in sources:
        score -= 15.0
    if "scope_veto" in sources:
        score -= 12.0
    if "civility_gate" in sources:
        score -= 10.0
    if "rule_match" in sources:
        score -= 6.0

    # Mild de-prioritization for clearly severe categories.
    if {"xenophobia", "violence", "abuse"} & categories:
        score -= 8.0
    if {"conspiracy", "scam_spam"} & categories:
        score -= 5.0

    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a borderline non-actionable review queue from guardrail logs for "
            "manual labeling."
        )
    )
    parser.add_argument(
        "--input",
        default="",
        help=(
            "Path to non_actionable_log CSV. If omitted, script auto-selects the latest "
            "non_actionable_log_*.csv in --queues-dir."
        ),
    )
    parser.add_argument(
        "--queues-dir",
        default="ml_artifacts/queues",
        help="Directory used when --input is omitted (default: ml_artifacts/queues).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path for borderline review queue.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200,
        help="Maximum number of rows to emit (default: 200).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.85,
        help="Minimum model confidence to consider (default: 0.85).",
    )
    parser.add_argument(
        "--exclude-sources",
        default="",
        help=(
            "Pipe-delimited decision sources to exclude entirely "
            '(example: "safety_veto|scope_veto").'
        ),
    )
    parser.add_argument(
        "--include-only-sources",
        default="",
        help=(
            "Optional pipe-delimited decision sources; if set, row must contain at least "
            "one of these."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.max_rows <= 0:
        raise SystemExit("[ERROR] --max-rows must be > 0")
    if not (0.0 <= args.min_confidence <= 1.0):
        raise SystemExit("[ERROR] --min-confidence must be in [0, 1]")

    input_path = Path(args.input) if args.input else find_latest_non_actionable_log(Path(args.queues_dir))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exclude_sources = parse_sources(args.exclude_sources)
    include_only_sources = parse_sources(args.include_only_sources)

    with input_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    ensure_columns(rows, input_path)

    candidates: List[Tuple[float, Dict[str, str]]] = []
    total_overridden_from_actionable = 0
    skipped_by_confidence = 0
    skipped_by_source_filters = 0

    for row in rows:
        predicted_label = (row.get("predicted_label") or "").strip()
        final_label = (row.get("final_label") or "").strip()

        if predicted_label != "actionable" or final_label != "non_actionable":
            continue

        total_overridden_from_actionable += 1

        confidence = parse_float(row.get("confidence") or "0")
        if confidence < args.min_confidence:
            skipped_by_confidence += 1
            continue

        sources = parse_sources(row.get("guardrail_decision_source") or "")
        categories = parse_sources(row.get("guardrail_categories") or "")

        if exclude_sources and (sources & exclude_sources):
            skipped_by_source_filters += 1
            continue

        if include_only_sources and not (sources & include_only_sources):
            skipped_by_source_filters += 1
            continue

        priority = score_candidate(confidence, sources, categories)
        hint = build_review_hint(sources, categories, confidence)

        candidate = {
            "comment_id": (row.get("comment_id") or "").strip(),
            "text": " ".join((row.get("text") or "").split()),
            "confidence": f"{confidence:.6f}",
            "final_confidence": (row.get("final_confidence") or "").strip(),
            "guardrail_decision_source": "|".join(sorted(sources)),
            "guardrail_rule_ids": (row.get("guardrail_rule_ids") or "").strip(),
            "guardrail_categories": "|".join(sorted(categories)),
            "review_priority_score": f"{priority:.3f}",
            "review_hint": hint,
        }
        candidates.append((priority, candidate))

    candidates.sort(
        key=lambda x: (
            x[0],
            parse_float(x[1].get("confidence") or "0"),
            x[1].get("comment_id") or "",
        ),
        reverse=True,
    )

    selected = [row for _, row in candidates[: args.max_rows]]

    fieldnames = [
        "rank",
        "comment_id",
        "confidence",
        "final_confidence",
        "guardrail_decision_source",
        "guardrail_rule_ids",
        "guardrail_categories",
        "review_priority_score",
        "review_hint",
        "text",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(selected, start=1):
            payload = {"rank": str(idx)}
            payload.update(row)
            writer.writerow(payload)

    print(f"[INFO] input={input_path}")
    print(f"[INFO] total_rows={len(rows)}")
    print(f"[INFO] overridden_from_actionable={total_overridden_from_actionable}")
    print(f"[INFO] skipped_by_confidence={skipped_by_confidence}")
    print(f"[INFO] skipped_by_source_filters={skipped_by_source_filters}")
    print(f"[INFO] candidate_pool={len(candidates)}")
    print(f"[INFO] wrote_rows={len(selected)}")
    print(f"[INFO] output={output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
