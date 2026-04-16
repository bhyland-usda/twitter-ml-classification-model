#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

REQUIRED_COLUMNS = {
    "comment_id",
    "text",
    "parent_text",
    "predicted_label",
    "confidence",
    "guardrail_decision_source",
    "guardrail_rule_ids",
    "guardrail_categories",
    "final_label",
    "final_confidence",
}

DEFAULT_INCLUDE_SOURCES = (
    "missing_concrete_action",
    "question_without_concrete_action",
    "missing_action_cue",
    "confidence_threshold",
)

DEFAULT_EXCLUDE_SOURCES = (
    "safety_veto",
    "scope_veto",
    "civility_gate",
    "rule_match",
)

DEFAULT_EXCLUDE_CATEGORIES = (
    "abuse",
    "violence",
    "xenophobia",
    "conspiracy",
    "scam_spam",
    "health_misinformation",
    "off_topic_geopolitics",
    "out_of_scope",
)


@dataclass(frozen=True)
class Candidate:
    bucket: str
    priority_score: float
    comment_id: str
    predicted_label: str
    final_label: str
    confidence: float
    final_confidence: float
    guardrail_decision_source: str
    guardrail_rule_ids: str
    guardrail_categories: str
    review_hint: str
    suggested_review_action: str
    parent_text: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a contextual manual relabel queue from guardrailed predictions. "
            "The queue includes final actionable survivors plus high-value downgraded "
            "actionable candidates, while preserving parent tweet context."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to guardrailed predictions CSV with columns including "
            "comment_id,text,parent_text,predicted_label,confidence,...,final_label."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path for the contextual relabel queue.",
    )
    parser.add_argument(
        "--max-survived-actionable",
        type=int,
        default=200,
        help="Maximum final actionable survivors to include (default: 200).",
    )
    parser.add_argument(
        "--max-downgraded",
        type=int,
        default=300,
        help="Maximum downgraded actionable rows to include (default: 300).",
    )
    parser.add_argument(
        "--min-downgraded-confidence",
        type=float,
        default=0.85,
        help="Minimum model confidence for downgraded actionable review rows (default: 0.85).",
    )
    parser.add_argument(
        "--include-sources",
        default=",".join(DEFAULT_INCLUDE_SOURCES),
        help=(
            "Comma-separated decision sources that qualify downgraded rows for review "
            f"(default: {','.join(DEFAULT_INCLUDE_SOURCES)})."
        ),
    )
    parser.add_argument(
        "--exclude-sources",
        default=",".join(DEFAULT_EXCLUDE_SOURCES),
        help=(
            "Comma-separated decision sources to exclude from downgraded review rows "
            f"(default: {','.join(DEFAULT_EXCLUDE_SOURCES)})."
        ),
    )
    parser.add_argument(
        "--exclude-categories",
        default=",".join(DEFAULT_EXCLUDE_CATEGORIES),
        help=(
            "Comma-separated guardrail categories to exclude from downgraded review rows "
            f"(default: {','.join(DEFAULT_EXCLUDE_CATEGORIES)})."
        ),
    )
    return parser.parse_args()


def parse_csv_set(raw: str) -> Set[str]:
    return {part.strip() for part in (raw or "").split(",") if part.strip()}


def parse_pipe_set(raw: str) -> Set[str]:
    return {part.strip() for part in (raw or "").split("|") if part.strip()}


def parse_float(raw: str | None, default: float = 0.0) -> float:
    try:
        value = float((raw or "").strip())
    except Exception:
        return default
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def normalize_text(value: str) -> str:
    return " ".join((value or "").split())


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"[ERROR] input CSV has zero rows: {path}")
    columns = set(rows[0].keys())
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise SystemExit(
            f"[ERROR] input CSV missing required columns {sorted(missing)}: {path}"
        )
    return rows


def make_survived_hint(text: str) -> str:
    lowered = text.lower()
    if "please" in lowered:
        return "Final actionable survivor with explicit request language; confirm it should remain actionable."
    if "?" in text:
        return "Final actionable survivor phrased as a question; verify operational relevance with full context."
    return "Final actionable survivor; confirm context makes the requested action relevant and in scope."


def make_downgraded_hint(
    sources: Set[str],
    confidence: float,
) -> str:
    if "question_without_concrete_action" in sources:
        return (
            "Downgraded high-confidence question/request. Review whether the parent context "
            "makes the ask operationally actionable despite missing concrete action syntax."
        )
    if "missing_concrete_action" in sources:
        return (
            "Downgraded for missing concrete action. Review whether parent context makes the "
            "requested action sufficiently specific."
        )
    if "missing_action_cue" in sources:
        return (
            "Downgraded for missing action cue. Review whether context implies a real request "
            "or policy ask even without explicit cue words."
        )
    if "confidence_threshold" in sources and confidence >= 0.95:
        return (
            "Very high-confidence actionable prediction suppressed by threshold interaction. "
            "Review whether this should become an affirmative contextual positive."
        )
    return (
        "Downgraded actionable candidate. Review whether parent tweet context changes the "
        "actionability judgment."
    )


def score_survived(row: Dict[str, str]) -> float:
    score = parse_float(row.get("final_confidence"), 0.0) * 100.0
    text = (row.get("text") or "").lower()
    if "please" in text:
        score += 4.0
    if "?" in (row.get("text") or ""):
        score += 2.0
    return score


def score_downgraded(
    row: Dict[str, str],
    sources: Set[str],
) -> float:
    score = parse_float(row.get("confidence"), 0.0) * 100.0
    text = (row.get("text") or "").lower()

    if "question_without_concrete_action" in sources:
        score += 8.0
    if "missing_concrete_action" in sources:
        score += 6.0
    if "missing_action_cue" in sources:
        score += 4.0
    if "confidence_threshold" in sources:
        score += 2.0

    if "please" in text:
        score += 3.0
    if "?" in (row.get("text") or ""):
        score += 2.0

    length_bonus = min(len(normalize_text(row.get("text", ""))) / 80.0, 3.0)
    score += length_bonus

    return score


def candidate_sort_key(candidate: Candidate) -> Tuple[float, float, str]:
    return (
        candidate.priority_score,
        candidate.confidence,
        candidate.comment_id,
    )


def dedupe_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        (row.get("comment_id") or "").strip(),
        normalize_text(row.get("parent_text") or "").lower(),
        normalize_text(row.get("text") or "").lower(),
    )


def select_survived_actionables(
    rows: Sequence[Dict[str, str]],
    max_rows: int,
) -> List[Candidate]:
    candidates: List[Candidate] = []
    seen: Set[Tuple[str, str, str]] = set()

    for row in rows:
        if (row.get("final_label") or "").strip() != "actionable":
            continue
        key = dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)

        text = normalize_text(row.get("text") or "")
        parent_text = normalize_text(row.get("parent_text") or "")

        candidates.append(
            Candidate(
                bucket="survived_actionable",
                priority_score=score_survived(row),
                comment_id=(row.get("comment_id") or "").strip(),
                predicted_label=(row.get("predicted_label") or "").strip(),
                final_label=(row.get("final_label") or "").strip(),
                confidence=parse_float(row.get("confidence"), 0.0),
                final_confidence=parse_float(row.get("final_confidence"), 0.0),
                guardrail_decision_source=(
                    row.get("guardrail_decision_source") or ""
                ).strip(),
                guardrail_rule_ids=(row.get("guardrail_rule_ids") or "").strip(),
                guardrail_categories=(row.get("guardrail_categories") or "").strip(),
                review_hint=make_survived_hint(text),
                suggested_review_action=(
                    "Confirm final actionable label or downgrade if the context is not actually actionable."
                ),
                parent_text=parent_text,
                text=text,
            )
        )

    candidates.sort(key=candidate_sort_key, reverse=True)
    return candidates[:max_rows]


def select_downgraded_actionables(
    rows: Sequence[Dict[str, str]],
    max_rows: int,
    min_confidence: float,
    include_sources: Set[str],
    exclude_sources: Set[str],
    exclude_categories: Set[str],
) -> List[Candidate]:
    candidates: List[Candidate] = []
    seen: Set[Tuple[str, str, str]] = set()

    for row in rows:
        predicted_label = (row.get("predicted_label") or "").strip()
        final_label = (row.get("final_label") or "").strip()

        if predicted_label != "actionable" or final_label != "non_actionable":
            continue

        confidence = parse_float(row.get("confidence"), 0.0)
        if confidence < min_confidence:
            continue

        sources = parse_pipe_set(row.get("guardrail_decision_source") or "")
        categories = parse_pipe_set(row.get("guardrail_categories") or "")

        if include_sources and not (sources & include_sources):
            continue
        if exclude_sources and (sources & exclude_sources):
            continue
        if exclude_categories and (categories & exclude_categories):
            continue

        parent_text = normalize_text(row.get("parent_text") or "")
        text = normalize_text(row.get("text") or "")
        if not parent_text or not text:
            continue

        key = dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)

        candidates.append(
            Candidate(
                bucket="downgraded_actionable",
                priority_score=score_downgraded(row, sources),
                comment_id=(row.get("comment_id") or "").strip(),
                predicted_label=predicted_label,
                final_label=final_label,
                confidence=confidence,
                final_confidence=parse_float(row.get("final_confidence"), 0.0),
                guardrail_decision_source="|".join(sorted(sources)),
                guardrail_rule_ids=(row.get("guardrail_rule_ids") or "").strip(),
                guardrail_categories="|".join(sorted(categories)),
                review_hint=make_downgraded_hint(sources, confidence),
                suggested_review_action=(
                    "Review with parent context. If context makes this a real operational ask, "
                    "change reviewer_label to actionable."
                ),
                parent_text=parent_text,
                text=text,
            )
        )

    candidates.sort(key=candidate_sort_key, reverse=True)
    return candidates[:max_rows]


def write_queue(path: Path, rows: Sequence[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "review_bucket": row.bucket,
                    "review_priority_score": f"{row.priority_score:.3f}",
                    "comment_id": row.comment_id,
                    "predicted_label": row.predicted_label,
                    "final_label": row.final_label,
                    "confidence": f"{row.confidence:.6f}",
                    "final_confidence": f"{row.final_confidence:.6f}",
                    "guardrail_decision_source": row.guardrail_decision_source,
                    "guardrail_rule_ids": row.guardrail_rule_ids,
                    "guardrail_categories": row.guardrail_categories,
                    "review_hint": row.review_hint,
                    "suggested_review_action": row.suggested_review_action,
                    "parent_text": row.parent_text,
                    "text": row.text,
                    "reviewer_label": "",
                    "reviewer_notes": "",
                }
            )


def main() -> int:
    args = parse_args()

    if args.max_survived_actionable < 0:
        raise SystemExit("[ERROR] --max-survived-actionable must be >= 0")
    if args.max_downgraded < 0:
        raise SystemExit("[ERROR] --max-downgraded must be >= 0")
    if not (0.0 <= args.min_downgraded_confidence <= 1.0):
        raise SystemExit("[ERROR] --min-downgraded-confidence must be in [0,1]")

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"[ERROR] missing input CSV: {input_path}")

    include_sources = parse_csv_set(args.include_sources)
    exclude_sources = parse_csv_set(args.exclude_sources)
    exclude_categories = parse_csv_set(args.exclude_categories)

    rows = load_rows(input_path)

    survived = select_survived_actionables(
        rows,
        max_rows=args.max_survived_actionable,
    )
    downgraded = select_downgraded_actionables(
        rows,
        max_rows=args.max_downgraded,
        min_confidence=args.min_downgraded_confidence,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
        exclude_categories=exclude_categories,
    )

    final_rows = list(survived) + list(downgraded)
    write_queue(output_path, final_rows)

    print(f"[INFO] input={input_path}")
    print(f"[INFO] total_rows={len(rows)}")
    print(f"[INFO] survived_actionable_selected={len(survived)}")
    print(f"[INFO] downgraded_actionable_selected={len(downgraded)}")
    print(f"[INFO] queue_rows_written={len(final_rows)}")
    print(f"[INFO] include_sources={sorted(include_sources)}")
    print(f"[INFO] exclude_sources={sorted(exclude_sources)}")
    print(f"[INFO] exclude_categories={sorted(exclude_categories)}")
    print(f"[INFO] output={output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
