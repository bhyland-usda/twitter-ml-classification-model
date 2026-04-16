#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

LABEL_ACTIONABLE = "actionable"
LABEL_NON_ACTIONABLE = "non_actionable"
ALLOWED_LABELS = {LABEL_ACTIONABLE, LABEL_NON_ACTIONABLE}
REVIEW_BUCKET_SURVIVED = "survived_actionable"
REVIEW_BUCKET_DOWNGRADED = "downgraded_actionable"

QUEUE_REQUIRED_COLUMNS = {
    "comment_id",
    "text",
    "parent_text",
    "predicted_label",
    "final_label",
    "confidence",
    "final_confidence",
    "review_bucket",
    "reviewer_label",
    "reviewer_notes",
}
BASE_REQUIRED_COLUMNS = {"comment_id", "text", "parent_text", "label"}

OUTPUT_FIELDNAMES = [
    "rank",
    "review_priority_score",
    "disagreement_direction",
    "label_source",
    "comment_id",
    "source_label",
    "reviewer_label",
    "predicted_label",
    "final_label",
    "review_bucket",
    "confidence",
    "final_confidence",
    "guardrail_decision_source",
    "guardrail_rule_ids",
    "guardrail_categories",
    "review_hint",
    "suggested_review_action",
    "reviewer_notes",
    "parent_text",
    "text",
]


@dataclass(frozen=True)
class QueueRow:
    comment_id: str
    text: str
    parent_text: str
    predicted_label: str
    final_label: str
    confidence: float
    final_confidence: float
    review_bucket: str
    reviewer_label: str
    reviewer_notes: str
    raw: Dict[str, str]

    def key(self) -> Tuple[str, str, str]:
        return (
            self.comment_id,
            normalize_for_key(self.parent_text),
            normalize_for_key(self.text),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a disagreement queue from reviewed contextual labels by comparing "
            "reviewer_label values against an existing binary training label source."
        )
    )
    parser.add_argument(
        "--reviewed-queue",
        required=True,
        help=(
            "Path to reviewed contextual queue CSV. Expected columns include "
            "comment_id,text,parent_text,predicted_label,final_label,reviewer_label."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path for the full disagreement queue.",
    )
    parser.add_argument(
        "--base-training-csv",
        default="",
        help=(
            "Optional existing binary training CSV with columns "
            "comment_id,text,parent_text,label. When provided, this is the primary "
            "comparison source."
        ),
    )
    parser.add_argument(
        "--high-priority-output",
        default="",
        help=(
            "Optional output CSV path for a smaller high-priority disagreement subset."
        ),
    )
    parser.add_argument(
        "--high-priority-max-rows",
        type=int,
        default=50,
        help="Maximum number of rows for --high-priority-output (default: 50).",
    )
    parser.add_argument(
        "--allow-blank-reviewer-labels",
        action="store_true",
        help=(
            "Skip rows with blank reviewer_label instead of treating them as an error."
        ),
    )
    return parser.parse_args()


def clean_cell(value: str | None) -> str:
    if value is None:
        return ""
    return value.replace("\ufeff", "").replace("\x00", "").strip()


def normalize_for_key(value: str) -> str:
    return " ".join(clean_cell(value).split()).lower()


def parse_float(value: str | None, default: float = 0.0) -> float:
    try:
        parsed = float(clean_cell(value))
    except Exception:
        return default
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"[ERROR] CSV has no header row: {path}")
        fieldnames = [clean_cell(name) for name in reader.fieldnames]
        rows: List[Dict[str, str]] = []
        for raw_row in reader:
            rows.append({clean_cell(k): clean_cell(v) for k, v in raw_row.items()})
    return rows, fieldnames


def require_columns(path: Path, fieldnames: Iterable[str], required: set[str]) -> None:
    missing = sorted(required - set(fieldnames))
    if missing:
        raise SystemExit(f"[ERROR] CSV missing required columns {missing}: {path}")


def load_queue_rows(
    path: Path,
    allow_blank_reviewer_labels: bool,
) -> Tuple[List[QueueRow], Dict[str, int]]:
    rows, fieldnames = read_csv_rows(path)
    require_columns(path, fieldnames, QUEUE_REQUIRED_COLUMNS)
    if not rows:
        raise SystemExit(f"[ERROR] reviewed queue has zero rows: {path}")

    stats: Dict[str, int] = {
        "queue_rows_total": 0,
        "queue_rows_blank_reviewer_label": 0,
        "queue_rows_invalid_reviewer_label": 0,
        "queue_rows_duplicate_keys": 0,
        "queue_rows_kept": 0,
    }

    deduped: Dict[Tuple[str, str, str], QueueRow] = {}

    for line_number, row in enumerate(rows, start=2):
        stats["queue_rows_total"] += 1

        comment_id = clean_cell(row.get("comment_id"))
        text = clean_cell(row.get("text"))
        parent_text = clean_cell(row.get("parent_text"))
        reviewer_label = clean_cell(row.get("reviewer_label")).lower()

        if not comment_id or not text or not parent_text:
            raise SystemExit(
                f"[ERROR] line {line_number}: reviewed queue row missing required "
                f"comment_id/text/parent_text in {path}"
            )

        if not reviewer_label:
            stats["queue_rows_blank_reviewer_label"] += 1
            if allow_blank_reviewer_labels:
                continue
            raise SystemExit(
                f"[ERROR] line {line_number}: blank reviewer_label in {path}"
            )

        if reviewer_label not in ALLOWED_LABELS:
            stats["queue_rows_invalid_reviewer_label"] += 1
            raise SystemExit(
                f"[ERROR] line {line_number}: invalid reviewer_label "
                f"'{reviewer_label}' in {path}; expected one of {sorted(ALLOWED_LABELS)}"
            )

        queue_row = QueueRow(
            comment_id=comment_id,
            text=text,
            parent_text=parent_text,
            predicted_label=clean_cell(row.get("predicted_label")).lower(),
            final_label=clean_cell(row.get("final_label")).lower(),
            confidence=parse_float(row.get("confidence"), 0.0),
            final_confidence=parse_float(row.get("final_confidence"), 0.0),
            review_bucket=clean_cell(row.get("review_bucket")),
            reviewer_label=reviewer_label,
            reviewer_notes=clean_cell(row.get("reviewer_notes")),
            raw=row,
        )

        key = queue_row.key()
        if key in deduped:
            stats["queue_rows_duplicate_keys"] += 1
        deduped[key] = queue_row

    queue_rows = sorted(
        deduped.values(),
        key=lambda row: (
            row.comment_id,
            normalize_for_key(row.parent_text),
            normalize_for_key(row.text),
        ),
    )
    stats["queue_rows_kept"] = len(queue_rows)
    return queue_rows, stats


def load_base_labels(path: Path) -> Dict[Tuple[str, str, str], str]:
    rows, fieldnames = read_csv_rows(path)
    require_columns(path, fieldnames, BASE_REQUIRED_COLUMNS)
    if not rows:
        raise SystemExit(f"[ERROR] base training CSV has zero rows: {path}")

    label_by_key: Dict[Tuple[str, str, str], str] = {}

    for line_number, row in enumerate(rows, start=2):
        comment_id = clean_cell(row.get("comment_id"))
        text = clean_cell(row.get("text"))
        parent_text = clean_cell(row.get("parent_text"))
        label = clean_cell(row.get("label")).lower()

        if not comment_id or not text or not parent_text or not label:
            raise SystemExit(
                f"[ERROR] line {line_number}: base training row missing required "
                f"comment_id/text/parent_text/label in {path}"
            )
        if label not in ALLOWED_LABELS:
            raise SystemExit(
                f"[ERROR] line {line_number}: invalid base training label '{label}' "
                f"in {path}; expected one of {sorted(ALLOWED_LABELS)}"
            )

        key = (
            comment_id,
            normalize_for_key(parent_text),
            normalize_for_key(text),
        )
        if key in label_by_key:
            raise SystemExit(
                f"[ERROR] line {line_number}: duplicate base training key "
                f"(comment_id,parent_text,text) in {path}"
            )
        label_by_key[key] = label

    return label_by_key


def resolve_source_label(
    queue_row: QueueRow,
    base_label_by_key: Dict[Tuple[str, str, str], str] | None,
) -> Tuple[str, str]:
    if base_label_by_key is not None:
        base_label = base_label_by_key.get(queue_row.key())
        if base_label:
            return base_label, "base_training"

    if queue_row.final_label in ALLOWED_LABELS:
        return queue_row.final_label, "queue_final_label"

    if queue_row.predicted_label in ALLOWED_LABELS:
        return queue_row.predicted_label, "queue_predicted_label"

    return "", "unresolved"


def compute_priority_score(
    queue_row: QueueRow,
    source_label: str,
) -> float:
    score = queue_row.confidence * 100.0

    if (
        source_label == LABEL_NON_ACTIONABLE
        and queue_row.reviewer_label == LABEL_ACTIONABLE
    ):
        score += 20.0
    if (
        source_label == LABEL_ACTIONABLE
        and queue_row.reviewer_label == LABEL_NON_ACTIONABLE
    ):
        score += 16.0

    if queue_row.review_bucket == REVIEW_BUCKET_SURVIVED:
        score += 8.0
    elif queue_row.review_bucket == REVIEW_BUCKET_DOWNGRADED:
        score += 4.0

    decision_sources = clean_cell(queue_row.raw.get("guardrail_decision_source"))
    categories = clean_cell(queue_row.raw.get("guardrail_categories"))
    text_lower = queue_row.text.lower()

    if "missing_concrete_action" in decision_sources:
        score += 3.0
    if "question_without_concrete_action" in decision_sources:
        score += 2.5
    if "missing_action_cue" in decision_sources:
        score += 1.5
    if "confidence_threshold" in decision_sources:
        score += 1.0

    if "please" in text_lower:
        score += 2.0
    if "mcool" in text_lower or "mandatory" in text_lower:
        score += 3.0
    if "contact me" in text_lower or "contact" in text_lower:
        score += 2.5
    if "fertilizer" in text_lower or "diesel" in text_lower:
        score += 2.0
    if "school lunch" in text_lower:
        score += 2.0
    if "wild horses" in text_lower:
        score += 3.0

    if categories:
        lowered_categories = categories.lower()
        if any(
            category in lowered_categories
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
            score -= 6.0

    return round(score, 6)


def disagreement_direction(source_label: str, reviewer_label: str) -> str:
    if source_label == LABEL_NON_ACTIONABLE and reviewer_label == LABEL_ACTIONABLE:
        return "upgraded_to_actionable"
    if source_label == LABEL_ACTIONABLE and reviewer_label == LABEL_NON_ACTIONABLE:
        return "downgraded_to_non_actionable"
    return "other_disagreement"


def build_disagreement_rows(
    queue_rows: Sequence[QueueRow],
    base_label_by_key: Dict[Tuple[str, str, str], str] | None,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    stats: Dict[str, int] = {
        "rows_with_resolved_source": 0,
        "rows_with_unresolved_source": 0,
        "rows_unchanged_vs_source": 0,
        "rows_disagreement": 0,
        "upgraded_to_actionable": 0,
        "downgraded_to_non_actionable": 0,
    }

    disagreements: List[Dict[str, str]] = []

    for queue_row in queue_rows:
        source_label, label_source = resolve_source_label(queue_row, base_label_by_key)
        if not source_label:
            stats["rows_with_unresolved_source"] += 1
            continue

        stats["rows_with_resolved_source"] += 1

        if queue_row.reviewer_label == source_label:
            stats["rows_unchanged_vs_source"] += 1
            continue

        direction = disagreement_direction(source_label, queue_row.reviewer_label)
        stats["rows_disagreement"] += 1
        if direction in stats:
            stats[direction] += 1

        disagreements.append(
            {
                "rank": "",
                "review_priority_score": f"{compute_priority_score(queue_row, source_label):.6f}",
                "disagreement_direction": direction,
                "label_source": label_source,
                "comment_id": queue_row.comment_id,
                "source_label": source_label,
                "reviewer_label": queue_row.reviewer_label,
                "predicted_label": queue_row.predicted_label,
                "final_label": queue_row.final_label,
                "review_bucket": queue_row.review_bucket,
                "confidence": f"{queue_row.confidence:.6f}",
                "final_confidence": f"{queue_row.final_confidence:.6f}",
                "guardrail_decision_source": clean_cell(
                    queue_row.raw.get("guardrail_decision_source")
                ),
                "guardrail_rule_ids": clean_cell(
                    queue_row.raw.get("guardrail_rule_ids")
                ),
                "guardrail_categories": clean_cell(
                    queue_row.raw.get("guardrail_categories")
                ),
                "review_hint": clean_cell(queue_row.raw.get("review_hint")),
                "suggested_review_action": clean_cell(
                    queue_row.raw.get("suggested_review_action")
                ),
                "reviewer_notes": queue_row.reviewer_notes,
                "parent_text": queue_row.parent_text,
                "text": queue_row.text,
            }
        )

    disagreements.sort(
        key=lambda row: (
            float(row["review_priority_score"]),
            row["disagreement_direction"] == "upgraded_to_actionable",
            row["comment_id"],
        ),
        reverse=True,
    )

    for rank, row in enumerate(disagreements, start=1):
        row["rank"] = str(rank)

    return disagreements, stats


def write_csv(
    path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def select_high_priority_rows(
    disagreement_rows: Sequence[Dict[str, str]],
    max_rows: int,
) -> List[Dict[str, str]]:
    if max_rows <= 0:
        raise SystemExit("[ERROR] --high-priority-max-rows must be > 0")

    upgraded = [
        row
        for row in disagreement_rows
        if row["disagreement_direction"] == "upgraded_to_actionable"
    ]
    downgraded = [
        row
        for row in disagreement_rows
        if row["disagreement_direction"] == "downgraded_to_non_actionable"
    ]

    prioritized = upgraded + downgraded
    prioritized.sort(
        key=lambda row: (float(row["review_priority_score"]), row["comment_id"]),
        reverse=True,
    )

    selected = prioritized[:max_rows]
    for rank, row in enumerate(selected, start=1):
        row["rank"] = str(rank)

    return selected


def main() -> int:
    args = parse_args()

    reviewed_queue_path = Path(args.reviewed_queue)
    output_path = Path(args.output)
    base_training_path = (
        Path(args.base_training_csv) if args.base_training_csv else None
    )
    high_priority_output_path = (
        Path(args.high_priority_output) if args.high_priority_output else None
    )

    if not reviewed_queue_path.exists():
        raise SystemExit(
            f"[ERROR] reviewed queue file not found: {reviewed_queue_path}"
        )
    if base_training_path and not base_training_path.exists():
        raise SystemExit(f"[ERROR] base training file not found: {base_training_path}")

    queue_rows, queue_stats = load_queue_rows(
        reviewed_queue_path,
        allow_blank_reviewer_labels=args.allow_blank_reviewer_labels,
    )
    base_label_by_key = (
        load_base_labels(base_training_path) if base_training_path else None
    )

    disagreement_rows, disagreement_stats = build_disagreement_rows(
        queue_rows,
        base_label_by_key,
    )
    write_csv(output_path, disagreement_rows, OUTPUT_FIELDNAMES)

    if high_priority_output_path:
        high_priority_rows = select_high_priority_rows(
            disagreement_rows,
            max_rows=args.high_priority_max_rows,
        )
        write_csv(high_priority_output_path, high_priority_rows, OUTPUT_FIELDNAMES)
    else:
        high_priority_rows = []

    print(f"[INFO] reviewed_queue={reviewed_queue_path}")
    print(f"[INFO] output={output_path}")
    if base_training_path:
        print(f"[INFO] base_training_csv={base_training_path}")
    if high_priority_output_path:
        print(f"[INFO] high_priority_output={high_priority_output_path}")
        print(f"[INFO] high_priority_rows={len(high_priority_rows)}")

    for key in (
        "queue_rows_total",
        "queue_rows_blank_reviewer_label",
        "queue_rows_duplicate_keys",
        "queue_rows_kept",
    ):
        print(f"[INFO] {key}={queue_stats[key]}")

    for key in (
        "rows_with_resolved_source",
        "rows_with_unresolved_source",
        "rows_unchanged_vs_source",
        "rows_disagreement",
        "upgraded_to_actionable",
        "downgraded_to_non_actionable",
    ):
        print(f"[INFO] {key}={disagreement_stats[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
