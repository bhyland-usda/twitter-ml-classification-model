#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

LABEL_ACTIONABLE = "actionable"
LABEL_NON_ACTIONABLE = "non_actionable"
ALLOWED_LABELS = {LABEL_ACTIONABLE, LABEL_NON_ACTIONABLE}
QUEUE_REQUIRED_COLUMNS = {
    "comment_id",
    "text",
    "parent_text",
    "reviewer_label",
    "reviewer_notes",
}
TRAIN_REQUIRED_COLUMNS = {"comment_id", "text", "parent_text", "label"}
OUTPUT_FIELDNAMES = ["comment_id", "text", "parent_text", "label"]


@dataclass(frozen=True)
class TrainingRow:
    comment_id: str
    text: str
    parent_text: str
    label: str

    def key(self) -> Tuple[str, str, str]:
        return (
            self.comment_id,
            normalize_for_key(self.parent_text),
            normalize_for_key(self.text),
        )

    def as_dict(self) -> Dict[str, str]:
        return {
            "comment_id": self.comment_id,
            "text": self.text,
            "parent_text": self.parent_text,
            "label": self.label,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a reviewed contextual relabel queue into clean training rows "
            "and optionally merge those reviewed labels into an existing binary "
            "training CSV."
        )
    )
    parser.add_argument(
        "--reviewed-queue",
        required=True,
        help=(
            "Path to reviewed queue CSV with columns including "
            "comment_id,text,parent_text,reviewer_label,reviewer_notes."
        ),
    )
    parser.add_argument(
        "--base-training-csv",
        default="",
        help=(
            "Optional existing training CSV with columns "
            "comment_id,text,parent_text,label. If provided, reviewed labels "
            "replace matching rows and new reviewed rows are appended."
        ),
    )
    parser.add_argument(
        "--output-training-csv",
        required=True,
        help=(
            "Output CSV path for merged clean training data with columns "
            "comment_id,text,parent_text,label."
        ),
    )
    parser.add_argument(
        "--output-reviewed-only-csv",
        default="",
        help=(
            "Optional output CSV path for reviewed rows only, converted into clean "
            "training schema."
        ),
    )
    parser.add_argument(
        "--allow-blank-reviewer-labels",
        action="store_true",
        help=(
            "If set, rows with blank reviewer_label are skipped instead of causing "
            "validation errors."
        ),
    )
    parser.add_argument(
        "--append-unmatched-reviewed-rows",
        action="store_true",
        help=(
            "If set, reviewed rows that do not already exist in the base training CSV "
            "are appended to the output. Leave unset for split-safe overlay-only merges."
        ),
    )
    return parser.parse_args()


def clean_cell(value: str | None) -> str:
    if value is None:
        return ""
    return value.replace("\ufeff", "").replace("\x00", "").strip()


def normalize_for_key(value: str) -> str:
    return " ".join(clean_cell(value).split()).lower()


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"[ERROR] CSV has no header row: {path}")
        fieldnames = [clean_cell(name) for name in reader.fieldnames]
        rows: List[Dict[str, str]] = []
        for raw in reader:
            rows.append({clean_cell(k): clean_cell(v) for k, v in raw.items()})
    return rows, fieldnames


def require_columns(path: Path, fieldnames: Iterable[str], required: set[str]) -> None:
    missing = sorted(required - set(fieldnames))
    if missing:
        raise SystemExit(f"[ERROR] CSV missing required columns {missing}: {path}")


def load_reviewed_rows(
    path: Path,
    allow_blank_reviewer_labels: bool,
) -> Tuple[List[TrainingRow], Dict[str, int]]:
    rows, fieldnames = read_csv_rows(path)
    require_columns(path, fieldnames, QUEUE_REQUIRED_COLUMNS)

    stats: Dict[str, int] = {
        "queue_rows_total": 0,
        "queue_rows_with_review_label": 0,
        "queue_rows_blank_review_label": 0,
        "queue_rows_invalid_review_label": 0,
        "queue_rows_invalid_payload": 0,
        "queue_rows_duplicate_review_key": 0,
        "queue_rows_kept": 0,
    }

    reviewed_by_key: Dict[Tuple[str, str, str], TrainingRow] = {}

    for line_number, row in enumerate(rows, start=2):
        stats["queue_rows_total"] += 1

        comment_id = clean_cell(row.get("comment_id"))
        text = clean_cell(row.get("text"))
        parent_text = clean_cell(row.get("parent_text"))
        reviewer_label = clean_cell(row.get("reviewer_label")).lower()

        if not reviewer_label:
            stats["queue_rows_blank_review_label"] += 1
            if allow_blank_reviewer_labels:
                continue
            raise SystemExit(
                f"[ERROR] line {line_number}: blank reviewer_label in {path}"
            )

        stats["queue_rows_with_review_label"] += 1

        if reviewer_label not in ALLOWED_LABELS:
            stats["queue_rows_invalid_review_label"] += 1
            raise SystemExit(
                f"[ERROR] line {line_number}: invalid reviewer_label "
                f"'{reviewer_label}' in {path}; expected one of {sorted(ALLOWED_LABELS)}"
            )

        if not comment_id or not text or not parent_text:
            stats["queue_rows_invalid_payload"] += 1
            raise SystemExit(
                f"[ERROR] line {line_number}: reviewed row missing required "
                f"comment_id/text/parent_text in {path}"
            )

        training_row = TrainingRow(
            comment_id=comment_id,
            text=text,
            parent_text=parent_text,
            label=reviewer_label,
        )
        key = training_row.key()
        if key in reviewed_by_key:
            stats["queue_rows_duplicate_review_key"] += 1
        reviewed_by_key[key] = training_row

    reviewed_rows = sorted(
        reviewed_by_key.values(),
        key=lambda row: (
            row.comment_id,
            normalize_for_key(row.parent_text),
            normalize_for_key(row.text),
        ),
    )
    stats["queue_rows_kept"] = len(reviewed_rows)
    return reviewed_rows, stats


def load_base_training_rows(path: Path) -> List[TrainingRow]:
    rows, fieldnames = read_csv_rows(path)
    require_columns(path, fieldnames, TRAIN_REQUIRED_COLUMNS)

    parsed_rows: List[TrainingRow] = []
    seen_keys: set[Tuple[str, str, str]] = set()

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
                f"[ERROR] line {line_number}: invalid base training label "
                f"'{label}' in {path}; expected one of {sorted(ALLOWED_LABELS)}"
            )

        parsed = TrainingRow(
            comment_id=comment_id,
            text=text,
            parent_text=parent_text,
            label=label,
        )
        key = parsed.key()
        if key in seen_keys:
            raise SystemExit(
                f"[ERROR] line {line_number}: duplicate base training key "
                f"(comment_id,parent_text,text) in {path}"
            )
        seen_keys.add(key)
        parsed_rows.append(parsed)

    if not parsed_rows:
        raise SystemExit(f"[ERROR] base training CSV has zero rows: {path}")

    return parsed_rows


def write_training_rows(path: Path, rows: Iterable[TrainingRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def merge_reviewed_into_base(
    base_rows: List[TrainingRow],
    reviewed_rows: List[TrainingRow],
    append_unmatched_reviewed_rows: bool,
) -> Tuple[List[TrainingRow], Dict[str, int]]:
    reviewed_by_key = {row.key(): row for row in reviewed_rows}

    merged_rows: List[TrainingRow] = []
    replaced_count = 0
    unchanged_count = 0

    for base_row in base_rows:
        key = base_row.key()
        reviewed_row = reviewed_by_key.pop(key, None)
        if reviewed_row is None:
            merged_rows.append(base_row)
            unchanged_count += 1
            continue

        merged_rows.append(reviewed_row)
        if reviewed_row.label != base_row.label:
            replaced_count += 1
        else:
            unchanged_count += 1

    if append_unmatched_reviewed_rows:
        appended_rows = sorted(
            reviewed_by_key.values(),
            key=lambda row: (
                row.comment_id,
                normalize_for_key(row.parent_text),
                normalize_for_key(row.text),
            ),
        )
        merged_rows.extend(appended_rows)
    else:
        appended_rows = []

    stats = {
        "base_rows_total": len(base_rows),
        "reviewed_rows_total": len(reviewed_rows),
        "merged_rows_total": len(merged_rows),
        "replaced_existing_rows": replaced_count,
        "unchanged_existing_rows": unchanged_count,
        "appended_new_rows": len(appended_rows),
    }
    return merged_rows, stats


def print_label_distribution(prefix: str, rows: Iterable[TrainingRow]) -> None:
    counts = Counter(row.label for row in rows)
    total = sum(counts.values())
    print(
        f"[INFO] {prefix}: rows={total} "
        f"actionable={counts.get(LABEL_ACTIONABLE, 0)} "
        f"non_actionable={counts.get(LABEL_NON_ACTIONABLE, 0)}"
    )


def main() -> int:
    args = parse_args()

    reviewed_queue_path = Path(args.reviewed_queue)
    base_training_path = (
        Path(args.base_training_csv) if args.base_training_csv else None
    )
    output_training_path = Path(args.output_training_csv)
    output_reviewed_only_path = (
        Path(args.output_reviewed_only_csv) if args.output_reviewed_only_csv else None
    )

    if not reviewed_queue_path.exists():
        raise SystemExit(
            f"[ERROR] reviewed queue file not found: {reviewed_queue_path}"
        )
    if base_training_path and not base_training_path.exists():
        raise SystemExit(f"[ERROR] base training file not found: {base_training_path}")

    reviewed_rows, review_stats = load_reviewed_rows(
        reviewed_queue_path,
        allow_blank_reviewer_labels=args.allow_blank_reviewer_labels,
    )

    if output_reviewed_only_path:
        write_training_rows(output_reviewed_only_path, reviewed_rows)

    if base_training_path:
        base_rows = load_base_training_rows(base_training_path)
        merged_rows, merge_stats = merge_reviewed_into_base(
            base_rows,
            reviewed_rows,
            append_unmatched_reviewed_rows=args.append_unmatched_reviewed_rows,
        )
    else:
        base_rows = []
        merged_rows = list(reviewed_rows)
        merge_stats = {
            "base_rows_total": 0,
            "reviewed_rows_total": len(reviewed_rows),
            "merged_rows_total": len(merged_rows),
            "replaced_existing_rows": 0,
            "unchanged_existing_rows": 0,
            "appended_new_rows": len(merged_rows),
        }

    write_training_rows(output_training_path, merged_rows)

    print(f"[INFO] reviewed_queue={reviewed_queue_path}")
    print(f"[INFO] output_training_csv={output_training_path}")
    if base_training_path:
        print(f"[INFO] base_training_csv={base_training_path}")
    if output_reviewed_only_path:
        print(f"[INFO] output_reviewed_only_csv={output_reviewed_only_path}")

    for key in (
        "queue_rows_total",
        "queue_rows_with_review_label",
        "queue_rows_blank_review_label",
        "queue_rows_duplicate_review_key",
        "queue_rows_kept",
    ):
        print(f"[INFO] {key}={review_stats[key]}")

    for key in (
        "base_rows_total",
        "reviewed_rows_total",
        "merged_rows_total",
        "replaced_existing_rows",
        "unchanged_existing_rows",
        "appended_new_rows",
    ):
        print(f"[INFO] {key}={merge_stats[key]}")

    print_label_distribution("reviewed_only", reviewed_rows)
    if base_rows:
        print_label_distribution("base_training", base_rows)
    print_label_distribution("merged_training", merged_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
