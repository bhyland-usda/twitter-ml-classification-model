#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

ALLOWED_LABELS = (
    "moderation_risk",
    "question_or_request",
    "actionable_feedback",
    "non_actionable_noise",
)

REQUIRED_COLUMNS = ("comment_id", "text", "parent_text", "label")


@dataclass(frozen=True)
class SplitConfig:
    train: float
    val: float
    test: float

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if not (abs(total - 1.0) < 1e-9):
            raise ValueError(f"Split fractions must sum to 1.0, got {total:.12f}")
        if min(self.train, self.val, self.test) <= 0:
            raise ValueError("All split fractions must be > 0")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate and quality-check labeled tweet/comment CSV for classifier training."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to labeled CSV with columns: comment_id,text,parent_text,label",
    )
    p.add_argument(
        "--write-clean",
        default="",
        help="Optional output path to write normalized CSV (trimmed fields, normalized label casing).",
    )
    p.add_argument(
        "--min-class-count",
        type=int,
        default=60,
        help="Warn if any class has fewer than this many examples (default: 60).",
    )
    p.add_argument(
        "--max-majority-ratio",
        type=float,
        default=0.70,
        help="Warn if largest class exceeds this fraction of dataset (default: 0.70).",
    )
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    return p.parse_args()


def _clean_cell(value: str) -> str:
    if value is None:
        return ""
    return value.replace("\ufeff", "").replace("\x00", "").strip()


def load_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        header = [_clean_cell(h) for h in reader.fieldnames]
        rows: List[Dict[str, str]] = []
        for raw in reader:
            row = {k: _clean_cell(v) for k, v in raw.items()}
            rows.append(row)
        return rows, header


def normalize_row(row: Dict[str, str]) -> Dict[str, str]:
    out = dict(row)
    out["comment_id"] = _clean_cell(out.get("comment_id", ""))
    out["text"] = _clean_cell(out.get("text", ""))
    out["parent_text"] = _clean_cell(out.get("parent_text", ""))
    out["label"] = _clean_cell(out.get("label", "")).lower()
    return out


def check_required_columns(header: List[str]) -> List[str]:
    return [c for c in REQUIRED_COLUMNS if c not in header]


def split_capacity_ok(class_count: int, frac: float) -> bool:
    return class_count * frac >= 1.0


def main() -> int:
    args = parse_args()
    split = SplitConfig(args.train_frac, args.val_frac, args.test_frac)
    try:
        split.validate()
    except ValueError as e:
        print(f"[ERROR] invalid split config: {e}")
        return 2

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] input file not found: {input_path}")
        return 2

    try:
        raw_rows, header = load_rows(input_path)
    except Exception as e:
        print(f"[ERROR] failed to read CSV: {e}")
        return 2

    missing_cols = check_required_columns(header)
    if missing_cols:
        print(f"[ERROR] missing required columns: {missing_cols}")
        print(f"[INFO] detected columns: {header}")
        return 2

    normalized_rows: List[Dict[str, str]] = []
    errors: List[str] = []
    warnings: List[str] = []

    seen_ids: Dict[str, int] = {}
    dup_ids: Dict[str, List[int]] = defaultdict(list)
    label_counts = Counter()

    for i, raw in enumerate(raw_rows, start=2):  # header on line 1
        row = normalize_row(raw)
        cid = row["comment_id"]
        text = row["text"]
        parent_text = row["parent_text"]
        label = row["label"]

        if not cid:
            errors.append(f"line {i}: empty comment_id")
        if not text:
            errors.append(f"line {i}: empty text")
        if not parent_text:
            errors.append(f"line {i}: empty parent_text")
        if not label:
            errors.append(f"line {i}: empty label")
        elif label not in ALLOWED_LABELS:
            errors.append(f"line {i}: invalid label '{label}'")

        if cid:
            if cid in seen_ids:
                dup_ids[cid].append(i)
            else:
                seen_ids[cid] = i

        if label in ALLOWED_LABELS:
            label_counts[label] += 1

        normalized_rows.append(row)

    if dup_ids:
        total_dup_rows = sum(len(v) for v in dup_ids.values())
        errors.append(
            f"duplicate comment_id detected: {len(dup_ids)} duplicate IDs affecting {total_dup_rows} rows"
        )

    total_rows = len(normalized_rows)
    print(f"[INFO] rows={total_rows}")
    print(f"[INFO] allowed_labels={list(ALLOWED_LABELS)}")
    print(
        f"[INFO] split=train:{split.train:.2f}, val:{split.val:.2f}, test:{split.test:.2f}"
    )

    if total_rows == 0:
        errors.append("CSV contains zero data rows")

    print("[INFO] class_distribution:")
    for lbl in ALLOWED_LABELS:
        c = label_counts.get(lbl, 0)
        pct = (c / total_rows * 100.0) if total_rows else 0.0
        print(f"  - {lbl}: {c} ({pct:.2f}%)")

    if total_rows:
        majority = max(label_counts.values()) if label_counts else 0
        majority_ratio = majority / total_rows
        if majority_ratio > args.max_majority_ratio:
            warnings.append(
                f"class imbalance: majority class ratio {majority_ratio:.3f} > {args.max_majority_ratio:.3f}"
            )

    for lbl in ALLOWED_LABELS:
        c = label_counts.get(lbl, 0)
        if c < args.min_class_count:
            warnings.append(
                f"low class count: {lbl} has {c}, below recommended minimum {args.min_class_count}"
            )
        for name, frac in (
            ("train", split.train),
            ("val", split.val),
            ("test", split.test),
        ):
            if c > 0 and not split_capacity_ok(c, frac):
                warnings.append(
                    f"stratified split risk: class '{lbl}' has {c}, insufficient for {name} fraction {frac:.2f}"
                )

    if warnings:
        print("[WARN] warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("[INFO] no warnings")

    if errors:
        print("[ERROR] validation errors:")
        for e in errors[:200]:
            print(f"  - {e}")
        if len(errors) > 200:
            print(f"  - ... and {len(errors) - 200} more")
        print("[FAIL] dataset failed validation")
        return 1

    if args.write_clean:
        out_path = Path(args.write_clean)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(REQUIRED_COLUMNS))
            writer.writeheader()
            for row in normalized_rows:
                writer.writerow(
                    {
                        "comment_id": row["comment_id"],
                        "text": row["text"],
                        "parent_text": row["parent_text"],
                        "label": row["label"],
                    }
                )
        print(f"[INFO] wrote cleaned CSV: {out_path}")

    print("[PASS] dataset passed validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
