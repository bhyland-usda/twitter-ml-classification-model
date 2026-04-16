#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Regression check for actionable-set stability.\n"
            "Compares predicted actionable IDs against a known-good actionable CSV."
        )
    )
    p.add_argument(
        "--predictions-csv",
        required=True,
        help="CSV containing model predictions (must include comment_id and label column).",
    )
    p.add_argument(
        "--known-actionable-csv",
        required=True,
        help="CSV containing known-good actionable rows (must include comment_id).",
    )
    p.add_argument(
        "--label-column",
        default="predicted_label",
        help="Label column in predictions CSV (default: predicted_label).",
    )
    p.add_argument(
        "--actionable-label",
        default="actionable",
        help="Value considered actionable in label column (default: actionable).",
    )
    p.add_argument(
        "--comment-id-column",
        default="comment_id",
        help="Comment ID column name in both CSVs (default: comment_id).",
    )
    p.add_argument(
        "--output-json",
        default="",
        help="Optional path to write JSON summary.",
    )
    p.add_argument(
        "--require-exact-match",
        action="store_true",
        default=True,
        help="Require exact set equality (default: true).",
    )
    p.add_argument(
        "--allow-subset-match",
        action="store_true",
        help="Pass if predictions are a subset of known-good IDs.",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=25,
        help="How many extra/missing IDs to include in sample outputs (default: 25).",
    )
    return p.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def require_columns(rows: List[Dict[str, str]], path: Path, columns: Set[str]) -> None:
    if not rows:
        raise SystemExit(f"[ERROR] CSV has zero rows: {path}")
    present = set(rows[0].keys())
    missing = columns - present
    if missing:
        raise SystemExit(f"[ERROR] CSV missing required columns {sorted(missing)}: {path}")


def collect_known_ids(rows: List[Dict[str, str]], comment_id_column: str) -> Set[str]:
    out: Set[str] = set()
    for r in rows:
        cid = (r.get(comment_id_column) or "").strip()
        if cid:
            out.add(cid)
    return out


def collect_pred_actionable_ids(
    rows: List[Dict[str, str]],
    comment_id_column: str,
    label_column: str,
    actionable_label: str,
) -> Set[str]:
    out: Set[str] = set()
    target = actionable_label.strip()
    for r in rows:
        cid = (r.get(comment_id_column) or "").strip()
        lbl = (r.get(label_column) or "").strip()
        if cid and lbl == target:
            out.add(cid)
    return out


def main() -> int:
    args = parse_args()

    predictions_path = Path(args.predictions_csv)
    known_path = Path(args.known_actionable_csv)
    out_json = Path(args.output_json) if args.output_json else None

    if not predictions_path.exists():
        raise SystemExit(f"[ERROR] missing predictions CSV: {predictions_path}")
    if not known_path.exists():
        raise SystemExit(f"[ERROR] missing known actionable CSV: {known_path}")

    pred_rows = read_rows(predictions_path)
    known_rows = read_rows(known_path)

    require_columns(
        pred_rows,
        predictions_path,
        {args.comment_id_column, args.label_column},
    )
    require_columns(
        known_rows,
        known_path,
        {args.comment_id_column},
    )

    known_ids = collect_known_ids(known_rows, args.comment_id_column)
    pred_actionable_ids = collect_pred_actionable_ids(
        pred_rows,
        args.comment_id_column,
        args.label_column,
        args.actionable_label,
    )

    intersection = known_ids & pred_actionable_ids
    missing = sorted(known_ids - pred_actionable_ids)
    extra = sorted(pred_actionable_ids - known_ids)

    precision_vs_known = (
        len(intersection) / len(pred_actionable_ids) if pred_actionable_ids else 0.0
    )
    recall_vs_known = len(intersection) / len(known_ids) if known_ids else 0.0
    exact_match = pred_actionable_ids == known_ids
    subset_match = pred_actionable_ids.issubset(known_ids)

    summary = {
        "predictions_csv": str(predictions_path),
        "known_actionable_csv": str(known_path),
        "comment_id_column": args.comment_id_column,
        "label_column": args.label_column,
        "actionable_label": args.actionable_label,
        "known_count": len(known_ids),
        "predicted_actionable_count": len(pred_actionable_ids),
        "intersection_count": len(intersection),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "precision_vs_known": precision_vs_known,
        "recall_vs_known": recall_vs_known,
        "exact_match": exact_match,
        "subset_match": subset_match,
        "sample_missing_ids": missing[: args.sample_size],
        "sample_extra_ids": extra[: args.sample_size],
    }

    print(json.dumps(summary, indent=2))

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[INFO] wrote summary: {out_json}")

    # Pass/fail policy
    if args.allow_subset_match:
        ok = subset_match
    else:
        ok = exact_match if args.require_exact_match else exact_match

    if ok:
        print("[PASS] actionable regression check passed")
        return 0

    print("[FAIL] actionable regression check failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
