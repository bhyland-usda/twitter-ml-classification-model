#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare student model predictions to teacher (guardrailed) output and "
            "report agreement, actionable-set overlap, and mismatch samples."
        )
    )
    p.add_argument(
        "--teacher-csv",
        required=True,
        help="Teacher CSV path (typically strict guardrailed output).",
    )
    p.add_argument(
        "--student-csv",
        required=True,
        help="Student CSV path (typically distilled model predictions).",
    )
    p.add_argument(
        "--comment-id-column",
        default="comment_id",
        help="Comment ID column name shared by both CSVs (default: comment_id).",
    )
    p.add_argument(
        "--teacher-label-column",
        default="final_label",
        help="Label column in teacher CSV (default: final_label).",
    )
    p.add_argument(
        "--student-label-column",
        default="predicted_label",
        help="Label column in student CSV (default: predicted_label).",
    )
    p.add_argument(
        "--actionable-label",
        default="actionable",
        help="Label value treated as actionable (default: actionable).",
    )
    p.add_argument(
        "--output-json",
        default="",
        help="Optional path to write JSON summary.",
    )
    p.add_argument(
        "--mismatches-csv",
        default="",
        help="Optional path to write per-row label mismatches.",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=25,
        help="Max sample size for mismatch and set-diff ID lists (default: 25).",
    )
    return p.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"[ERROR] CSV has no header row: {path}")
        return list(reader)


def require_columns(
    rows: List[Dict[str, str]], path: Path, required: List[str]
) -> None:
    if not rows:
        raise SystemExit(f"[ERROR] CSV has zero rows: {path}")
    cols = set(rows[0].keys())
    missing = [c for c in required if c not in cols]
    if missing:
        raise SystemExit(f"[ERROR] CSV missing required columns {missing}: {path}")


def build_label_map(
    rows: List[Dict[str, str]],
    id_col: str,
    label_col: str,
    csv_path: Path,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    dupes: List[str] = []

    for r in rows:
        cid = (r.get(id_col) or "").strip()
        lbl = (r.get(label_col) or "").strip()
        if not cid:
            continue
        if cid in out:
            dupes.append(cid)
            continue
        out[cid] = lbl

    if dupes:
        sample = sorted(set(dupes))[:10]
        raise SystemExit(
            f"[ERROR] duplicate {id_col} values in {csv_path}: sample={sample}"
        )

    return out


def write_mismatches_csv(
    path: Path,
    mismatches: List[Tuple[str, str, str]],
    id_col: str,
    teacher_col: str,
    student_col: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[id_col, teacher_col, student_col],
        )
        writer.writeheader()
        for cid, t, s in mismatches:
            writer.writerow(
                {
                    id_col: cid,
                    teacher_col: t,
                    student_col: s,
                }
            )


def main() -> int:
    args = parse_args()

    teacher_path = Path(args.teacher_csv)
    student_path = Path(args.student_csv)

    if not teacher_path.exists():
        raise SystemExit(f"[ERROR] missing teacher CSV: {teacher_path}")
    if not student_path.exists():
        raise SystemExit(f"[ERROR] missing student CSV: {student_path}")

    teacher_rows = read_rows(teacher_path)
    student_rows = read_rows(student_path)

    require_columns(
        teacher_rows,
        teacher_path,
        [args.comment_id_column, args.teacher_label_column],
    )
    require_columns(
        student_rows,
        student_path,
        [args.comment_id_column, args.student_label_column],
    )

    teacher_map = build_label_map(
        teacher_rows, args.comment_id_column, args.teacher_label_column, teacher_path
    )
    student_map = build_label_map(
        student_rows, args.comment_id_column, args.student_label_column, student_path
    )

    teacher_ids = set(teacher_map.keys())
    student_ids = set(student_map.keys())
    common_ids = sorted(teacher_ids & student_ids)

    teacher_only_ids = sorted(teacher_ids - student_ids)
    student_only_ids = sorted(student_ids - teacher_ids)

    match_count = 0
    mismatches: List[Tuple[str, str, str]] = []

    for cid in common_ids:
        t = teacher_map[cid]
        s = student_map[cid]
        if t == s:
            match_count += 1
        else:
            mismatches.append((cid, t, s))

    actionable_label = args.actionable_label.strip()
    teacher_actionable = {
        cid for cid in common_ids if teacher_map[cid] == actionable_label
    }
    student_actionable = {
        cid for cid in common_ids if student_map[cid] == actionable_label
    }

    agreement = (match_count / len(common_ids)) if common_ids else 0.0

    summary = {
        "teacher_csv": str(teacher_path),
        "student_csv": str(student_path),
        "comment_id_column": args.comment_id_column,
        "teacher_label_column": args.teacher_label_column,
        "student_label_column": args.student_label_column,
        "actionable_label": actionable_label,
        "teacher_rows": len(teacher_rows),
        "student_rows": len(student_rows),
        "teacher_unique_ids": len(teacher_ids),
        "student_unique_ids": len(student_ids),
        "common_ids": len(common_ids),
        "teacher_only_ids": len(teacher_only_ids),
        "student_only_ids": len(student_only_ids),
        "agreement": agreement,
        "mismatches": len(mismatches),
        "teacher_actionable": len(teacher_actionable),
        "student_actionable": len(student_actionable),
        "exact_actionable_set_match": teacher_actionable == student_actionable,
        "actionable_intersection": len(teacher_actionable & student_actionable),
        "teacher_only_actionable": len(teacher_actionable - student_actionable),
        "student_only_actionable": len(student_actionable - teacher_actionable),
        "teacher_only_id_sample": teacher_only_ids[: args.sample_size],
        "student_only_id_sample": student_only_ids[: args.sample_size],
        "mismatch_id_sample": [cid for cid, _, _ in mismatches[: args.sample_size]],
        "teacher_only_actionable_sample": sorted(
            teacher_actionable - student_actionable
        )[: args.sample_size],
        "student_only_actionable_sample": sorted(
            student_actionable - teacher_actionable
        )[: args.sample_size],
    }

    print(json.dumps(summary, indent=2))

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[INFO] wrote summary JSON: {out_json}")

    if args.mismatches_csv:
        mismatch_path = Path(args.mismatches_csv)
        write_mismatches_csv(
            mismatch_path,
            mismatches,
            args.comment_id_column,
            args.teacher_label_column,
            args.student_label_column,
        )
        print(f"[INFO] wrote mismatches CSV: {mismatch_path}")

    if (
        summary["teacher_only_ids"] == 0
        and summary["student_only_ids"] == 0
        and summary["mismatches"] == 0
    ):
        print("[PASS] parity check passed")
        return 0

    print("[FAIL] parity check failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
