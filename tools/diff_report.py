#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base prediction CSV vs guardrailed CSV and report changed rows."
    )
    parser.add_argument(
        "--before",
        required=True,
        help="Base prediction CSV (e.g., raw_comments_export_predictions_response.csv)",
    )
    parser.add_argument(
        "--after",
        required=True,
        help="Guardrailed prediction CSV (e.g., raw_comments_export_predictions_response_guardrailed.csv)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV containing only changed rows",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=30,
        help="Max changed rows to print to stdout (default: 30)",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        return list(reader)


def main() -> int:
    args = parse_args()

    before_path = Path(args.before)
    after_path = Path(args.after)
    output_path = Path(args.output)

    if not before_path.exists():
        raise SystemExit(f"[ERROR] before file not found: {before_path}")
    if not after_path.exists():
        raise SystemExit(f"[ERROR] after file not found: {after_path}")

    before_rows = load_csv(before_path)
    after_rows = load_csv(after_path)

    if not before_rows:
        raise SystemExit("[ERROR] before file has zero rows")
    if not after_rows:
        raise SystemExit("[ERROR] after file has zero rows")

    before_map = {}
    for row in before_rows:
        comment_id = (row.get("comment_id") or "").strip()
        if not comment_id:
            continue
        before_map[comment_id] = row

    changed_rows: list[dict[str, str]] = []
    transition_counter = Counter()

    for after_row in after_rows:
        comment_id = (after_row.get("comment_id") or "").strip()
        if not comment_id:
            continue
        before_row = before_map.get(comment_id)
        if before_row is None:
            continue

        before_label = (before_row.get("predicted_label") or "").strip()
        after_label = (after_row.get("final_label") or "").strip()

        if before_label != after_label:
            transition_counter[(before_label, after_label)] += 1
            changed_rows.append(
                {
                    "comment_id": comment_id,
                    "before_label": before_label,
                    "after_label": after_label,
                    "before_confidence": (before_row.get("confidence") or "").strip(),
                    "after_confidence": (after_row.get("final_confidence") or "").strip(),
                    "guardrail_triggered": (after_row.get("guardrail_triggered") or "").strip(),
                    "guardrail_rule_ids": (after_row.get("guardrail_rule_ids") or "").strip(),
                    "guardrail_categories": (after_row.get("guardrail_categories") or "").strip(),
                    "text": (after_row.get("text") or "").strip(),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "comment_id",
                "before_label",
                "after_label",
                "before_confidence",
                "after_confidence",
                "guardrail_triggered",
                "guardrail_rule_ids",
                "guardrail_categories",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerows(changed_rows)

    print(f"[INFO] before_file={before_path}")
    print(f"[INFO] after_file={after_path}")
    print(f"[INFO] output_file={output_path}")
    print(f"[INFO] total_before_rows={len(before_rows)}")
    print(f"[INFO] total_after_rows={len(after_rows)}")
    print(f"[INFO] changed_rows={len(changed_rows)}")

    if transition_counter:
        print("[INFO] label_transitions:")
        for (before_label, after_label), count in sorted(
            transition_counter.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        ):
            print(f"  - {before_label} -> {after_label}: {count}")
    else:
        print("[INFO] no label changes detected")

    max_print = max(0, args.max_print)
    if changed_rows and max_print > 0:
        print(f"[INFO] sample_changed_rows (up to {max_print}):")
        for i, row in enumerate(changed_rows[:max_print], start=1):
            snippet = row["text"].replace("\n", " ")[:200]
            print(
                f"{i:02d}. id={row['comment_id']} {row['before_label']} -> {row['after_label']} "
                f"rules={row['guardrail_rule_ids']} text={snippet}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
