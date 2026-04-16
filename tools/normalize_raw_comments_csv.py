#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

REQUIRED_COLUMNS = {"comment_id", "text", "parent_text"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize and deduplicate a raw comments CSV. "
            "The script trims fields, removes BOM/null chars, drops rows with empty "
            "comment_id/text/parent_text, and writes a clean CSV."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input raw CSV (must include comment_id, text, and parent_text columns).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output normalized CSV.",
    )
    parser.add_argument(
        "--dedupe-by-id-only",
        action="store_true",
        help=(
            "If set, deduplicate only by comment_id (keeps first seen id). "
            "Default dedupe key is (comment_id, parent_text, text)."
        ),
    )
    return parser.parse_args()


def clean_cell(value: str | None) -> str:
    if value is None:
        return ""
    # Remove common file contamination artifacts and trim.
    return value.replace("\ufeff", "").replace("\x00", "").strip()


def load_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {path}")

        fieldnames = [clean_cell(c) for c in reader.fieldnames]
        rows: List[Dict[str, str]] = []
        for raw in reader:
            row = {k: clean_cell(v) for k, v in raw.items()}
            rows.append(row)

    return rows, fieldnames


def write_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"[ERROR] input file not found: {input_path}")

    rows, fieldnames = load_rows(input_path)

    missing = REQUIRED_COLUMNS - set(fieldnames)
    if missing:
        raise SystemExit(
            f"[ERROR] missing required columns: {sorted(missing)}. "
            f"Found columns: {fieldnames}"
        )

    seen: set[tuple[str, str, str] | str] = set()
    normalized: List[Dict[str, str]] = []

    dropped_empty = 0
    dropped_dupe = 0

    for row in rows:
        comment_id = clean_cell(row.get("comment_id"))
        text = clean_cell(row.get("text"))
        parent_text = clean_cell(row.get("parent_text"))

        if not comment_id or not text or not parent_text:
            dropped_empty += 1
            continue

        # Keep all original columns, but ensure required ones are normalized.
        out_row = {k: clean_cell(row.get(k, "")) for k in fieldnames}
        out_row["comment_id"] = comment_id
        out_row["text"] = text
        out_row["parent_text"] = parent_text

        if args.dedupe_by_id_only:
            key: tuple[str, str, str] | str = comment_id
        else:
            key = (comment_id, parent_text, text)

        if key in seen:
            dropped_dupe += 1
            continue

        seen.add(key)
        normalized.append(out_row)

    write_rows(output_path, fieldnames, normalized)

    print(f"[INFO] input={input_path}")
    print(f"[INFO] output={output_path}")
    print(f"[INFO] input_rows={len(rows)}")
    print(f"[INFO] output_rows={len(normalized)}")
    print(f"[INFO] dropped_empty_rows={dropped_empty}")
    print(f"[INFO] dropped_duplicate_rows={dropped_dupe}")
    print(
        f"[INFO] dedupe_mode={'comment_id' if args.dedupe_by_id_only else '(comment_id,parent_text,text)'}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
