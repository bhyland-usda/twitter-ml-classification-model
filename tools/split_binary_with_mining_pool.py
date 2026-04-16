#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

REQUIRED_COLUMNS = ("comment_id", "text", "parent_text", "label")
ALLOWED_LABELS = {"actionable", "non_actionable"}
SPLIT_ORDER = ("train", "val", "test", "mining")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically split a binary contextual dataset into "
            "train/val/test plus a reserved mining pool."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input binary CSV with columns: comment_id,text,parent_text,label",
    )
    parser.add_argument("--train-out", required=True, help="Output train CSV path")
    parser.add_argument("--val-out", required=True, help="Output validation CSV path")
    parser.add_argument("--test-out", required=True, help="Output test CSV path")
    parser.add_argument(
        "--mining-out",
        required=True,
        help="Output reserved mining-pool CSV path",
    )
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--mining-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_cell(value: str | None) -> str:
    if value is None:
        return ""
    return value.replace("\ufeff", "").replace("\x00", "").strip()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"[ERROR] CSV has no header row: {path}")

        header = [clean_cell(h) for h in reader.fieldnames]
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise SystemExit(
                f"[ERROR] input CSV missing required columns {missing}: {path}"
            )

        rows: List[Dict[str, str]] = []
        for line_number, raw in enumerate(reader, start=2):
            row = {k: clean_cell(v) for k, v in raw.items()}
            comment_id = row.get("comment_id", "")
            text = row.get("text", "")
            parent_text = row.get("parent_text", "")
            label = row.get("label", "")

            if not comment_id:
                raise SystemExit(f"[ERROR] line {line_number}: empty comment_id")
            if not text:
                raise SystemExit(f"[ERROR] line {line_number}: empty text")
            if not parent_text:
                raise SystemExit(f"[ERROR] line {line_number}: empty parent_text")
            if label not in ALLOWED_LABELS:
                raise SystemExit(
                    f"[ERROR] line {line_number}: invalid label '{label}', "
                    f"expected one of {sorted(ALLOWED_LABELS)}"
                )

            rows.append(
                {
                    "comment_id": comment_id,
                    "text": text,
                    "parent_text": parent_text,
                    "label": label,
                }
            )

    if not rows:
        raise SystemExit(f"[ERROR] input CSV has zero rows: {path}")

    return rows


def write_rows(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "comment_id": row["comment_id"],
                    "text": row["text"],
                    "parent_text": row["parent_text"],
                    "label": row["label"],
                }
            )


def validate_fractions(fractions: Dict[str, float]) -> None:
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise SystemExit(f"[ERROR] split fractions must sum to 1.0, got {total:.12f}")
    for name, value in fractions.items():
        if value <= 0.0:
            raise SystemExit(f"[ERROR] {name}-frac must be > 0, got {value}")


def allocate_counts(n: int, fractions: Dict[str, float]) -> Dict[str, int]:
    active_names = [name for name in SPLIT_ORDER if fractions[name] > 0.0]
    if n == 0:
        return {name: 0 for name in SPLIT_ORDER}

    counts = {name: 0 for name in SPLIT_ORDER}

    if n >= len(active_names):
        # Reserve at least one row per active split when feasible.
        for name in active_names:
            counts[name] = 1
        remaining = n - len(active_names)
        if remaining == 0:
            return counts

        raw_targets = {
            name: remaining * fractions[name] / sum(fractions[a] for a in active_names)
            for name in active_names
        }
    else:
        # Not enough rows to populate all active splits.
        # Give one row to the highest-priority fractions first.
        ranked = sorted(
            active_names,
            key=lambda name: (fractions[name], -SPLIT_ORDER.index(name)),
            reverse=True,
        )
        for name in ranked[:n]:
            counts[name] = 1
        return counts

    floors = {name: math.floor(raw_targets[name]) for name in active_names}
    for name in active_names:
        counts[name] += floors[name]

    assigned = sum(counts.values())
    remainder = n - assigned
    if remainder < 0:
        raise RuntimeError("internal error: allocated too many rows")

    ranked_by_remainder = sorted(
        active_names,
        key=lambda name: (
            raw_targets[name] - floors[name],
            fractions[name],
            -SPLIT_ORDER.index(name),
        ),
        reverse=True,
    )

    for name in ranked_by_remainder[:remainder]:
        counts[name] += 1

    if sum(counts.values()) != n:
        raise RuntimeError(
            f"internal error: count allocation mismatch (expected {n}, got {sum(counts.values())})"
        )

    return counts


def stratified_split(
    rows: Sequence[Dict[str, str]],
    fractions: Dict[str, float],
    seed: int,
) -> Dict[str, List[Dict[str, str]]]:
    by_label: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(dict(row))

    rng = random.Random(seed)
    split_rows: Dict[str, List[Dict[str, str]]] = {name: [] for name in SPLIT_ORDER}

    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        counts = allocate_counts(len(group), fractions)

        start = 0
        for split_name in SPLIT_ORDER:
            end = start + counts[split_name]
            split_rows[split_name].extend(group[start:end])
            start = end

        if start != len(group):
            raise RuntimeError(
                f"internal error: split slicing mismatch for label={label}"
            )

    for split_name in SPLIT_ORDER:
        rng.shuffle(split_rows[split_name])

    return split_rows


def print_distribution(name: str, rows: Sequence[Dict[str, str]]) -> None:
    counts = Counter(row["label"] for row in rows)
    print(
        f"[INFO] {name}: rows={len(rows)} "
        f"actionable={counts.get('actionable', 0)} "
        f"non_actionable={counts.get('non_actionable', 0)}"
    )


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"[ERROR] input file not found: {input_path}")

    fractions = {
        "train": args.train_frac,
        "val": args.val_frac,
        "test": args.test_frac,
        "mining": args.mining_frac,
    }
    validate_fractions(fractions)

    rows = load_rows(input_path)
    splits = stratified_split(rows, fractions, args.seed)

    outputs = {
        "train": Path(args.train_out),
        "val": Path(args.val_out),
        "test": Path(args.test_out),
        "mining": Path(args.mining_out),
    }

    for split_name, path in outputs.items():
        if not splits[split_name]:
            raise SystemExit(
                f"[ERROR] {split_name} split would be empty; adjust fractions or dataset size"
            )
        write_rows(path, splits[split_name])

    print("[INFO] wrote binary contextual splits with reserved mining pool")
    print(f"[INFO] input={input_path}")
    print(
        "[INFO] fractions="
        f"train:{fractions['train']:.2f} "
        f"val:{fractions['val']:.2f} "
        f"test:{fractions['test']:.2f} "
        f"mining:{fractions['mining']:.2f}"
    )
    print(f"[INFO] seed={args.seed}")
    print_distribution("full", rows)
    for split_name in SPLIT_ORDER:
        print_distribution(split_name, splits[split_name])
        print(f"[INFO] {split_name}_out={outputs[split_name]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
