#!/usr/env python3

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

INPUT = Path("raw_comments_labeled.clean.csv")
OUT_BINARY = Path("raw_comments_binary_response.csv")
OUT_TRAIN = Path("train_binary_response.csv")
OUT_VAL = Path("val_binary_response.csv")
OUT_TEST = Path("test_binary_response.csv")

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

SOURCE_TO_BINARY = {
    "question_or_request": "actionable",
    "actionable_feedback": "actionable",
    "moderation_risk": "non_actionable",
    "non_actionable_noise": "non_actionable",
}


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["comment_id", "text", "parent_text", "label"])
        w.writeheader()
        w.writerows(rows)


def stratified_split(
    rows,
    label_key: str,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
):
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")

    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r[label_key]].append(r)

    train, val, test = [], [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        n = len(group)

        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_test = n - n_train - n_val

        if n >= 3:
            if n_train == 0:
                n_train = 1
            if n_val == 0:
                n_val = 1
            n_test = n - n_train - n_val
            if n_test == 0:
                if n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1
                n_test = n - n_train - n_val

        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


rows = read_rows(INPUT)
if not rows:
    raise SystemExit("input CSV has no rows")

required_columns = {"comment_id", "text", "parent_text", "label"}
missing_columns = required_columns - set(rows[0].keys())
if missing_columns:
    raise SystemExit(f"missing required columns: {sorted(missing_columns)}")

binary_rows = []
for r in rows:
    original_label = (r.get("label") or "").strip()
    if original_label not in SOURCE_TO_BINARY:
        raise SystemExit(f"unexpected source label: {original_label}")
    binary_rows.append(
        {
            "comment_id": (r.get("comment_id") or "").strip(),
            "text": (r.get("text") or "").strip(),
            "parent_text": (r.get("parent_text") or "").strip(),
            "label": SOURCE_TO_BINARY[original_label],
        }
    )

write_rows(OUT_BINARY, binary_rows)
train, val, test = stratified_split(
    binary_rows, "label", SEED, TRAIN_FRAC, VAL_FRAC, TEST_FRAC
)
write_rows(OUT_TRAIN, train)
write_rows(OUT_VAL, val)
write_rows(OUT_TEST, test)


def show_dist(name, split_rows):
    c = Counter(r["label"] for r in split_rows)
    n = len(split_rows)
    print(
        f"[INFO] {name}: rows={n}, actionable={c.get('actionable', 0)}, non_actionable={c.get('non_actionable', 0)}"
    )


print("[INFO] wrote response-actionability binary datasets")
show_dist("full", binary_rows)
show_dist("train", train)
show_dist("val", val)
show_dist("test", test)
