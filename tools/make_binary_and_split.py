from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_TO_BINARY = {
    "moderation_risk": "actionable",
    "question_or_request": "actionable",
    "actionable_feedback": "actionable",
    "non_actionable_noise": "non_actionable",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="raw_comments_labeled.clean.csv")
    p.add_argument("--out-binary", required=True, help="binary labeled csv")
    p.add_argument("--train-out", required=True, help="train split csv")
    p.add_argument("--val-out", required=True, help="val split csv")
    p.add_argument("--test-out", required=True, help="test split csv")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def stratified_split(rows, label_col, train_frac, val_frac, test_frac, seed):
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Fractions must sum to 1.0; got {total}")

    by_label = defaultdict(list)
    for r in rows:
        by_label[r[label_col]].append(r)

    rng = random.Random(seed)
    train, val, test = [], [], []

    for label, group in by_label.items():
        rng.shuffle(group)
        n = len(group)

        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_test = n - n_train - n_val

        # ensure no split is empty when possible
        if n >= 3:
            if n_train == 0:
                n_train = 1
            if n_val == 0:
                n_val = 1
            n_test = n - n_train - n_val
            if n_test == 0:
                if n_train > n_val and n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1
                n_test = 1

        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def main():
    args = parse_args()

    inp = Path(args.input)
    rows = read_csv(inp)
    required = {"comment_id", "text", "parent_text", "label"}
    if not rows:
        raise SystemExit("Input CSV has no rows")
    missing = required - set(rows[0].keys())
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    binary_rows = []
    for r in rows:
        src = (r.get("label") or "").strip()
        if src not in SOURCE_TO_BINARY:
            raise SystemExit(f"Unexpected label: {src}")
        binary_rows.append(
            {
                "comment_id": (r.get("comment_id") or "").strip(),
                "text": (r.get("text") or "").strip(),
                "parent_text": (r.get("parent_text") or "").strip(),
                "label": SOURCE_TO_BINARY[src],
            }
        )

    # write full binary file
    write_csv(
        Path(args.out_binary),
        binary_rows,
        ["comment_id", "text", "parent_text", "label"],
    )

    # stratified splits
    train, val, test = stratified_split(
        binary_rows,
        "label",
        args.train_frac,
        args.val_frac,
        args.test_frac,
        args.seed,
    )

    write_csv(
        Path(args.train_out), train, ["comment_id", "text", "parent_text", "label"]
    )
    write_csv(Path(args.val_out), val, ["comment_id", "text", "parent_text", "label"])
    write_csv(Path(args.test_out), test, ["comment_id", "text", "parent_text", "label"])

    def dist(name, split_rows):
        c = Counter(r["label"] for r in split_rows)
        n = len(split_rows)
        print(
            f"[INFO] {name}: rows={n}, actionable={c.get('actionable', 0)}, non_actionable={c.get('non_actionable', 0)}"
        )

    print("[INFO] wrote binary dataset + splits")
    dist("full", binary_rows)
    dist("train", train)
    dist("val", val)
    dist("test", test)


if __name__ == "__main__":
    main()
