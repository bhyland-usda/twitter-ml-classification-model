#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

REQUIRED_TRAIN_COLUMNS = {"comment_id", "text", "parent_text", "label"}
REQUIRED_LOG_COLUMNS = {
    "comment_id",
    "text",
    "parent_text",
    "predicted_label",
    "confidence",
    "guardrail_triggered",
    "guardrail_decision_source",
    "final_label",
}

DEFAULT_INCLUDED_REASONS = [
    "safety_veto",
    "scope_veto",
    "missing_concrete_action",
    "question_without_concrete_action",
    "missing_action_cue",
    "civility_gate",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Deterministically mine hard negatives from guardrailed inference logs "
            "and append them to an existing train split for model fine-tuning."
        )
    )
    p.add_argument(
        "--train-csv",
        required=True,
        help="Existing train CSV (comment_id,text,parent_text,label)",
    )
    p.add_argument(
        "--val-csv", default="", help="Optional val CSV used for leakage exclusion"
    )
    p.add_argument(
        "--test-csv", default="", help="Optional test CSV used for leakage exclusion"
    )
    p.add_argument(
        "--non-actionable-log",
        required=True,
        help="Guardrail non_actionable_log CSV from latest strict inference run",
    )
    p.add_argument(
        "--output-train",
        required=True,
        help="Output augmented train CSV path (comment_id,text,parent_text,label)",
    )
    p.add_argument(
        "--summary-out",
        default="",
        help="Optional output JSON summary path (defaults next to output train)",
    )
    p.add_argument(
        "--include-reasons",
        default=",".join(DEFAULT_INCLUDED_REASONS),
        help="Comma-separated guardrail decision sources to mine (default is strict hard-negative set)",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.90,
        help="Minimum model confidence for mined hard negatives (default: 0.90)",
    )
    p.add_argument(
        "--max-added",
        type=int,
        default=250,
        help="Maximum mined hard negatives to append (default: 250)",
    )
    p.add_argument(
        "--max-per-reason",
        type=int,
        default=120,
        help="Maximum candidates selected per decision reason before global fill (default: 120)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic tie-breaking (default: 42)",
    )
    return p.parse_args()


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _parse_bool(v: str) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_float(v: str, default: float = 0.0) -> float:
    try:
        x = float(v)
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x
    except Exception:
        return default


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ensure_columns(rows: List[Dict[str, str]], required: Set[str], path: Path) -> None:
    if not rows:
        raise SystemExit(f"[ERROR] CSV has zero rows: {path}")
    cols = set(rows[0].keys())
    missing = required - cols
    if missing:
        raise SystemExit(
            f"[ERROR] CSV missing required columns {sorted(missing)}: {path}"
        )


def make_exclusion_sets(
    train_rows: List[Dict[str, str]],
    val_rows: List[Dict[str, str]],
    test_rows: List[Dict[str, str]],
) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    exclude_ids: Set[str] = set()
    exclude_texts: Set[Tuple[str, str]] = set()

    for rows in (train_rows, val_rows, test_rows):
        for r in rows:
            cid = (r.get("comment_id") or "").strip()
            parent_txt = _norm_text(r.get("parent_text") or "")
            txt = _norm_text(r.get("text") or "")
            if cid:
                exclude_ids.add(cid)
            if parent_txt or txt:
                exclude_texts.add((parent_txt, txt))

    return exclude_ids, exclude_texts


def split_sources(s: str) -> Set[str]:
    return {x.strip() for x in (s or "").split("|") if x.strip()}


def candidate_sort_key(c: Dict[str, str]) -> Tuple[float, str, str, str]:
    conf = _parse_float(c.get("confidence", "0"), 0.0)
    cid = (c.get("comment_id") or "").strip()
    parent_txt = _norm_text(c.get("parent_text") or "")
    txt = _norm_text(c.get("text") or "")
    return (conf, cid, parent_txt, txt)


def select_candidates(
    candidates: List[Dict[str, str]],
    include_reasons: List[str],
    max_per_reason: int,
    max_added: int,
    seed: int,
) -> Tuple[List[Dict[str, str]], Counter[str]]:
    rng = random.Random(seed)

    # deterministic stable base ordering
    ordered = sorted(candidates, key=candidate_sort_key, reverse=True)

    by_reason: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for c in ordered:
        sources = split_sources(c.get("guardrail_decision_source", ""))
        hit = [r for r in include_reasons if r in sources]
        if not hit:
            continue
        # Assign primary reason by configured priority order
        primary = hit[0]
        by_reason[primary].append(c)

    selected: List[Dict[str, str]] = []
    selected_keys: Set[Tuple[str, str, str]] = set()
    reason_counts: Counter[str] = Counter()

    def key_of(row: Dict[str, str]) -> Tuple[str, str, str]:
        return (
            (row.get("comment_id") or "").strip(),
            _norm_text(row.get("parent_text") or ""),
            _norm_text(row.get("text") or ""),
        )

    # round 1: per-reason cap
    for reason in include_reasons:
        bucket = list(by_reason.get(reason, []))
        # deterministic tie noise within equal confidence/comment_id clusters
        rng.shuffle(bucket)
        bucket.sort(key=candidate_sort_key, reverse=True)

        taken = 0
        for row in bucket:
            if len(selected) >= max_added or taken >= max_per_reason:
                break
            k = key_of(row)
            if k in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(k)
            reason_counts[reason] += 1
            taken += 1

    # round 2: global fill up to max_added
    if len(selected) < max_added:
        remaining = []
        for row in ordered:
            k = key_of(row)
            if k not in selected_keys:
                remaining.append(row)

        for row in remaining:
            if len(selected) >= max_added:
                break
            sources = split_sources(row.get("guardrail_decision_source", ""))
            hit = [r for r in include_reasons if r in sources]
            if not hit:
                continue
            reason = hit[0]
            selected.append(row)
            selected_keys.add(key_of(row))
            reason_counts[reason] += 1

    return selected, reason_counts


def main() -> int:
    args = parse_args()

    train_path = Path(args.train_csv)
    val_path = Path(args.val_csv) if args.val_csv else None
    test_path = Path(args.test_csv) if args.test_csv else None
    log_path = Path(args.non_actionable_log)
    out_train_path = Path(args.output_train)

    if not train_path.exists():
        raise SystemExit(f"[ERROR] missing train CSV: {train_path}")
    if val_path and not val_path.exists():
        raise SystemExit(f"[ERROR] missing val CSV: {val_path}")
    if test_path and not test_path.exists():
        raise SystemExit(f"[ERROR] missing test CSV: {test_path}")
    if not log_path.exists():
        raise SystemExit(f"[ERROR] missing non-actionable log CSV: {log_path}")

    include_reasons = [x.strip() for x in args.include_reasons.split(",") if x.strip()]
    if not include_reasons:
        raise SystemExit("[ERROR] --include-reasons resolved to empty list")
    include_reasons = list(dict.fromkeys(include_reasons))  # dedupe, preserve order

    if not (0.0 <= args.min_confidence <= 1.0):
        raise SystemExit("[ERROR] --min-confidence must be in [0,1]")
    if args.max_added <= 0:
        raise SystemExit("[ERROR] --max-added must be > 0")
    if args.max_per_reason <= 0:
        raise SystemExit("[ERROR] --max-per-reason must be > 0")

    train_rows = read_csv(train_path)
    ensure_columns(train_rows, REQUIRED_TRAIN_COLUMNS, train_path)

    val_rows: List[Dict[str, str]] = []
    test_rows: List[Dict[str, str]] = []
    if val_path:
        val_rows = read_csv(val_path)
        ensure_columns(val_rows, REQUIRED_TRAIN_COLUMNS, val_path)
    if test_path:
        test_rows = read_csv(test_path)
        ensure_columns(test_rows, REQUIRED_TRAIN_COLUMNS, test_path)

    log_rows = read_csv(log_path)
    ensure_columns(log_rows, REQUIRED_LOG_COLUMNS, log_path)

    exclude_ids, exclude_texts = make_exclusion_sets(train_rows, val_rows, test_rows)

    eligible: List[Dict[str, str]] = []
    filtered_counts = Counter()

    for r in log_rows:
        predicted = (r.get("predicted_label") or "").strip()
        final_label = (r.get("final_label") or "").strip()
        triggered = _parse_bool(r.get("guardrail_triggered") or "")
        conf = _parse_float(r.get("confidence") or "0", 0.0)
        cid = (r.get("comment_id") or "").strip()
        parent_text_raw = (r.get("parent_text") or "").strip()
        txt_raw = (r.get("text") or "").strip()
        parent_text_norm = _norm_text(parent_text_raw)
        txt_norm = _norm_text(txt_raw)
        sources = split_sources(r.get("guardrail_decision_source") or "")

        if predicted != "actionable":
            filtered_counts["not_predicted_actionable"] += 1
            continue
        if final_label != "non_actionable":
            filtered_counts["not_final_non_actionable"] += 1
            continue
        if not triggered:
            filtered_counts["not_guardrail_triggered"] += 1
            continue
        if conf < args.min_confidence:
            filtered_counts["below_min_confidence"] += 1
            continue
        if not (sources & set(include_reasons)):
            filtered_counts["no_included_reason"] += 1
            continue
        if cid and cid in exclude_ids:
            filtered_counts["already_in_splits_by_id"] += 1
            continue
        if (parent_text_norm or txt_norm) and (
            parent_text_norm,
            txt_norm,
        ) in exclude_texts:
            filtered_counts["already_in_splits_by_text"] += 1
            continue
        if not txt_raw:
            filtered_counts["empty_text"] += 1
            continue
        if not parent_text_raw:
            filtered_counts["empty_parent_text"] += 1
            continue

        eligible.append(
            {
                "comment_id": cid,
                "text": txt_raw,
                "parent_text": parent_text_raw,
                "predicted_label": predicted,
                "confidence": f"{conf:.6f}",
                "guardrail_decision_source": "|".join(sorted(sources)),
                "guardrail_rule_ids": (r.get("guardrail_rule_ids") or "").strip(),
                "guardrail_categories": (r.get("guardrail_categories") or "").strip(),
                "final_label": final_label,
            }
        )

    selected, reason_counts = select_candidates(
        candidates=eligible,
        include_reasons=include_reasons,
        max_per_reason=args.max_per_reason,
        max_added=args.max_added,
        seed=args.seed,
    )

    added_rows = []
    seen_train_keys = {
        (
            (r.get("comment_id") or "").strip(),
            _norm_text(r.get("parent_text") or ""),
            _norm_text(r.get("text") or ""),
        )
        for r in train_rows
    }
    dedup_skips = 0

    for row in selected:
        key = (
            (row.get("comment_id") or "").strip(),
            _norm_text(row.get("parent_text") or ""),
            _norm_text(row.get("text") or ""),
        )
        if key in seen_train_keys:
            dedup_skips += 1
            continue
        added_rows.append(
            {
                "comment_id": (row.get("comment_id") or "").strip(),
                "text": (row.get("text") or "").strip(),
                "parent_text": (row.get("parent_text") or "").strip(),
                "label": "non_actionable",
            }
        )
        seen_train_keys.add(key)

    augmented_train = list(train_rows) + added_rows

    out_train_path.parent.mkdir(parents=True, exist_ok=True)
    with out_train_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["comment_id", "text", "parent_text", "label"]
        )
        writer.writeheader()
        writer.writerows(augmented_train)

    # summary
    summary_path = (
        Path(args.summary_out)
        if args.summary_out
        else out_train_path.with_suffix(".summary.json")
    )
    label_counts_before = Counter((r.get("label") or "").strip() for r in train_rows)
    label_counts_after = Counter(
        (r.get("label") or "").strip() for r in augmented_train
    )

    summary = {
        "seed": args.seed,
        "train_csv": str(train_path),
        "val_csv": str(val_path) if val_path else "",
        "test_csv": str(test_path) if test_path else "",
        "non_actionable_log_csv": str(log_path),
        "output_train_csv": str(out_train_path),
        "included_reasons": include_reasons,
        "min_confidence": args.min_confidence,
        "max_added": args.max_added,
        "max_per_reason": args.max_per_reason,
        "train_rows_before": len(train_rows),
        "train_rows_after": len(augmented_train),
        "added_hard_negatives": len(added_rows),
        "eligible_candidates": len(eligible),
        "selected_candidates_pre_dedup": len(selected),
        "dedup_skips_against_train": dedup_skips,
        "reason_counts_added": dict(reason_counts),
        "label_counts_before": dict(label_counts_before),
        "label_counts_after": dict(label_counts_after),
        "filter_counts": dict(filtered_counts),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] eligible_candidates={len(eligible)}")
    print(f"[INFO] selected_candidates_pre_dedup={len(selected)}")
    print(f"[INFO] added_hard_negatives={len(added_rows)}")
    print(f"[INFO] train_rows_before={len(train_rows)}")
    print(f"[INFO] train_rows_after={len(augmented_train)}")
    print(f"[INFO] output_train={out_train_path}")
    print(f"[INFO] summary={summary_path}")
    print(f"[INFO] reason_counts_added={dict(reason_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
